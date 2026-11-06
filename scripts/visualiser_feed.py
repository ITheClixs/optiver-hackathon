#!/usr/bin/env python3
"""Capture the dashboard feed and keep run/participants.json fresh. Read-only, no exchange login.

    uv run --extra feed python scripts/visualiser_feed.py --host <event-host> --me "<our team name>"

Autopilot reads the participants file, so run this beside the bot from the moment the host is known.
The feed is a separate connection from the exchange session, but no one has confirmed it cannot
disturb that session. Check `reconnects` in run/status.json during practice before relying on it.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.participants import ParticipantBook
from otk.recorder import JsonlRecorder
from otk.visualiser import (
    DEFAULT_MODULES,
    FeedSink,
    feed_url,
    hello_messages,
    run_feed,
    websocket_frames,
)

logger = logging.getLogger("otk.visualiser_feed")


def write_report(path: Path, book: ParticipantBook) -> None:
    """Replaced atomically, so a reader never sees half a file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(book.report(), indent=2, default=str))
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="dashboard host, for example 2026-x.optibook.net")
    parser.add_argument("--username", help="adds ?username= to the URL, if the event needs it")
    parser.add_argument("--me", help="our team's name as the dashboard shows it")
    parser.add_argument("--modules", default=",".join(DEFAULT_MODULES))
    parser.add_argument("--no-subscribe", action="store_true", help="send SELECT only")
    parser.add_argument("--cookie", help="browser session cookie, if the feed turns out to need login")
    parser.add_argument("--subprotocol", help="websocket subprotocol, as the Delta Dash player socket used")
    parser.add_argument("--out", default="captures/feed.jsonl.gz")
    parser.add_argument("--participants", default="run/participants.json")
    parser.add_argument("--every", type=float, default=5.0, help="seconds between participant reports")
    parser.add_argument("--seconds", type=float, help="stop after this long; default runs until killed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    modules = [name.strip() for name in args.modules.split(",") if name.strip()]
    url = feed_url(args.host, args.username)
    hello = hello_messages(args.username, modules, subscribe=not args.no_subscribe)
    report_path = Path(args.participants)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    recorder = JsonlRecorder(args.out)
    book = ParticipantBook(me=args.me)
    sink = FeedSink(recorder.record, book)

    stopping: list[bool] = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
    signal.signal(signal.SIGINT, lambda *_: stopping.append(True))
    deadline = time.monotonic() + args.seconds if args.seconds else float("inf")
    state = {"next_report": 0.0}

    def on_frame(frame: bytes | str) -> None:
        sink.on_frame(frame)
        if time.monotonic() >= state["next_report"]:
            state["next_report"] = time.monotonic() + args.every
            try:
                write_report(report_path, book)
            except (OSError, TypeError, ValueError):
                logger.exception("could not write %s", report_path)

    try:
        run_feed(
            lambda: websocket_frames(url, hello, cookie=args.cookie, subprotocol=args.subprotocol),
            on_frame,
            lambda: bool(stopping) or time.monotonic() >= deadline,
        )
    finally:
        recorder.close()
    print(f"{sink.frames} frames, {sink.undecoded} undecoded, {sink.normaliser.errors} parse errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
