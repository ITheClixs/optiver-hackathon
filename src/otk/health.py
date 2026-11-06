"""Is the bot really trading? One verdict for the Driver, from files alone, so it needs no login and no tokens.

A bot can be up and connected and still do nothing, with no error in the log: no books, no instruments, every order
rejected. `python optibook_bot.py --health` names those. Each comes with what to do next.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

GREEN, AMBER, RED = "GREEN", "AMBER", "RED"
HEALTH_EXIT_CODES = {GREEN: 0, AMBER: 1, RED: 2}
HEALTH_STALE_S = 20.0
HEALTH_QUIET_S = 60.0
HEALTH_RECONNECTS = 3
HEALTH_ERRORS = 5
HEALTH_MIN_ORDERS = 10
HEALTH_REJECT_SHARE = 0.5
BACKUP_PLAN = "the backup plan: restart the bot, and if it still cannot trade, run a minimal quoting bot instead"


def _problem(status: Mapping, age_s: float) -> tuple[str, str] | None:
    """The worst thing wrong, or None. RED means it is not trading; AMBER means look now."""
    sent, rejects = status.get("orders_sent", 0), status.get("rejects", 0)
    kill = status.get("kill") or {}
    if age_s > HEALTH_STALE_S:
        return RED, f"not running: status is {age_s:.0f} s old"
    if not status.get("connected"):
        return RED, "not connected to the exchange"
    if kill.get("latched"):
        return RED, f"stopped for good by the kill switch: {kill.get('reason')}"
    if status.get("elapsed_s", 0) > HEALTH_QUIET_S and not sent:
        return RED, "connected but has sent no orders: check the instruments and books in run/bot.log"
    if sent >= HEALTH_MIN_ORDERS and rejects / sent > HEALTH_REJECT_SHARE:
        return RED, f"{rejects} of {sent} orders rejected, last: {status.get('last_reject')}"
    if status.get("halted"):
        return AMBER, "halted by a kill switch or a kill file, and may resume by itself"
    if status.get("reconnects", 0) >= HEALTH_RECONNECTS:
        return AMBER, f"{status['reconnects']} reconnects: is someone else logging in?"
    if status.get("errors", 0) >= HEALTH_ERRORS:
        return AMBER, f"{status['errors']} errors: read run/bot.log"
    return None


def health_report(run_dir: Path, now: float | None = None) -> tuple[str, list[str]]:
    path = Path(run_dir) / "status.json"
    try:
        age_s = (time.time() if now is None else now) - path.stat().st_mtime
        status = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return RED, [f"no usable {path}: {exc}", "It never got as far as trading. Read run/bot.log, then " + BACKUP_PLAN]
    found = _problem(status, age_s)
    facts = (
        f"pnl {status.get('pnl')}, positions {status.get('positions')}, orders sent {status.get('orders_sent')}, "
        f"rejects {status.get('rejects')}, errors {status.get('errors')}, reconnects {status.get('reconnects')}, "
        f"writes/s {status.get('writes_last_second')}, up {status.get('elapsed_s')} s"
    )
    if found is None:
        return GREEN, [facts]
    return found[0], [found[1], facts, "What to do: " + BACKUP_PLAN]


def print_health(run_dir: Path) -> int:
    verdict, lines = health_report(run_dir)
    print(f"{verdict}: " + "\n  ".join(lines))
    return HEALTH_EXIT_CODES[verdict]
