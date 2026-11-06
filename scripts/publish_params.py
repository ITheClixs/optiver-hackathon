#!/usr/bin/env python3
"""Send live parameters or a kill to the running bot, through the `live` branch. No login to the exchange.

The bot's updater picks this up within seconds. Each change is a commit saying why, so the branch is the log.

    python scripts/publish_params.py '{"size": 5}' --why "informed flow on ASML"
    python scripts/publish_params.py --kill --why "reject storm"
    python scripts/publish_params.py --flatten --why "..."
    python scripts/publish_params.py --clear --why "cause fixed"
    python scripts/publish_params.py --reset --why "start of the event"   # nothing left over from a rehearsal

Keep one writer: a second one spoils a live test, and any key published outranks Autopilot. Anyone may pull.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from otk.params import merge_params, validate_params

BRANCH = "live"
ROLE_ENV = "OTK_ROLE"  # set by a role's loop for its jobs, and absent for a person
PUBLISHER = "A"
LEFTOVERS = ("board.json", "test.json")  # what a rehearsal believed and tried must not reach the event
Gh = Callable[[list[str]], subprocess.CompletedProcess]


def refused_to_role(role: str | None, args: argparse.Namespace) -> str | None:
    """Why this job may not do this, if it may not. It stops accidents, not an agent set on getting round it."""
    if not role:
        return None
    if args.reset:
        return "only a person resets what is published"
    if role != PUBLISHER and (args.changes or args.clear):
        return f"only role {PUBLISHER} publishes parameters or clears a kill. Put your proposal in a line starting @{PUBLISHER}"
    return None


def run_gh(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=False)


def read_file(gh: Gh, repo: str, name: str) -> tuple[dict, str | None]:
    done = gh(["api", f"repos/{repo}/contents/{name}?ref={BRANCH}"])
    if done.returncode != 0:
        return {}, None
    found = json.loads(done.stdout)
    return json.loads(base64.b64decode(found["content"])), found["sha"]


def ensure_branch(gh: Gh, repo: str) -> None:
    if gh(["api", f"repos/{repo}/branches/{BRANCH}"]).returncode == 0:
        return
    sha = gh(["api", f"repos/{repo}/git/ref/heads/main", "--jq", ".object.sha"]).stdout.strip()
    done = gh(["api", f"repos/{repo}/git/refs", "-f", f"ref=refs/heads/{BRANCH}", "-f", f"sha={sha}"])
    if done.returncode != 0:
        raise SystemExit(f"could not create the {BRANCH} branch: {done.stderr}")


def write_file(gh: Gh, repo: str, name: str, payload: dict, sha: str | None, why: str) -> None:
    content = base64.b64encode(json.dumps(payload, indent=2, sort_keys=True).encode()).decode()
    fields = ["-f", f"message={name}: {why}", "-f", f"content={content}", "-f", f"branch={BRANCH}"]
    if sha:
        fields += ["-f", f"sha={sha}"]
    done = gh(["api", "--method", "PUT", f"repos/{repo}/contents/{name}", *fields])
    if done.returncode != 0:
        raise SystemExit(f"not published: {done.stderr}")


def publish_params(gh: Gh, repo: str, changes: dict, why: str) -> dict:
    current, sha = read_file(gh, repo, "params.json")
    merged = merge_params(current, changes)
    validate_params(merged)  # refuse here, so a typo never reaches the branch
    write_file(gh, repo, "params.json", merged, sha, why)
    return merged


def publish_control(gh: Gh, repo: str, control: dict, why: str) -> None:
    _current, sha = read_file(gh, repo, "control.json")
    write_file(gh, repo, "control.json", control, sha, why)


def main(argv: list[str] | None = None, gh: Gh = run_gh) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("changes", nargs="?", help="JSON object of parameter changes, merged onto what is published")
    parser.add_argument("--why", required=True, help="reason; becomes the commit message")
    parser.add_argument("--repo", required=True, help="owner/name of the repository whose live branch the bot reads")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--kill", action="store_true", help="pull all quotes and halt")
    group.add_argument("--flatten", action="store_true", help="pull quotes and close out")
    group.add_argument("--clear", action="store_true", help="undo a kill or flatten set this way")
    parser.add_argument("--reset", action="store_true", help="drop every published parameter and any live test first, so rehearsal leftovers cannot reach the bot")
    args = parser.parse_args(argv)
    refusal = refused_to_role(os.environ.get(ROLE_ENV), args)
    if refusal:
        print(f"nothing published: {refusal}", file=sys.stderr)
        return 2
    ensure_branch(gh, args.repo)
    if args.kill or args.flatten or args.clear:
        publish_control(gh, args.repo, {"kill": args.kill, "flatten": args.flatten}, args.why)
    if args.reset:
        for name in ("params.json", *LEFTOVERS):
            _current, sha = read_file(gh, args.repo, name)
            write_file(gh, args.repo, name, {}, sha, args.why)
    if args.changes:
        try:
            print(json.dumps(publish_params(gh, args.repo, json.loads(args.changes), args.why), indent=2, sort_keys=True))
        except ValueError as exc:
            print(f"rejected, nothing published: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
