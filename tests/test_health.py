"""`--health` tells the Driver in one line whether the bot is really trading, with no login and no tokens."""

import json
import os
import time

import pytest

from otk.health import AMBER, GREEN, RED, health_report

HEALTHY = {
    "elapsed_s": 300.0, "connected": True, "pnl": 120.5, "positions": {"A": 5}, "errors": 0, "rejects": 1, "reconnects": 0,
    "halted": False, "kill": {"trips": 0, "latched": False, "reason": None}, "orders_sent": 400, "writes_last_second": 6,
}


def write(tmp_path, age_s=1.0, **changes):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({**HEALTHY, **changes}))
    old = time.time() - age_s
    os.utime(path, (old, old))


def test_healthy(tmp_path):
    write(tmp_path)
    verdict, lines = health_report(tmp_path)
    assert verdict == GREEN
    assert "120.5" in " ".join(lines)


def test_no_status_means_it_never_got_going(tmp_path):
    verdict, lines = health_report(tmp_path)
    assert verdict == RED
    assert "backup plan" in " ".join(lines)


def test_a_broken_status_file_is_not_a_crash(tmp_path):
    (tmp_path / "status.json").write_text("{not json")
    assert health_report(tmp_path)[0] == RED


@pytest.mark.parametrize(
    ("changes", "age", "verdict", "word"),
    [
        ({}, 60.0, RED, "not running"),
        ({"connected": False}, 1.0, RED, "not connected"),
        ({"orders_sent": 0, "elapsed_s": 120.0}, 1.0, RED, "no orders"),
        ({"orders_sent": 0, "elapsed_s": 10.0}, 1.0, GREEN, ""),
        ({"orders_sent": 100, "rejects": 60}, 1.0, RED, "rejected"),
        ({"kill": {"trips": 3, "latched": True, "reason": "repeated losses"}}, 1.0, RED, "repeated losses"),
        ({"halted": True}, 1.0, AMBER, "halted"),
        ({"reconnects": 3}, 1.0, AMBER, "reconnect"),
        ({"errors": 5}, 1.0, AMBER, "errors"),
    ],
)
def test_each_way_of_not_really_trading_is_named(tmp_path, changes, age, verdict, word):
    write(tmp_path, age_s=age, **changes)
    got, lines = health_report(tmp_path)
    assert got == verdict
    assert word in " ".join(lines)
