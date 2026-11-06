"""On the day, one command builds the right file. These settings and that command must be real."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import make_single_file

from otk.params import validate_params
from otk.updater import parse_freeze

PRESETS = sorted((ROOT / "config" / "presets").glob("*.json"))


def test_there_is_one_for_every_format_the_bot_names_and_for_each_route():
    names = {path.stem for path in PRESETS}
    assert {"dual_listing", "stocks", "basket", "full_board", "news", "updater", "self_tune"} <= names


@pytest.mark.parametrize("path", PRESETS, ids=lambda path: path.stem)
def test_every_preset_holds_valid_settings(path):
    settings = json.loads(path.read_text())
    validate_params(settings.get("params", {}))
    if "update" in settings:
        assert parse_freeze(settings["update"]["freeze_at"]) is not None


def test_events_combine_left_to_right_and_objects_merge():
    merged = make_single_file.combine_events([{"params": {"size": 5, "skew_ticks": 3}, "a": 1}, {"params": {"size": 8}, "a": 2}])
    assert merged == {"params": {"size": 8, "skew_ticks": 3}, "a": 2}


def test_a_weight_from_another_event_never_ships_in_a_preset():
    """They differ per event, and a wrong one misprices the ETF. Left out, it quotes off the ETF's own book."""
    for path in PRESETS:
        assert "index_weights" not in json.loads(path.read_text()), path.name


def test_one_command_builds_a_file_that_passes_its_self_check(tmp_path):
    out = tmp_path / "bot.py"
    presets = ROOT / "config" / "presets"
    routes = [flag for name in ("full_board", "updater", "self_tune") for flag in ("--event", str(presets / f"{name}.json"))]
    assert make_single_file.main(["--out", str(out), *routes]) == 0
    text = out.read_text()
    assert '"option_vol": 3.0' not in text
    assert '"self_tune"' in text
    done = subprocess.run([sys.executable, str(out), "--selfcheck"], capture_output=True, text=True, timeout=120, cwd=tmp_path, check=False)
    assert done.returncode == 0, done.stderr[-600:]
