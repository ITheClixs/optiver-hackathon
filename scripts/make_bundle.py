#!/usr/bin/env python3
"""Build one runnable file for the devenv when git is unavailable there.

The output is a zipapp: `python otk_bot.pyz --help`. Upload it through the browser IDE. It contains
only stdlib-compatible runtime modules, and research modules that need numpy are left out.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipapp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESEARCH_ONLY = {"overfit.py"}
MAIN = """import sys
from run_bot import main
sys.exit(main())
"""


def build(output: Path) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copytree(
            ROOT / "src" / "otk",
            stage / "otk",
            ignore=lambda _d, names: {
                n for n in names if n in RESEARCH_ONLY or n == "__pycache__"
            },
        )
        source = (ROOT / "scripts" / "run_bot.py").read_text()
        (stage / "run_bot.py").write_text(source.replace('sys.path.insert(0, str(ROOT / "src"))\n', ""))
        (stage / "__main__.py").write_text(MAIN)
        output.parent.mkdir(parents=True, exist_ok=True)
        zipapp.create_archive(stage, output, interpreter="/usr/bin/env python3")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dist/otk_bot.pyz")
    args = parser.parse_args(argv)
    print(build(Path(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
