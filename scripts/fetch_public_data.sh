#!/usr/bin/env bash
# Real Optibook data for backtesting before the event: three weeks of ASML and ASML_DUAL level-5 quotes
# and named trades from a public repo (exwaii/asml-data, about 3 GB). data/ is gitignored, and it stays
# that way because the files hold real participants' names.
#
#   bash scripts/fetch_public_data.sh
#   H=2026-07-14T03; D=data/public/asml-data
#   uv run python scripts/book_lab.py \
#     --quotes $D/ASML/quotes_$H.csv:ASML --quotes $D/ASML_DUAL/quotes_$H.csv:ASML_DUAL \
#     --trades $D/ASML/trades_$H.csv:ASML --trades $D/ASML_DUAL/trades_$H.csv:ASML_DUAL \
#     --latency 0,0.05,0.25,1,3
set -euo pipefail
target="$(cd "$(dirname "$0")/.." && pwd)/data/public/asml-data"
if [ -d "$target/.git" ]; then
  git -C "$target" pull --ff-only
else
  mkdir -p "$(dirname "$target")"
  git clone --depth 1 https://github.com/exwaii/asml-data.git "$target"
fi
echo "ready: $target"
