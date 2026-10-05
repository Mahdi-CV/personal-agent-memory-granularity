#!/usr/bin/env bash
# Build the workshop paper: regenerate every table and figure from the frozen
# results, then compile. No number in the PDF is typed by hand.
#
#   ./paper/build.sh
#
# Requires tectonic (no sudo: https://tectonic-typesetting.github.io) and a
# python with matplotlib. Override either via TECTONIC / PYTHON.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

TECTONIC="${TECTONIC:-$(command -v tectonic || echo "$HOME/.local/bin/tectonic")}"
PYTHON="${PYTHON:-$HOME/.venvs/paper/bin/python}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

if [ ! -x "$TECTONIC" ]; then
  echo "tectonic not found. Set TECTONIC=/path/to/tectonic" >&2
  exit 1
fi

echo "== tables (also asserts the headline claim still holds)"
python3 "$HERE/make_tables.py"

echo "== figures"
"$PYTHON" "$HERE/make_figs.py"

echo "== compile"
mkdir -p "$HERE/build"
cd "$HERE"
"$TECTONIC" -X compile main.tex --outdir build

echo
echo "OK -> $HERE/build/main.pdf"
