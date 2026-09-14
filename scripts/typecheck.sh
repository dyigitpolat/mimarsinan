#!/usr/bin/env bash
# Curated basedpyright gate (pyrightconfig.json). Too slow for the pytest
# suite; run before committing: ./scripts/typecheck.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Either venv: .venv is what `make install` (uv sync) creates, env/ is the
# long-lived pip venv. --pythonpath overrides pyrightconfig.json's venv entry
# so the search paths follow whichever one is present.
if [ -x .venv/bin/python ]; then
    VENV=.venv
elif [ -x env/bin/python ]; then
    VENV=env
else
    echo "no virtualenv found: run \`make install\`" >&2
    exit 1
fi
source "$VENV/bin/activate"
exec basedpyright --pythonpath "$PWD/$VENV/bin/python" --level error src/mimarsinan "$@"
