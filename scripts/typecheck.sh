#!/usr/bin/env bash
# Curated basedpyright gate (pyrightconfig.json). Too slow for the pytest
# suite; run before committing: ./scripts/typecheck.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source env/bin/activate
exec basedpyright --level error src/mimarsinan "$@"
