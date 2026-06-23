#!/usr/bin/env bash
# Make a git worktree runnable by symlinking the gitignored runtime deps from the
# main checkout (the venv, datasets, built nevresim, spikingjelly, etc. are not in
# git, so a fresh worktree lacks them). Idempotent; safe to re-run.
#
# Usage:  scripts/gpu/bootstrap_worktree.sh [MAIN_ROOT]
#   MAIN_ROOT defaults to $MIM_MAIN_ROOT or /home/yigit/repos/research_stuff/mimarsinan
set -euo pipefail

MAIN_ROOT="${1:-${MIM_MAIN_ROOT:-/home/yigit/repos/research_stuff/mimarsinan}}"
HERE="$(pwd -P)"

if [ "$HERE" = "$(cd "$MAIN_ROOT" && pwd -P)" ]; then
  echo "bootstrap: cwd IS the main checkout; nothing to link."
  exit 0
fi

for dep in env datasets build spikingjelly nevresim; do
  src="$MAIN_ROOT/$dep"
  if [ -e "$src" ] && [ ! -e "$HERE/$dep" ]; then
    ln -s "$src" "$HERE/$dep"
    echo "bootstrap: linked $dep -> $src"
  fi
done
echo "bootstrap: worktree ready at $HERE"
