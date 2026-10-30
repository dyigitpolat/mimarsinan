#!/usr/bin/env bash
# What the detached bring-up is doing right now. Safe to run at any time: it
# reads, it never touches the run.
#
#   scripts/status.sh
#
# Three questions, in the order you ask them: is it alive, how far has it got,
# and what did it just say.

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="${HERE}/results"
LOG="${RESULTS}/run_all.log"
LOCK="${HERE}/.run_all.lock"
TAIL_LINES="${ODIN_STATUS_TAIL:-15}"

say() { printf '%s\n' "$*"; }
rule() { say "----------------------------------------------------------------------"; }

say "ODIN bring-up status — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "package ${HERE}"
rule

pid=""
[ -f "${LOCK}/pid" ] && pid="$(cat "${LOCK}/pid")"
if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
    say "LIVE: run_all.sh is running as pid ${pid} (since $(cat "${LOCK}/since" 2>/dev/null || echo '?'))."
elif [ -n "${pid}" ]; then
    say "NOT RUNNING: a stale lock names pid ${pid}, which is gone. Re-running"
    say "  ./run_all.sh takes the lock over and keeps every completed phase."
else
    say "NOT RUNNING: no lock is held. ./run_all.sh resumes from the artifacts."
fi

rule
say "PHASES"
"${HERE}/run_all.sh" --status

rule
say "SLURM (squeue --me)"
if command -v squeue > /dev/null 2>&1; then
    squeue --me 2>&1 || true
else
    say "no squeue on PATH — not on the head node?"
fi

rule
say "LAST ${TAIL_LINES} LINES OF ${LOG}"
if [ -f "${LOG}" ]; then
    tail -n "${TAIL_LINES}" "${LOG}"
else
    say "(no log yet — bootstrap_hacc.sh writes it as soon as run_all.sh starts)"
fi
