#!/usr/bin/env bash
# Merge a tested branch into main WITHOUT corrupting in-flight GPU-runner jobs.
#
# The runner's jobs `import` from main's LIVE working tree at process start, so a
# git op that leaves a half-written or conflict-marked .py there crashes any job
# that loads it (we lost 17 jobs to exactly this). RULE: resolve ALL conflicts in a
# throwaway worktree FIRST (`git worktree add ../wt <branch>`), so the merge into
# main is CONFLICT-FREE; then this script pauses new launches, lets any just-started
# job finish importing, does the clean merge, and resumes.
#
# Usage: scripts/campaign/safe_merge.sh <conflict-free-branch> [settle_seconds]
set -euo pipefail
BRANCH="${1:?branch required}"; SETTLE="${2:-15}"
Q="${MIM_CAMPAIGN_DIR:-runs/campaign}/q"
mkdir -p "$Q"
touch "$Q/PAUSE"
trap 'rm -f "$Q/PAUSE"' EXIT
echo "[safe_merge] runner paused; settling ${SETTLE}s so in-flight jobs finish importing..."
sleep "$SETTLE"
git checkout main
# A conflict here means you did NOT resolve in a worktree first — abort, do not
# leave markers in the runner's checkout.
git merge --no-ff --no-edit "$BRANCH"
echo "[safe_merge] merged $BRANCH into main; resuming runner."
