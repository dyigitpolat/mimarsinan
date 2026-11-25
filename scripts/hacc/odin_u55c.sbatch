#!/bin/bash
#SBATCH --chdir /tmp/
#SBATCH --job-name=odin_u55c
#SBATCH --output=slurm-odin-%j.out
#SBATCH --error=slurm-odin-%j.err
#
# The ODIN U55C parity campaign, submitted from the head node.
#
# LOG PATHS: slurm opens --output/--error BEFORE the job body runs, so a path
# under a directory that does not exist yet makes the submission fail to launch
# with nothing to read. These are therefore relative to --chdir (/tmp on the
# allocated node, which always exists); the run's own artifacts still land under
# ${ODIN_DATA_ROOT:-/data}/${USER}/log/odin_<timestamp>/, which run_board.sh
# creates, and RUNBOOK §4 creates that log directory before the first
# submission so the copy step at the end of the job has somewhere to go.
#
# ACCOUNT AND PARTITION ARE NOT SET HERE ON PURPOSE: both are the site's. A site
# that requires an account sets SBATCH_ACCOUNT=<account> (sbatch reads it) or
# passes --account=<account>. The partition decides the time limit, and the two
# useful kinds are very different:
#
#   sbatch -p <U55C board partition> scripts/hacc/odin_u55c.sbatch
#       a bare U55C shell partition, typically hour-capped. Enough for a
#       bring-up run or a small campaign.
#
#   sbatch -p <U55C + GPU partition> scripts/hacc/odin_u55c.sbatch
#       a node that carries the U55C and a GPU (an MI210) in one chassis, with
#       a longer cap: the only way to run host segments on the GPU in the same
#       job. Such nodes are often busy, so expect to queue.
#
# A node that pairs its U55C with other cards instead of a GPU is fine for a
# board-only run and wrong for the heterogeneous one.
#
# NEVER submit a BUILD here: the shell partitions are hour-capped and a
# place-and-route is hours. Build on a compile partition or the head node
# (build_xclbn.sh).

set -euo pipefail

DATA_ROOT="${ODIN_DATA_ROOT:-/data}"
STAGE="${ODIN_STAGE:-${DATA_ROOT}/${USER}/odin}"
CONFIG="${ODIN_CONFIG:-odin_u55c.json}"

echo "[sbatch] job ${SLURM_JOB_ID:-?} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
mkdir -p "${DATA_ROOT}/${USER}/log"
exec "${STAGE}/mimarsinan/scripts/hacc/run_board.sh" "${STAGE}" "${CONFIG}"
