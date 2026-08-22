#!/bin/bash
#SBATCH --chdir /tmp/
#SBATCH --account=slurm
#SBATCH --job-name=odin_u55c
#SBATCH --output=slurm-odin-%j.out
#SBATCH --error=slurm-odin-%j.err
#
# The ODIN U55C parity campaign, submitted from hacchead.
#
# LOG PATHS: slurm opens --output/--error BEFORE the job body runs, so a path
# under a directory that does not exist yet makes the submission fail to launch
# with nothing to read. These are therefore relative to --chdir (/tmp on the
# allocated node, which always exists); the run's own artifacts still land under
# /data/${USER}/log/odin_<timestamp>/, which run_board.sh creates, and RUNBOOK
# §4 creates /data/${USER}/log before the first submission so the copy step at
# the end of the job has somewhere to go.
#
# The first two #SBATCH lines are the cluster's own required preamble
# (Xtra-Computing/hacc_demo/example.sh). PARTITION IS NOT SET HERE ON PURPOSE:
# it decides the time limit, and the two useful choices are very different.
#
#   sbatch -p xilinx_u55c_gen3x16_xdma_3_202210_1 scripts/hacc/odin_u55c.sbatch
#       the bare U55C shell partition — ONE HOUR cap. Enough for a bring-up
#       run or a small campaign, on hacc-gpu1/2/3.
#
#   sbatch -p mi210_vck_u55c scripts/hacc/odin_u55c.sbatch
#       the true U55C + MI210 nodes (hacc-gpu2, hacc-gpu3) — 7-day cap, and the
#       only way to run host segments on the MI210 in the same job. Both were
#       busy in the sampled sinfo, so expect to queue.
#
# hacc-gpu1 pairs its U55C with U250/U280 instead of an MI210: fine for a
# board-only run, wrong for the heterogeneous one.
#
# NEVER submit a BUILD here: the shell partitions are hour-capped and a
# place-and-route is hours. Build on cpu_only/hacchead (build_xclbn.sh).

set -euo pipefail

STAGE="${ODIN_STAGE:-/data/${USER}/odin}"
CONFIG="${ODIN_CONFIG:-odin_u55c.json}"

echo "[sbatch] job ${SLURM_JOB_ID:-?} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
mkdir -p "/data/${USER}/log"
exec "${STAGE}/mimarsinan/scripts/hacc/run_board.sh" "${STAGE}" "${CONFIG}"
