#!/bin/bash
#SBATCH --chdir /tmp/
#SBATCH --account=slurm
#SBATCH --job-name=odin_joint
#SBATCH --output=slurm-odin-joint-%j.out
#SBATCH --error=slurm-odin-joint-%j.err
#
# THE TWO-COMPONENT RUN: the board and an independent reference, in ONE job,
# joined on the shared filesystem and certified A == B.
#
# WHY NOT `--het-group`. The task asked for a slurm heterogeneous job. The
# cluster's own documentation (Xtra-Computing/hacc_demo: README.md, doc/0-login
# through doc/7-coyote, and doc/1-FPGA-allocation.md's example.sh) documents
# exactly two submission forms — `srun -p <partition>` and `sbatch -p
# <partition> script.sh` — and mentions neither heterogeneous jobs nor job
# dependencies anywhere. What it DOES document is better for this purpose:
# `mi210_vck_u55c` (doc/1-FPGA-allocation.md lines 32-33) is a 7-day partition
# whose nodes, hacc-gpu2 and hacc-gpu3, carry BOTH a U55C and an MI210
# (README.md lines 32 and 38). One allocation on that partition therefore
# already holds both devices, so the two components are two concurrent steps of
# one job rather than two het-groups that must be co-scheduled across
# hour-capped partitions. Same guarantee — one submission, two components, a
# filesystem join — with syntax the cluster is documented to accept.
#
# FIELD-OBSERVED 2026-08-25: `mi210_vck_u55c` is not usable from this account —
# it does not appear in `scontrol show partition` and a submission answers
# "User's group not permitted to use this partition". The live joint venue is
# `mi210_u280_u55c` (AllowGroups=lab,hgpu,fpga_u280; 12 h, or 5 days on
# mi210_u280_u55c_long_reservation) on hacc-gpu1, which carries the MI210, the
# U280 AND a U55C in one chassis. run_all.sh picks it by asking the cluster.
#
#   ODIN_PKG=... ODIN_XCLBIN=... \
#       sbatch --wait -p mi210_u280_u55c scripts/hacc/odin_joint.sbatch

set -euo pipefail

# Slurm's own --output/--error live on the node's /tmp (see the preamble above),
# so everything this job prints is ALSO mirrored under ODIN_LOG on /data — that
# mirror, not the slurm file, is what run_all.sh reads and what comes home.
if [[ -n "${ODIN_LOG:-}" ]]; then
    mkdir -p "$(dirname "${ODIN_LOG}")"
    exec > >(tee -a "${ODIN_LOG}") 2>&1
fi

PKG="${ODIN_PKG:?ODIN_PKG (the unpacked package root) must be exported}"
XCLBIN="${ODIN_XCLBIN:?ODIN_XCLBIN (the staged bitstream) must be exported}"
JOIN="${ODIN_JOIN:-${ODIN_DATA_ROOT:-/data}/${USER}/odin/join}"
XRT_SETUP="${XRT_SETUP:-/opt/xilinx/xrt/setup.sh}"

A_DIR="${JOIN}/component_a_board"
B_DIR="${JOIN}/component_b_reference"
mkdir -p "${A_DIR}" "${B_DIR}"

echo "[joint] job ${SLURM_JOB_ID:-?} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
echo "[joint] join directory ${JOIN}  (the shared filesystem is the only path the head node and"
echo "        the board VMs share — hacc_demo/doc/1-FPGA-allocation.md line 155)"

# --- component B: the reference. No device, no XRT, no card. -----------------
# It re-derives every fixture's window counts from the SHIPPED per-cycle table
# under the driver's own window rule, verifies each fixture's self-hash, and
# writes its own attestation. It is an independent recomputation of the frozen
# evidence on a different process and a different device context, NOT a second
# simulation of the network: nothing on this cluster runs the RTL.
(
    echo "[joint/B] reference component starting on $(hostname)"
    python3 "${PKG}/host/odin_board_driver.py" --reference \
        --fixtures "${PKG}/fixtures" --results "${B_DIR}"
    echo "[joint/B] reference component done"
) &
component_b=$!

# --- component A: the board. -------------------------------------------------
(
    echo "[joint/A] board component starting on $(hostname)"
    if [[ ! -f "${XRT_SETUP}" ]]; then
        echo "REFUSING: no XRT at ${XRT_SETUP}; component A needs a board node." >&2
        exit 2
    fi
    # shellcheck disable=SC1090  # cluster-side script, absent in this repo
    source "${XRT_SETUP}"
    /opt/xilinx/xrt/bin/xbutil examine | tee "${A_DIR}/xbutil_examine.log"
    python3 "${PKG}/host/odin_board_driver.py" \
        --xclbin "${XCLBIN}" --fixtures "${PKG}/fixtures" --results "${A_DIR}"
    echo "[joint/A] board component done"
) &
component_a=$!

status=0
wait "${component_b}" || status=$?
wait "${component_a}" || status=$?
if [[ "${status}" -ne 0 ]]; then
    echo "REFUSING: a component failed (exit ${status}); the join is not run." >&2
    exit "${status}"
fi

# --- the join ----------------------------------------------------------------
python3 "${PKG}/host/odin_board_driver.py" \
    --compare "${A_DIR}/summary_board.json" "${B_DIR}/summary_reference.json" \
    --results "${JOIN}"
echo "[joint] join written under ${JOIN}"
