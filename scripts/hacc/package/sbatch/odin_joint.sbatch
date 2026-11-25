#!/bin/bash
#SBATCH --chdir /tmp/
#SBATCH --job-name=odin_joint
#SBATCH --output=slurm-odin-joint-%j.out
#SBATCH --error=slurm-odin-joint-%j.err
#
# THE TWO-COMPONENT RUN: the board and an independent reference, in ONE job,
# joined on the shared filesystem and certified A == B.
#
# WHY NOT `--het-group`. The task asked for a slurm heterogeneous job. The
# cluster's own documentation describes exactly two submission forms — `srun -p
# <partition>` and `sbatch -p <partition> script.sh` — and mentions neither
# heterogeneous jobs nor job dependencies anywhere. What it DOES describe is
# better for this purpose: a long partition whose nodes carry BOTH a U55C and
# an MI210. One allocation on such a partition therefore already holds both
# devices, so the two components are two concurrent steps of one job rather
# than two het-groups that must be co-scheduled across hour-capped partitions.
# Same guarantee — one submission, two components, a filesystem join — with
# syntax the cluster is documented to accept.
#
# FIELD-OBSERVED 2026-08-25: a partition the documentation names may not be
# usable from a given account — it does not appear in `scontrol show partition`
# and a submission answers "User's group not permitted to use this partition".
# run_all.sh therefore picks the joint venue from the site's candidates
# (ODIN_JOINT_CANDIDATES, or ODIN_JOINT_PARTITION) by asking the cluster, and a
# node that carries the MI210 beside the U55C in one chassis serves it.
#
# No account is set here: a site that requires one sets SBATCH_ACCOUNT=<account>
# (sbatch reads it) or passes --account.
#
# AND ON THE U250 (v3, the card the cluster can actually build), the two-device
# chassis argument does not apply at all — the U250 nodes carry no GPU. It
# never had to. Component B has always been a CPU recomputation of the frozen
# evidence, not a second simulation: it imports no device library, opens no
# card and asks for no accelerator. So on a U250 allocation the SAME two
# components run side by side on the SAME job — the board process on the card,
# the reference process on that node's CPUs — and the join is the same
# filesystem join. What the card profile changes is the claim we print, not the
# structure of the run: `joint_model` in scripts/hacc/cards.sh says which of
# the two situations this partition is, and this script says so out loud rather
# than asserting a GPU that is not there.
#
#   ODIN_PKG=... ODIN_XCLBIN=... ODIN_CARD=u250 \
#       sbatch --wait -p <U250 partition> scripts/hacc/odin_joint.sbatch

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

# shellcheck source=scripts/hacc/cards.sh
source "${PKG}/scripts/hacc/cards.sh"
CARD="$(odin_card)"

echo "[joint] job ${SLURM_JOB_ID:-?} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
echo "[joint] card ${CARD}"
case "$(odin_card_field joint_model "${CARD}")" in
    board_and_gpu_in_one_chassis)
        echo "[joint] this partition's node carries the board AND an MI210 in one"
        echo "        chassis, so one allocation holds both components."
        ;;
    *)
        echo "[joint] this partition's nodes carry NO GPU, and the join never needed"
        echo "        one: component B is a CPU recomputation of the frozen evidence,"
        echo "        so both components run CPU+board on this one allocation."
        ;;
esac
echo "[joint] join directory ${JOIN}  (the shared filesystem is the only path the head node and"
echo "        the board VMs share — ODIN_DATA_ROOT, /data by default)"

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
