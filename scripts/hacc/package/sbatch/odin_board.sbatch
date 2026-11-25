#!/bin/bash
#SBATCH --chdir /tmp/
#SBATCH --job-name=odin_board
#SBATCH --output=slurm-odin-board-%j.out
#SBATCH --error=slurm-odin-board-%j.err
#
# Run the ODIN driver against an allocated card. run_all.sh picks the partition
# from the site's candidates (ODIN_BOARD_CANDIDATES, or ODIN_BOARD_PARTITION)
# and the cluster's own answers: a partition that allocates the card you built
# for, typically an hour-capped shell partition or a longer pool.
#
# ODIN_MODE=probe is B0: introspect the xclbin, load it, open the CU EXCLUSIVE,
# execute a one-token NULL PROGRAM and read back the capture header the fabric
# DMA'd home. It reads NO control register -- pyxrt binds none, which is what
# field-falsified the previous driver on this cluster on 2026-08-25.
#
# ODIN_MODE=deploy is "HACC ODIN Deployment": the sealed bundle under
# deployment/ executed as one host-mediated PASS per core -- program the core
# once, run each sample, transcode the counts into the next core's axon slots.
# ODIN_BUNDLE names a bundle other than the packaged default; ODIN_CHIP names
# WHICH FABRIC this bitstream holds (the executor refuses a bundle whose own
# chip_config claims name the other one); ODIN_DEPLOY_SAMPLES bounds the
# campaign.
#
# FIELD-OBSERVED 2026-08-25: a partition's AllowGroups decides who may use it,
# and a partition the documentation names may still refuse an account outright;
# the picker in run_all.sh checks both before it submits anything.
#
# THE CARD IS ALLOCATED, NOT AMBIENT. A board node answered '0 devices found'
# under xbutil from a non-FPGA partition on 2026-08-25: the cards are passed
# into the job's VM at allocation time (the cluster's board-VM model, now
# field-confirmed), so `xbutil examine` finding nothing here means the wrong
# partition, not a dead card.
#
# No account is set here: a site that requires one sets SBATCH_ACCOUNT=<account>
# (sbatch reads it) or passes --account. --output/--error stay under --chdir
# (/tmp) because slurm opens them before the job body runs.
#
#   ODIN_PKG=... ODIN_XCLBIN=... ODIN_MODE=probe|run \
#       sbatch --wait -p <partition> scripts/hacc/odin_board.sbatch

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
MODE="${ODIN_MODE:-run}"
RESULTS="${ODIN_RESULTS:-${PKG}/results/board}"
XRT_SETUP="${XRT_SETUP:-/opt/xilinx/xrt/setup.sh}"

echo "[board] job ${SLURM_JOB_ID:-?} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
echo "[board] package ${PKG}"
echo "[board] xclbin  ${XCLBIN}"
echo "[board] mode    ${MODE}"

if [[ ! -f "${XRT_SETUP}" ]]; then
    echo "REFUSING: no XRT at ${XRT_SETUP}. This job runs ONLY on an allocated" >&2
    echo "  HACC board node; /opt/xilinx/xrt is the stock XRT root" >&2
    echo "  (set XRT_SETUP and XRT_BIN if your site keeps it elsewhere)." >&2
    exit 2
fi

mkdir -p "${RESULTS}"
# shellcheck disable=SC1090  # cluster-side script, absent in this repo
source "${XRT_SETUP}"
/opt/xilinx/xrt/bin/xbutil examine | tee "${RESULTS}/xbutil_examine.log"

case "${MODE}" in
    probe)
        python3 "${PKG}/host/odin_board_driver.py" --probe \
            --xclbin "${XCLBIN}" --results "${RESULTS}"
        ;;
    run)
        python3 "${PKG}/host/odin_board_driver.py" \
            --xclbin "${XCLBIN}" --fixtures "${PKG}/fixtures" \
            --results "${RESULTS}"
        ;;
    deploy)
        DEPLOY_ARGS=(--xclbin "${XCLBIN}" --results "${RESULTS}")
        if [[ -n "${ODIN_BUNDLE:-}" ]]; then
            DEPLOY_ARGS+=(--bundle "${ODIN_BUNDLE}")
        fi
        if [[ -n "${ODIN_CHIP:-}" ]]; then
            DEPLOY_ARGS+=(--chip "${ODIN_CHIP}")
        fi
        python3 "${PKG}/host/odin_deployment_executor.py" "${DEPLOY_ARGS[@]}"
        ;;
    *)
        echo "REFUSING: unknown ODIN_MODE '${MODE}' (probe | run | deploy)." >&2
        exit 2
        ;;
esac
