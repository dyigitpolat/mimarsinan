#!/bin/bash
#SBATCH --chdir /tmp/
#SBATCH --account=slurm
#SBATCH --job-name=odin_board
#SBATCH --output=slurm-odin-board-%j.out
#SBATCH --error=slurm-odin-board-%j.err
#
# Run the ODIN driver against an allocated U55C. Submit with
#   -p xilinx_u55c_gen3x16_xdma_3_202210_1   (hacc-gpu1/2/3, ONE HOUR cap)
# or
#   -p mi210_vck_u55c                        (hacc-gpu2/3, 7-day cap)
# per Xtra-Computing/hacc_demo/doc/1-FPGA-allocation.md lines 32-33 and 44-46.
#
# The first two #SBATCH lines are the cluster's required preamble (that doc's
# example.sh lines 143-145). --output/--error stay under --chdir (/tmp) because
# slurm opens them before the job body runs.
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
    echo "  HACC@NUS board node; /opt/xilinx/xrt is the cluster's XRT root" >&2
    echo "  (hacc_demo/doc/0-login.md line 55)." >&2
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
    *)
        echo "REFUSING: unknown ODIN_MODE '${MODE}' (probe | run)." >&2
        exit 2
        ;;
esac
