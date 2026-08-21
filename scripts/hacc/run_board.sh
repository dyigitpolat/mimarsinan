#!/usr/bin/env bash
# Run the ODIN parity campaign on an allocated HACC@NUS U55C board.
#
# This is the script an sbatch job (scripts/hacc/odin_u55c.sbatch) executes ON
# THE NODE, and it follows the cluster's own convention from
# Xtra-Computing/hacc_demo/example.sh: relocate to a /tmp workspace, log into
# /data/${USER}/log, source the XRT setup, scan the cards, copy the staged
# artifacts in, run, and leave every artifact behind under the log path.
#
#   scripts/hacc/run_board.sh [stage-dir] [config]
#
# stage-dir defaults to /data/${USER}/odin and must already hold the xclbin and
# the deployment config; nothing is built here (see build_xclbn.sh).
set -euo pipefail

STAGE="${1:-/data/${USER}/odin}"
CONFIG="${2:-odin_u55c.json}"
XRT_SETUP="${XRT_SETUP:-/opt/xilinx/xrt/setup.sh}"

# --- REFUSE LOUD off-cluster -------------------------------------------------
if [[ ! -f "${XRT_SETUP}" ]]; then
    echo "REFUSING: no XRT at ${XRT_SETUP}." >&2
    echo "  This script runs ONLY on an allocated HACC@NUS board node." >&2
    echo "  Allocate one per scripts/hacc/RUNBOOK.md; locally, set" >&2
    echo "  odin_fpga_transport='rtl_cosim' and run the same program with no" >&2
    echo "  board at all." >&2
    exit 2
fi
if [[ ! -d "${STAGE}" ]]; then
    echo "REFUSING: staging directory ${STAGE} does not exist." >&2
    echo "  Stage the xclbin + config there first (RUNBOOK step 4);" >&2
    echo "  /data is the only path shared between the head node and the VMs." >&2
    exit 2
fi

workdir="/tmp/${USER}_odin"
rm -rf "${workdir}"
mkdir -p "${workdir}"
cd "${workdir}"

time_string="$(date +%Y_%m_%d_%H_%M_%S)"
log_path="/data/${USER}/log/odin_${time_string}"
mkdir -p "${log_path}"

# shellcheck disable=SC1090  # cluster-side script, absent in this repo
source "${XRT_SETUP}"
/opt/xilinx/xrt/bin/xbutil examine | tee "${log_path}/scan.log"

cp -r "${STAGE}"/. "${workdir}/" > "${log_path}/copy.log" 2>&1
xclbin="$(find "${workdir}" -maxdepth 1 -name '*.xclbin' | head -n 1)"
if [[ -z "${xclbin}" ]]; then
    echo "REFUSING: no .xclbin in ${STAGE}; build it first (build_xclbn.sh)." >&2
    exit 2
fi

echo "[hacc-run] node    : $(hostname)"
echo "[hacc-run] xclbin  : ${xclbin}"
echo "[hacc-run] config  : ${CONFIG}"
echo "[hacc-run] logs    : ${log_path}"

repo="${ODIN_REPO:-${workdir}/mimarsinan}"
if [[ ! -d "${repo}" ]]; then
    echo "REFUSING: no repository at ${repo}; stage it or set ODIN_REPO." >&2
    exit 2
fi

cd "${repo}"
MIMARSINAN_ODIN_XCLBIN="${xclbin}" \
    ./env/bin/python run.py "${CONFIG}" 2>&1 | tee "${log_path}/exec.log"

cp -r generated/. "${log_path}/generated/" 2>/dev/null || true
echo "[hacc-run] artifacts under ${log_path}"

# Hot-reset the card when a run wedges it (hacc_demo/example.sh's own recipe).
function reset_fpga {
    board_id="$(/opt/xilinx/xrt/bin/xbutil examine \
        | grep "\[" | awk '{print$1}' | sed 's/\[//' | sed 's/\]//')"
    /opt/xilinx/xrt/bin/xbutil reset -d "${board_id}" --force
}
