#!/usr/bin/env bash
# The ODIN board bring-up, end to end, from one unzipped package.
#
#   ./run_all.sh                 every phase, resuming where it stopped
#   ./run_all.sh --dry-run       print every command, run nothing
#   ./run_all.sh --only 1        just the no-hardware selftest
#   ./run_all.sh --from 5        skip the builds, start at the board
#   ./run_all.sh --force         re-run phases whose stamp already exists
#
# PHASES
#   0  env probe            what this cluster has, written to results/env.txt
#   1  selftest             the whole driver against a fake pyxrt, no hardware
#   2  build hw_emu         cpu_only + an XCL_EMULATION_MODE=hw_emu smoke there
#   3  build hw             cpu_only, the real bitstream (2-6 h)
#   4  stage                the xclbin onto /data per the runbook
#   5  B0 smoke             load the xclbin on a U55C, read the CSRs
#   6  B1 campaign          every shipped fixture, certified
#   7  joint run            board + independent reference in one job, joined
#
# Every phase writes a stamp under .state/ and is skipped when its stamp is
# there, so an interrupted run resumes and a completed one is a no-op.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
STATE="${HERE}/.state"
RESULTS="${HERE}/results"
DRIVER="${HERE}/host/odin_board_driver.py"
FAKE="${HERE}/host/fake_pyxrt_for_selftest.py"
FIXTURES="${HERE}/fixtures"

# Cluster facts, all from Xtra-Computing/hacc_demo (cited per line below).
PLATFORM="${ODIN_PLATFORM:-xilinx_u55c_gen3x16_xdma_3_202210_1}"
BUILD_PARTITION="${ODIN_BUILD_PARTITION:-cpu_only}"          # doc/1 line 20, 7 days
BOARD_PARTITION="${ODIN_BOARD_PARTITION:-${PLATFORM}}"        # doc/1 line 44, 1 hour
JOINT_PARTITION="${ODIN_JOINT_PARTITION:-mi210_vck_u55c}"     # doc/1 line 32, 7 days
EXPECTED_XRT="${ODIN_EXPECTED_XRT:-2.18.179}"                 # README.md line 32
XILINX_ROOT="${XILINX_ROOT:-/tools/xilinx}"                   # doc/0-login line 56
VITIS_VERSION="${VITIS_VERSION:-2022.2}"                      # README.md line 32
XRT_ROOT="${XRT_ROOT:-/opt/xilinx/xrt}"                       # doc/0-login line 55

STAGE_DIR="${ODIN_STAGE:-/data/${USER}/odin}"
BUILD_DIR="${HERE}/build/hacc"
XCLBIN_SRC="${BUILD_DIR}/hw_nc1/odin_fpga_hw.xclbin"
XCLBIN_STAGED="${STAGE_DIR}/odin_fpga_hw.xclbin"

DRY_RUN=0
FORCE=0
ONLY=""
FROM=0

usage() { sed -n '2,28p' "$0"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        --only) ONLY="${2:?--only needs a phase number}"; shift ;;
        --from) FROM="${2:?--from needs a phase number}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "REFUSING: unknown argument '$1' (see --help)." >&2; exit 2 ;;
    esac
    shift
done

mkdir -p "${STATE}" "${RESULTS}"

say() { printf '%s\n' "$*"; }
head_line() { say ""; say "=== $* ==="; }

# Print or run. Every side effect in this script goes through here, so
# --dry-run genuinely runs nothing.
do_cmd() {
    if [ "${DRY_RUN}" = "1" ]; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

# The same, but with the command's output mirrored into a results/ log.
do_cmd_logged() {
    local log="$1"
    shift
    if [ "${DRY_RUN}" = "1" ]; then
        printf '[dry-run] (tee %s)' "${log}"
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    mkdir -p "$(dirname "${log}")"
    "$@" 2>&1 | tee -a "${log}"
    return "${PIPESTATUS[0]}"
}

stamped() {
    [ "${FORCE}" = "0" ] && [ -f "${STATE}/phase${1}.ok" ]
}

stamp() {
    [ "${DRY_RUN}" = "1" ] && return 0
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "${STATE}/phase${1}.ok"
}

want() {
    local phase="$1"
    if [ -n "${ONLY}" ]; then
        [ "${ONLY}" = "${phase}" ]
        return
    fi
    [ "${phase}" -ge "${FROM}" ]
}

require_file() {
    if [ ! -e "$1" ]; then
        say "REFUSING: $1 is missing."
        say "  FIX: $2"
        exit 2
    fi
}

# ---------------------------------------------------------------------------
# 0. env probe
# ---------------------------------------------------------------------------
phase_0() {
    head_line "phase 0: env probe"
    local report="${RESULTS}/env.txt"
    local failures=0
    if [ "${DRY_RUN}" = "1" ]; then
        say "[dry-run] would probe modules/vitis/xrt/slurm into ${report}"
        return 0
    fi
    {
        say "host      : $(hostname)"
        say "user      : ${USER}"
        say "date(utc) : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        say "package   : ${HERE}"
        say "python3   : $(command -v python3 || echo MISSING) $(python3 -V 2>&1 || true)"
        say "sbatch    : $(command -v sbatch || echo MISSING)"
        say "sinfo     : $(command -v sinfo || echo MISSING)"
        say "vitis     : ${XILINX_ROOT}/Vitis/${VITIS_VERSION}"
        say "xrt       : ${XRT_ROOT}"
        say "platform  : ${PLATFORM}"
        say "expect XRT: ${EXPECTED_XRT}"
        say ""
        say "--- module avail (if any) ---"
        if command -v module > /dev/null 2>&1; then
            module avail 2>&1 || true
            module list 2>&1 || true
        else
            say "no 'module' command on this host; HACC exposes its toolchains"
            say "through /tools/xilinx and /opt/xilinx directly."
        fi
        say ""
        say "--- sinfo ---"
        if command -v sinfo > /dev/null 2>&1; then sinfo 2>&1 || true; fi
        say ""
        say "--- xbutil examine (board nodes only) ---"
        if [ -x "${XRT_ROOT}/bin/xbutil" ]; then
            "${XRT_ROOT}/bin/xbutil" examine 2>&1 || true
        else
            say "no ${XRT_ROOT}/bin/xbutil here (expected on hacchead)"
        fi
    } > "${report}" 2>&1
    say "[phase0] wrote ${report}"

    case "${HERE}" in
        /data/*) : ;;
        *)
            say "REFUSING: the package is unpacked at ${HERE}, not under /data."
            say "  /data is the ONLY path shared between the head node and the"
            say "  board VMs (hacc_demo/doc/1-FPGA-allocation.md line 155), so a"
            say "  job would not be able to read this directory at all."
            say "  FIX: mkdir -p /data/\${USER} && unzip odin_hacc_package.zip -d /data/\${USER}"
            failures=$((failures + 1))
            ;;
    esac
    if [ ! -f "${XILINX_ROOT}/Vitis/${VITIS_VERSION}/settings64.sh" ]; then
        say "REFUSING: no Vitis ${VITIS_VERSION} at ${XILINX_ROOT}/Vitis/${VITIS_VERSION}."
        say "  FIX: run this on hacchead or a cpu_only node; off-cluster there"
        say "  is no toolchain and no shell to link against. Override the root"
        say "  with XILINX_ROOT / VITIS_VERSION if the cluster moved it."
        failures=$((failures + 1))
    fi
    if [ ! -f "${XRT_ROOT}/setup.sh" ]; then
        say "REFUSING: no XRT at ${XRT_ROOT}/setup.sh."
        say "  FIX: the cluster keeps XRT at /opt/xilinx/xrt"
        say "  (hacc_demo/doc/0-login.md line 55). Set XRT_ROOT if it moved."
        failures=$((failures + 1))
    fi
    if ! command -v sbatch > /dev/null 2>&1; then
        say "REFUSING: no sbatch on PATH — phases 2-7 all submit slurm jobs."
        say "  FIX: run this from hacchead (source /home/hacc_env)."
        failures=$((failures + 1))
    fi
    if grep -q "${EXPECTED_XRT}" "${report}"; then
        say "[phase0] XRT ${EXPECTED_XRT} seen in the probe."
    else
        say "[phase0] NOTE: XRT ${EXPECTED_XRT} (hacc_demo/README.md line 32) was"
        say "         not seen in ${report}. hacc_demo/doc/0-login.md still lists"
        say "         2.14.384 for this shell and is stale; confirm the real"
        say "         version with xbutil examine ON THE BOARD NODE. A shell/XRT"
        say "         mismatch against what the xclbin was linked with is triage"
        say "         step 1, not a footnote."
    fi
    if [ "${failures}" -gt 0 ]; then
        say "[phase0] ${failures} blocking problem(s); fix them and re-run."
        exit 2
    fi
    say "[phase0] OK"
}

# ---------------------------------------------------------------------------
# 1. selftest — the whole driver, no hardware
# ---------------------------------------------------------------------------
phase_1() {
    head_line "phase 1: driver selftest against a fake pyxrt (NO hardware)"
    require_file "${DRIVER}" "re-unzip the package; host/odin_board_driver.py is missing"
    require_file "${FAKE}" "re-unzip the package; the fake pyxrt is missing"
    require_file "${FIXTURES}" "re-unzip the package; fixtures/ is missing"
    do_cmd_logged "${RESULTS}/phase1_selftest.log" \
        python3 "${DRIVER}" --selftest --fake-pyxrt "${FAKE}" \
        --fixtures "${FIXTURES}" --results "${RESULTS}/selftest"
}

# ---------------------------------------------------------------------------
# 2/3. the builds
# ---------------------------------------------------------------------------
submit_build() {
    local target="$1"
    local log="${RESULTS}/phase_build_${target}.log"
    do_cmd env \
        ODIN_PKG="${HERE}" ODIN_TARGET="${target}" ODIN_PLATFORM="${PLATFORM}" \
        ODIN_LOG="${log}" \
        sbatch --wait -p "${BUILD_PARTITION}" \
        "${HERE}/scripts/hacc/odin_build.sbatch"
    if [ "${DRY_RUN}" = "0" ] && [ -f "${log}" ]; then
        tail -n 20 "${log}"
    fi
}

phase_2() {
    head_line "phase 2: build hw_emu on ${BUILD_PARTITION}, then smoke it THERE"
    say "hw_emu needs NO CARD: XCL_EMULATION_MODE=hw_emu binds XRT to the"
    say "emulation model v++ packaged into the xclbin, so the smoke runs on the"
    say "build node. It proves the packaged kernel opens by name, the register"
    say "map answers, the AXI master moves the payloads and the capture decodes"
    say "into the frozen counts. It cannot prove HBM ordering behind a real XDMA"
    say "shell, XRT's allocation on silicon, or timing closure — that is B0/B1."
    submit_build hw_emu
}

phase_3() {
    head_line "phase 3: build hw on ${BUILD_PARTITION} (the real bitstream)"
    say "Budget 2-6 h: place-and-route of one ODIN core plus the program and"
    say "capture RAMs. NEVER submit this to a board partition — those are capped"
    say "at one hour."
    submit_build hw
}

# ---------------------------------------------------------------------------
# 4. stage onto /data
# ---------------------------------------------------------------------------
phase_4() {
    head_line "phase 4: stage the bitstream under ${STAGE_DIR}"
    if [ "${DRY_RUN}" = "0" ]; then
        require_file "${XCLBIN_SRC}" "run phase 3 first (the hw build produces it)"
    fi
    do_cmd mkdir -p "${STAGE_DIR}" "/data/${USER}/log"
    do_cmd cp "${XCLBIN_SRC}" "${XCLBIN_STAGED}"
    do_cmd ls -l "${XCLBIN_STAGED}"
    say "[phase4] /data is the only path the head node and the board VMs share."
}

# ---------------------------------------------------------------------------
# 5/6. the board
# ---------------------------------------------------------------------------
submit_board() {
    local mode="$1"
    local partition="$2"
    local log="${RESULTS}/phase_board_${mode}.log"
    do_cmd env \
        ODIN_PKG="${HERE}" ODIN_XCLBIN="${XCLBIN_STAGED}" ODIN_MODE="${mode}" \
        ODIN_RESULTS="${RESULTS}/board_${mode}" ODIN_LOG="${log}" \
        sbatch --wait -p "${partition}" \
        "${HERE}/scripts/hacc/odin_board.sbatch"
    if [ "${DRY_RUN}" = "0" ] && [ -f "${log}" ]; then
        tail -n 40 "${log}"
    fi
}

phase_5() {
    head_line "phase 5: B0 smoke on ${BOARD_PARTITION}"
    say "Load the xclbin, resolve odin_fpga_kernel_top with EXCLUSIVE access,"
    say "read capture_capacity (0x54), program_capacity (0x5C) and status"
    say "(0x4C). A zero capacity refuses at open by design: it means the host is"
    say "talking to something that is not this kernel."
    submit_board probe "${BOARD_PARTITION}"
}

phase_6() {
    head_line "phase 6: B1 parity campaign on ${BOARD_PARTITION}"
    say "Every shipped fixture, one certificate line each. Anything other than"
    say "PASS exact=1.000000 max|dcount|=0 is a finding, not a tolerance."
    submit_board run "${BOARD_PARTITION}"
}

# ---------------------------------------------------------------------------
# 7. the two-component run
# ---------------------------------------------------------------------------
phase_7() {
    head_line "phase 7: board + independent reference in ONE job on ${JOINT_PARTITION}"
    say "hacc-gpu2/hacc-gpu3 carry a U55C and an MI210 in the same node, so one"
    say "allocation on ${JOINT_PARTITION} already holds both components; see the"
    say "'WHY NOT --het-group' note in scripts/hacc/odin_joint.sbatch."
    local log="${RESULTS}/phase_joint.log"
    do_cmd env \
        ODIN_PKG="${HERE}" ODIN_XCLBIN="${XCLBIN_STAGED}" \
        ODIN_JOIN="${RESULTS}/joint" ODIN_LOG="${log}" \
        sbatch --wait -p "${JOINT_PARTITION}" \
        "${HERE}/scripts/hacc/odin_joint.sbatch"
    if [ "${DRY_RUN}" = "0" ] && [ -f "${log}" ]; then
        tail -n 40 "${log}"
    fi
}

# ---------------------------------------------------------------------------

for phase in 0 1 2 3 4 5 6 7; do
    if ! want "${phase}"; then
        continue
    fi
    if stamped "${phase}"; then
        say "[phase${phase}] already done ($(cat "${STATE}/phase${phase}.ok")) — skipping"
        continue
    fi
    "phase_${phase}"
    stamp "${phase}"
done

say ""
say "Done. Evidence is under ${RESULTS}; bring it home with ./collect_results.sh"
