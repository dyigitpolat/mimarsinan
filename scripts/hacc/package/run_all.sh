#!/usr/bin/env bash
# The ODIN board bring-up, end to end, from one unzipped package.
#
#   ./run_all.sh                 every phase; work already done is detected, not redone
#   ./run_all.sh --status        what is done, what is live, and stop
#   ./run_all.sh --dry-run       print every command, run nothing
#   ./run_all.sh --only 1        just the no-hardware selftest
#   ./run_all.sh --from 5        skip the builds, start at the board
#   ./run_all.sh --force         redo a phase even though its artifact is there
#
# PHASES
#   0  env probe            what this cluster has, written to results/env.txt
#   1  selftest             the whole driver against a fake pyxrt, no hardware
#   2  build hw_emu         a compile partition + a bounded hw_emu smoke there
#   3  build hw             a compile partition, the real bitstream (2-6 h)
#   4  stage                the xclbin onto the shared filesystem
#   5  B0 smoke             load the xclbin on a U55C, read the CSRs
#   6  B1 campaign          every shipped fixture, certified
#   7  joint run            board + independent reference in one job, joined
#
# RESUME IS ARTIFACT-BASED — there are no stamps to keep in sync. A phase is
# done when the thing it was supposed to produce EXISTS: the xclbin plus a
# sidecar naming the build script's sha256 it was built with, the staged
# bitstream, the driver's own result JSONs. Edit build_xclbn.sh and the build
# runs again; kill this script mid-phase and re-running it picks up exactly
# where the artifacts stop.
#
# ONE AT A TIME. A live run holds .run_all.lock; a second invocation refuses and
# names the pid that holds it. bootstrap_hacc.sh launches this detached, so the
# normal way to watch it is `tail -f results/run_all.log` and `scripts/status.sh`.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RESULTS="${HERE}/results"
DRIVER="${HERE}/host/odin_board_driver.py"
FAKE="${HERE}/host/fake_pyxrt_for_selftest.py"
FIXTURES="${HERE}/fixtures"
BUILD_SCRIPT="${HERE}/scripts/hacc/build_xclbn.sh"
JOURNAL="${RESULTS}/phase_journal.tsv"
PICK_LOG="${RESULTS}/partition_picks.txt"
LOCK="${HERE}/.run_all.lock"

# shellcheck source=scripts/hacc/toolchain.sh
source "${HERE}/scripts/hacc/toolchain.sh"

# /data is the only path the head node and the board VMs share
# (hacc_demo/doc/1-FPGA-allocation.md line 155). ODIN_DATA_ROOT moves it so the
# whole flow can be exercised off-cluster.
DATA_ROOT="${ODIN_DATA_ROOT:-/data}"
PLATFORM="${ODIN_PLATFORM:-xilinx_u55c_gen3x16_xdma_3_202210_1}"
EXPECTED_XRT="${ODIN_EXPECTED_XRT:-2.18.179}"                 # hacc_demo/README.md line 32
XRT_ROOT="${XRT_ROOT:-/opt/xilinx/xrt}"                       # doc/0-login line 55

# Candidate partitions, best first. NOTHING here is trusted: each is kept only
# if scontrol says your groups may use it and sinfo says it has a node that is
# up. See pick_partition() — and the field notes in README_HACC.md.
BUILD_CANDIDATES=(cpu_only vck5000_compile mi210_u280_u55c_long_reservation)
JOINT_CANDIDATES=(mi210_vck_u55c mi210_u280_u55c)
BOARD_CANDIDATES=("${PLATFORM}")

STAGE_DIR="${ODIN_STAGE:-${DATA_ROOT}/${USER}/odin}"
BUILD_DIR="${HERE}/build/hacc"
XCLBIN_STAGED="${STAGE_DIR}/odin_fpga_hw.xclbin"

DRY_RUN=0
FORCE=0
STATUS_ONLY=0
ONLY=""
FROM=0

usage() { sed -n '2,40p' "$0"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        --status) STATUS_ONLY=1 ;;
        --only) ONLY="${2:?--only needs a phase number}"; shift ;;
        --from) FROM="${2:?--from needs a phase number}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "REFUSING: unknown argument '$1' (see --help)." >&2; exit 2 ;;
    esac
    shift
done

mkdir -p "${RESULTS}"

say() { printf '%s\n' "$*"; }
head_line() { say ""; say "=== $* ==="; }
utc() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# The one line every failure path ends with, so the user always knows that
# stopping here costs nothing already paid for.
RERUN_LINE="fix, then re-run ./run_all.sh — completed work is kept"

journal() {
    [ "${DRY_RUN}" = "1" ] && return 0
    printf '%s\t%s\t%s\t%s\n' "$(utc)" "$1" "$2" "${3:-}" >> "${JOURNAL}"
}

# The pick is a decision worth keeping next to the evidence it produced — but
# not on a --dry-run, which writes nothing anywhere.
record_pick() {
    [ "${DRY_RUN}" = "1" ] && return 0
    printf '%s\t%s\t%s\n' "$(utc)" "$1" "$2" >> "${PICK_LOG}"
}

xclbin_of() { printf '%s/%s_nc1/odin_fpga_%s.xclbin\n' "${BUILD_DIR}" "$1" "$1"; }

# ---------------------------------------------------------------------------
# One live run at a time
# ---------------------------------------------------------------------------
release_lock() { rm -rf "${LOCK}"; }

acquire_lock() {
    if [ "${DRY_RUN}" = "1" ] || [ "${STATUS_ONLY}" = "1" ]; then
        return 0
    fi
    local pid=""
    if ! mkdir "${LOCK}" 2>/dev/null; then
        pid="$(cat "${LOCK}/pid" 2>/dev/null || true)"
        if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
            say "REFUSING: run_all.sh is already running here as pid ${pid}"
            say "  (since $(cat "${LOCK}/since" 2>/dev/null || echo '?'))."
            say "  Watch it :  tail -f ${RESULTS}/run_all.log"
            say "  Status   :  ${HERE}/scripts/status.sh"
            say "  Stop it  :  kill ${pid}    — completed work is kept"
            exit 3
        fi
        say "[lock] the lock is held by pid ${pid:-?}, which is gone; taking it over."
    fi
    mkdir -p "${LOCK}"
    printf '%s\n' "$$" > "${LOCK}/pid"
    utc > "${LOCK}/since"
    trap release_lock EXIT INT TERM
}

# ---------------------------------------------------------------------------
# Print or run. Every side effect goes through here, so --dry-run runs nothing.
# ---------------------------------------------------------------------------
do_cmd() {
    if [ "${DRY_RUN}" = "1" ]; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

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
        return 2
    fi
}

# ---------------------------------------------------------------------------
# PARTITION AUTO-PICK. Two gates, both from the cluster itself, both at phase
# time: scontrol must show the partition AND admit one of your groups, and
# sinfo must show it a node that is not down or drained. Every rejected
# candidate is printed with the reason it lost.
# ---------------------------------------------------------------------------
MY_GROUPS=""
SINFO_TABLE=""
SINFO_READ=0
PICKED=""

# Read the cluster ONCE per invocation: these two are globals on purpose,
# because every helper that needs them runs inside a command substitution and a
# subshell's cache would be thrown away.
load_groups() {
    [ -n "${MY_GROUPS}" ] && return 0
    local raw
    raw="$(groups 2>/dev/null || id -nG 2>/dev/null || true)"
    raw="${raw#*" : "}"          # some `groups` print "user : g1 g2"
    MY_GROUPS=" ${raw} "
}

load_sinfo() {
    [ "${SINFO_READ}" = "1" ] && return 0
    SINFO_READ=1
    if command -v sinfo > /dev/null 2>&1; then
        SINFO_TABLE="$(sinfo -a -N -h -o '%N %P %t' 2>/dev/null || true)"
    fi
}

my_groups() {
    load_groups
    printf '%s' "${MY_GROUPS}"
}

# Prints the group that lets you in, or returns 1.
group_admits() {
    local allow="$1" mine group
    if [ "${allow}" = "ALL" ] || [ -z "${allow}" ]; then
        printf 'ALL'
        return 0
    fi
    mine="$(my_groups)"
    while IFS= read -r group; do
        [ -n "${group}" ] || continue
        case "${mine}" in
            *" ${group} "*) printf '%s' "${group}"; return 0 ;;
        esac
    done < <(printf '%s' "${allow}" | tr ',' '\n')
    return 1
}

# node:state pairs sinfo reports for one partition, one per line.
partition_nodes() {
    local partition="$1"
    printf '%s\n' "${SINFO_TABLE}" | awk -v want="${partition}" '
        {
            part = $2; sub(/\*$/, "", part);
            if (part == want) printf "%s:%s\n", $1, $3;
        }'
}

node_state_is_live() {
    case "$1" in
        idle|mix|mixed|alloc|allocated|comp|completing|resv) return 0 ;;
        *) return 1 ;;
    esac
}

pick_partition() {
    local role="$1" override="$2"
    shift 2
    local candidate dump allow admitted nodes node state live maxtime
    PICKED=""
    if [ -n "${override}" ]; then
        PICKED="${override}"
        say "[pick/${role}] ${PICKED} — set explicitly by the environment; no probing done."
        record_pick "${role}" "${PICKED} (env override)"
        return 0
    fi
    if ! command -v scontrol > /dev/null 2>&1 || ! command -v sinfo > /dev/null 2>&1; then
        PICKED="$1"
        say "[pick/${role}] no scontrol/sinfo on PATH — cannot filter anything."
        say "[pick/${role}] falling back to the first candidate: ${PICKED}"
        return 0
    fi
    load_groups
    load_sinfo
    say "[pick/${role}] candidates, best first: $*"
    say "[pick/${role}] your groups:$(my_groups)"
    for candidate in "$@"; do
        if ! dump="$(scontrol show partition "${candidate}" 2>&1)"; then
            say "[pick/${role}] REJECT ${candidate}: scontrol has no such partition."
            say "               Slurm does not show it to this account at all. In the"
            say "               field (2026-08-25) a submission to it answered"
            say "               'User's group not permitted to use this partition'."
            continue
        fi
        allow="$(printf '%s' "${dump}" | tr ' ' '\n' | sed -n 's/^AllowGroups=//p' | head -1)"
        if ! admitted="$(group_admits "${allow}")"; then
            say "[pick/${role}] REJECT ${candidate}: AllowGroups=${allow}, and you are in"
            say "               $(my_groups)— no overlap."
            continue
        fi
        nodes="$(partition_nodes "${candidate}")"
        if [ -z "${nodes}" ]; then
            say "[pick/${role}] REJECT ${candidate}: sinfo lists no node for it."
            continue
        fi
        live=""
        while IFS= read -r node; do
            [ -n "${node}" ] || continue
            state="${node##*:}"
            if node_state_is_live "${state}"; then
                live="${live} ${node}"
            fi
        done <<< "${nodes}"
        if [ -z "${live}" ]; then
            say "[pick/${role}] REJECT ${candidate}: every node is down or drained —" \
                "$(printf '%s' "${nodes}" | tr '\n' ' ')"
            continue
        fi
        maxtime="$(printf '%s' "${dump}" | tr ' ' '\n' | sed -n 's/^MaxTime=//p' | head -1)"
        PICKED="${candidate}"
        say "[pick/${role}] CHOSE ${candidate}: AllowGroups=${allow} (via ${admitted})," \
            "MaxTime=${maxtime:-?}, up:${live}"
        record_pick "${role}" \
            "${candidate} allow=${allow} via=${admitted} maxtime=${maxtime:-?} up=${live# }"
        return 0
    done
    say "REFUSING: no ${role} partition survived both gates (your groups, and a node"
    say "  that is up). The rejections above are the whole reason. Override with"
    say "  ODIN_BUILD_PARTITION / ODIN_BOARD_PARTITION / ODIN_JOINT_PARTITION if you"
    say "  know better than sinfo does."
    say "${RERUN_LINE}"
    return 2
}

# ---------------------------------------------------------------------------
# 0. env probe
# ---------------------------------------------------------------------------
phase_0() {
    head_line "phase 0: env probe"
    local report="${RESULTS}/env.txt"
    local failures=0 vitis="" vitis_root="" vitis_version=""
    if vitis="$(odin_vitis_settings)"; then
        vitis_root="${vitis%%|*}"
        vitis_version="${vitis#*|}"; vitis_version="${vitis_version%%|*}"
    fi
    if [ "${DRY_RUN}" = "1" ]; then
        say "[dry-run] would probe modules/vitis/xrt/slurm into ${report}"
        return 0
    fi
    {
        say "host      : $(hostname)"
        say "user      : ${USER}"
        say "groups    : $(my_groups)"
        say "date(utc) : $(utc)"
        say "package   : ${HERE}"
        say "data root : ${DATA_ROOT}"
        say "python3   : $(command -v python3 || echo MISSING) $(python3 -V 2>&1 || true)"
        say "sbatch    : $(command -v sbatch || echo MISSING)"
        say "sinfo     : $(command -v sinfo || echo MISSING)"
        say "vitis     : ${vitis_root:-NOT FOUND}${vitis_version:+/Vitis/${vitis_version}}"
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
            say "through /tools/Xilinx and /opt/xilinx directly."
        fi
        say ""
        say "--- sinfo -a -N ---"
        if command -v sinfo > /dev/null 2>&1; then sinfo -a -N 2>&1 || true; fi
        say ""
        say "--- scontrol show partition ---"
        if command -v scontrol > /dev/null 2>&1; then
            scontrol show partition 2>&1 || true
        fi
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
        "${DATA_ROOT}"/*) : ;;
        *)
            say "REFUSING: the package is unpacked at ${HERE}, not under ${DATA_ROOT}."
            say "  ${DATA_ROOT} is the ONLY path shared between the head node and the"
            say "  board VMs (hacc_demo/doc/1-FPGA-allocation.md line 155), so a"
            say "  job would not be able to read this directory at all."
            say "  FIX: run ./bootstrap_hacc.sh next to the uploaded zip; it unpacks"
            say "  to ${DATA_ROOT}/\${USER}/odin_hacc_package and launches this for you."
            failures=$((failures + 1))
            ;;
    esac
    if [ -z "${vitis}" ]; then
        say "REFUSING: no Vitis found under /tools/Xilinx, /tools/xilinx, /opt/Xilinx"
        say "  or /opt/xilinx. FIELD NOTE (2026-08-25): this cluster carries Vitis"
        say "  2024.2 under /tools/Xilinx — the capital X matters, and the vendor"
        say "  docs' 2022.2//tools/xilinx is stale. Pin it with XILINX_ROOT and"
        say "  VITIS_VERSION if it moved again."
        failures=$((failures + 1))
    else
        say "[phase0] Vitis ${vitis_version} at ${vitis_root} (discovered, not assumed)."
    fi
    if [ ! -f "$(odin_xrt_setup)" ]; then
        say "REFUSING: no XRT at $(odin_xrt_setup)."
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
        say "[phase0] ${failures} blocking problem(s)."
        return 2
    fi
    say "[phase0] OK"
}

# ---------------------------------------------------------------------------
# 1. selftest — the whole driver, no hardware
# ---------------------------------------------------------------------------
phase_1() {
    head_line "phase 1: driver selftest against a fake pyxrt (NO hardware)"
    require_file "${DRIVER}" "re-unzip the package; host/odin_board_driver.py is missing" || return 2
    require_file "${FAKE}" "re-unzip the package; the fake pyxrt is missing" || return 2
    require_file "${FIXTURES}" "re-unzip the package; fixtures/ is missing" || return 2
    do_cmd_logged "${RESULTS}/phase1_selftest.log" \
        python3 "${DRIVER}" --selftest --fake-pyxrt "${FAKE}" \
        --fixtures "${FIXTURES}" --results "${RESULTS}/selftest" || return 1
}

# ---------------------------------------------------------------------------
# 2/3. the builds — resumed from the xclbin and the sidecar that names the
# build script it came from, never from a stamp.
# ---------------------------------------------------------------------------
build_script_sha() {
    sha256sum "${BUILD_SCRIPT}" 2>/dev/null | cut -d' ' -f1
}

sidecar_of() { printf '%s.built_with\n' "$(xclbin_of "$1")"; }

write_sidecar() {
    local target="$1" sidecar
    sidecar="$(sidecar_of "${target}")"
    [ -f "$(xclbin_of "${target}")" ] || return 0
    {
        printf 'build_script_sha256=%s\n' "$(build_script_sha)"
        printf 'target=%s\n' "${target}"
        printf 'platform=%s\n' "${PLATFORM}"
        printf 'partition=%s\n' "${PICKED}"
        printf 'built_utc=%s\n' "$(utc)"
    } > "${sidecar}"
    say "[phase] recorded ${sidecar}"
}

submit_build() {
    local target="$1"
    local log="${RESULTS}/phase_build_${target}.log"
    local status=0
    do_cmd env \
        ODIN_PKG="${HERE}" ODIN_TARGET="${target}" ODIN_PLATFORM="${PLATFORM}" \
        ODIN_LOG="${log}" ODIN_EMU_SMOKE="${ODIN_EMU_SMOKE:-smallest}" \
        ODIN_EMU_SMOKE_TIMEOUT="${ODIN_EMU_SMOKE_TIMEOUT:-5400}" \
        sbatch --wait -p "${PICKED}" \
        "${HERE}/scripts/hacc/odin_build.sbatch" || status=$?
    if [ "${DRY_RUN}" = "0" ] && [ -f "${log}" ]; then
        tail -n 20 "${log}"
    fi
    if [ "${status}" -ne 0 ]; then
        say "[build/${target}] sbatch on '${PICKED}' exited ${status}."
        say "  Read ${log} and the slurm-odin-build-*.out on the node's /tmp."
        say "  Pin another partition with ODIN_BUILD_PARTITION=<name> if the pick"
        say "  above was wrong."
        say "${RERUN_LINE}"
        return "${status}"
    fi
    if [ "${DRY_RUN}" = "0" ]; then
        require_file "$(xclbin_of "${target}")" \
            "the job returned 0 but produced no xclbin; read ${log}" || return 2
        write_sidecar "${target}"
    fi
}

phase_2() {
    pick_partition build "${ODIN_BUILD_PARTITION:-}" "${BUILD_CANDIDATES[@]}" || return 2
    head_line "phase 2: build hw_emu on ${PICKED}, then smoke it THERE"
    say "hw_emu needs NO CARD: XCL_EMULATION_MODE=hw_emu binds XRT to the"
    say "emulation model v++ packaged into the xclbin, so the smoke runs on the"
    say "build node. It proves the packaged kernel opens by name, the register"
    say "map answers, the AXI master moves the payloads and the capture decodes"
    say "into the frozen counts. It cannot prove HBM ordering behind a real XDMA"
    say "shell, XRT's allocation on silicon, or timing closure — that is B0/B1."
    say "The smoke is BOUNDED: the smallest fixture only, under"
    say "${ODIN_EMU_SMOKE_TIMEOUT:-5400}s. ODIN_EMU_SMOKE=all runs all five."
    submit_build hw_emu
}

phase_3() {
    pick_partition build "${ODIN_BUILD_PARTITION:-}" "${BUILD_CANDIDATES[@]}" || return 2
    head_line "phase 3: build hw on ${PICKED} (the real bitstream)"
    say "Budget 2-6 h: place-and-route of one ODIN core plus the program and"
    say "capture RAMs. NEVER submit this to a board partition — those are capped"
    say "at one hour."
    submit_build hw
}

# ---------------------------------------------------------------------------
# 4. stage onto the shared filesystem
# ---------------------------------------------------------------------------
phase_4() {
    head_line "phase 4: stage the bitstream under ${STAGE_DIR}"
    if [ "${DRY_RUN}" = "0" ]; then
        require_file "$(xclbin_of hw)" "run phase 3 first (the hw build produces it)" \
            || return 2
    fi
    do_cmd mkdir -p "${STAGE_DIR}" "${DATA_ROOT}/${USER}/log" || return 1
    do_cmd cp "$(xclbin_of hw)" "${XCLBIN_STAGED}" || return 1
    do_cmd ls -l "${XCLBIN_STAGED}" || return 1
    say "[phase4] ${DATA_ROOT} is the only path the head node and the board VMs share."
}

# ---------------------------------------------------------------------------
# 5/6. the board
# ---------------------------------------------------------------------------
submit_board() {
    local mode="$1"
    local log="${RESULTS}/phase_board_${mode}.log"
    local status=0
    do_cmd env \
        ODIN_PKG="${HERE}" ODIN_XCLBIN="${XCLBIN_STAGED}" ODIN_MODE="${mode}" \
        ODIN_RESULTS="${RESULTS}/board_${mode}" ODIN_LOG="${log}" \
        sbatch --wait -p "${PICKED}" \
        "${HERE}/scripts/hacc/odin_board.sbatch" || status=$?
    if [ "${DRY_RUN}" = "0" ] && [ -f "${log}" ]; then
        tail -n 40 "${log}"
    fi
    if [ "${status}" -ne 0 ]; then
        say "[board/${mode}] sbatch on '${PICKED}' exited ${status}."
        say "  Read ${log}; the board partition is capped at ONE HOUR, so a"
        say "  timeout there is a queue fact, not a design finding."
        say "${RERUN_LINE}"
        return "${status}"
    fi
}

phase_5() {
    pick_partition board "${ODIN_BOARD_PARTITION:-}" "${BOARD_CANDIDATES[@]}" || return 2
    head_line "phase 5: B0 smoke on ${PICKED}"
    say "Load the xclbin, resolve odin_fpga_kernel_top with EXCLUSIVE access,"
    say "read capture_capacity (0x54), program_capacity (0x5C) and status"
    say "(0x4C). A zero capacity refuses at open by design: it means the host is"
    say "talking to something that is not this kernel."
    submit_board probe
}

phase_6() {
    pick_partition board "${ODIN_BOARD_PARTITION:-}" "${BOARD_CANDIDATES[@]}" || return 2
    head_line "phase 6: B1 parity campaign on ${PICKED}"
    say "Every shipped fixture, one certificate line each. Anything other than"
    say "PASS exact=1.000000 max|dcount|=0 is a finding, not a tolerance."
    submit_board run
}

# ---------------------------------------------------------------------------
# 7. the two-component run
# ---------------------------------------------------------------------------
phase_7() {
    pick_partition joint "${ODIN_JOINT_PARTITION:-}" "${JOINT_CANDIDATES[@]}" || return 2
    head_line "phase 7: board + independent reference in ONE job on ${PICKED}"
    say "The node behind this partition carries the U55C and the MI210 in the"
    say "same chassis, so one allocation already holds both components; see the"
    say "'WHY NOT --het-group' note in scripts/hacc/odin_joint.sbatch."
    local log="${RESULTS}/phase_joint.log"
    local status=0
    do_cmd env \
        ODIN_PKG="${HERE}" ODIN_XCLBIN="${XCLBIN_STAGED}" \
        ODIN_JOIN="${RESULTS}/joint" ODIN_LOG="${log}" \
        sbatch --wait -p "${PICKED}" \
        "${HERE}/scripts/hacc/odin_joint.sbatch" || status=$?
    if [ "${DRY_RUN}" = "0" ] && [ -f "${log}" ]; then
        tail -n 40 "${log}"
    fi
    if [ "${status}" -ne 0 ]; then
        say "[joint] sbatch on '${PICKED}' exited ${status}. Read ${log}."
        say "${RERUN_LINE}"
        return "${status}"
    fi
}

# ---------------------------------------------------------------------------
# What counts as done: the artifact, and nothing else.
# ---------------------------------------------------------------------------
DONE_WHY=""

built_artifact_ready() {
    local target="$1" xclbin sidecar recorded
    xclbin="$(xclbin_of "${target}")"
    sidecar="$(sidecar_of "${target}")"
    [ -f "${xclbin}" ] || return 1
    if [ ! -f "${sidecar}" ]; then
        DONE_WHY="rebuilding: ${xclbin} has no .built_with sidecar, so nothing"
        DONE_WHY="${DONE_WHY} says which build_xclbn.sh produced it"
        return 1
    fi
    recorded="$(sed -n 's/^build_script_sha256=//p' "${sidecar}" | head -1)"
    if [ "${recorded}" != "$(build_script_sha)" ]; then
        DONE_WHY="rebuilding: ${xclbin} was built with build_xclbn.sh"
        DONE_WHY="${DONE_WHY} ${recorded:0:12}…, the package now carries"
        DONE_WHY="${DONE_WHY} $(build_script_sha | cut -c1-12)…"
        return 1
    fi
    DONE_WHY="${xclbin} exists and its sidecar names this build_xclbn.sh"
    return 0
}

phase_done() {
    DONE_WHY=""
    case "$1" in
        2|3)
            local target="hw_emu"
            [ "$1" = "3" ] && target="hw"
            built_artifact_ready "${target}" && return 0
            [ -n "${DONE_WHY}" ] && say "[phase$1] ${DONE_WHY}"
            return 1
            ;;
        4)
            if [ -f "${XCLBIN_STAGED}" ] && [ -f "$(xclbin_of hw)" ] \
               && cmp -s "${XCLBIN_STAGED}" "$(xclbin_of hw)"; then
                DONE_WHY="${XCLBIN_STAGED} is already the built bitstream, byte for byte"
                return 0
            fi
            return 1
            ;;
        5)
            if [ -f "${RESULTS}/board_probe/probe.json" ]; then
                DONE_WHY="${RESULTS}/board_probe/probe.json — the CSRs were read"
                return 0
            fi
            return 1
            ;;
        6)
            if [ -f "${RESULTS}/board_run/summary_board.json" ]; then
                DONE_WHY="${RESULTS}/board_run/summary_board.json — the campaign certified"
                return 0
            fi
            return 1
            ;;
        7)
            if [ -f "${RESULTS}/joint/summary_join.json" ]; then
                DONE_WHY="${RESULTS}/joint/summary_join.json — the join is written"
                return 0
            fi
            return 1
            ;;
        *) return 1 ;;
    esac
}

print_status() {
    local phase
    say "phase   state      evidence"
    for phase in 0 1 2 3 4 5 6 7; do
        if phase_done "${phase}"; then
            printf '%-7s %-10s %s\n' "${phase}" "done" "${DONE_WHY}"
        elif [ "${phase}" -le 1 ]; then
            printf '%-7s %-10s %s\n' "${phase}" "always" "cheap; runs on every invocation"
        else
            printf '%-7s %-10s %s\n' "${phase}" "pending" "${DONE_WHY:-no artifact yet}"
        fi
    done
    if [ -s "${JOURNAL}" ]; then
        say ""
        say "last journal lines:"
        tail -n 8 "${JOURNAL}"
    fi
}

# ---------------------------------------------------------------------------

if [ "${STATUS_ONLY}" = "1" ]; then
    print_status
    exit 0
fi

acquire_lock

say "ODIN bring-up starting $(utc) — package ${HERE}"
say "pid $$; watch with tail -f ${RESULTS}/run_all.log"

for phase in 0 1 2 3 4 5 6 7; do
    if ! want "${phase}"; then
        continue
    fi
    if [ "${FORCE}" = "0" ] && [ "${phase}" -ge 2 ] && phase_done "${phase}"; then
        say "[phase${phase}] already done — ${DONE_WHY}"
        journal "phase${phase}" SKIP "${DONE_WHY}"
        continue
    fi
    journal "phase${phase}" START ""
    status=0
    if ! "phase_${phase}"; then
        status=1
    fi
    if [ "${status}" -ne 0 ]; then
        journal "phase${phase}" FAIL ""
        say ""
        say "[phase${phase}] STOPPED."
        say "${RERUN_LINE}"
        exit 1
    fi
    journal "phase${phase}" DONE ""
done

say ""
say "Done $(utc). Evidence is under ${RESULTS}; bring it home with ./collect_results.sh"
