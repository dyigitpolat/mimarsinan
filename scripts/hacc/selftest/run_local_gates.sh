#!/usr/bin/env bash
# The whole HACC package, exercised off-cluster against a STUB cluster.
#
#   scripts/hacc/selftest/run_local_gates.sh [workdir]
#
# It puts stub sbatch/sinfo/scontrol/squeue/groups binaries on PATH, backed by
# the 2026-08-25 field transcripts under field_2026_08_25/, points
# ODIN_DATA_ROOT at a scratch directory, and then runs the SHIPPED bootstrap and
# run_all.sh exactly as the owner would. Every gate below is a claim the package
# makes about itself; a red line here is a package defect.
#
#   1  the build pick is vck5000_compile, and cpu_only loses on a DOWN node
#   2  the joint pick is mi210_u280_u55c after mi210_vck_u55c is not permitted
#   3  the board pick is the platform partition, reached through `hgpu`
#   4  bootstrap verifies the zip's sha256, extracts, launches DETACHED
#   5  bootstrap ARCHIVES an existing install instead of deleting it
#   6  a corrupted upload is refused before anything is unpacked
#   7  kill -9 mid-phase, then re-run: the finished build is kept, not redone
#   8  a second live instance is refused by name and pid
#   9  a changed build_xclbn.sh makes the xclbin stale and the build run again
#  10  a failed sbatch ends with the one line that says work is kept
#  11  the hw_emu smoke is bounded: it times out, notes it honestly, continues
#  12  the driver selftest passes from the fresh extract
#
# Exit 0 all green, 1 a gate failed, 77 nothing to test (no dist package: run
# scripts/hacc/make_package.py first).

set -uo pipefail

SELFTEST="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${SELFTEST}/../../.." && pwd)"
STUBS="${SELFTEST}/stubs"
FIELD="${SELFTEST}/field_2026_08_25"
ZIP="${REPO}/dist/odin_hacc_package.zip"
BOOTSTRAP="${REPO}/dist/bootstrap_hacc.sh"

if [ ! -f "${ZIP}" ] || [ ! -f "${BOOTSTRAP}" ]; then
    echo "SKIPPING: no ${ZIP} / ${BOOTSTRAP}."
    echo "  Build them first:  env/bin/python scripts/hacc/make_package.py"
    exit 77
fi

WORK="${1:-$(mktemp -d)}"
mkdir -p "${WORK}"
FAILED=0

export ODIN_STUB_FIELD="${FIELD}"
ODIN_REAL_PYTHON3="$(command -v python3)"
export ODIN_REAL_PYTHON3
VITIS="${WORK}/tools/Xilinx/Vitis/2024.2"
export XILINX_ROOT="${WORK}/tools/Xilinx"
export XRT_ROOT="${WORK}/xrt"
export PATH="${STUBS}:${PATH}"

mkdir -p "${VITIS}/bin" "${XRT_ROOT}"
for tool in vivado v++ emconfigutil; do
    cp "${STUBS}/vitis_bin/${tool}" "${VITIS}/bin/${tool}"
done
# The nounset trap the 34216ceb fix exists for: Vitis 2024.x's settings read
# $PYTHONPATH, which `set -u` treats as fatal unless it is relaxed first.
cat > "${VITIS}/settings64.sh" <<EOF
export XILINX_VITIS="${VITIS}"
export PYTHONPATH="\${PYTHONPATH}:${VITIS}/python"
export PATH="${VITIS}/bin:\${PATH}"
EOF
: > "${XRT_ROOT}/setup.sh"

say() { printf '%s\n' "$*"; }
gate() { printf '\n--- gate %s: %s\n' "$1" "$2"; }
pass() { printf 'GATE %-4s PASS  %s\n' "$1" "$2"; }
fail() { printf 'GATE %-4s FAIL  %s\n' "$1" "$2"; FAILED=1; }

# Assert a pattern is (or is not) in a file, and say which.
expect() {
    local id="$1" file="$2" pattern="$3"
    if grep -qF -- "${pattern}" "${file}"; then
        pass "${id}" "saw: ${pattern}"
    else
        fail "${id}" "missing from ${file}: ${pattern}"
    fi
}
expect_file() {
    local id="$1" path="$2"
    if [ -e "${path}" ]; then pass "${id}" "exists: ${path}"
    else fail "${id}" "missing: ${path}"; fi
}

# A fresh extract under its own data root, so scenarios cannot leak into
# each other's artifacts.
fresh_package() {
    local root="$1"
    rm -rf "${root}"
    mkdir -p "${root}/${USER}"
    unzip -q "${ZIP}" -d "${root}/${USER}"
    printf '%s/%s/odin_hacc_package\n' "${root}" "${USER}"
}

wait_for() {                      # wait_for <seconds> <file> <pattern>
    local deadline=$(( SECONDS + $1 )) file="$2" pattern="$3"
    while [ "${SECONDS}" -lt "${deadline}" ]; do
        [ -f "${file}" ] && grep -qF -- "${pattern}" "${file}" && return 0
        sleep 0.2
    done
    return 1
}

# ===========================================================================
gate 1 "the build pick, from the cluster's own answers"
PICK_ROOT="${WORK}/picks"
PKG="$(fresh_package "${PICK_ROOT}")"
LOG="${WORK}/pick_build.log"
( cd "${PKG}" && ODIN_DATA_ROOT="${PICK_ROOT}" ./run_all.sh --dry-run --only 2 ) > "${LOG}" 2>&1
say "--- transcript ---"; sed -n '1,14p' "${LOG}"
expect 1a "${LOG}" "REJECT cpu_only: every node is down or drained"
expect 1b "${LOG}" "CHOSE vck5000_compile"
expect 1c "${LOG}" "AllowGroups=ALL"
expect 1d "${LOG}" "MaxTime=7-00:00:00"

gate 2 "the joint pick, after the partition that refused this account"
LOG="${WORK}/pick_joint.log"
( cd "${PKG}" && ODIN_DATA_ROOT="${PICK_ROOT}" ./run_all.sh --dry-run --only 7 ) > "${LOG}" 2>&1
say "--- transcript ---"; sed -n '1,14p' "${LOG}"
expect 2a "${LOG}" "REJECT mi210_vck_u55c: scontrol has no such partition"
expect 2b "${LOG}" "User's group not permitted to use this partition"
expect 2c "${LOG}" "CHOSE mi210_u280_u55c: AllowGroups=lab,hgpu,fpga_u280 (via hgpu)"

gate 3 "the board pick"
LOG="${WORK}/pick_board.log"
( cd "${PKG}" && ODIN_DATA_ROOT="${PICK_ROOT}" ./run_all.sh --dry-run --only 5 ) > "${LOG}" 2>&1
expect 3a "${LOG}" "CHOSE xilinx_u55c_gen3x16_xdma_3_202210_1: AllowGroups=lab,hgpu,fpga_u55c (via hgpu)"
expect 3b "${LOG}" "MaxTime=01:00:00"

# ===========================================================================
gate 4 "bootstrap: verify, place, launch detached, run to the end"
BOOT_ROOT="${WORK}/boot"
mkdir -p "${BOOT_ROOT}/${USER}"
LOG="${WORK}/bootstrap.log"
( cd "${WORK}" && ODIN_DATA_ROOT="${BOOT_ROOT}" "${BOOTSTRAP}" "${ZIP}" ) > "${LOG}" 2>&1
TARGET="${BOOT_ROOT}/${USER}/odin_hacc_package"
RUN_LOG="${TARGET}/results/run_all.log"
say "--- transcript ---"; cat "${LOG}"
expect 4a "${LOG}" "matches the packaged value"
expect 4b "${LOG}" "run_all.sh is detached"
expect 4c "${LOG}" "Safe to log out now"
if wait_for 180 "${RUN_LOG}" "Evidence is under"; then
    pass 4d "the detached run finished on its own"
else
    fail 4d "the detached run did not finish in 180s"
fi
expect 4e "${RUN_LOG}" "[selftest] refusals: 5/5 typed correctly"
expect_file 4f "${TARGET}/build/hacc/hw_nc1/odin_fpga_hw.xclbin.built_with"
expect_file 4g "${TARGET}/results/board_run/summary_board.json"
expect_file 4h "${TARGET}/results/joint/summary_join.json"
expect 4i "${TARGET}/results/partition_picks.txt" "vck5000_compile"
( cd "${TARGET}" && ./scripts/status.sh ) > "${WORK}/status.log" 2>&1
say "--- status.sh ---"; cat "${WORK}/status.log"
expect 4j "${WORK}/status.log" "NOT RUNNING: no lock is held"
expect 4k "${WORK}/status.log" "squeue --me"

gate 5 "bootstrap archives what is already there"
printf 'owner note\n' > "${TARGET}/OWNER_NOTE.txt"
LOG="${WORK}/bootstrap_again.log"
( cd "${WORK}" && ODIN_DATA_ROOT="${BOOT_ROOT}" "${BOOTSTRAP}" "${ZIP}" ) > "${LOG}" 2>&1
expect 5a "${LOG}" "archiving it whole to"
ARCHIVE="$(find "${BOOT_ROOT}/${USER}" -maxdepth 1 -name 'odin_prev_*.tar.gz' -print -quit)"
LISTING="${WORK}/archive_listing.txt"
[ -n "${ARCHIVE}" ] && tar -tzf "${ARCHIVE}" > "${LISTING}" 2>/dev/null
if [ -n "${ARCHIVE}" ] && grep -q 'OWNER_NOTE.txt' "${LISTING}"; then
    pass 5b "the previous install is inside $(basename "${ARCHIVE}"), OWNER_NOTE.txt and all"
else
    fail 5b "no odin_prev_*.tar.gz carrying the previous install"
fi
if [ -e "${TARGET}/OWNER_NOTE.txt" ]; then
    fail 5c "the replaced install still carries the previous OWNER_NOTE.txt"
else
    pass 5c "the new install is a clean extract, not a merge over the old one"
fi
wait_for 180 "${TARGET}/results/run_all.log" "Evidence is under" || true

gate 6 "a corrupted upload is refused"
cp "${ZIP}" "${WORK}/corrupt.zip"
printf 'trailing damage' >> "${WORK}/corrupt.zip"
LOG="${WORK}/bootstrap_corrupt.log"
( cd "${WORK}" && ODIN_DATA_ROOT="${WORK}/never" "${BOOTSTRAP}" "${WORK}/corrupt.zip" ) \
    > "${LOG}" 2>&1
status=$?
expect 6a "${LOG}" "not the package this bootstrap was cut for"
if [ "${status}" -eq 2 ]; then pass 6b "refused with exit 2"; else fail 6b "exit ${status}, wanted 2"; fi
if [ -d "${WORK}/never/${USER}/odin_hacc_package" ]; then
    fail 6c "it unpacked the corrupt zip anyway"
else
    pass 6c "nothing was unpacked"
fi

# ===========================================================================
gate 7 "kill -9 mid-phase, then resume"
KILL_ROOT="${WORK}/kill"
PKG="$(fresh_package "${KILL_ROOT}")"
mkdir -p "${PKG}/results"
RUN_LOG="${PKG}/results/run_all.log"
(
    cd "${PKG}" && ODIN_DATA_ROOT="${KILL_ROOT}" \
        ODIN_STUB_SBATCH_SLEEP=120 ODIN_STUB_SBATCH_SLEEP_TARGET=hw \
        setsid ./run_all.sh > "${RUN_LOG}" 2>&1 &
)
if wait_for 120 "${RUN_LOG}" "phase 3: build hw"; then
    victim="$(cat "${PKG}/.run_all.lock/pid")"
    kill -9 -"${victim}" 2>/dev/null || kill -9 "${victim}" 2>/dev/null || true
    sleep 0.5
    pass 7a "killed pid ${victim} (whole process group) during phase 3"
else
    fail 7a "phase 3 never started"
fi
expect_file 7b "${PKG}/build/hacc/hw_emu_nc1/odin_fpga_hw_emu.xclbin"
if [ -f "${PKG}/build/hacc/hw_nc1/odin_fpga_hw.xclbin" ]; then
    fail 7c "the hw build somehow completed; the resume gate would prove nothing"
else
    pass 7c "the hw xclbin is absent, as an interrupted build leaves it"
fi
LOG="${WORK}/resume.log"
( cd "${PKG}" && ODIN_DATA_ROOT="${KILL_ROOT}" ./run_all.sh ) > "${LOG}" 2>&1
say "--- transcript ---"; sed -n '1,12p' "${LOG}"
expect 7d "${LOG}" "which is gone; taking it over"
expect 7e "${LOG}" "[phase2] already done"
expect 7f "${LOG}" "exists and its sidecar names this build_xclbn.sh"
expect 7g "${LOG}" "phase 3: build hw"
expect 7h "${LOG}" "Evidence is under"

gate 8 "a second live instance is refused"
(
    cd "${PKG}" && ODIN_DATA_ROOT="${KILL_ROOT}" ODIN_STUB_SBATCH_SLEEP=60 \
        ODIN_STUB_SBATCH_SLEEP_TARGET=probe \
        setsid ./run_all.sh --force > "${PKG}/results/live.log" 2>&1 &
)
if wait_for 60 "${PKG}/results/live.log" "phase 5: B0 smoke"; then
    LOG="${WORK}/second_instance.log"
    ( cd "${PKG}" && ODIN_DATA_ROOT="${KILL_ROOT}" ./run_all.sh ) > "${LOG}" 2>&1
    status=$?
    say "--- transcript ---"; cat "${LOG}"
    expect 8a "${LOG}" "REFUSING: run_all.sh is already running here as pid"
    if [ "${status}" -eq 3 ]; then pass 8b "refused with exit 3"; else fail 8b "exit ${status}, wanted 3"; fi
    holder="$(cat "${PKG}/.run_all.lock/pid" 2>/dev/null || echo '')"
    [ -n "${holder}" ] && kill -9 -"${holder}" 2>/dev/null
    rm -rf "${PKG}/.run_all.lock"
else
    fail 8a "the live instance never reached a slow phase"
fi

gate 9 "a changed build script makes the xclbin stale"
printf '\n# owner edit, %s\n' "$(date -u +%s)" >> "${PKG}/scripts/hacc/build_xclbn.sh"
LOG="${WORK}/stale.log"
( cd "${PKG}" && ODIN_DATA_ROOT="${KILL_ROOT}" ./run_all.sh ) > "${LOG}" 2>&1
expect 9a "${LOG}" "rebuilding:"
expect 9b "${LOG}" "the package now carries"
expect 9c "${LOG}" "phase 2: build hw_emu"
expect 9d "${LOG}" "[phase6] already done"

gate 10 "a failed sbatch ends with the one line"
FAIL_ROOT="${WORK}/failing"
PKG="$(fresh_package "${FAIL_ROOT}")"
LOG="${WORK}/sbatch_failure.log"
( cd "${PKG}" && ODIN_DATA_ROOT="${FAIL_ROOT}" ODIN_STUB_SBATCH_FAIL=hw_emu ./run_all.sh ) \
    > "${LOG}" 2>&1
status=$?
say "--- transcript ---"; tail -n 8 "${LOG}"
expect 10a "${LOG}" "fix, then re-run ./run_all.sh — completed work is kept"
if [ "${status}" -eq 1 ]; then pass 10b "stopped with exit 1"; else fail 10b "exit ${status}, wanted 1"; fi

# ===========================================================================
gate 11 "the hw_emu smoke is bounded and says so"
EMU_ROOT="${WORK}/emu"
PKG="$(fresh_package "${EMU_ROOT}")"
LOG="${WORK}/emu_smoke.log"
(
    cd "${PKG}" && PATH="${STUBS}/hang:${PATH}" \
        ODIN_PKG="${PKG}" ODIN_TARGET=hw_emu ODIN_EMU_SMOKE_TIMEOUT=2 \
        bash scripts/hacc/odin_build.sbatch
) > "${LOG}" 2>&1
status=$?
say "--- transcript ---"; tail -n 16 "${LOG}"
if [ "${status}" -eq 0 ]; then pass 11a "the build stands (exit 0) after the smoke timed out"
else fail 11a "exit ${status}, wanted 0"; fi
expect 11b "${LOG}" "emu smoke: nc1_single_core_ceiling only, timeout 2s"
expect_file 11c "${PKG}/results/hw_emu/EMU_SMOKE_SKIPPED.txt"
expect 11d "${PKG}/results/hw_emu/EMU_SMOKE_SKIPPED.txt" "wall is UNKNOWN"
expect 11e "${PKG}/results/hw_emu/EMU_SMOKE_SKIPPED.txt" "B0/B1 on silicon supersede"
LOG="${WORK}/emu_smoke_all.log"
(
    cd "${PKG}" && PATH="${STUBS}/hang:${PATH}" \
        ODIN_PKG="${PKG}" ODIN_TARGET=hw_emu ODIN_EMU_SMOKE=all ODIN_EMU_SMOKE_TIMEOUT=2 \
        bash scripts/hacc/odin_build.sbatch
) > "${LOG}" 2>&1
expect 11f "${LOG}" "emu smoke: every shipped fixture"

gate 12 "the driver selftest, from the fresh extract"
LOG="${WORK}/selftest.log"
(
    cd "${PKG}" && python3 host/odin_board_driver.py --selftest \
        --fake-pyxrt host/fake_pyxrt_for_selftest.py \
        --fixtures fixtures --results "${WORK}/selftest_results"
) > "${LOG}" 2>&1
status=$?
if [ "${status}" -eq 0 ]; then pass 12a "green from the extracted package"
else fail 12a "exit ${status}; see ${LOG}"; fi
expect 12b "${LOG}" "refusals: 5/5 typed correctly"

# ===========================================================================
say ""
if [ "${FAILED}" = "0" ]; then
    say "ALL GATES GREEN (work under ${WORK})"
else
    say "GATES FAILED (work under ${WORK})"
fi
exit "${FAILED}"
