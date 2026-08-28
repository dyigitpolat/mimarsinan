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
#  13  ODIN_CARD=u250 picks the 202020_1 shell, Vitis 2020.2, the U250 part and
#      the DDR connectivity config — asserted on the v++ stub's own argv
#  14  ODIN_CARD=u55c on the CURRENT field evidence REFUSES EARLY, names the
#      admin fix, submits nothing — and SELF-HEALS the moment 2022.2 appears
#  15  bootstrap --card u250 threads the card into the detached run
#  16  the DEPLOYMENT executor runs the two-core bundle as host-mediated passes
#      against the fake pyxrt, certifies every pass, and reports ACCURACY
#  17  the deployment mutations: tampered expected counts go RED, a tampered
#      self-hash refuses as OdinBundleCorrupt, and a wrong stimulus gets no
#      verdict instead of another pass's counts
#  18  the die-map renderer draws on both paths and refuses without a checkpoint
#  19  run_all.sh PHASE 8 stages the bundle, runs it, collects the report, and
#      is artifact-resume aware; collect_results.sh brings it home
#  21  the DEPLOYMENT package (when dist/odin_hacc_deployment.zip exists):
#      its deployment/DEPLOYMENT.json names the exported network, and phase 8
#      runs THAT bundle rather than the committed witness one
#  20  the CHIP CACHE: a build publishes, a second install hits it and spends no
#      compile slot, adoption files an existing install under the key a build
#      looks up, a held lock never clobbers, and a changed recipe misses
#
# Exit 0 all green, 1 a gate failed, 77 nothing to test (no dist package: run
# scripts/hacc/make_package.py first).

set -uo pipefail

SELFTEST="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${SELFTEST}/../../.." && pwd)"
STUBS="${SELFTEST}/stubs"
FIELD="${SELFTEST}/field_2026_08_25"
ZIP="${REPO}/dist/odin_hacc_package.zip"
DEPLOY_ZIP="${REPO}/dist/odin_hacc_deployment.zip"
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
export XILINX_ROOT="${WORK}/tools/Xilinx"
export XRT_ROOT="${WORK}/xrt"
export PATH="${STUBS}:${PATH}"
mkdir -p "${XRT_ROOT}"
: > "${XRT_ROOT}/setup.sh"

# The installed Vitis set, straight out of the field transcript — 2020.1 2020.2
# 2021.2 2022.1 2023.2 2024.2, and NO 2022.2. Each one gets the stub binaries
# and a settings64.sh that exports XILINX_VITIS to ITSELF, so a gate can prove
# which version the card profile actually chose.
install_vitis() {                  # install_vitis <root> <version...>
    local root="$1" version dir
    shift
    for version in "$@"; do
        dir="${root}/Vitis/${version}"
        mkdir -p "${dir}/bin"
        for tool in vivado v++ emconfigutil; do
            cp "${STUBS}/vitis_bin/${tool}" "${dir}/bin/${tool}"
        done
        # The nounset trap the 34216ceb fix exists for: Vitis 2024.x's settings
        # read $PYTHONPATH, which `set -u` treats as fatal unless relaxed first.
        cat > "${dir}/settings64.sh" <<EOF
export XILINX_VITIS="${dir}"
export PYTHONPATH="\${PYTHONPATH}:${dir}/python"
export PATH="${dir}/bin:\${PATH}"
EOF
    done
}
# shellcheck disable=SC2046  # the transcript is one version per line, by design
install_vitis "${XILINX_ROOT}" $(cat "${FIELD}/vitis_installed.txt")

# /opt/xilinx/platforms as the field found it: one 2022.2-locked U55C shell and
# the two U250 shells. NOT exported globally — the card gate must stay silent
# where there is no evidence to read, which is every gate below that does not
# opt in with ODIN_PLATFORM_ROOT.
PLATFORM_ROOT="${WORK}/platforms"
while IFS= read -r platform; do
    [ -n "${platform}" ] || continue
    mkdir -p "${PLATFORM_ROOT}/${platform}"
done < "${FIELD}/platforms_installed.txt"

say() { printf '%s\n' "$*"; }
gate() { printf '\n--- gate %s: %s\n' "$1" "$2"; }
pass() { printf 'GATE %-4s PASS  %s\n' "$1" "$2"; }
skip() { printf 'GATE %-4s SKIP  %s\n' "$1" "$2"; }
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
    local root="$1" archive="${2:-${ZIP}}"
    rm -rf "${root}"
    mkdir -p "${root}/${USER}"
    unzip -q "${archive}" -d "${root}/${USER}"
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
expect 4e "${RUN_LOG}" "[selftest] refusals: 4/4 typed correctly"
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
expect 12b "${LOG}" "refusals: 4/4 typed correctly"

# ===========================================================================
# THE CARD IS A PARAMETER (v3). Everything above ran on the default card with
# no platform evidence to read, which is exactly the off-cluster case. The
# three gates below hand the package the FIELD's evidence — the installed
# platform list and the installed Vitis list — and check both answers it must
# give: build the U250, refuse the U55C.
# ===========================================================================
gate 13 "ODIN_CARD=u250: the 202020_1 shell, Vitis 2020.2, the U250 part, DDR"
U250_ROOT="${WORK}/u250"
PKG="$(fresh_package "${U250_ROOT}")"
LOG="${WORK}/u250_pick.log"
(
    cd "${PKG}" && ODIN_DATA_ROOT="${U250_ROOT}" ODIN_CARD=u250 \
        ODIN_PLATFORM_ROOT="${PLATFORM_ROOT}" ./run_all.sh --dry-run --only 5
) > "${LOG}" 2>&1
say "--- transcript ---"; sed -n '1,14p' "${LOG}"
expect 13a "${LOG}" "card     : u250   (platform xilinx_u250_gen3x16_xdma_3_1_202020_1, config scripts/hacc/odin_u250.cfg)"
expect 13b "${LOG}" "CHOSE xilinx_u250_gen3x16_xdma_3_1_202020_1: AllowGroups=lab,fpga_u250 (via fpga_u250)"
expect 13c "${LOG}" "MaxTime=01:00:00"

LOG="${WORK}/u250_joint_pick.log"
(
    cd "${PKG}" && ODIN_DATA_ROOT="${U250_ROOT}" ODIN_CARD=u250 \
        ODIN_PLATFORM_ROOT="${PLATFORM_ROOT}" ./run_all.sh --dry-run --only 7
) > "${LOG}" 2>&1
expect 13d "${LOG}" "CHOSE u250_standard_reservation_pool"
expect 13e "${LOG}" "These nodes carry no GPU, and the join never needed one"

# The build itself, off-cluster, through the SHIPPED sbatch and build script.
VXX_TRACE="${WORK}/u250_vxx_trace.txt"
LOG="${WORK}/u250_build.log"
(
    cd "${PKG}" && ODIN_PKG="${PKG}" ODIN_TARGET=hw ODIN_CARD=u250 \
        ODIN_PLATFORM_ROOT="${PLATFORM_ROOT}" ODIN_STUB_VXX_TRACE="${VXX_TRACE}" \
        bash scripts/hacc/odin_build.sbatch
) > "${LOG}" 2>&1
status=$?
say "--- transcript ---"; sed -n '1,20p' "${LOG}"
if [ "${status}" -eq 0 ]; then pass 13f "the u250 build ran to the end (exit 0)"
else fail 13f "exit ${status}; see ${LOG}"; fi
expect 13g "${LOG}" "[hacc-build] card     : u250"
expect 13h "${LOG}" "[hacc-build] platform : xilinx_u250_gen3x16_xdma_3_1_202020_1"
expect 13i "${LOG}" "/Vitis/2020.2 (prefer 2020.2)"
expect 13j "${PKG}/build/hacc/hw_nc1/pack_kernel.tcl" "-part xcu250-figd2104-2L-e"
if [ -f "${VXX_TRACE}" ]; then
    say "--- v++ argv ---"; cat "${VXX_TRACE}"
    expect 13k "${VXX_TRACE}" "--config scripts/hacc/odin_u250.cfg"
    expect 13l "${VXX_TRACE}" "--platform xilinx_u250_gen3x16_xdma_3_1_202020_1"
else
    fail 13k "the stub v++ was never invoked; no argv to assert"
fi
expect 13m "${PKG}/scripts/hacc/odin_u250.cfg" "sp=odin_0.m_axi_gmem:DDR[0]"

gate 14 "ODIN_CARD=u55c refuses EARLY on the field's own evidence"
DEAD_ROOT="${WORK}/deadlock"
PKG="$(fresh_package "${DEAD_ROOT}")"
LOG="${WORK}/u55c_refusal.log"
TRACE="${WORK}/u55c_sbatch_trace.txt"
: > "${TRACE}"
(
    cd "${PKG}" && ODIN_DATA_ROOT="${DEAD_ROOT}" ODIN_CARD=u55c \
        ODIN_PLATFORM_ROOT="${PLATFORM_ROOT}" ODIN_STUB_TRACE="${TRACE}" ./run_all.sh
) > "${LOG}" 2>&1
status=$?
say "--- transcript ---"; cat "${LOG}"
if [ "${status}" -eq 2 ]; then pass 14a "refused with exit 2"; else fail 14a "exit ${status}, wanted 2"; fi
expect 14b "${LOG}" "REFUSING: ODIN_CARD=u55c is DEADLOCKED on this cluster (field-observed 2026-08-25)."
expect 14c "${LOG}" "THE ADMIN FIX, and it is the only one: install Vitis/Vivado 2022.2 alongside"
expect 14d "${LOG}" "ODIN_CARD=u250 ./run_all.sh"
if [ -s "${TRACE}" ]; then
    fail 14e "it submitted something before refusing: $(cat "${TRACE}")"
else
    pass 14e "no sbatch was issued — the refusal costs no queue slot"
fi
if [ -d "${PKG}/.run_all.lock" ]; then
    fail 14f "it took the run lock before refusing"
else
    pass 14f "the run lock was never taken"
fi
# SELF-HEAL: the moment 2022.2 exists, the same package stops refusing.
install_vitis "${WORK}/tools_healed" 2020.2 2022.2 2024.2
LOG="${WORK}/u55c_selfheal.log"
(
    cd "${PKG}" && ODIN_DATA_ROOT="${DEAD_ROOT}" ODIN_CARD=u55c \
        XILINX_ROOT="${WORK}/tools_healed" ODIN_PLATFORM_ROOT="${PLATFORM_ROOT}" \
        ./run_all.sh --dry-run --only 5
) > "${LOG}" 2>&1
if grep -qF 'DEADLOCKED' "${LOG}"; then
    fail 14g "it still refuses with 2022.2 installed — the guard is a hard-coded verdict"
else
    pass 14g "with 2022.2 installed the refusal is gone by itself"
fi
expect 14h "${LOG}" "CHOSE xilinx_u55c_gen3x16_xdma_3_202210_1"

gate 15 "bootstrap --card u250 threads the card into the detached run"
CARD_BOOT="${WORK}/cardboot"
mkdir -p "${CARD_BOOT}/${USER}"
LOG="${WORK}/bootstrap_card.log"
(
    cd "${WORK}" && ODIN_DATA_ROOT="${CARD_BOOT}" ODIN_PLATFORM_ROOT="${PLATFORM_ROOT}" \
        "${BOOTSTRAP}" --card u250 "${ZIP}"
) > "${LOG}" 2>&1
TARGET="${CARD_BOOT}/${USER}/odin_hacc_package"
RUN_LOG="${TARGET}/results/run_all.log"
say "--- transcript ---"; cat "${LOG}"
expect 15a "${LOG}" "card u250 — platform xilinx_u250_gen3x16_xdma_3_1_202020_1"
expect 15b "${LOG}" "run_all.sh is detached"
if wait_for 180 "${RUN_LOG}" "Evidence is under"; then
    pass 15c "the detached u250 run finished on its own"
else
    fail 15c "the detached u250 run did not finish in 180s"
fi
expect 15d "${RUN_LOG}" "card     : u250"
expect 15e "${TARGET}/results/partition_picks.txt" "xilinx_u250_gen3x16_xdma_3_1_202020_1"
expect 15f "${TARGET}/build/hacc/hw_nc1/odin_fpga_hw.xclbin.built_with" "card=u250"
LOG="${WORK}/bootstrap_bad_card.log"
(
    cd "${WORK}" && ODIN_DATA_ROOT="${WORK}/nevercard" "${BOOTSTRAP}" --card u9000 "${ZIP}"
) > "${LOG}" 2>&1
status=$?
expect 15g "${LOG}" "unknown --card 'u9000'"
if [ "${status}" -eq 2 ]; then pass 15h "refused with exit 2"; else fail 15h "exit ${status}, wanted 2"; fi


# ===========================================================================
# THE DEPLOYMENT (P8). Everything above proves the package can build and drive
# a bitstream; the gates below prove it can DEPLOY a multi-core network on the
# NC=1 one it builds, by running each core as its own pass and transcoding
# between them on the host.
# ===========================================================================
gate 16 "the deployment executor: two host-mediated passes, certified and timed"
DEPLOY_ROOT="${WORK}/deploy"
PKG="$(fresh_package "${DEPLOY_ROOT}")"
LOG="${WORK}/deployment.log"
(
    cd "${PKG}" && python3 host/odin_deployment_executor.py \
        --xclbin /selftest/no-such.xclbin \
        --fake-pyxrt host/fake_pyxrt_for_selftest.py \
        --replay deployment/nc1_two_core_passes_capture.json \
        --results "${PKG}/results/deployment"
) > "${LOG}" 2>&1
status=$?
say "--- transcript ---"; cat "${LOG}"
if [ "${status}" -eq 0 ]; then pass 16a "the campaign closed green (exit 0)"
else fail 16a "exit ${status}; see ${LOG}"; fi
expect 16b "${LOG}" "[SpikeCountCertificate] spike-count certificate [odin_fpga/exact]: PASS exact=1.000000 max|dcount|=0"
expect 16c "${LOG}" "core 0: programmed once"
expect 16d "${LOG}" "core 1: programmed once"
expect 16e "${LOG}" "[deploy] ACCURACY : 0.750000"
expect 16f "${LOG}" "transcode_s"
expect_file 16g "${PKG}/results/deployment/deployment_report.json"
expect_file 16h "${PKG}/results/deployment/deployment_samples.tsv"
REPORT="${PKG}/results/deployment/deployment_report.json"
if python3 - "${REPORT}" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
stages = ("bo_write_s", "sync_s", "run_s", "readback_s", "decode_s",
          "transcode_s", "pass_total_s")
missing = [s for s in stages if s not in report["walls"]["per_stage"]]
assert not missing, missing
assert report["walls"]["passes"] == 8, report["walls"]["passes"]
assert report["passed"] and report["accuracy"] == 0.75
assert all(row["passed"] for row in report["certificates"])
assert len(report["per_core"]) == 2
PY
then pass 16i "the report carries every stage's percentiles, 8 passes, 2 cores"
else fail 16i "the report is missing a wall or a pass"; fi

gate 17 "the deployment mutations"
python3 - "${PKG}" <<'PY'
import json, os, sys
root = sys.argv[1]
sys.path.insert(0, os.path.join(root, "host"))
import odin_deployment_bundle as b
deploy = os.path.join(root, "deployment")
def load(name): return json.load(open(os.path.join(deploy, name)))
def dump(name, doc):
    with open(os.path.join(deploy, name), "w") as handle:
        json.dump(b.seal(doc), handle, sort_keys=True, separators=(",", ":"))
# (1) a legitimately SEALED bundle whose frozen counts are wrong by one
bad = load("nc1_two_core_passes.json")
bad["certification"]["windows"]["0"]["1"][0][0] += 1
dump("mutant_counts.json", bad)
rep = load("nc1_two_core_passes_capture.json")
rep["bundle_self_hash"] = json.load(open(
    os.path.join(deploy, "mutant_counts.json")))["self_hash"]
dump("mutant_counts_capture.json", rep)
# (2) a bundle whose bytes were edited and NOT resealed
raw = open(os.path.join(deploy, "nc1_two_core_passes.json")).read()
open(os.path.join(deploy, "mutant_seal.json"), "w").write(
    raw.replace('"NC=1', '"nc=1', 1))
# (3) a replay whose stimulus keys are for another program: no verdict
rep = load("nc1_two_core_passes_capture.json")
for run in rep["runs"]:
    run["stimulus_sha256"] = "0" * 64
dump("mutant_replay.json", rep)
PY
LOG="${WORK}/deploy_mutant_counts.log"
(
    cd "${PKG}" && python3 host/odin_deployment_executor.py --xclbin x \
        --fake-pyxrt host/fake_pyxrt_for_selftest.py \
        --bundle deployment/mutant_counts.json \
        --replay deployment/mutant_counts_capture.json \
        --results "${PKG}/results/mutant_counts"
) > "${LOG}" 2>&1
status=$?
if [ "${status}" -eq 1 ]; then pass 17a "a tampered expectation exits 1, not 0"
else fail 17a "exit ${status}, wanted 1"; fi
expect 17b "${LOG}" "FAIL exact="

LOG="${WORK}/deploy_mutant_seal.log"
(
    cd "${PKG}" && python3 host/odin_deployment_executor.py --xclbin x \
        --fake-pyxrt host/fake_pyxrt_for_selftest.py \
        --bundle deployment/mutant_seal.json \
        --replay deployment/nc1_two_core_passes_capture.json \
        --results "${PKG}/results/mutant_seal"
) > "${LOG}" 2>&1
status=$?
if [ "${status}" -eq 2 ]; then pass 17c "a tampered self-hash refuses with exit 2"
else fail 17c "exit ${status}, wanted 2"; fi
expect 17d "${LOG}" "OdinBundleCorrupt"
expect 17e "${LOG}" "frozen evidence"

LOG="${WORK}/deploy_mutant_replay.log"
(
    cd "${PKG}" && python3 host/odin_deployment_executor.py --xclbin x \
        --fake-pyxrt host/fake_pyxrt_for_selftest.py \
        --replay deployment/mutant_replay.json \
        --results "${PKG}/results/mutant_replay"
) > "${LOG}" 2>&1
status=$?
if [ "${status}" -eq 2 ]; then pass 17f "a stimulus the fabric never saw gets NO VERDICT"
else fail 17f "exit ${status}, wanted 2"; fi
expect 17g "${LOG}" "NO-VERDICT sentinel"

gate 18 "the die-map renderer, on both paths"
CSV="${WORK}/placement.csv"
python3 - "${CSV}" <<'PY'
import sys
rows = ["name,class,site,x,y"]
for index in range(600):
    kind = ("odin_core" if index % 3 == 0
            else "sequencer" if index % 3 == 1 else "shell")
    rows.append(f"cell_{index},{kind},SLICE_X{index}Y{index},"
                f"{index % 48},{index % 41}")
open(sys.argv[1], "w").write("\n".join(rows) + "\n")
PY
LOG="${WORK}/die_map.log"
( cd "${PKG}" && python3 host/render_die_map.py --csv "${CSV}" \
    --out "${WORK}/die.svg" --format svg ) > "${LOG}" 2>&1
status=$?
if [ "${status}" -eq 0 ]; then pass 18a "the stdlib SVG path drew the map"
else fail 18a "exit ${status}; see ${LOG}"; fi
expect 18b "${LOG}" "svg(stdlib)"
expect 18c "${WORK}/die.svg" "<rect"
expect 18d "${WORK}/die.svg" "#9aa0a6"
LOG="${WORK}/die_map_auto.log"
( cd "${PKG}" && python3 host/render_die_map.py --csv "${CSV}" \
    --out "${WORK}/die_auto.png" ) > "${LOG}" 2>&1
if [ -f "${WORK}/die_auto.png" ] || [ -f "${WORK}/die_auto.svg" ]; then
    pass 18e "the auto path produced a map, and named which one: $(cat "${LOG}")"
else
    fail 18e "the auto path drew nothing; see ${LOG}"
fi
LOG="${WORK}/die_map_absent.log"
( cd "${PKG}" && python3 host/render_die_map.py --csv "${WORK}/nope.csv" \
    --out "${WORK}/x.svg" ) > "${LOG}" 2>&1
status=$?
if [ "${status}" -eq 2 ]; then pass 18f "no checkpoint CSV refuses with exit 2"
else fail 18f "exit ${status}, wanted 2"; fi
expect 18g "${LOG}" "mine_checkpoint.sh"


gate 19 "phase 8 — HACC NUS - ODIN Deployment, end to end under stub slurm"
P8_ROOT="${WORK}/phase8"
PKG="$(fresh_package "${P8_ROOT}")"
LOG="${WORK}/phase8.log"
( cd "${PKG}" && ODIN_DATA_ROOT="${P8_ROOT}" ODIN_CHIP_CACHE="${WORK}/cache_p8" \
    ./run_all.sh ) > "${LOG}" 2>&1
status=$?
say "--- transcript (phase 8) ---"; sed -n '/phase 8:/,$p' "${LOG}" | head -30
if [ "${status}" -eq 0 ]; then pass 19a "the whole flow reached the end (exit 0)"
else fail 19a "exit ${status}; see ${LOG}"; fi
expect 19b "${LOG}" "phase 8: HACC NUS - ODIN Deployment"
expect 19c "${LOG}" "one host-mediated PASS per core"
expect 19d "${LOG}" "nc1_two_core_passes.json"
expect_file 19e "${PKG}/results/board_deploy/deployment_report.json"
expect_file 19f "${PKG}/results/board_deploy/deployment_samples.tsv"
expect 19g "${PKG}/results/phase_deployment.log" "ACCURACY : 0.750000"
LOG="${WORK}/phase8_resume.log"
( cd "${PKG}" && ODIN_DATA_ROOT="${P8_ROOT}" ODIN_CHIP_CACHE="${WORK}/cache_p8" \
    ./run_all.sh ) > "${LOG}" 2>&1
expect 19h "${LOG}" "[phase8] already done"
expect 19i "${LOG}" "deployment ran and reported its accuracy"
LOG="${WORK}/phase8_status.log"
( cd "${PKG}" && ODIN_DATA_ROOT="${P8_ROOT}" ./run_all.sh --status ) > "${LOG}" 2>&1
expect 19j "${LOG}" "deployment_report.json"
LOG="${WORK}/collect.log"
( cd "${PKG}" && ODIN_DATA_ROOT="${P8_ROOT}" ODIN_CHIP_CACHE="${WORK}/cache_p8" \
    ./collect_results.sh "${WORK}/collected" ) > "${LOG}" 2>&1
say "--- collect_results.sh ---"; cat "${LOG}"
TARBALL="$(find "${WORK}/collected" -name 'odin_hacc_results_*.tar.gz' -print -quit)"
# The listing is written out first: `tar | grep -q` closes the pipe on the first
# match and SIGPIPEs tar, which `pipefail` then reports as a failure.
COLLECTED_LIST="${WORK}/collected_listing.txt"
: > "${COLLECTED_LIST}"
[ -n "${TARBALL}" ] && tar -tzf "${TARBALL}" > "${COLLECTED_LIST}"
expect 19k "${COLLECTED_LIST}" "results/board_deploy/deployment_report.json"
expect 19l "${COLLECTED_LIST}" "chip_cache/hw/key_inputs.txt"
expect 19m "${COLLECTED_LIST}" "results/board_deploy/deployment_samples.tsv"

gate 21 "the DEPLOYMENT package: its index is the network phase 8 runs"
# scripts/hacc/make_package.py --deployment <bundle> writes this second
# artifact: the same bring-up package plus an EXPORTED network and the index
# that names it. There is one bootstrap flow either way; what changes is which
# bundle phase 8 picks up.
if [ ! -f "${DEPLOY_ZIP}" ]; then
    skip 21a "no ${DEPLOY_ZIP}; build it with make_package.py --deployment BUNDLE"
else
    D_ROOT="${WORK}/deploypkg"
    PKG="$(fresh_package "${D_ROOT}" "${DEPLOY_ZIP}")"
    expect_file 21a "${PKG}/deployment/DEPLOYMENT.json"
    DEFAULT_BUNDLE="$(sed -n 's/.*"default"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        "${PKG}/deployment/DEPLOYMENT.json" | head -n 1)"
    expect_file 21b "${PKG}/${DEFAULT_BUNDLE}"
    LOG="${WORK}/deploypkg.log"
    ( cd "${PKG}" && ODIN_DATA_ROOT="${D_ROOT}" ODIN_CHIP_CACHE="${WORK}/cache_d" \
        ./run_all.sh ) > "${LOG}" 2>&1
    status=$?
    say "--- transcript (deployment phase 8) ---"
    sed -n '/phase 8:/,$p' "${LOG}" | head -20
    if [ "${status}" -eq 0 ]; then pass 21c "the deployment flow reached the end"
    else fail 21c "exit ${status}; see ${LOG}"; fi
    expect 21d "${LOG}" "deployment index names $(basename "${DEFAULT_BUNDLE}")"
    expect_file 21e "${PKG}/results/board_deploy/deployment_report.json"
    expect 21f "${PKG}/results/phase_deployment.log" "ACCURACY"
fi

gate 20 "the chip cache: publish, hit, adopt, lock, and a changed recipe"
CACHE_ROOT="${WORK}/chipcache"
C1_ROOT="${WORK}/cache_one"
PKG="$(fresh_package "${C1_ROOT}")"
LOG="${WORK}/cache_build.log"
( cd "${PKG}" && ODIN_DATA_ROOT="${C1_ROOT}" ODIN_CHIP_CACHE="${CACHE_ROOT}" \
    ./run_all.sh ) > "${LOG}" 2>&1
KEY="$( cd "${PKG}" && ODIN_DATA_ROOT="${C1_ROOT}" ODIN_CHIP_CACHE="${CACHE_ROOT}" \
    ./scripts/chip_cache.sh key hw | sed -n 's/^key=//p' )"
expect_file 20a "${CACHE_ROOT}/${KEY}/odin_fpga.xclbin"
expect_file 20b "${CACHE_ROOT}/${KEY}/key_inputs.txt"
expect 20c "${CACHE_ROOT}/${KEY}/key_inputs.txt" "fifo_words=1024"
expect 20d "${PKG}/results/phase_journal.tsv" "CACHE_PUBLISH"

C2_ROOT="${WORK}/cache_two"
PKG2="$(fresh_package "${C2_ROOT}")"
LOG="${WORK}/cache_hit.log"
TRACE="${WORK}/cache_hit_trace.txt"
: > "${TRACE}"
( cd "${PKG2}" && ODIN_DATA_ROOT="${C2_ROOT}" ODIN_CHIP_CACHE="${CACHE_ROOT}" \
    ODIN_STUB_TRACE="${TRACE}" ./run_all.sh ) > "${LOG}" 2>&1
say "--- transcript (cache) ---"; grep -E 'cache|phase 2|phase 3' "${LOG}" | head -12
expect 20e "${LOG}" "cache hit ${KEY}"
expect 20f "${LOG}" "no compile slot spent"
expect 20g "${PKG2}/results/phase_journal.tsv" "cache hit ${KEY}"
if grep -q 'target=hw ' "${TRACE}"; then
    fail 20h "it submitted a build anyway: $(grep 'target=hw ' "${TRACE}")"
else
    pass 20h "no build was submitted — the cache paid for both targets"
fi
expect_file 20i "${PKG2}/build/hacc/hw_nc1/odin_fpga_hw.xclbin"

ADOPT_ROOT="${WORK}/adopt"
ADOPT_CACHE="${WORK}/adopt_cache"
PKG3="$(fresh_package "${ADOPT_ROOT}")"
( cd "${PKG3}" && ODIN_DATA_ROOT="${ADOPT_ROOT}" ODIN_CHIP_CACHE_DISABLE=1 \
    ./run_all.sh --only 3 ) > "${WORK}/adopt_build.log" 2>&1
LOG="${WORK}/adopt.log"
( cd "${PKG3}" && ODIN_DATA_ROOT="${ADOPT_ROOT}" ODIN_CHIP_CACHE="${ADOPT_CACHE}" \
    ./scripts/chip_cache.sh adopt "${PKG3}" --alias v4 ) > "${LOG}" 2>&1
status=$?
say "--- adoption ---"; cat "${LOG}"
if [ "${status}" -eq 0 ]; then pass 20j "adoption closed (exit 0)"
else fail 20j "exit ${status}; see ${LOG}"; fi
ADOPT_KEY="$( cd "${PKG3}" && ODIN_DATA_ROOT="${ADOPT_ROOT}" \
    ODIN_CHIP_CACHE="${ADOPT_CACHE}" ./scripts/chip_cache.sh key hw \
    | sed -n 's/^key=//p' )"
if [ -f "${ADOPT_CACHE}/${ADOPT_KEY}/odin_fpga.xclbin" ]; then
    pass 20k "the adopted entry sits under the key a BUILD would look up"
else
    fail 20k "adoption filed under a key no build resolves (${ADOPT_KEY})"
fi
if [ -L "${ADOPT_CACHE}/v4_hw" ]; then
    pass 20l "the alias symlink v4_hw points at $(basename "$(readlink "${ADOPT_CACHE}/v4_hw")")"
else
    fail 20l "no v4_hw alias symlink"
fi
LOG="${WORK}/adopt_nosidecar.log"
rm -f "${PKG3}/build/hacc/hw_nc1/odin_fpga_hw.xclbin.built_with"
( cd "${PKG3}" && ODIN_CHIP_CACHE="${WORK}/adopt_cache2" \
    ./scripts/chip_cache.sh adopt "${PKG3}" ) > "${LOG}" 2>&1
status=$?
if [ "${status}" -eq 2 ]; then pass 20m "an install with no sidecar refuses to be adopted"
else fail 20m "exit ${status}, wanted 2"; fi
expect 20n "${LOG}" "keyed on a guess is worse"

mkdir -p "${CACHE_ROOT}/${KEY}.lock"
COLLIDE_ROOT="${WORK}/collide"
PKG4="$(fresh_package "${COLLIDE_ROOT}")"
mkdir -p "${PKG4}/build/hacc/hw_nc1"
printf 'A DIFFERENT BITSTREAM\n' > "${PKG4}/build/hacc/hw_nc1/odin_fpga_hw.xclbin"
cp "${CACHE_ROOT}/${KEY}/built_with.txt" \
   "${PKG4}/build/hacc/hw_nc1/odin_fpga_hw.xclbin.built_with"
BEFORE="$(sha256sum "${CACHE_ROOT}/${KEY}/odin_fpga.xclbin" | cut -d' ' -f1)"
LOG="${WORK}/cache_lock.log"
( cd "${PKG4}" && ODIN_DATA_ROOT="${COLLIDE_ROOT}" ODIN_CHIP_CACHE="${CACHE_ROOT}" \
    ./scripts/chip_cache.sh publish hw ) > "${LOG}" 2>&1
expect 20o "${LOG}" "already published; leaving it alone"
AFTER="$(sha256sum "${CACHE_ROOT}/${KEY}/odin_fpga.xclbin" | cut -d' ' -f1)"
if [ "${BEFORE}" = "${AFTER}" ]; then
    pass 20p "the published entry was not clobbered"
else
    fail 20p "the entry changed under a second publisher"
fi
rm -rf "${CACHE_ROOT}/${KEY}" "${CACHE_ROOT}/${KEY}.lock"
mkdir -p "${CACHE_ROOT}/${KEY}.lock"
LOG="${WORK}/cache_lock2.log"
( cd "${PKG4}" && ODIN_DATA_ROOT="${COLLIDE_ROOT}" ODIN_CHIP_CACHE="${CACHE_ROOT}" \
    ./scripts/chip_cache.sh publish hw ) > "${LOG}" 2>&1
expect 20q "${LOG}" "another job holds the lock"
if [ -d "${CACHE_ROOT}/${KEY}" ]; then
    fail 20r "it published into a key another job had claimed"
else
    pass 20r "a claimed key is left to its claimant"
fi
rm -rf "${CACHE_ROOT}/${KEY}.lock"

printf '\n# owner edit, %s\n' "$(date -u +%s)" >> "${PKG2}/scripts/hacc/build_xclbn.sh"
NEWKEY="$( cd "${PKG2}" && ODIN_DATA_ROOT="${C2_ROOT}" ODIN_CHIP_CACHE="${CACHE_ROOT}" \
    ./scripts/chip_cache.sh key hw | sed -n 's/^key=//p' )"
if [ "${NEWKEY}" != "${KEY}" ]; then
    pass 20s "a changed build_xclbn.sh moves the key — no stale hit"
else
    fail 20s "the key ignored the recipe that produces the bitstream"
fi

# ===========================================================================
say ""
if [ "${FAILED}" = "0" ]; then
    say "ALL GATES GREEN (work under ${WORK})"
else
    say "GATES FAILED (work under ${WORK})"
fi
exit "${FAILED}"
