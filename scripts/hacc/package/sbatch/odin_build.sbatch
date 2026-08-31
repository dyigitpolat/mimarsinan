#!/bin/bash
#SBATCH --chdir /tmp/
#SBATCH --account=slurm
#SBATCH --job-name=odin_build
#SBATCH --output=slurm-odin-build-%j.out
#SBATCH --error=slurm-odin-build-%j.err
#
# Build the ODIN kernel into an xclbin. run_all.sh picks the partition from the
# cluster's own scontrol/sinfo answers and submits with -p <that>.
#
# The first two #SBATCH lines are the cluster's required preamble
# (Xtra-Computing/hacc_demo/doc/1-FPGA-allocation.md, example.sh lines 143-145),
# and --output/--error are relative to --chdir (/tmp on the node) because slurm
# opens them BEFORE the job body runs: a path under a directory that does not
# exist yet makes the submission fail to launch with nothing to read.
#
# NEVER submit this to a board partition: the bare shell partitions
# (xilinx_u55c_gen3x16_xdma_3_202210_1, xilinx_u250_gen3x16_xdma_3_1_202020_1)
# are capped at ONE HOUR (doc/1-FPGA-allocation.md lines 42-44) and a
# place-and-route is hours. FIELD-OBSERVED 2026-08-25: the live build venue is
# `vck5000_compile` (AllowGroups=ALL, MaxTime 7d, hacc-node0 idle). `cpu_only`
# maps to hacc-gpu0, which is DOWN with 'NO NETWORK ADDRESS FOUND' — the docs'
# hacc-node2 is stale. Build venues are card-independent; ODIN_CARD only
# decides what gets linked, not where.
#
#   ODIN_PKG=/data/$USER/odin_hacc_package ODIN_TARGET=hw_emu ODIN_CARD=u250 \
#       sbatch --wait -p vck5000_compile scripts/hacc/odin_build.sbatch

set -euo pipefail

# Slurm's own --output/--error live on the node's /tmp (see the preamble above),
# so everything this job prints is ALSO mirrored under ODIN_LOG on /data — that
# mirror, not the slurm file, is what run_all.sh reads and what comes home.
if [[ -n "${ODIN_LOG:-}" ]]; then
    mkdir -p "$(dirname "${ODIN_LOG}")"
    exec > >(tee -a "${ODIN_LOG}") 2>&1
fi

PKG="${ODIN_PKG:?ODIN_PKG (the unpacked package root) must be exported}"
TARGET="${ODIN_TARGET:-hw_emu}"

# shellcheck source=scripts/hacc/cards.sh
source "${PKG}/scripts/hacc/cards.sh"
CARD="$(odin_card)"
PLATFORM="$(odin_effective_platform "${CARD}")"

echo "[build] job ${SLURM_JOB_ID:-?} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
echo "[build] package ${PKG}"
echo "[build] card    ${CARD}"
echo "[build] target  ${TARGET}"

cd "${PKG}"
scripts/hacc/build_xclbn.sh "${TARGET}"

# The build directory carries the CHIP's suffix (empty for stock) — resolve it
# the same way build_xclbn.sh just did, or a generated fabric's artifacts are
# reported at a directory that does not exist and set -e kills a good build.
# shellcheck source=scripts/hacc/chips.sh
source scripts/hacc/chips.sh
odin_chip_resolve
BUILD_OUT="${PKG}/build/hacc/${TARGET}_nc1${CHIP_SUFFIX}"

echo "[build] artefacts under ${BUILD_OUT}/"
ls -l "${BUILD_OUT}/"

# --- the hw_emu smoke, ON THIS BUILD NODE, with NO card ----------------------
# XCL_EMULATION_MODE=hw_emu makes XRT bind to the QEMU/RTL emulation model that
# v++ packaged into the xclbin instead of to a device, so this runs on cpu_only.
# WHAT IT PROVES: the packaged kernel opens by name with its six frozen
# arguments, the AXI master moves the program and stimulus payloads, the
# sequencer executes them, and the capture buffer comes home carrying a header
# the fabric wrote, decoding into the frozen counts. (No register is read
# anywhere: pyxrt binds no read_register -- field-falsified 2026-08-25.) WHAT IT CANNOT PROVE: HBM ordering behind a real XDMA
# shell, XRT's buffer allocation on silicon, and timing closure — that is B0/B1.
#
# AND IT IS BOUNDED. An xsim-backed hw_emu run has no published wall: the
# smallest shipped fixture is the DEFAULT smoke, under a timeout, and a timeout
# writes an honest note and lets the build stand. ODIN_EMU_SMOKE=all runs all
# five; ODIN_EMU_SMOKE=off skips it. B0/B1 on silicon supersede it either way.
SMOKE="${ODIN_EMU_SMOKE:-smallest}"
SMOKE_TIMEOUT="${ODIN_EMU_SMOKE_TIMEOUT:-5400}"
SMALLEST_FIXTURE="nc1_single_core_ceiling"

if [[ "${TARGET}" == "hw_emu" && "${SMOKE}" != "off" && "${SMOKE}" != "0" ]]; then
    emu_dir="${BUILD_OUT}"
    emu_results="${PKG}/results/hw_emu"
    mkdir -p "${emu_results}"
    # build_xclbn.sh sourced the toolchain in a CHILD process, so this shell has
    # never seen emconfigutil or XRT's python bindings. Source it here too, or
    # the smoke skips itself on every cluster on earth.
    # shellcheck source=scripts/hacc/toolchain.sh
    source "${PKG}/scripts/hacc/toolchain.sh"
    if vitis="$(odin_vitis_settings)"; then
        odin_source_toolchain "${vitis##*|}"
    fi
    if ! command -v emconfigutil > /dev/null 2>&1; then
        echo "[build] EMU SMOKE SKIPPED: emconfigutil is not on PATH after"
        echo "        sourcing settings64.sh, so no emconfig.json can be made."
        echo "        Set ODIN_REQUIRE_EMU_SMOKE=1 to make this a hard failure."
        if [[ "${ODIN_REQUIRE_EMU_SMOKE:-0}" == "1" ]]; then
            exit 3
        fi
        exit 0
    fi
    selection=()
    if [[ "${SMOKE}" != "all" ]]; then
        selection=(--fixture "${SMALLEST_FIXTURE}")
        echo "[build] emu smoke: ${SMALLEST_FIXTURE} only, timeout ${SMOKE_TIMEOUT}s"
    else
        echo "[build] emu smoke: every shipped fixture, timeout ${SMOKE_TIMEOUT}s"
    fi
    status=0
    (
        cd "${emu_dir}"
        emconfigutil --platform "${PLATFORM}" --nd 1
        export XCL_EMULATION_MODE=hw_emu
        export EMCONFIG_PATH="${emu_dir}"
        timeout "${SMOKE_TIMEOUT}" \
            python3 "${PKG}/host/odin_board_driver.py" \
                --xclbin "${emu_dir}/odin_fpga_hw_emu.xclbin" \
                --fixtures "${PKG}/fixtures" \
                ${selection[@]+"${selection[@]}"} \
                --results "${emu_results}"
    ) || status=$?
    if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
        {
            echo "EMU SMOKE SKIPPED (timeout)"
            echo "fixture   : ${SMOKE}"
            echo "timeout_s : ${SMOKE_TIMEOUT}"
            echo "utc       : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo
            echo "The xsim-backed emulation did not finish inside the budget. Its"
            echo "wall is UNKNOWN, not slow-but-fine: nothing here measured it."
            echo "This is not a failure of the build — the xclbin stands — and it"
            echo "is not evidence of anything either. B0/B1 on silicon supersede"
            echo "this smoke entirely; raise ODIN_EMU_SMOKE_TIMEOUT if you want the"
            echo "number for its own sake."
        } > "${emu_results}/EMU_SMOKE_SKIPPED.txt"
        cat "${emu_results}/EMU_SMOKE_SKIPPED.txt"
    elif [[ "${status}" -ne 0 ]]; then
        echo "[build] the hw_emu smoke FAILED (exit ${status}) — that is a" >&2
        echo "        packaging or decode finding, not a timeout." >&2
        exit "${status}"
    fi
fi
