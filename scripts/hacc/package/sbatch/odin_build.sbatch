#!/bin/bash
#SBATCH --chdir /tmp/
#SBATCH --account=slurm
#SBATCH --job-name=odin_build
#SBATCH --output=slurm-odin-build-%j.out
#SBATCH --error=slurm-odin-build-%j.err
#
# Build the ODIN kernel into an xclbin. Submit with -p cpu_only.
#
# The first two #SBATCH lines are the cluster's required preamble
# (Xtra-Computing/hacc_demo/doc/1-FPGA-allocation.md, example.sh lines 143-145),
# and --output/--error are relative to --chdir (/tmp on the node) because slurm
# opens them BEFORE the job body runs: a path under a directory that does not
# exist yet makes the submission fail to launch with nothing to read.
#
# NEVER submit this to a board partition: xilinx_u55c_gen3x16_xdma_3_202210_1 is
# capped at ONE HOUR (doc/1-FPGA-allocation.md line 44) and a place-and-route is
# hours. cpu_only (hacc-node2) is capped at 7 days (line 20).
#
#   ODIN_PKG=/data/$USER/odin_hacc_package ODIN_TARGET=hw_emu \
#       sbatch --wait -p cpu_only scripts/hacc/odin_build.sbatch

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

echo "[build] job ${SLURM_JOB_ID:-?} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
echo "[build] package ${PKG}"
echo "[build] target  ${TARGET}"

cd "${PKG}"
scripts/hacc/build_xclbn.sh "${TARGET}"

echo "[build] artefacts under ${PKG}/build/hacc/${TARGET}_nc1/"
ls -l "${PKG}/build/hacc/${TARGET}_nc1/"

# --- the hw_emu smoke, ON THIS BUILD NODE, with NO card ----------------------
# XCL_EMULATION_MODE=hw_emu makes XRT bind to the QEMU/RTL emulation model that
# v++ packaged into the xclbin instead of to a device, so this runs on cpu_only.
# WHAT IT PROVES: the packaged kernel opens by name, the s_axilite register map
# answers (capacity registers, status), the AXI master moves the program and
# stimulus payloads, the sequencer executes them and the capture buffer decodes
# into the frozen counts. WHAT IT CANNOT PROVE: HBM ordering behind a real XDMA
# shell, XRT's buffer allocation on silicon, and timing closure — that is B0/B1.
if [[ "${TARGET}" == "hw_emu" && "${ODIN_EMU_SMOKE:-1}" == "1" ]]; then
    emu_dir="${PKG}/build/hacc/hw_emu_nc1"
    if ! command -v emconfigutil > /dev/null 2>&1; then
        echo "[build] EMU SMOKE SKIPPED: emconfigutil is not on PATH after"
        echo "        sourcing settings64.sh, so no emconfig.json can be made."
        echo "        Set ODIN_REQUIRE_EMU_SMOKE=1 to make this a hard failure."
        if [[ "${ODIN_REQUIRE_EMU_SMOKE:-0}" == "1" ]]; then
            exit 3
        fi
        exit 0
    fi
    (
        cd "${emu_dir}"
        emconfigutil --platform "${ODIN_PLATFORM:-xilinx_u55c_gen3x16_xdma_3_202210_1}" --nd 1
        export XCL_EMULATION_MODE=hw_emu
        export EMCONFIG_PATH="${emu_dir}"
        python3 "${PKG}/host/odin_board_driver.py" \
            --xclbin "${emu_dir}/odin_fpga_hw_emu.xclbin" \
            --fixtures "${PKG}/fixtures" \
            --results "${PKG}/results/hw_emu"
    )
fi
