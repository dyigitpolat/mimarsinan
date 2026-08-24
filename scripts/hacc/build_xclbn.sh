#!/usr/bin/env bash
# Build the ODIN RTL kernel into a U55C xclbin on HACC@NUS.
#
# WHERE: hacchead or a compile partition — NEVER on a board reservation: the
# bare U55C shell partitions are capped at ONE HOUR and a placed-and-routed
# build is hours. FIELD-OBSERVED 2026-08-25: `vck5000_compile` (hacc-node0,
# AllowGroups=ALL, 7 days) is the live build venue; `cpu_only` maps to
# hacc-gpu0, which is DOWN. Vitis is 2024.2 under /tools/Xilinx, discovered by
# toolchain.sh rather than assumed.
#
# WHAT: package_xo turns the Verilog kernel + its kernel.xml into an .xo, then
# v++ --link places it into the U55C XDMA shell. The emulation targets
# (sw_emu/hw_emu) build in minutes and are the right first step; `hw` is the
# real bitstream.
#
# Run it from the repository root (it reads hw/ from there):
#   scripts/hacc/build_xclbn.sh hw_emu      # ~15 min, functional
#   scripts/hacc/build_xclbn.sh hw          # hours, the real bitstream
set -euo pipefail

TARGET="${1:-hw_emu}"
NC="${ODIN_KERNEL_CORES:-1}"
if [[ "${NC}" != "1" ]]; then
    echo "REFUSING: ODIN_KERNEL_CORES=${NC}, but the packaging flow plumbs NC=1 only." >&2
    echo "  The RTL parameter exists; passing it through package_xo is a P7b" >&2
    echo "  follow-up. Build NC=1, bring the board up, then widen." >&2
    exit 2
fi

PLATFORM="${ODIN_PLATFORM:-xilinx_u55c_gen3x16_xdma_3_202210_1}"
KERNEL="odin_fpga_kernel_top"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/hacc/toolchain.sh
source "$(dirname "${BASH_SOURCE[0]}")/toolchain.sh"
cd "${here}"

# --- REFUSE LOUD off-cluster -------------------------------------------------
# The version is DISCOVERED (field-observed 2026-08-25: Vitis 2024.2 under
# /tools/Xilinx, not the 2022.2 under /tools/xilinx the vendor docs list);
# XILINX_ROOT / VITIS_VERSION still pin it if you need a specific one.
if ! vitis="$(odin_vitis_settings)"; then
    echo "REFUSING: no Vitis under ${XILINX_ROOT:-/tools/Xilinx or /tools/xilinx}." >&2
    echo "  This script builds ONLY on HACC@NUS (hacchead or a compile" >&2
    echo "  partition). Log in per scripts/hacc/RUNBOOK.md and run it there;" >&2
    echo "  off-cluster there is no toolchain and no shell to link against." >&2
    echo "  Pin one with XILINX_ROOT=/tools/Xilinx VITIS_VERSION=2024.2." >&2
    exit 2
fi
XILINX_ROOT="${vitis%%|*}"
VITIS_VERSION="${vitis#*|}"; VITIS_VERSION="${VITIS_VERSION%%|*}"
VITIS_SETTINGS="${vitis##*|}"
if [[ "${TARGET}" != "sw_emu" && "${TARGET}" != "hw_emu" && "${TARGET}" != "hw" ]]; then
    echo "REFUSING: unknown target '${TARGET}' (sw_emu | hw_emu | hw)." >&2
    exit 2
fi

# The vendor setup scripts expand variables that may be unset in a fresh slurm
# shell (Vitis 2024.x's .settings64-Vitis.sh reads $PYTHONPATH), which nounset
# treats as fatal — odin_source_toolchain relaxes it for exactly these two.
odin_source_toolchain "${VITIS_SETTINGS}"

BUILD="build/hacc/${TARGET}_nc${NC}"
mkdir -p "${BUILD}"

echo "[hacc-build] vitis    : ${XILINX_ROOT}/Vitis/${VITIS_VERSION}"
echo "[hacc-build] platform : ${PLATFORM}"
echo "[hacc-build] target   : ${TARGET}"
echo "[hacc-build] cores    : ${NC}"
echo "[hacc-build] build dir: ${BUILD}"
echo "[hacc-build] host     : $(hostname)"

# --- 1. the kernel description v++ packages ---------------------------------
python3 scripts/hacc/gen_kernel_xml.py \
    --output "${BUILD}/kernel.xml" --kernel "${KERNEL}"

# --- 2. package_xo: Verilog + kernel.xml -> .xo ------------------------------
SOURCES=(
    hw/fpga/kernel/odin_spi_master.v
    hw/fpga/kernel/odin_aer_bridge.v
    hw/fpga/kernel/odin_fpga_kernel.v
    hw/fpga/kernel/odin_fpga_kernel_top.v
    hw/fpga/mem/SRAM_256x128_wrapper.v
    hw/fpga/mem/SRAM_8192x32_wrapper.v
)
while IFS= read -r -d '' src; do SOURCES+=("${src}"); done \
    < <(find hw/vendor/odin/src -name '*.v' -print0 | sort -z)

rm -f "${BUILD}/${KERNEL}.xo"
"${XILINX_VITIS}/bin/package_xo" \
    -xo_path "${BUILD}/${KERNEL}.xo" \
    -kernel_name "${KERNEL}" \
    -kernel_xml "${BUILD}/kernel.xml" \
    -ip_directory "${BUILD}/ip" \
    -force \
    -design_xml "" \
    -kernel_files "${SOURCES[@]}"

# --- 3. v++ --link: .xo -> .xclbin against the U55C shell --------------------
"${XILINX_VITIS}/bin/v++" --link \
    --target "${TARGET}" \
    --platform "${PLATFORM}" \
    --kernel "${KERNEL}" \
    --config scripts/hacc/odin_u55c.cfg \
    --temp_dir "${BUILD}/tmp" \
    --report_dir "${BUILD}/reports" \
    --log_dir "${BUILD}/logs" \
    --save-temps \
    -o "${BUILD}/odin_fpga_${TARGET}.xclbin" \
    "${BUILD}/${KERNEL}.xo"

echo "[hacc-build] wrote ${BUILD}/odin_fpga_${TARGET}.xclbin"
echo "[hacc-build] stage it:  cp ${BUILD}/odin_fpga_${TARGET}.xclbin /data/${USER}/odin/"
