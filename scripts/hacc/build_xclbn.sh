#!/usr/bin/env bash
# Build the ODIN RTL kernel into an xclbin on HACC@NUS, for the card you name.
#
# THE CARD IS A PARAMETER (v3). ODIN_CARD selects a profile in
# scripts/hacc/cards.sh — platform, Vivado part, v++ connectivity config,
# preferred Vitis — and nothing card-shaped is written down anywhere else:
#   ODIN_CARD=u250 scripts/hacc/build_xclbn.sh hw
# Default is u55c, which on this cluster REFUSES with the deadlock reason
# (only a 2022.2-locked U55C shell installed, and no 2022.2 Vitis).
#
# WHERE: hacchead or a compile partition — NEVER on a board reservation: the
# bare board partitions are capped at ONE HOUR and a placed-and-routed build is
# hours. FIELD-OBSERVED 2026-08-25: `vck5000_compile` (hacc-node0,
# AllowGroups=ALL, 7 days) is the live build venue; `cpu_only` maps to
# hacc-gpu0, which is DOWN. Vitis is discovered by toolchain.sh, and the card
# profile's preferred version wins when it is installed.
#
# WHAT: package_xo turns the Verilog kernel + its kernel.xml into an .xo, then
# v++ --link places it into the card's XDMA shell. The emulation targets
# (sw_emu/hw_emu) build in minutes and are the right first step; `hw` is the
# real bitstream.
#
# Run it from the repository root (it reads hw/ from there):
#   ODIN_CARD=u250 scripts/hacc/build_xclbn.sh hw_emu   # ~15 min, functional
#   ODIN_CARD=u250 scripts/hacc/build_xclbn.sh hw       # hours, the bitstream
set -euo pipefail

TARGET="${1:-hw_emu}"
NC="${ODIN_KERNEL_CORES:-1}"
if [[ "${NC}" != "1" ]]; then
    echo "REFUSING: ODIN_KERNEL_CORES=${NC}, but the packaging flow plumbs NC=1 only." >&2
    echo "  The RTL parameter exists; passing it through package_xo is a P7b" >&2
    echo "  follow-up. Build NC=1, bring the board up, then widen." >&2
    exit 2
fi

KERNEL="odin_fpga_kernel_top"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/hacc/toolchain.sh
source "$(dirname "${BASH_SOURCE[0]}")/toolchain.sh"
# shellcheck source=scripts/hacc/cards.sh
source "$(dirname "${BASH_SOURCE[0]}")/cards.sh"
cd "${here}"

# CARD / PLATFORM / PART / CFG, and ODIN_VITIS_PREFER for the probe below.
# A card this cluster cannot build refuses HERE, before a compile slot is spent.
odin_card_resolve

# --- REFUSE LOUD off-cluster -------------------------------------------------
# The version is DISCOVERED (field-observed 2026-08-25: 2020.1 2020.2 2021.2
# 2022.1 2023.2 2024.2 under /tools/Xilinx, and NO 2022.2); the card profile
# prefers the release its shell was built with, and XILINX_ROOT / VITIS_VERSION
# still pin it if you need a specific one.
if ! vitis="$(odin_vitis_settings)"; then
    echo "REFUSING: no Vitis under ${XILINX_ROOT:-/tools/Xilinx or /tools/xilinx}." >&2
    echo "  This script builds ONLY on HACC@NUS (hacchead or a compile" >&2
    echo "  partition). Log in per scripts/hacc/RUNBOOK.md and run it there;" >&2
    echo "  off-cluster there is no toolchain and no shell to link against." >&2
    echo "  Pin one with XILINX_ROOT=/tools/Xilinx VITIS_VERSION=2020.2." >&2
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

echo "[hacc-build] card     : ${CARD}"
echo "[hacc-build] vitis    : ${XILINX_ROOT}/Vitis/${VITIS_VERSION} (prefer ${ODIN_VITIS_PREFER:-none})"
echo "[hacc-build] platform : ${PLATFORM}"
echo "[hacc-build] part     : ${PART}"
echo "[hacc-build] config   : ${CFG}"
echo "[hacc-build] target   : ${TARGET}"
echo "[hacc-build] cores    : ${NC}"
echo "[hacc-build] build dir: ${BUILD}"
echo "[hacc-build] host     : $(hostname)"

if [[ ! -f "${CFG}" ]]; then
    echo "REFUSING: the card profile names ${CFG}, which is not in this tree." >&2
    echo "  Each card ships its own v++ connectivity config; add one beside" >&2
    echo "  scripts/hacc/odin_u55c.cfg and name it in scripts/hacc/cards.sh." >&2
    exit 2
fi

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

# package_xo is a Vivado Tcl command, NOT a Vitis binary (field-observed
# 2026-08-25 on hacc-node0: ${XILINX_VITIS}/bin has no package_xo). The
# documented batch flow (UG1393 ch. RTL Kernels; AMD Vitis-Tutorials
# 05-bottom_up_rtl_kernel pack_kernel.tcl) packages the sources as a Vivado
# IP with ipx::package_project, marks it sdx_kernel/rtl, and calls package_xo
# inside `vivado -mode batch`. Our audited kernel.xml stays authoritative.
#
# EVERY COMMAND BELOW IS 2020.2 SYNTAX, checked one by one, because the U250
# profile pins Vivado 2020.2 and the tutorial this flow follows is itself a
# 2020.x-era one: create_project -force -part, add_files -norecurse,
# set_property top, update_compile_order -fileset, ipx::package_project with
# -root_dir/-vendor/-library/-taxonomy/-import_files/-set_current, the
# sdx_kernel / sdx_kernel_type properties, ipx::associate_bus_interfaces
# with -busif/-clock (the krnl_aes step that tells the system linker ap_clk
# drives both AXI interfaces — its absence is the field-observed
# 'Could not identify a clock source pin of /odin_0/m_axi_gmem'),
# ipx::update_source_project_archive
# -component, ipx::save_core, and package_xo with
# -xo_path/-kernel_name/-kernel_xml/-ip_directory. package_xo's `-force` is the
# one option whose 2020.2 availability we could not confirm off-cluster, so it
# is gone: `file delete -force` on the target does the same job in every
# release, and the bash `rm -f` below already did it once.
if ! command -v vivado > /dev/null 2>&1; then
    echo "REFUSING: no vivado on PATH after sourcing the Vitis settings." >&2
    echo "  package_xo runs inside Vivado; settings64.sh normally adds" >&2
    echo "  \${XILINX_ROOT}/Vivado/<version>/bin to PATH. Check the toolchain." >&2
    exit 2
fi
rm -f "${BUILD}/${KERNEL}.xo"
abs_build="$(cd "${BUILD}" && pwd)"
{
    printf 'set xo_path "%s"\n' "${abs_build}/${KERNEL}.xo"
    printf 'set kernel_name "%s"\n' "${KERNEL}"
    printf 'set kernel_xml "%s"\n' "${abs_build}/kernel.xml"
    printf 'set ip_dir "%s"\n' "${abs_build}/ip"
    printf 'create_project -force odin_pack "%s/pack_prj" -part %s\n' \
        "${abs_build}" "${PART}"
    for src in "${SOURCES[@]}"; do
        printf 'add_files -norecurse "%s/%s"\n' "${here}" "${src}"
    done
    cat <<'TCL'
set_property top $kernel_name [current_fileset]
update_compile_order -fileset sources_1
ipx::package_project -root_dir $ip_dir -vendor nus.edu -library user \
    -taxonomy /UserIP -import_files -set_current true
ipx::associate_bus_interfaces -busif m_axi_gmem -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk [ipx::current_core]
set_property sdx_kernel true [ipx::current_core]
set_property sdx_kernel_type rtl [ipx::current_core]
ipx::update_source_project_archive -component [ipx::current_core]
ipx::save_core [ipx::current_core]
file delete -force $xo_path
package_xo -xo_path $xo_path -kernel_name $kernel_name \
    -kernel_xml $kernel_xml -ip_directory $ip_dir
TCL
} > "${BUILD}/pack_kernel.tcl"
if ! vivado -mode batch -nojournal -log "${BUILD}/pack_kernel.log" \
        -source "${BUILD}/pack_kernel.tcl"; then
    echo "[hacc-build] packaging FAILED — last 40 lines of ${BUILD}/pack_kernel.log:" >&2
    tail -n 40 "${BUILD}/pack_kernel.log" >&2 || true
    exit 2
fi
if [[ ! -f "${BUILD}/${KERNEL}.xo" ]]; then
    echo "REFUSING: vivado exited 0 but ${BUILD}/${KERNEL}.xo was not written;" >&2
    echo "  read ${BUILD}/pack_kernel.log." >&2
    exit 2
fi

# --- 3. v++ --link: .xo -> .xclbin against the card's shell -------------------
"${XILINX_VITIS}/bin/v++" --link \
    --target "${TARGET}" \
    --platform "${PLATFORM}" \
    --kernel "${KERNEL}" \
    --config "${CFG}" \
    --temp_dir "${BUILD}/tmp" \
    --report_dir "${BUILD}/reports" \
    --log_dir "${BUILD}/logs" \
    --save-temps \
    -o "${BUILD}/odin_fpga_${TARGET}.xclbin" \
    "${BUILD}/${KERNEL}.xo"

echo "[hacc-build] wrote ${BUILD}/odin_fpga_${TARGET}.xclbin"
echo "[hacc-build] stage it:  cp ${BUILD}/odin_fpga_${TARGET}.xclbin /data/${USER}/odin/"
