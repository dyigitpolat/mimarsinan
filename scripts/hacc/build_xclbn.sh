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
# THE CHIP IS A PARAMETER TOO (v4). ODIN_CHIP selects a profile in
# scripts/hacc/chips.sh — which core the kernel instantiates and therefore which
# RTL v++ compiles — and the two axes are independent:
#   ODIN_CARD=u250 ODIN_CHIP=odin_wide_1024x256_mb16 scripts/hacc/build_xclbn.sh hw
# Default is odin_stock_256x256, the vendored fabric, whose source list and
# build directory are byte-identical to what this script compiled before the
# chip axis existed. A generated fabric ships its RTL under hw/gen/chips/<chip>/
# (emitted and gated by scripts/hacc/gen_chip_rtl.py) and its artifacts land in
# a chip-suffixed build directory, so the two never overwrite each other.
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
# COMPUTE UNITS ARE A PARAMETER TOO (route D). ODIN_COMPUTE_UNITS=N links N
# compute units of the same kernel, unit k on bank k of the card's memory
# (scripts/hacc/gen_link_cfg.py derives that [connectivity] from the card's own
# config; cards.sh declares the bank limit). ODIN_CU_SLRS optionally places the
# units, one SLR name per unit, as data. N = 1 (the default) links the card's
# config itself into the archived directory; N > 1 builds into `<dir>_cu<N>`:
#   ODIN_COMPUTE_UNITS=10 ODIN_CHIP=odin_wide_1024x256_mb16 scripts/hacc/build_xclbn.sh hw
#
# CORES PER UNIT ARE A PARAMETER TOO (multi-core Stage 1). ODIN_KERNEL_CORES=NC
# (scripts/hacc/chips.sh; default 1) packages the kernel with its NC parameter
# set to that value, so each compute unit instantiates NC cores of the chip
# (the `core_gen` loop of the kernel body, each core with its own SPI master and
# AER-in bridge, one shared capture; no core-to-core path in fabric). The value
# is written into the packaged core's user and HDL parameters, and the build
# REFUSES before the link unless the packaging log confirms both. It builds into
# `<target>_nc<NC><chip suffix>` (odin_build_leaf), the compute-unit suffix after it:
#   ODIN_KERNEL_CORES=4 scripts/hacc/build_xclbn.sh hw     # build/hacc/hw_nc4/
#
# Run it from the repository root (it reads hw/ from there):
#   ODIN_CARD=u250 scripts/hacc/build_xclbn.sh hw_emu   # ~15 min, functional
#   ODIN_CARD=u250 scripts/hacc/build_xclbn.sh hw       # hours, the bitstream
set -euo pipefail

TARGET="${1:-hw_emu}"

KERNEL="odin_fpga_kernel_top"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/hacc/toolchain.sh
source "$(dirname "${BASH_SOURCE[0]}")/toolchain.sh"
# shellcheck source=scripts/hacc/cards.sh
source "$(dirname "${BASH_SOURCE[0]}")/cards.sh"
# shellcheck source=scripts/hacc/chips.sh
source "$(dirname "${BASH_SOURCE[0]}")/chips.sh"
cd "${here}"

# CARD / PLATFORM / PART / CFG, and ODIN_VITIS_PREFER for the probe below.
# A card this cluster cannot build refuses HERE, before a compile slot is spent.
odin_card_resolve
# CHIP / CHIP_CORE_KIND / CHIP_VARIANT / CHIP_SUFFIX / CHIP_RTL_DIR. A generated
# fabric whose RTL is not in this tree refuses here too.
odin_chip_resolve
# NC: how many cores each compute unit instantiates. A value the kernel cannot
# address (not an integer from 1 to 256) refuses here too.
if ! NC="$(odin_kernel_cores)"; then
    exit 2
fi
# CUS / CU_SUFFIX: how many compute units the link replicates. A count that is
# not a positive integer, or one past the card's banks, refuses here too.
if ! CUS="$(odin_compute_units)"; then
    exit 2
fi
CU_SUFFIX="$(odin_cu_suffix "${CUS}")"
CU_BANKS="$(odin_card_field memory_banks "${CARD}")"
if [[ "${CUS}" -gt "${CU_BANKS:-1}" ]]; then
    echo "REFUSING: ODIN_COMPUTE_UNITS=${CUS}, but card ${CARD} declares ${CU_BANKS:-1}" >&2
    echo "  memory bank(s) (memory_banks in scripts/hacc/cards.sh): one bank per unit." >&2
    exit 2
fi

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

BUILD="build/hacc/$(odin_build_leaf "${TARGET}")${CU_SUFFIX}"
mkdir -p "${BUILD}"

echo "[hacc-build] card     : ${CARD}"
echo "[hacc-build] chip     : ${CHIP} (${CHIP_CORE_KIND}${CHIP_VARIANT:+ ${CHIP_VARIANT}})"
echo "[hacc-build] vitis    : ${XILINX_ROOT}/Vitis/${VITIS_VERSION} (prefer ${ODIN_VITIS_PREFER:-none})"
echo "[hacc-build] platform : ${PLATFORM}"
echo "[hacc-build] part     : ${PART}"
echo "[hacc-build] config   : ${CFG}"
echo "[hacc-build] target   : ${TARGET}"
echo "[hacc-build] cores    : ${NC}"
echo "[hacc-build] compute units: ${CUS}"
echo "[hacc-build] build dir: ${BUILD}"
echo "[hacc-build] host     : $(hostname)"

if [[ ! -f "${CFG}" ]]; then
    echo "REFUSING: the card profile names ${CFG}, which is not in this tree." >&2
    echo "  Each card ships its own v++ connectivity config; add one beside" >&2
    echo "  scripts/hacc/odin_u55c.cfg and name it in scripts/hacc/cards.sh." >&2
    exit 2
fi
# N > 1: the card's config with its [connectivity] derived for N units, written
# beside the build it describes; N = 1 links the card's config itself.
if [[ "${CUS}" != "1" ]]; then
    LINK_CFG="${BUILD}/link_cu${CUS}.cfg"
    if ! python3 scripts/hacc/gen_link_cfg.py --base "${CFG}" \
            --compute-units "${CUS}" --memory-banks "${CU_BANKS}" \
            --slrs "${ODIN_CU_SLRS:-}" --output "${LINK_CFG}"; then
        echo "REFUSING: no link config for ${CUS} compute units (above)." >&2
        exit 2
    fi
    CFG="${LINK_CFG}"
    echo "[hacc-build] link cfg : ${CFG} (${CUS} compute units${ODIN_CU_SLRS:+, SLRs ${ODIN_CU_SLRS}})"
fi

# --- 1. the kernel description v++ packages ---------------------------------
python3 scripts/hacc/gen_kernel_xml.py \
    --output "${BUILD}/kernel.xml" --kernel "${KERNEL}"

# --- 2. package_xo: Verilog + kernel.xml -> .xo ------------------------------
# The RTL is the CHIP's, in compile order: scripts/hacc/chips.sh is the only
# place that knows which files a fabric is made of, and the stock chip's list is
# the one this script used to spell out literally.
SOURCES=()
while IFS= read -r src; do
    [ -n "${src}" ] || continue
    if [[ ! -f "${src}" ]]; then
        echo "REFUSING: the ${CHIP} source list names ${src}, which is not in" >&2
        echo "  this tree. Emit a generated fabric with" >&2
        echo "  scripts/hacc/gen_chip_rtl.py, or re-unzip the package." >&2
        exit 2
    fi
    SOURCES+=("${src}")
done < <(odin_chip_sources "${CHIP}")
echo "[hacc-build] sources  : ${#SOURCES[@]} RTL file(s)"

# --- 2a. the SPI clock domain of every vendored core --------------------------
# Each vendored ODIN core is programmed through its SPI slave, and the slave is
# clocked by SCK: the output of the `sck` register of the kernel's SPI master
# (hw/fpga/kernel/odin_spi_master.v), CLK/4 on the gated core clock, clocking
# 381 slave registers (319 on the rising edge, 62 on the falling one). Without a
# clock on it the domain is never timed: every routed summary lists those pins
# as unclocked, and on a multi-unit build, where opt_design leaves most SCK nets
# on fabric routing, its same-edge hold races go unseen. This writes the
# constraint set for every core of the kernel (NC is data), scoped to the kernel
# top so it names no compute unit and applies to each one the link instantiates:
#   * a generated clock per SCK register, divide by 4 of the register's own clock;
#   * hold into the domain relaxed by 3 core cycles: MOSI is set one core cycle
#     before SCK rises and held three after it (odin_spi_master.v, phases 0-3),
#     and the slave's reset is released with SCK stopped;
#   * the slave's outputs into the core bounded at two core periods, data path
#     only: its events pass two-flop synchronizers and its buses are stable when
#     the core acts on the synchronized event (vendored controller.v);
#   * ASYNC_REG on those synchronizers;
#   * a global buffer asked for, by property, on every kernel clock: each SCK net
#     and, once per kernel, the gated core clock. opt_design's BUFG optimization
#     inserts at most eight buffers per build and counts the requested ones
#     against them, so asking for SCK alone left six of seven units' core clocks
#     unbuffered at N = 7 (build 71055); asking for all of them leaves none to it.
#     The core-clock request sits on the net at its leaf driver (the gating LUT).
# The domain's own paths keep their default (true) relations.
# A fabric without the vendored slave (a generated core) has no SCK: no file.
odin_spi_domain_xdc() {   # odin_spi_domain_xdc NC
    local nc="$1" c core sck_c sck_q master sck syncs reg
    printf '# ODIN SPI clock domain: written by scripts/hacc/build_xclbn.sh for NC = %s,\n' "${nc}"
    printf '# scoped to the kernel top (odin_fpga_kernel_top); one block per core.\n'
    # The gated core clock (odin_fpga_kernel.v, core_clk) clocks every core's SPI
    # master, so core 0's master names it at any NC. The request goes on the net at
    # its LEAF DRIVER (the gating LUT), so one buffer carries all its clock pins: on
    # the hierarchical segment at the master's clock pin it buffered only the
    # master's 58 and the LUT kept driving the other 3,929 (probe 71068, build 71057).
    printf '# the gated core clock of the kernel, at its leaf driver\n'
    printf 'set_property CLOCK_BUFFER_TYPE BUFG [get_nets -of_objects [get_pins -leaf -filter {DIRECTION == OUT} -of_objects [get_nets -of_objects [get_pins {kernel_i/core_gen[0].spi_i/sck_reg/C}]]]]\n'
    for ((c = 0; c < nc; c++)); do
        core="kernel_i/core_gen[${c}]"
        sck_c="[get_pins {${core}.spi_i/sck_reg/C}]"
        sck_q="[get_pins {${core}.spi_i/sck_reg/Q}]"
        master="[get_clocks -of_objects ${sck_c}]"
        sck="[get_clocks -of_objects ${sck_q}]"
        syncs=""
        for reg in controller_0/SPI_GATE_ACTIVITY_sync_int_reg controller_0/SPI_GATE_ACTIVITY_sync_reg \
                   controller_0/CTRL_READBACK_EVENT_sync_int_reg controller_0/CTRL_READBACK_EVENT_sync_reg \
                   controller_0/CTRL_PROG_EVENT_sync_int_reg controller_0/CTRL_PROG_EVENT_sync_reg \
                   scheduler_0/SPI_OPEN_LOOP_sync_int_reg scheduler_0/SPI_OPEN_LOOP_sync_reg; do
            syncs+="${syncs:+ }${core}.dut/${reg}"
        done
        printf '# core %s\n' "${c}"
        printf 'create_generated_clock -source %s -divide_by 4 %s\n' "${sck_c}" "${sck_q}"
        printf 'set_multicycle_path -hold -start 3 -from %s -to %s\n' "${master}" "${sck}"
        printf 'set_max_delay -datapath_only -from %s -to %s [expr {2.0 * [get_property PERIOD %s]}]\n' \
            "[get_cells -filter {IS_SEQUENTIAL} {${core}.dut/spi_slave_0/*}]" "${master}" "${master}"
        printf 'set_property CLOCK_BUFFER_TYPE BUFG [get_nets -of_objects %s]\n' "${sck_q}"
        printf 'set_property ASYNC_REG TRUE [get_cells {%s}]\n' "${syncs}"
    done
}
SPI_XDC_NAME="odin_spi_domain.xdc"
SPI_XDC=""
if [[ "${CHIP_CORE_KIND}" == "vendored" ]]; then
    SPI_XDC="${BUILD}/${SPI_XDC_NAME}"
    odin_spi_domain_xdc "${NC}" > "${SPI_XDC}"
    echo "[hacc-build] spi xdc  : ${NC} SPI clock domain(s) (${SPI_XDC})"
else
    echo "[hacc-build] spi xdc  : none (no SPI clock domain in this fabric)"
fi

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
    printf 'set kernel_cores %s\n' "${NC}"
    printf 'create_project -force odin_pack "%s/pack_prj" -part %s\n' \
        "${abs_build}" "${PART}"
    for src in "${SOURCES[@]}"; do
        printf 'add_files -norecurse "%s/%s"\n' "${here}" "${src}"
    done
    if [[ -n "${SPI_XDC}" ]]; then
        printf 'set spi_xdc_name "%s"\n' "${SPI_XDC_NAME}"
        printf 'add_files -fileset constrs_1 -norecurse "%s/%s"\n' "${abs_build}" "${SPI_XDC_NAME}"
        cat <<'TCL'
# The SPI clock-domain constraints: implementation only (they time the linked
# design; the kernel's out-of-context synthesis needs none of them and must not
# act on the buffer request) and processed late, after the shell's own clocks
# are defined.
set spi_xdc [get_files -of_objects [get_filesets constrs_1] *$spi_xdc_name]
set_property USED_IN_SYNTHESIS false $spi_xdc
set_property USED_IN_IMPLEMENTATION true $spi_xdc
set_property PROCESSING_ORDER LATE $spi_xdc
TCL
    fi
    cat <<'TCL'
set_property top $kernel_name [current_fileset]
update_compile_order -fileset sources_1
ipx::package_project -root_dir $ip_dir -vendor nus.edu -library user \
    -taxonomy /UserIP -import_files -set_current true
ipx::associate_bus_interfaces -busif m_axi_gmem -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk [ipx::current_core]
# NC, the cores this compute unit instantiates, is DATA: the packaged core's
# user parameter (what an instance is configured with) and its HDL parameter
# (what elaborates the RTL) are both set to it, read back, and printed for the
# build script to check before any link starts.
set nc_user [ipx::get_user_parameters NC -of_objects [ipx::current_core]]
set nc_hdl [ipx::get_hdl_parameters NC -of_objects [ipx::current_core]]
if {$nc_user eq "" || $nc_hdl eq ""} {
    error "ODIN: the packaged core declares no NC parameter"
}
set_property value $kernel_cores $nc_user
set_property value $kernel_cores $nc_hdl
foreach nc_param [list $nc_user $nc_hdl] {
    if {[get_property value $nc_param] != $kernel_cores} {
        error "ODIN: NC reads back [get_property value $nc_param], not $kernel_cores"
    }
}
puts "ODIN_PACKAGED_NC [get_property value $nc_user] [get_property value $nc_hdl]"
TCL
    if [[ -n "${SPI_XDC}" ]]; then
        cat <<'TCL'
# The packaged core keeps the SPI constraints in its implementation file group
# (never its synthesis one), used in implementation and processed late: set on
# the packaged file, read back, and printed for the build script to check
# before any link starts.
set spi_found {}
foreach fg [ipx::get_file_groups -of_objects [ipx::current_core]] {
    foreach f [ipx::get_files -of_objects $fg] {
        if {[file tail [get_property NAME $f]] eq $spi_xdc_name} {
            lappend spi_found [list [get_property NAME $fg] $f]
        }
    }
}
if {[llength $spi_found] == 1} {
    lassign [lindex $spi_found 0] spi_group spi_file
    set_property USED_IN {implementation} $spi_file
    set_property PROCESSING_ORDER late $spi_file
    puts "ODIN_PACKAGED_XDC $spi_xdc_name [get_property USED_IN $spi_file] [get_property PROCESSING_ORDER $spi_file] $spi_group"
} else {
    puts "ODIN_PACKAGED_XDC $spi_xdc_name found [llength $spi_found] times"
}
TCL
    fi
    cat <<'TCL'
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
# The packaged core must carry NC in BOTH parameters, or every core past the
# first would be missing from a bitstream that took hours to route.
packaged_nc="$(sed -n 's/^ODIN_PACKAGED_NC \([0-9]*\) \([0-9]*\)$/\1 \2/p' \
    "${BUILD}/pack_kernel.log" | tail -n 1)"
if [[ "${packaged_nc}" != "${NC} ${NC}" ]]; then
    echo "REFUSING: the packaging log confirms NC '${packaged_nc:-none}' (user, HDL" >&2
    echo "  parameter of the packaged core), not '${NC} ${NC}'; read" >&2
    echo "  ${BUILD}/pack_kernel.log. No link is started from this .xo." >&2
    exit 2
fi
echo "[hacc-build] packaged : NC ${NC} (user and HDL parameter, read back)"
# A vendored core's SPI constraints must be in the packaged core, for
# implementation only and processed late, or the link would route an untimed
# SPI domain for hours.
if [[ -n "${SPI_XDC}" ]]; then
    packaged_xdc="$(sed -n 's/^ODIN_PACKAGED_XDC \(.*\)$/\1/p' "${BUILD}/pack_kernel.log" \
        | tail -n 1)"
    if [[ "${packaged_xdc}" != "${SPI_XDC_NAME} implementation late xilinx_implementation" ]]; then
        echo "REFUSING: the packaging log reads '${packaged_xdc:-nothing}' for ${SPI_XDC_NAME}," >&2
        echo "  not '${SPI_XDC_NAME} implementation late xilinx_implementation' (used in," >&2
        echo "  processing order, file group of the packaged core); read" >&2
        echo "  ${BUILD}/pack_kernel.log. No link is started from this .xo." >&2
        exit 2
    fi
    echo "[hacc-build] packaged : spi xdc for implementation, late (${SPI_XDC_NAME})"
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

# --- 4. the build's identity, NC included, beside the xclbin --------------------
# A host that loads this xclbin checks its declared core count against this
# record (odin_board_driver.require_build_record_agrees): the NC a bitstream was
# built with travels WITH the bitstream instead of living only in a variable.
# The UUID is the link's own (`<xclbin>.info`, "UUID (xclbin)"); a link that
# wrote none records "unknown", and the host then checks the sha256 alone.
XCLBIN_OUT="${BUILD}/odin_fpga_${TARGET}.xclbin"
xclbin_uuid="$(sed -n 's/^ *UUID (xclbin): *\([0-9a-fA-F-]*\) *$/\1/p' \
    "${XCLBIN_OUT}.info" 2>/dev/null | head -n 1 || true)"
{
    printf 'xclbin_sha256=%s\n' "$(sha256sum "${XCLBIN_OUT}" | cut -d' ' -f1)"
    printf 'xclbin_uuid=%s\n' "${xclbin_uuid:-unknown}"
    printf 'kernel_cores=%s\n' "${NC}"
    printf 'compute_units=%s\n' "${CUS}"
    printf 'chip_config=%s\n' "${CHIP}"
    printf 'target=%s\n' "${TARGET}"
    printf 'card=%s\n' "${CARD}"
    printf 'platform=%s\n' "${PLATFORM}"
} > "${XCLBIN_OUT}.identity"
echo "[hacc-build] identity : NC ${NC}, UUID ${xclbin_uuid:-unknown} (${XCLBIN_OUT}.identity)"
echo "[hacc-build] stage it:  cp ${BUILD}/odin_fpga_${TARGET}.xclbin /data/${USER}/odin/"
