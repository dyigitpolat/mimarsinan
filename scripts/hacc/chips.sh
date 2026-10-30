# shellcheck shell=bash
# The CHIP CONFIGURATION is a parameter, and this file is the only place the
# SHELL side knows what one implies. It is to the fabric what cards.sh is to the
# board: one name selects the core the kernel instantiates, the RTL source set
# v++ compiles, and the build directory the artifact lands in.
#
# CROSS-LANGUAGE CONTRACT. The table below is the shell copy of
# `src/mimarsinan/chip_simulation/odin_fpga/chip_configs.py`, which is the SSOT
# and reads its geometry straight off the `CoreSpec` the cosimulation gates
# prove. The packaged build ships without `src/`, which is why a second copy
# exists at all; `tests/unit/chip_simulation/test_odin_chip_configs.py` parses
# this file and REFUSES any drift between the two.
#
#   odin_chip                      the selected chip (ODIN_CHIP, default stock)
#   odin_chip_profile [chip]       KEY=VALUE lines, or 1 for an unknown chip
#   odin_chip_field KEY [chip]     one field of that profile
#   odin_chip_resolve              CHIP/CHIP_CORE_KIND/CHIP_VARIANT/CHIP_SUFFIX/
#                                  CHIP_RTL_DIR set, or exit 2 on an unknown one
#   odin_kernel_cores              NC, the cores one compute unit instantiates
#                                  (ODIN_KERNEL_CORES, default 1), or 2
#   odin_build_leaf TARGET [chip]  <target>_nc<NC><chip suffix>: the build
#                                  directory every script names
#
# WHY THE FABRIC IS A SOURCE SET AND NOT A PARAMETER. The stock kernel body
# (`hw/fpga/kernel/odin_fpga_kernel.v`) is the RTL a routed bitstream was built
# from, and the chip cache keys that bitstream on a digest over the vendored +
# kernel source sets. A generate branch inside that file would move the digest
# and orphan the routed artifact for a branch it never elaborates. So a variant
# chip ships its own file declaring the SAME module `odin_fpga_kernel`, and the
# two are never compiled together. Everything above the core domain -- the AXI4
# wrapper, the register map, kernel.xml, the host protocol table -- is shared.

ODIN_CHIP_DEFAULT="odin_stock_256x256"

odin_chip() { printf '%s' "${ODIN_CHIP:-${ODIN_CHIP_DEFAULT}}"; }

odin_chip_profile() {
    local chip="${1:-}"
    [ -n "${chip}" ] || chip="$(odin_chip)"
    case "${chip}" in
        odin_stock_256x256)
            cat <<'PROFILE'
chip=odin_stock_256x256
core_kind=vendored
variant=
max_axons=128
max_neurons=256
physical_axon_rows=256
effective_max_axons=127
weight_bits=4
weight_sign_granularity=per_axon
membrane_bits=8
theta_ceiling=255
build_suffix=
rtl_dir=
PROFILE
            ;;
        odin_wide_1024x256_mb16)
            cat <<'PROFILE'
chip=odin_wide_1024x256_mb16
core_kind=generated
variant=gen_a1024n256_mb16w8_per_event
max_axons=1024
max_neurons=256
physical_axon_rows=1024
effective_max_axons=1023
weight_bits=8
weight_sign_granularity=per_synapse
membrane_bits=16
theta_ceiling=65535
build_suffix=_odin_wide_1024x256_mb16
rtl_dir=hw/gen/chips/odin_wide_1024x256_mb16
PROFILE
            ;;
        *)
            return 1
            ;;
    esac
}

odin_chip_field() {
    local key="$1" chip="${2:-}" profile
    profile="$(odin_chip_profile "${chip}")" || return 1
    printf '%s' "${profile}" | sed -n "s/^${key}=//p" | head -1
}

odin_chip_names() { printf 'odin_stock_256x256 odin_wide_1024x256_mb16'; }

# CORES PER COMPUTE UNIT (NC) is a parameter of every fabric: the kernel body
# instantiates NC cores of the chip in its `core_gen` loop, each with its own SPI
# master and AER-in bridge, and one capture shared by all of them
# (hw/fpga/kernel/odin_fpga_kernel.v; the generated kernels carry the same
# loop). NC is passed to the shared wrapper `odin_fpga_kernel_top` when the
# kernel is packaged, so the chip's source set does not change with it. The
# sequencer and the capture address a core with 8 bits (`active_core[7:0]`,
# `cap_core[7:0]`), so 256 is the most one unit can address.
ODIN_KERNEL_CORES_MAX=256

odin_kernel_cores() {
    local n="${ODIN_KERNEL_CORES:-1}"
    if ! [[ "${n}" =~ ^[1-9][0-9]{0,2}$ ]] || [ "${n}" -gt "${ODIN_KERNEL_CORES_MAX}" ]; then
        echo "REFUSING: ODIN_KERNEL_CORES='${n}' is not an integer from 1 to" >&2
        echo "  ${ODIN_KERNEL_CORES_MAX}: it is the number of ODIN cores one compute unit" >&2
        echo "  instantiates (default 1), and the kernel addresses a core with 8 bits" >&2
        echo "  (hw/fpga/kernel/odin_fpga_kernel.v, active_core[7:0])." >&2
        return 2
    fi
    printf '%s' "${n}"
}

# The build directory leaf of TARGET for the selected chip (or the one named) at
# the selected NC. Every script that names a build directory takes it from here,
# so NC = 1 stays `<target>_nc1<suffix>`, where every archived build sits; the
# compute-unit suffix (cards.sh, odin_cu_suffix) follows it where units apply.
odin_build_leaf() {
    local target="$1" chip="${2:-}" nc suffix
    nc="$(odin_kernel_cores)" || return 2
    [ -n "${chip}" ] || chip="$(odin_chip)"
    odin_chip_profile "${chip}" > /dev/null 2>&1 || return 1
    suffix="$(odin_chip_field build_suffix "${chip}")"
    printf '%s_nc%s%s' "${target}" "${nc}" "${suffix}"
}

# The RTL v++ compiles for this chip, one path per line, in compile order. The
# stock fabric is the committed kernel tree plus the vendored design; a
# generated one is the SHARED wrapper plus the two files the generator emitted,
# and it compiles neither the stock kernel body nor the vendored tree.
odin_chip_sources() {
    local chip="${1:-}" kind dir
    [ -n "${chip}" ] || chip="$(odin_chip)"
    kind="$(odin_chip_field core_kind "${chip}")" || return 1
    if [ "${kind}" = "vendored" ]; then
        cat <<'SOURCES'
hw/fpga/kernel/odin_spi_master.v
hw/fpga/kernel/odin_aer_bridge.v
hw/fpga/kernel/odin_fpga_kernel.v
hw/fpga/kernel/odin_fpga_kernel_top.v
hw/fpga/mem/SRAM_256x128_wrapper.v
hw/fpga/mem/SRAM_8192x32_wrapper.v
SOURCES
        find hw/vendor/odin/src -name '*.v' | sort
        return 0
    fi
    dir="$(odin_chip_field rtl_dir "${chip}")"
    printf 'hw/fpga/kernel/odin_aer_bridge.v\n'
    printf 'hw/fpga/kernel/odin_fpga_kernel_top.v\n'
    printf '%s/odin_gen_core.v\n' "${dir}"
    printf '%s/odin_fpga_kernel.v\n' "${dir}"
}

# Resolve the chip once. An unknown chip, or a generated one whose RTL is not in
# this tree, exits 2 HERE rather than hours into a v++ link.
odin_chip_resolve() {
    CHIP="$(odin_chip)"
    if ! odin_chip_profile "${CHIP}" > /dev/null 2>&1; then
        echo "REFUSING: unknown ODIN_CHIP='${CHIP}'. Known chips: $(odin_chip_names)." >&2
        echo "  The per-chip facts live in scripts/hacc/chips.sh and its SSOT" >&2
        echo "  src/mimarsinan/chip_simulation/odin_fpga/chip_configs.py — add a" >&2
        echo "  profile there rather than special-casing a fabric anywhere else." >&2
        exit 2
    fi
    CHIP_CORE_KIND="$(odin_chip_field core_kind "${CHIP}")"
    CHIP_VARIANT="$(odin_chip_field variant "${CHIP}")"
    CHIP_SUFFIX="$(odin_chip_field build_suffix "${CHIP}")"
    CHIP_RTL_DIR="$(odin_chip_field rtl_dir "${CHIP}")"
    if [ "${CHIP_CORE_KIND}" != "vendored" ]; then
        local missing=0 src
        for src in "${CHIP_RTL_DIR}/odin_gen_core.v" \
                   "${CHIP_RTL_DIR}/odin_fpga_kernel.v"; do
            [ -f "${src}" ] || { echo "  missing: ${src}" >&2; missing=1; }
        done
        if [ "${missing}" != "0" ]; then
            echo "REFUSING: ODIN_CHIP='${CHIP}' is a GENERATED fabric and its RTL" >&2
            echo "  is not in this tree. In the repository, emit it with" >&2
            echo "    env/bin/python scripts/hacc/gen_chip_rtl.py --chip ${CHIP}" >&2
            echo "  In a package, the files are shipped and a missing one means" >&2
            echo "  the package was built before this chip existed." >&2
            exit 2
        fi
    fi
}
