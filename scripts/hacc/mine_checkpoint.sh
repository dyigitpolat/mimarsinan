#!/usr/bin/env bash
# Mine a ROUTED checkpoint for everything only the real shell can tell us, and
# file it in the chip cache next to the bitstream it describes.
#
#   scripts/hacc/mine_checkpoint.sh [hw|hw_emu] [--dcp PATH] [--out DIR]
#
# WHY IT EXISTS. Local synthesis (yosys) gives a per-core census and nothing
# else: no placement, no congestion, no post-route timing. Those live in the
# routed checkpoint v++ leaves behind under --temp_dir, and nothing off-cluster
# can produce them. The bitstream is hundreds of megabytes and stays where it
# is; these reports are kilobytes and come home.
#
# WHAT IT WRITES, under <out>/ (default: the chip-cache entry for this build):
#   reports/utilization.rpt        report_utilization
#   reports/utilization_hier.rpt   report_utilization -hierarchical
#   reports/congestion.rpt         report_design_analysis -congestion
#   reports/timing_summary.rpt     report_timing_summary
#   reports/route_status.rpt       report_route_status
#   reports/placement.csv          one row per placed primitive: name,class,site,x,y
#   maps/die_map.(png|svg)         the picture host/render_die_map.py draws
#
# THE CSV IS THE CONTRACT with host/render_die_map.py. Site names on every
# Xilinx device carry their grid position (SLICE_X12Y34, RAMB36_X4Y17), so the
# coordinates are parsed out of the name rather than asked of a property that
# differs between families. `class` is the kernel sub-block a cell belongs to,
# or `shell` for everything the platform brought.

set -euo pipefail

TARGET="hw"
DCP=""
OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        hw|hw_emu) TARGET="$1" ;;
        --dcp) DCP="${2:?--dcp needs a path}"; shift ;;
        --out) OUT="${2:?--out needs a directory}"; shift ;;
        -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
        *) echo "REFUSING: unknown argument '$1' (see --help)." >&2; exit 2 ;;
    esac
    shift
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "${HERE}/.." && pwd)"
[ -f "${PKG}/MANIFEST.json" ] || PKG="$(cd "${HERE}/../.." && pwd)"
BUILD="${ODIN_BUILD_DIR:-${PKG}/build/hacc}/${TARGET}_nc1"
KERNEL_INSTANCE="${ODIN_KERNEL_INSTANCE:-odin_0}"
RENDERER="${PKG}/host/render_die_map.py"
CHIP_CACHE="${PKG}/scripts/chip_cache.sh"

# shellcheck source=scripts/hacc/toolchain.sh
source "${HERE}/toolchain.sh"
# shellcheck source=scripts/hacc/cards.sh
source "${HERE}/cards.sh"

say() { printf '%s\n' "$*"; }
die() { printf 'REFUSING: %s\n' "$*" >&2; exit 2; }

# --- 1. the checkpoint -------------------------------------------------------
if [ -z "${DCP}" ]; then
    DCP="$(find "${BUILD}" -name '*routed*.dcp' -print 2>/dev/null | sort | tail -1)"
fi
[ -n "${DCP}" ] && [ -f "${DCP}" ] || die \
    "no routed checkpoint under ${BUILD}. v++ leaves one in --temp_dir when it
  is run with --save-temps (scripts/hacc/build_xclbn.sh passes it); if the
  build ran elsewhere, point at it with --dcp. Nothing else on this cluster or
  off it can produce placement, congestion or post-route timing."

# --- 2. where the evidence lands --------------------------------------------
if [ -z "${OUT}" ]; then
    if [ -x "${CHIP_CACHE}" ] && OUT="$("${CHIP_CACHE}" path "${TARGET}" 2>/dev/null)"; then
        say "[mine] filing into the chip-cache entry ${OUT}"
    else
        OUT="${BUILD}/mined"
        say "[mine] no chip cache reachable; filing into ${OUT}"
    fi
fi
mkdir -p "${OUT}/reports" "${OUT}/maps"

if ! vitis="$(odin_vitis_settings)"; then
    die "no Vitis/Vivado here. Mining opens a checkpoint, which is a Vivado
  operation; run this on hacchead or the compile partition the build ran on."
fi
odin_source_toolchain "${vitis##*|}"
command -v vivado > /dev/null 2>&1 || die "no vivado on PATH after the settings"

say "[mine] checkpoint : ${DCP}"
say "[mine] target     : ${TARGET}"
say "[mine] kernel inst: ${KERNEL_INSTANCE}"
say "[mine] out        : ${OUT}"

# --- 3. the batch Vivado recipe ---------------------------------------------
TCL="${OUT}/mine_checkpoint.tcl"
cat > "${TCL}" <<TCLEOF
open_checkpoint {${DCP}}
report_utilization                       -file {${OUT}/reports/utilization.rpt}
report_utilization -hierarchical         -file {${OUT}/reports/utilization_hier.rpt}
report_design_analysis -congestion       -file {${OUT}/reports/congestion.rpt}
report_timing_summary -max_paths 10      -file {${OUT}/reports/timing_summary.rpt}
report_route_status                      -file {${OUT}/reports/route_status.rpt}

# One row per PLACED primitive. The site name carries the grid position on every
# Xilinx family (SLICE_X12Y34), so it is parsed rather than asked of a property
# whose name differs between families.
set fh [open {${OUT}/reports/placement.csv} w]
puts \$fh "name,class,site,x,y"
set placed 0
foreach cell [get_cells -hierarchical -quiet -filter {IS_PRIMITIVE == 1}] {
    set loc [get_property -quiet LOC \$cell]
    if {\$loc eq ""} { continue }
    if {![regexp {_X(\\d+)Y(\\d+)} \$loc -> x y]} { continue }
    set name [get_property NAME \$cell]
    set klass "shell"
    if {[string first "${KERNEL_INSTANCE}" \$name] >= 0} {
        set tail [string range \$name [expr {[string first "${KERNEL_INSTANCE}" \$name] \\
            + [string length "${KERNEL_INSTANCE}"] + 1}] end]
        set parts [split \$tail "/"]
        if {[llength \$parts] > 1} {
            set klass [lindex \$parts 0]
        } else {
            set klass "${KERNEL_INSTANCE}"
        }
    }
    puts \$fh "\$name,\$klass,\$loc,\$x,\$y"
    incr placed
}
close \$fh
puts "\[mine\] wrote \$placed placed primitives"
TCLEOF

vivado -mode batch -nojournal -log "${OUT}/reports/mine_checkpoint.log" \
    -source "${TCL}"

CSV="${OUT}/reports/placement.csv"
[ -s "${CSV}" ] || die "vivado exited 0 but wrote no placement rows; read
  ${OUT}/reports/mine_checkpoint.log"
say "[mine] placement : $(($(wc -l < "${CSV}") - 1)) placed primitives"

# --- 4. the picture ----------------------------------------------------------
if [ -f "${RENDERER}" ]; then
    python3 "${RENDERER}" --csv "${CSV}" \
        --out "${OUT}/maps/die_map.png" \
        --title "ODIN ${TARGET} on $(odin_card) — $(basename "${DCP}")"
else
    say "[mine] no ${RENDERER}; the CSV is filed and can be drawn later."
fi

say "[mine] done. Reports under ${OUT}/reports, maps under ${OUT}/maps."
say "[mine] collect_results.sh brings both home; the bitstream stays put."
