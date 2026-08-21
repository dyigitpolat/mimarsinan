#!/usr/bin/env bash
# The NAMED runner for the RTL cosimulation gates (plan §7 rows 15-18) and the
# local synthesizability gate (row 19, P5.5a).
#
# These rows are marked `slow` + `integration` and the default suite does NOT
# run them: they build and execute a Verilog testbench around the byte-identical
# vendored ODIN core, and one full-memory SPI programming pass alone is ~6M
# simulated cycles per core. Run them here, deliberately, and read the per-test
# walls off the `[odin-rtl] ...` lines.
#
# [ODIN6] `test_odin_gen_*` are plan row 20: a GENERATED variant core
# (`hw/gen/odin_gen_core.v.tmpl` expanded from a `CoreSpec`) proved equal to its
# nevresim and torch twins at zero difference, plus the stock spec's vendored
# passthrough. Their walls print on `[odin-gen] ...` lines.
#
# [ODIN7a] `test_odin_fpga_*` are the physical backend: the END-TO-END gate
# (the REAL pipeline step deploying onto the RTL cosimulation transport, its
# counts certified against the HCM reference and nevresim at zero difference)
# and the Vitis KERNEL gates (the packaged wrapper elaborates under iverilog;
# the on-fabric sequencer reproduces the host testbench's counts on the same
# program). Their walls print on `[odin-fpga] ...` / `[odin-kernel] ...` lines.
#
#   scripts/hw_tests/run_hw_tests.sh                 # every RTL gate
#   scripts/hw_tests/run_hw_tests.sh -k r11a         # one of them
#   scripts/hw_tests/run_hw_tests.sh -k synth        # the yosys synthesis gate
#   scripts/hw_tests/run_hw_tests.sh -k fpga         # the P7a backend gates
#
# The simulator (iverilog/vvp/verilator) AND the synthesizer (yosys) are looked
# up in MIMARSINAN_HW_SIM_BIN (default build/tools/oss-cad-suite/bin); when one
# is absent the gates that need it skip LOUDLY, naming that path, instead of
# reporting green. Regenerate the committed synthesis evidence deliberately with
# scripts/hw_tests/regen_synth_report.py.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON="${PYTHON:-env/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

echo "[hw-tests] repo:      $(pwd)"
echo "[hw-tests] python:    ${PYTHON}"
echo "[hw-tests] simulator: ${MIMARSINAN_HW_SIM_BIN:-build/tools/oss-cad-suite/bin}"

exec "${PYTHON}" -m pytest \
    tests/integration/test_odin_rtl_micro.py \
    tests/integration/test_odin_rtl_r11a.py \
    tests/integration/test_odin_rtl_spi.py \
    tests/integration/test_odin_rtl_barrier.py \
    tests/integration/test_odin_rtl_overlay.py \
    tests/integration/test_odin_rtl_engines.py \
    tests/integration/test_odin_rtl_synth.py \
    tests/integration/test_odin_gen_geometry.py \
    tests/integration/test_odin_gen_sync_fire.py \
    tests/integration/test_odin_fpga_e2e.py \
    tests/integration/test_odin_fpga_kernel.py \
    -m "slow and integration" \
    -p no:randomly -n0 -v -s --timeout=5400 --durations=0 \
    "$@"
