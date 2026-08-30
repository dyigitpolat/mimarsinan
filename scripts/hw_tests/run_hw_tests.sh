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
# program; and the WRAPPER's own AXI4 DMA engine STREAMS that program through
# a behavioural AXI4 memory model into the fabric's elastic FIFO while the
# sequencer executes it, and drains the capture back into it, with the
# capacity/status registers refusing an overflow and a bad opcode). Their walls
# print on `[odin-fpga] ...` / `[odin-kernel] ...` / `[odin-wrapper] ...` lines.
#
# [ODIN C2] `test_odin_fpga_wide_kernel.py` is the WIDE chip configuration in the
# wrapper: the same `odin_fpga_kernel_top`, byte-untouched, driving the
# GENERATED 1024x256 core through its direct configuration port instead of the
# vendored core over SPI. What selects the fabric is the SOURCE SET
# (`chip_configs.ChipConfig.rtl_sources`, mirrored in `scripts/hacc/chips.sh`),
# so the stock kernel file the chip cache's RTL digest covers is never touched.
# Its walls print on `[odin-wide] ...` / `[odin-wide-stall] ...` lines.
#
# [ODIN9] the STALL-INVARIANCE gate lives in `test_odin_fpga_kernel.py`: the
# same fixture is delivered under the no-stall baseline and five seeded
# starvations of the AXI read data channel, and every run must produce
# byte-identical events at byte-identical ENABLED-cycle timestamps. The op
# stream arrives live now, so WHEN a word arrives is a degree of freedom the
# atol=0 certificates cannot afford; core-enable gating is what closes it, and
# this is the gate that would go red if the gating were removed. Its per-seed
# lines print on `[odin-stall] ...`.
#
#   scripts/hw_tests/run_hw_tests.sh                 # every RTL gate
#   scripts/hw_tests/run_hw_tests.sh -k r11a         # one of them
#   scripts/hw_tests/run_hw_tests.sh -k synth        # the yosys synthesis gate
#   scripts/hw_tests/run_hw_tests.sh -k limits       # the P8 compile-limits sweep
#   scripts/hw_tests/run_hw_tests.sh -k fpga         # the P7a backend gates
#   scripts/hw_tests/run_hw_tests.sh -k hacc         # the P8 deployment bundle
#
# The simulator (iverilog/vvp/verilator) AND the synthesizer (yosys) are looked
# up in MIMARSINAN_HW_SIM_BIN (default build/tools/oss-cad-suite/bin); when one
# is absent the gates that need it skip LOUDLY, naming that path, instead of
# reporting green. Regenerate the committed synthesis evidence deliberately with
# scripts/hw_tests/regen_synth_report.py, and the committed compile-limits study
# (hw/fpga/compile_limits.json + docs/odin_fpga_compile_limits_study.md) with
# scripts/hw_tests/regen_compile_limits.py.
#
# [ODIN8] `test_odin_hacc_cosim.py` freezes the unit loop-closer's tiny
# classifier with the RTL COSIMULATION as the bundle's witness instead of the
# cycle-accurate twin, then runs the SHIPPED board executor against those
# measured answers: the same four golden gates, the same seal, the same
# executor, with every expectation earned on the vendored core.
#
# [ODIN8] `test_odin_compile_limits.py` is the P8 compile-limits sweep: every
# studied configuration re-synthesized and matched against the committed record,
# with the two findings the packing bounds rest on re-derived rather than
# trusted. Its walls print on `[odin-limits] ...` lines.
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
    tests/integration/test_odin_compile_limits.py \
    tests/integration/test_odin_gen_geometry.py \
    tests/integration/test_odin_gen_sync_fire.py \
    tests/integration/test_odin_fpga_e2e.py \
    tests/integration/test_odin_fpga_kernel.py \
    tests/integration/test_odin_fpga_wide_kernel.py \
    tests/integration/test_odin_hacc_cosim.py \
    -m "slow and integration" \
    -p no:randomly -n0 -v -s --timeout=5400 --durations=0 \
    "$@"
