#!/usr/bin/env bash
# The NAMED runner for the RTL cosimulation gates (plan §7 rows 15-18).
#
# These rows are marked `slow` + `integration` and the default suite does NOT
# run them: they build and execute a Verilog testbench around the byte-identical
# vendored ODIN core, and one full-memory SPI programming pass alone is ~6M
# simulated cycles per core. Run them here, deliberately, and read the per-test
# walls off the `[odin-rtl] ...` lines.
#
#   scripts/hw_tests/run_hw_tests.sh                 # every RTL gate
#   scripts/hw_tests/run_hw_tests.sh -k r11a         # one of them
#
# The simulator is looked up in MIMARSINAN_HW_SIM_BIN (default
# build/tools/oss-cad-suite/bin); when it is absent every gate skips LOUDLY,
# naming that path, instead of reporting green.
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
    -m "slow and integration" \
    -p no:randomly -n0 -v -s --timeout=5400 --durations=0 \
    "$@"
