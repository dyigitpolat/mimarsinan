#!/usr/bin/env bash
# Bootstrap the SANA-FE submodule + optional plugin build.
#
# SANA-FE is GPL-3.0 and is therefore opt-in: ``pip install -e .`` of
# mimarsinan does NOT pull it.  Run this script once (with the project
# venv active) before enabling ``enable_sanafe_simulation`` in a config.
#
# Optional plugin build:
#   MIMARSINAN_SANAFE_PLUGIN=1 bash scripts/bootstrap_sanafe.sh
#
# That compiles
# ``src/mimarsinan/chip_simulation/sanafe/plugins/mimarsinan_subtractive_lif.cpp``
# into a soma plugin SANA-FE can load at runtime.  Only needed when the
# built-in ``loihi_lif_soma`` cannot reproduce mimarsinan's
# ``SubtractiveLIFReset`` semantics exactly (the single-core parity test
# pinpoints this).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==> Pulling SANA-FE submodule"
git submodule update --init --recursive sana_fe

echo "==> Installing SANA-FE into the active Python environment"
# Build deps:  CMake >= 3.13, GCC >= 8, flex, pybind11 >= 2.6.
pip install -e ./sana_fe

if [ "${MIMARSINAN_SANAFE_PLUGIN:-0}" = "1" ]; then
    PLUGIN_SRC="$PROJECT_ROOT/src/mimarsinan/chip_simulation/sanafe/plugins"
    PLUGIN_BUILD="$PROJECT_ROOT/sana_fe/plugins/build"
    if [ ! -f "$PLUGIN_SRC/CMakeLists.txt" ]; then
        echo "MIMARSINAN_SANAFE_PLUGIN=1 set but $PLUGIN_SRC/CMakeLists.txt is missing." >&2
        echo "(Strategy B plugin source is added on demand by the single-core parity test." >&2
        echo " Leave this flag unset until then.)" >&2
        exit 2
    fi
    echo "==> Building mimarsinan_subtractive_lif soma plugin"
    mkdir -p "$PLUGIN_BUILD"
    cd "$PLUGIN_BUILD"
    cmake -DSANAFE_ROOT="$PROJECT_ROOT/sana_fe" "$PLUGIN_SRC"
    cmake --build . --parallel
fi

echo "==> SANA-FE bootstrap complete."
echo "    Enable ``enable_sanafe_simulation: true`` in deployment_parameters"
echo "    (or via the wizard) to run the new step."
