#!/usr/bin/env bash
# Bootstrap the SANA-FE submodule + mimarsinan custom plugins.
#
# SANA-FE is GPL-3.0 and therefore opt-in: ``pip install -e .`` of
# mimarsinan does NOT pull it.  Run this script once (with the project
# venv active) before enabling ``enable_sanafe_simulation`` in a config.
#
# What this script does:
#   1. Pulls the SANA-FE submodule.
#   2. Installs SANA-FE into the active venv (defaults to PyPI wheel —
#      override with MIMARSINAN_SANAFE_FROM_SOURCE=1 to build from the
#      submodule source).
#   3. Builds the mimarsinan-owned plugins
#      (``libmimarsinan_dendrite.so``, ``libmimarsinan_soma.so``) into
#      ``build/mimarsinan_sanafe_plugins/``.  The plugins replace
#      SANA-FE's built-in ``accumulator`` dendrite and ``leaky_integrate_fire``
#      soma so the per-core neuron count is not capped at the Loihi-derived
#      1024.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==> Pulling SANA-FE submodule"
git submodule update --init --recursive sana_fe

if [ "${MIMARSINAN_SANAFE_FROM_SOURCE:-0}" = "1" ]; then
    echo "==> Building SANA-FE from source (MIMARSINAN_SANAFE_FROM_SOURCE=1)"
    pip uninstall -y sanafe || true
    pip install -e ./sana_fe
else
    echo "==> Installing SANA-FE wheel from PyPI"
    pip install sanafe
fi

echo "==> Building mimarsinan SANA-FE plugins"
PLUGIN_SRC="$PROJECT_ROOT/src/mimarsinan/chip_simulation/sanafe/plugins"
PLUGIN_BUILD="$PROJECT_ROOT/build/mimarsinan_sanafe_plugins"
rm -rf "$PLUGIN_BUILD"
mkdir -p "$PLUGIN_BUILD"
cmake -S "$PLUGIN_SRC" -B "$PLUGIN_BUILD" \
    -DSANAFE_SRC="$PROJECT_ROOT/sana_fe/src"
cmake --build "$PLUGIN_BUILD" --parallel

echo "==> SANA-FE bootstrap complete."
echo "    Plugin binaries:"
ls -la "$PLUGIN_BUILD"/*.so 2>/dev/null || true
echo
echo "    Enable ``enable_sanafe_simulation: true`` in deployment_parameters"
echo "    (or via the wizard) to run the new step."
