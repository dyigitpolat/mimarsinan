#!/usr/bin/env bash
# Bootstrap the optional SANA-FE backend: the PyPI wheel + the mimarsinan plugins.
#
# SANA-FE is GPL-3.0 and therefore opt-in: the base install does NOT pull it,
# and no SANA-FE source enters this tree. The plugin build fetches SANA-FE's
# v2.1.1 header archive (pinned by SHA256 in the plugins' CMakeLists.txt) into
# build/, exactly as the old submodule supplied it.
#
# Equivalent one-liner:  make install SANAFE=1
#
# What this script does:
#   1. `uv sync --extra sanafe` — installs sanafe 2.1.1 alongside the project.
#      PINNED: the integration (arch YAML, soma model_attributes, plugins)
#      targets 2.1.1. An unpinned install upgraded it to 2.2.x on 2026-06-17,
#      which SIGFPEs on arch load (docs/.../SANAFE_fpe_investigation.md). Bump
#      only after re-validating the SANA-FE parity gate and
#      _SUPPORTED_SANAFE_VERSIONS.
#   2. Builds the six mimarsinan-owned plugins into
#      build/mimarsinan_sanafe_plugins/. They replace SANA-FE's built-in
#      `accumulator` dendrite and `leaky_integrate_fire` soma so the per-core
#      neuron count is not capped at the Loihi-derived 1024, and add the four
#      TTFS soma variants.
#
# MIMARSINAN_SANAFE_SRC=/path/to/sana_fe/src builds the plugins against an
# existing SANA-FE checkout instead of the pinned archive.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==> Installing SANA-FE (pinned 2.1.1) through the project manifest"
uv sync --extra dev --extra loihi --extra sanafe

echo "==> Building mimarsinan SANA-FE plugins"
uv run python scripts/build_sanafe_plugins.py

echo
echo "    Enable \`enable_sanafe_simulation: true\` in deployment_parameters"
echo "    (or via the wizard) to run the new step."
