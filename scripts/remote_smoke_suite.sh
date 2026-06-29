#!/usr/bin/env bash
# Remote Slurm smoke suite for hypervolume closure Phase 0.
#
# Intended to run inside a GPU Slurm allocation (or locally on a CUDA node)
# after workspace sync and setup_remote_env.sh. Validates imports, compiler
# discovery, Nevresim compile, and one tiny Simulation before campaign runs.
set -euo pipefail

REPO="${MIMARSINAN_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO}"

if [ -x "${MIMARSINAN_REMOTE_VENV:-$HOME/mimarsinan_slurmech/env}/bin/python" ]; then
  # shellcheck disable=SC1091
  source "${MIMARSINAN_REMOTE_VENV:-$HOME/mimarsinan_slurmech/env}/bin/activate"
elif [ -x "${REPO}/env/bin/python" ]; then
  # shellcheck disable=SC1091
  source "${REPO}/env/bin/activate"
fi

export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

echo "[remote_smoke_suite] repo=${REPO}"
echo "[remote_smoke_suite] python=$(python --version 2>&1)"

python - <<'PY'
import importlib
import sys

checks = [
    ("torch", lambda m: getattr(m, "__version__", "ok")),
    ("torchvision", lambda m: getattr(m, "__version__", "ok")),
    ("einops", lambda m: getattr(m, "__version__", "ok")),
    ("mimarsinan", lambda m: getattr(m, "__version__", "ok")),
]
for name, fmt in checks:
    mod = importlib.import_module(name)
    print(f"[remote_smoke_suite] import {name} ok ({fmt(mod)})")

import torch
if torch.cuda.is_available():
    print(f"[remote_smoke_suite] cuda device={torch.cuda.get_device_name(0)}")
else:
    print("[remote_smoke_suite] warning: CUDA unavailable on this node")
PY

python - <<'PY'
from mimarsinan.common.build_utils import find_cpp20_compiler

cmd, family = find_cpp20_compiler()
if not cmd:
    raise SystemExit("no C++20 compiler found")
print(f"[remote_smoke_suite] compiler={cmd} family={family}")
PY

python - <<'PY'
import importlib.util
from pathlib import Path

root = Path("spikingjelly")
if not root.is_dir():
    raise SystemExit("spikingjelly vendored tree missing from sync")
spec = importlib.util.find_spec("spikingjelly")
if spec is None:
    raise SystemExit("spikingjelly import failed")
print("[remote_smoke_suite] spikingjelly import ok")
PY

python - <<'PY'
from pathlib import Path

from mimarsinan.chip_simulation.nevresim import compile_nevresim as nev_compile
from mimarsinan.common.build_utils import find_cpp20_compiler

if not Path("nevresim/include").is_dir():
    raise SystemExit("nevresim headers missing from sync")
cmd, family = find_cpp20_compiler()
if not cmd:
    raise SystemExit("no C++20 compiler for nevresim compile")
print(f"[remote_smoke_suite] nevresim module ok compiler={cmd} family={family}")
print(f"[remote_smoke_suite] compile entrypoint={nev_compile.compile_simulator.__name__}")
PY

python - <<'PY'
from pathlib import Path

from sanafecpp import *  # noqa: F403

plugin_dir = Path("build/mimarsinan_sanafe_plugins")
required = [
    "libmimarsinan_dendrite.so",
    "libmimarsinan_soma.so",
    "libmimarsinan_ttfs_continuous_soma.so",
    "libmimarsinan_ttfs_quantized_soma.so",
    "libmimarsinan_ttfs_cycle_soma.so",
    "libmimarsinan_ttfs_cascade_soma.so",
]
missing = [name for name in required if not (plugin_dir / name).exists()]
if missing:
    raise SystemExit(
        "SANA-FE plugins missing under build/mimarsinan_sanafe_plugins: "
        + ", ".join(missing)
        + ". Run scripts/setup_remote_env.sh or scripts/bootstrap_sanafe.sh."
    )
print("[remote_smoke_suite] sanafe import + plugins ok")
PY

python - <<'PY'
import numpy as np
from mimarsinan.chip_simulation.ttfs.ttfs_executor import TtfsAnalyticalExecutor
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping
from mimarsinan.code_generation.cpp_chip_model import SpikeSource

core = HardCore(axons_per_core=2, neurons_per_core=2)
core.core_matrix = np.array([[2.0, -1.0], [0.5, 1.0]], dtype=np.float64)
core.axon_sources = [
    SpikeSource(-2, 0, is_input=True, is_off=False),
    SpikeSource(-2, 1, is_input=True, is_off=False),
]
core.threshold = 2.0
core.hardware_bias = np.array([0.1, -0.2], dtype=np.float64)
core.latency = 0
core.available_axons = 0
core.available_neurons = 0
mapping = HardCoreMapping(chip_cores=[])
mapping.cores = [core]
mapping.output_sources = np.array([SpikeSource(0, 0), SpikeSource(0, 1)])

inp = np.array([[0.25, 0.75]], dtype=np.float64)
exec_ = TtfsAnalyticalExecutor()
out = exec_.run_segment(mapping, inp, simulation_length=4, spiking_mode="ttfs")
assert out.per_core_activations[0].shape == (1, 2)
print("[remote_smoke_suite] tiny TTFS analytical segment ok")
PY

echo "[remote_smoke_suite] all checks passed"
