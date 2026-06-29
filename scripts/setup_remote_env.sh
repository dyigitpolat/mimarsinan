#!/usr/bin/env bash
set -euo pipefail

# Build a shared remote Python environment for all slurmech runs.
#
# This script is intended to run on the Slurm login node after the workspace has
# been synced. Jobs should then use [env].mode="reuse" and point at the venv path
# below; the script is not run inside every Slurm allocation.

VENV_PATH="${MIMARSINAN_REMOTE_VENV:-$HOME/mimarsinan_slurmech/env}"
BASE_VENV="${MIMARSINAN_BASE_VENV:-$HOME/imagenetshaq/env}"
PYTHON_BIN="${MIMARSINAN_REMOTE_PYTHON:-python3}"
INSTALL_SEARCH_DEPS="${MIMARSINAN_INSTALL_SEARCH_DEPS:-0}"
FORCE_NEW_ENV="${MIMARSINAN_FORCE_NEW_ENV:-0}"

echo "[setup_remote_env] venv=${VENV_PATH}"
echo "[setup_remote_env] python=$(${PYTHON_BIN} --version 2>&1)"

if [ "${FORCE_NEW_ENV}" != "1" ] && [ -x "${BASE_VENV}/bin/python" ]; then
  if [ -e "${VENV_PATH}" ] && [ ! -L "${VENV_PATH}" ]; then
    if "${VENV_PATH}/bin/python" - <<'PY' >/dev/null 2>&1
import torch
PY
    then
      echo "[setup_remote_env] existing env imports torch"
    else
      backup="${VENV_PATH}.broken.$(date +%Y%m%d-%H%M%S)"
      echo "[setup_remote_env] moving broken env to ${backup}"
      mv "${VENV_PATH}" "${backup}"
    fi
  fi
  if [ ! -e "${VENV_PATH}" ]; then
    echo "[setup_remote_env] linking shared env to ${BASE_VENV}"
    ln -s "${BASE_VENV}" "${VENV_PATH}"
  fi
elif [ ! -x "${VENV_PATH}/bin/python" ]; then
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_PATH}"
fi

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
# Runtime/deployment dependencies used by run.py and the MNIST mixer campaign.
# Keep large CUDA packages and optional NAS/search dependencies out of the default
# path when reusing an existing CUDA-capable venv; installing another torch wheel
# commonly exceeds home-directory quota.
python -m pip install \
  git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git \
  einops \
  python-dotenv \
  pytest \
  pyflakes

python - <<'PY'
missing = []
warnings = []
for name in ["numpy", "scipy", "matplotlib", "einops"]:
    try:
        __import__(name)
    except Exception as exc:
        missing.append(f"{name}: {exc!r}")
for name in ["torch", "torchvision", "warmup_scheduler"]:
    try:
        __import__(name)
    except Exception as exc:
        warnings.append(f"{name}: {exc!r}")
if missing:
    raise SystemExit("missing required base dependencies:\n" + "\n".join(missing))
for warning in warnings:
    print("[setup_remote_env] login-node warning:", warning)
PY

if [ "${INSTALL_SEARCH_DEPS}" = "1" ]; then
  python -m pip install nni pymoo pydantic-ai plotly
fi

if [ "${MIMARSINAN_INSTALL_SANAFE:-1}" = "1" ]; then
  if ! python - <<'PY' >/dev/null 2>&1
from sanafecpp import *  # noqa: F403
PY
  then
    echo "[setup_remote_env] installing sanafe wheel and building plugins"
    python -m pip install "sanafe==2.1.1"
    PLUGIN_SRC="${MIMARSINAN_REPO:-$PWD}/src/mimarsinan/chip_simulation/sanafe/plugins"
    PLUGIN_BUILD="${MIMARSINAN_REPO:-$PWD}/build/mimarsinan_sanafe_plugins"
    rm -rf "${PLUGIN_BUILD}"
    mkdir -p "${PLUGIN_BUILD}"
    cmake -S "${PLUGIN_SRC}" -B "${PLUGIN_BUILD}" \
      -DSANAFE_SRC="${MIMARSINAN_REPO:-$PWD}/sana_fe/src"
    cmake --build "${PLUGIN_BUILD}" --parallel
  else
    echo "[setup_remote_env] sanafe already importable"
  fi
fi

python - <<'PY'
import sys
print("[setup_remote_env] python", sys.version)
for name in ["numpy", "scipy", "einops", "matplotlib"]:
    mod = __import__(name)
    print("[setup_remote_env]", name, getattr(mod, "__version__", "ok"))
for name in ["torch", "torchvision"]:
    try:
        mod = __import__(name)
        print("[setup_remote_env]", name, getattr(mod, "__version__", "ok"))
    except Exception as exc:
        print("[setup_remote_env] login-node warning:", name, repr(exc))
PY

echo "[setup_remote_env] done"
