#!/usr/bin/env bash
# Shared helpers for checkpoint reproduction sweeps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CHECKPOINT_DIR}/../.." && pwd)"
OUT_DIR="${CHECKPOINT_DIR}/data"
GEN_DIR="${REPO_ROOT}/generated"
BASE_CONFIG="${SCRIPT_DIR}/base_configs/cifar10_d4_synchronized.json"

export MIMARSINAN_DISABLE_FFCV=1

run_config() {
  local cfg_path="$1"
  local label="$2"
  echo "=== Running ${label} ==="
  echo "  config: ${cfg_path}"
  (cd "${REPO_ROOT}" && env/bin/python run.py --headless "${cfg_path}")
  local exp
  exp="$(python3 -c "import json; print(json.load(open('${cfg_path}'))['experiment_name'])")"
  local metric_file="${GEN_DIR}/${exp}_phased_deployment_run/__target_metric.json"
  if [[ -f "${metric_file}" ]]; then
    echo "  deployed metric: $(cat "${metric_file}")"
  else
    echo "  WARNING: metric file not found: ${metric_file}"
  fi
}

make_variant_config() {
  local label="$1"
  shift
  local out="${SCRIPT_DIR}/generated/${label}.json"
  mkdir -p "${SCRIPT_DIR}/generated"
  python3 - "${BASE_CONFIG}" "${out}" "${label}" "$@" <<'PY'
import json, sys
base_path, out_path, label = sys.argv[1], sys.argv[2], sys.argv[3]
patches = dict(a.split("=", 1) for a in sys.argv[4:])
with open(base_path) as f:
    cfg = json.load(f)
cfg["experiment_name"] = f"checkpoint_repro_{label}"
dp = cfg.setdefault("deployment_parameters", {})
pc = cfg.setdefault("platform_constraints", {})
for k, v in patches.items():
    if k.startswith("deployment_parameters."):
        dp[k.split(".", 1)[1]] = json.loads(v)
    elif k.startswith("platform_constraints."):
        pc[k.split(".", 1)[1]] = json.loads(v)
    elif k == "spiking_mode":
        dp["spiking_mode"] = json.loads(v)
    elif k == "data_provider_name":
        cfg["data_provider_name"] = json.loads(v)
    elif k == "depth":
        dp.setdefault("model_config", {})["depth"] = json.loads(v)
    elif k == "ttfs_cycle_schedule":
        dp["ttfs_cycle_schedule"] = json.loads(v)
    else:
        dp[k] = json.loads(v)
with open(out_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(out_path)
PY
  echo "${out}"
}
