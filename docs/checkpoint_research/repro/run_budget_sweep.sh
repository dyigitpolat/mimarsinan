#!/usr/bin/env bash
# Budget and epoch sweep on CIFAR-10 d4 synchronized (+ LIF variant).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

run_variant() {
  local label="$1"
  shift
  local cfg
  cfg="$(make_variant_config "${label}" "$@")"
  run_config "${cfg}" "${label}"
}

run_variant "ttfs_budget4_ep20"   'tuning_budget_scale=4'  'training_epochs=20'
run_variant "ttfs_budget16_ep40"  'tuning_budget_scale=16' 'training_epochs=40'
run_variant "ttfs_budget40_ep60"  'tuning_budget_scale=40' 'training_epochs=60'
run_variant "lif_budget16_ep40"   'spiking_mode="lif"' 'tuning_budget_scale=16' 'training_epochs=40'

echo "Done. Compare to data/03_budget_sweep.jsonl"
