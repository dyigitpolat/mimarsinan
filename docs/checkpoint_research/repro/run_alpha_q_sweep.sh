#!/usr/bin/env bash
# Alpha, quantile, and LIF baseline sweep on CIFAR-10 d4.
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

run_variant "ttfs_a0.3_base" 'kd_ce_alpha=0.3'
run_variant "ttfs_a0.6"      'kd_ce_alpha=0.6'
run_variant "ttfs_a1.0"      'kd_ce_alpha=1.0'
run_variant "ttfs_q1.0"      'activation_scale_quantile=1.0'
run_variant "ttfs_a1.0_q1.0" 'kd_ce_alpha=1.0' 'activation_scale_quantile=1.0'
run_variant "lif_a0.3_base"  'spiking_mode="lif"'

echo "Done. Compare to data/02_ttfs_alpha_q_sweep.jsonl"
