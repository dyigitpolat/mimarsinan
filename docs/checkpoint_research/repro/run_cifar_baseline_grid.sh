#!/usr/bin/env bash
# Full 8-cell CIFAR baseline grid (see data/00_cifar_baseline_grid.jsonl).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

run_cell() {
  local label="$1"
  shift
  local cfg
  cfg="$(make_variant_config "${label}" "$@")"
  run_config "${cfg}" "${label}"
}

# CIFAR-10
run_cell "cifar10_d4_synchronized" 'ttfs_cycle_schedule="synchronized"' 'depth=4' 'data_provider_name="CIFAR10_DataProvider"'
run_cell "cifar10_d6_synchronized" 'ttfs_cycle_schedule="synchronized"' 'depth=6' 'data_provider_name="CIFAR10_DataProvider"'
run_cell "cifar10_d4_cascaded"     'ttfs_cycle_schedule="cascaded"'     'depth=4' 'data_provider_name="CIFAR10_DataProvider"'
run_cell "cifar10_d6_cascaded"     'ttfs_cycle_schedule="cascaded"'     'depth=6' 'data_provider_name="CIFAR10_DataProvider"'

# CIFAR-100
run_cell "cifar100_d4_synchronized" 'ttfs_cycle_schedule="synchronized"' 'depth=4' 'data_provider_name="CIFAR100_DataProvider"'
run_cell "cifar100_d6_synchronized" 'ttfs_cycle_schedule="synchronized"' 'depth=6' 'data_provider_name="CIFAR100_DataProvider"'
run_cell "cifar100_d4_cascaded"     'ttfs_cycle_schedule="cascaded"'     'depth=4' 'data_provider_name="CIFAR100_DataProvider"'
run_cell "cifar100_d6_cascaded"     'ttfs_cycle_schedule="cascaded"'     'depth=6' 'data_provider_name="CIFAR100_DataProvider"'

echo "Done. Compare outputs to docs/checkpoint_research/data/00_cifar_baseline_grid.jsonl"
