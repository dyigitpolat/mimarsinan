#!/usr/bin/env bash
# T sweep on CIFAR-10 d4 synchronized (T=4 is the base config default).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

for T in 8 16 32 64; do
  cfg="$(make_variant_config "ttfs_T${T}" \
    "platform_constraints.target_tq=${T}" \
    "platform_constraints.simulation_steps=${T}")"
  run_config "${cfg}" "ttfs_T${T}"
done

echo "Done. Compare to data/01_ttfs_T_sweep.jsonl"
