#!/usr/bin/env bash
# Smoke test: one CIFAR-10 d4 synchronized cell (~8 min GPU).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
run_config "${BASE_CONFIG}" "cifar10_d4_synchronized_smoke"
