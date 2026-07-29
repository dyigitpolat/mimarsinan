"""The ``computeop_liveness_transfers`` config axis (SSOT): full | identity_only."""

from __future__ import annotations

from typing import Any, Mapping

COMPUTEOP_LIVENESS_TRANSFERS_KEY = "computeop_liveness_transfers"

# The intended capability: per-op transfer functions relay deadness through
# elementwise activations, index bijections, and pooling regions.
COMPUTEOP_LIVENESS_TRANSFERS_FULL = "full"

# Pre-W4b behavior for A/B: only declared-identity 1:1 ops relay, every
# op-referenced producer is starvation-guarded.
COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY = "identity_only"

COMPUTEOP_LIVENESS_TRANSFERS_MODES = (
    COMPUTEOP_LIVENESS_TRANSFERS_FULL,
    COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY,
)

DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS = COMPUTEOP_LIVENESS_TRANSFERS_FULL


def require_computeop_liveness_transfers(policy: Any) -> str:
    """Validate and return a transfer policy; unknown values fail loud."""
    if policy not in COMPUTEOP_LIVENESS_TRANSFERS_MODES:
        raise ValueError(
            f"{COMPUTEOP_LIVENESS_TRANSFERS_KEY} must be one of "
            f"{COMPUTEOP_LIVENESS_TRANSFERS_MODES}, got {policy!r}."
        )
    return str(policy)


def resolve_computeop_liveness_transfers(config: Mapping[str, Any]) -> str:
    """Resolve the policy from a deployment config; absent = default (full)."""
    return require_computeop_liveness_transfers(
        config.get(
            COMPUTEOP_LIVENESS_TRANSFERS_KEY,
            DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
        )
    )
