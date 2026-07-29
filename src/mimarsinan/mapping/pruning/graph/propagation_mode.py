"""Elimination-propagation mode axis (SSOT): masked | closure | cascade."""

from __future__ import annotations

from typing import Any, Mapping

ELIMINATION_PROPAGATION_KEY = "elimination_propagation"

# Allocation-naive LOWER BOUND: exactly the seeded, exemption-filtered sets.
# No coupling, no propagation — what an allocator reclaims if it only trusts
# the criterion masks. Deliberately NOT called a "structured-pruning baseline":
# real single-layer structured pruners still couple paired rows (see closure).
ELIMINATION_PROPAGATION_MASKED = "masked"

# One-hop seed-group coupling (the DepGraph / torch-pruning-equivalent
# baseline): a seeded neuron kill removes its axon row in every direct
# consumer, and a seeded axon-row kill removes its paired producer neuron
# when every reader of that neuron is seed-dead. No emergent-deadness
# discovery (a neuron whose inputs all died stays alive) and no iteration.
ELIMINATION_PROPAGATION_CLOSURE = "closure"

# The full bidirectional cross-core liveness fixpoint (default path).
ELIMINATION_PROPAGATION_CASCADE = "cascade"

ELIMINATION_PROPAGATION_MODES = (
    ELIMINATION_PROPAGATION_MASKED,
    ELIMINATION_PROPAGATION_CLOSURE,
    ELIMINATION_PROPAGATION_CASCADE,
)

DEFAULT_ELIMINATION_PROPAGATION = ELIMINATION_PROPAGATION_CASCADE


def require_elimination_propagation(mode: Any) -> str:
    """Validate and return a propagation mode; unknown values fail loud."""
    if mode not in ELIMINATION_PROPAGATION_MODES:
        raise ValueError(
            f"elimination_propagation must be one of "
            f"{ELIMINATION_PROPAGATION_MODES}, got {mode!r}."
        )
    return str(mode)


def resolve_elimination_propagation(config: Mapping[str, Any]) -> str:
    """Resolve the mode from a deployment config; absent = default (cascade)."""
    return require_elimination_propagation(
        config.get(ELIMINATION_PROPAGATION_KEY, DEFAULT_ELIMINATION_PROPAGATION)
    )
