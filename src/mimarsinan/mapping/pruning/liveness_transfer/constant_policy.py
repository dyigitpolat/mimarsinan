"""The ``elimination_constant_folding`` config axis (SSOT) + the exactness domain gate.

Two orthogonal switches decide whether a CONST line may be folded:

- the POLICY axis (``full`` default / ``off`` kill-switch), resolved from the
  deployment config exactly like ``computeop_liveness_transfers``; the
  ``identity_only`` transfer kill-switch forces it ``off`` because constant
  propagation is built on the per-op transfer relations;
- the DOMAIN gate, derived from ``spiking_mode``. Folding a CONST(c) axon row
  into a core's constant carrier replaces a per-timestep signal by a
  per-timestep bias. For ``c == 0`` that is exact in every domain (no signal
  either way — the existing elimination invariant). For ``c != 0`` it is exact
  only where a core's emission is value-linear, i.e. the value/MVM domain
  (``INERT_SPIKING_MODE``): a rate- or time-coded spike train matches the
  folded bias only in its WINDOW INTEGRAL, not per timestep, and the
  intervening threshold nonlinearity can turn that difference into different
  spike counts. So the gate REFUSES non-zero constants outside the value
  domain rather than guessing — the same refuse-to-certify honesty the
  cascade certificate uses for grid-breaking ops.
"""

from __future__ import annotations

from typing import Any, Mapping

from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from mimarsinan.mapping.pruning.liveness_transfer.transfer_policy import (
    COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY,
    DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
)

ELIMINATION_CONSTANT_FOLDING_KEY = "elimination_constant_folding"

# The intended capability: the constant lattice runs and folds every CONST
# line the exactness gate admits.
ELIMINATION_CONSTANT_FOLDING_FULL = "full"

# Kill-switch: the lattice never descends, so every kill set is byte-identical
# to the W4b-1 (zero-only) cascade.
ELIMINATION_CONSTANT_FOLDING_OFF = "off"

ELIMINATION_CONSTANT_FOLDING_MODES = (
    ELIMINATION_CONSTANT_FOLDING_FULL,
    ELIMINATION_CONSTANT_FOLDING_OFF,
)

DEFAULT_ELIMINATION_CONSTANT_FOLDING = ELIMINATION_CONSTANT_FOLDING_FULL


def require_elimination_constant_folding(policy: Any) -> str:
    """Validate and return a constant-folding policy; unknown values fail loud."""
    if policy not in ELIMINATION_CONSTANT_FOLDING_MODES:
        raise ValueError(
            f"{ELIMINATION_CONSTANT_FOLDING_KEY} must be one of "
            f"{ELIMINATION_CONSTANT_FOLDING_MODES}, got {policy!r}."
        )
    return str(policy)


def resolve_elimination_constant_folding(config: Mapping[str, Any]) -> str:
    """Resolve the policy from a deployment config; absent = default (full)."""
    return require_elimination_constant_folding(
        config.get(
            ELIMINATION_CONSTANT_FOLDING_KEY,
            DEFAULT_ELIMINATION_CONSTANT_FOLDING,
        )
    )


def effective_constant_folding(
    *,
    policy: str = DEFAULT_ELIMINATION_CONSTANT_FOLDING,
    computeop_liveness_transfers: str = DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
) -> str:
    """The policy after the ``identity_only`` implication (transfers -> folding).

    Constant propagation composes the per-op ``LivenessTransfer`` relations,
    so the transfer kill-switch necessarily disables it too: ``identity_only``
    must reproduce the pre-W4b barrier exactly, folds included.
    """
    resolved = require_elimination_constant_folding(policy)
    if computeop_liveness_transfers == COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY:
        return ELIMINATION_CONSTANT_FOLDING_OFF
    return resolved


def domain_admits_nonzero_constants(spiking_mode: Any) -> bool:
    """Whether CONST(c != 0) folds are exact for this chip domain.

    True only for the value/MVM domain (``INERT_SPIKING_MODE``), where a core
    emits ``(x @ W + b) / theta`` — a bias entry and an always-on axon row are
    then the SAME structure, so moving ``c * W[r, :]`` onto the carrier is an
    identity on the value function. Spiking domains get CONST(0) folding only.
    """
    return str(spiking_mode or "").lower() == INERT_SPIKING_MODE
