"""[W6c] Which analysis inputs the ledger uses when the arms are precomputed.

``compute_elimination_ledger`` accepts an already-computed
:class:`EliminationArms` so a deployment pays for the masked / closure /
requested runs once. Those runs have ALREADY consumed the seeds, the zero
threshold, the transfer/folding registries and the spiking mode — before W6c
the ledger kept its own copies of all of them, silently ignored whatever the
caller passed, and leaked its OWN defaults into the liveness pass (so a
precomputed-arms caller measured liveness at ``zero_threshold=1e-8`` and
``spiking_mode="lif"`` no matter what the arms ran under).

This module is the one place that decides, and it never drops an input on the
floor: with arms present every arm input is READ BACK OFF THEM, a contradicting
value RAISES, and a second seed set RAISES.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from mimarsinan.mapping.pruning.elimination_ledger.arm_runs import (
    EliminationArms,
)
from mimarsinan.mapping.pruning.elimination_ledger.ledger_types import (
    EliminationLedgerError,
)
from mimarsinan.mapping.pruning.liveness_transfer import (
    DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
    DEFAULT_ELIMINATION_CONSTANT_FOLDING,
)


class Unset:
    """Sentinel: this analysis input was not supplied by the caller."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


UNSET = Unset()


@dataclass(frozen=True)
class ArmInputs:
    """The analysis inputs one ledger run actually uses.

    Field names match BOTH the ``compute_elimination_ledger`` parameters and
    the :class:`EliminationArms` fields, which is what lets the reconciliation
    below be a single generic loop instead of four hand-written comparisons.
    """

    zero_threshold: float = 1e-8
    computeop_liveness_transfers: str = DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS
    elimination_constant_folding: str = DEFAULT_ELIMINATION_CONSTANT_FOLDING
    spiking_mode: str = "lif"


_ARM_INPUT_NAMES = tuple(ArmInputs.__dataclass_fields__)


def resolve_arm_inputs(
    supplied: Mapping[str, Any],
    seeds: Mapping[str, Any],
    arms: EliminationArms | None,
    mode: str,
) -> ArmInputs:
    """The analysis inputs to USE — off the arms when they exist, else off the
    caller — after refusing every input that contradicts the arms."""
    if arms is None:
        given: Dict[str, Any] = {
            name: value for name, value in supplied.items()
            if not isinstance(value, Unset)
        }
        return ArmInputs(**given)

    if arms.mode != mode:
        raise EliminationLedgerError(
            f"precomputed arms were run under mode={arms.mode!r} but the "
            f"ledger was asked for mode={mode!r}; the attribution would "
            "difference against the wrong arms."
        )
    for name, value in seeds.items():
        if value is not None:
            raise EliminationLedgerError(
                f"{name} was passed alongside precomputed arms, which were "
                "already seeded and exemption-filtered; the ledger cannot "
                "honour a second seed set. Pass the seeds to "
                "compute_elimination_arms instead."
            )
    for name in _ARM_INPUT_NAMES:
        value = supplied[name]
        if isinstance(value, Unset):
            continue
        ran_with = getattr(arms, name)
        if value != ran_with:
            raise EliminationLedgerError(
                f"{name}={value!r} was passed alongside precomputed arms that "
                f"were run with {name}={ran_with!r}; the ledger would report "
                "an attribution the arms never computed."
            )
    return ArmInputs(**{name: getattr(arms, name) for name in _ARM_INPUT_NAMES})
