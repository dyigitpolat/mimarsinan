"""The input shift a perceptron's bias compensates, recorded at its bake and re-derived at every weight projection.

A negative-value boundary feeds its consumer ``x + s`` (a lifted host op, or Soft Core Mapping's shift), and the
consumer's bias is corrected for the field each axon receives, ``B' = B - W~ s`` on the effective weights of that moment
(``mapping.support.received_shift``, the one writer). A weight projection that replaces ``W~`` by ``Q(W~)`` must move
the bias with it, ``B' <- B' - (Q(W~) - W~) s``, or every lifted entry deploys an extra ``(Q(W~) - W~) s`` (phase C
defect D-WQ, cause B). The correction therefore records ``s`` here (one value, one per input, or one per element of the
input), and the projection compensates every weight write it makes (``received_shift.compensate_projected_weights``).
"""

from __future__ import annotations

import torch

from mimarsinan.common.refusal import ToolchainRefusal

INPUT_SHIFT = "input_shift"
"""The buffer holding the shift a correction compensates (effective units: one value, one per input, or one per element
of the input)."""

BAKED_FLAG = "_neg_shift_baked"
"""Marks that a correction's record exists (it no longer decides a correction: ``received_shift`` keeps the sources)."""


class UnrecordedShiftBakeError(ToolchainRefusal, ValueError):
    """A perceptron whose bias was baked for an input shift that no record names (a model saved before the record)."""


def record_input_shift(perceptron, shift: torch.Tensor) -> None:
    """Keep the shift a bake compensated as a non-persistent buffer beside the bake's flag: it moves with the module,
    travels in deep copies and pickled caches, and leaves the state_dict layout unchanged."""
    perceptron.register_buffer(INPUT_SHIFT, shift.detach().clone(), persistent=False)


def recorded_input_shift(perceptron) -> torch.Tensor | None:
    """The shift ``perceptron``'s bias compensates, or None when no bake recorded one."""
    return getattr(perceptron, INPUT_SHIFT, None)


def refuse_unrecorded_shift_bakes(perceptrons) -> None:
    """A baked perceptron without its record cannot have the compensation re-derived on projected weights: refuse it
    by name before a projection deploys the stale bake (rerun from the step that bakes, Activation Quantization's
    install or LIF Adaptation's entry, on a tree that records the shift)."""
    missing = [getattr(p, "name", None) or type(p).__name__ for p in perceptrons
               if getattr(p, BAKED_FLAG, False) and recorded_input_shift(p) is None]
    if missing:
        raise UnrecordedShiftBakeError(
            f"weight projection: {len(missing)} perceptron(s) carry a negative-shift bake (B - W s) with no recorded "
            f"{INPUT_SHIFT}, so the compensation cannot follow the projected weights (a model baked before the "
            f"record): {missing[:8]}{' ...' if len(missing) > 8 else ''}; rerun from the step that bakes the shift")
