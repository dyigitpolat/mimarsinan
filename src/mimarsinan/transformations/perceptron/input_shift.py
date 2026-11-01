"""The input shift a perceptron's bias compensates, recorded at its bake and re-derived at every weight projection.

A negative-value boundary feeds its consumer ``x + s`` (a lifted host op, or Soft Core Mapping's shift), and the
consumer's bias is baked once to ``B' = B - W~ s`` on the effective weights of that moment
(``mapping.support.negative_shift.apply_negative_shift_bias``). A weight projection that replaces ``W~`` by ``Q(W~)``
must move the bias with it, ``B' <- B' - (Q(W~) - W~) s``, or every lifted entry deploys an extra ``(Q(W~) - W~) s``
(phase C defect D-WQ, cause B). The bake therefore records ``s`` here, and the projection compensates every weight
write it makes.
"""

from __future__ import annotations

import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

INPUT_SHIFT = "input_shift"
"""The buffer holding the shift a bake compensated (effective units, a scalar or one value per axon)."""

BAKED_FLAG = "_neg_shift_baked"
"""The bake's once-only flag (``apply_negative_shift_bias``)."""


class UnrecordedShiftBakeError(ToolchainRefusal, ValueError):
    """A perceptron whose bias was baked for an input shift that no record names (a model saved before the record)."""


def record_input_shift(perceptron, shift: torch.Tensor) -> None:
    """Keep the shift a bake compensated as a non-persistent buffer beside the bake's flag: it moves with the module,
    travels in deep copies and pickled caches, and leaves the state_dict layout unchanged."""
    perceptron.register_buffer(INPUT_SHIFT, shift.detach().clone(), persistent=False)


def recorded_input_shift(perceptron) -> torch.Tensor | None:
    """The shift ``perceptron``'s bias compensates, or None when no bake recorded one."""
    return getattr(perceptron, INPUT_SHIFT, None)


def compensate_projected_weights(perceptron, transformer: PerceptronTransformer, weight_before: torch.Tensor) -> None:
    """``B' <- B' - (W~_after - W~_before) s``: the bias keeps compensating the recorded shift through the effective
    weights the projection just installed, at any rate (the delta is whatever the projection wrote)."""
    shift = recorded_input_shift(perceptron)
    if shift is None:
        return
    delta = transformer.get_effective_weight(perceptron) - weight_before
    correction = (delta * shift.to(dtype=delta.dtype, device=delta.device)).sum(dim=-1)
    transformer.apply_effective_bias_transform(perceptron, lambda b, c=correction: b - c)


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
