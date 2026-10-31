"""The one gauge-establishment seam of the deployed composition, and the precondition every twin that walks it checks
(U-SC S-1).

The chip-aligned LIF walk is a wire-currency twin: every perceptron output travels as a rate, and a host op that
computes on values decodes each source at its producer's gauge through the wrap slots ``establish_wire_gauge`` arms (the
calls Weight Quantization and Soft Core Mapping make). A walk over a graph whose gauge was never established reads every
such op in rate units and measures a composition nothing deploys (SD-10a: the C5 T-LIF D-hat at 0.0092 against 0.2252
armed, and the LIF retention gate refusing at 0.0118). ``establish_deployed_wire_gauge`` is the seam the first stage that
walks the deployed composition runs, unconditionally: the arming, the mixed-domain repair and the one-writer coherence
certificate, and a stamp on the graph. ``require_established_wire_gauge`` is what a twin checks before it walks: a
stamped graph is re-derived at its current thetas (idempotent in ``activation_scales``: the exact-QAT ladder trains theta
after the seam), a graph on which the seam would change nothing passes as it is, and any other graph is refused by name.
"""

from __future__ import annotations

import torch

from mimarsinan.common.refusal.toolchain_refusal import ToolchainRefusal
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.spiking.scale_aware_boundaries import (
    establish_gauge_for_mixed_domain_seams,
    establish_wire_gauge,
    stamped_input_boundary_scale,
    verify_boundary_currency_coherence,
)
from mimarsinan.spiking.segment_partition import perceptron_of

__all__ = [
    "WireGaugeNotEstablishedError",
    "establish_deployed_wire_gauge",
    "require_established_wire_gauge",
    "wire_gauge_established",
]

_STAMP = "wire_gauge_established"
_RTOL = 1e-6


class WireGaugeNotEstablishedError(ToolchainRefusal, RuntimeError):
    """A twin asked to walk a wire-currency graph whose gauge no stage established."""


def _mapper_repr(model_or_repr):
    if hasattr(model_or_repr, "_ensure_exec_graph"):
        return model_or_repr
    get = getattr(model_or_repr, "get_mapper_repr", None)
    return get() if get is not None else None


def wire_gauge_established(model_or_repr) -> bool:
    """Whether the seam established this graph (the stamp travels with clones and step-cache pickles)."""
    repr_ = _mapper_repr(model_or_repr)
    return bool(getattr(repr_, _STAMP, False)) if repr_ is not None else False


def establish_deployed_wire_gauge(model_or_repr, *, input_data_scale: float) -> int:
    """Arm the wrap slots and propagate the boundary currencies (``establish_wire_gauge``), repair every mixed-domain
    join, certify one-writer coherence, and stamp the graph. Returns the number of joins repaired."""
    establish_wire_gauge(model_or_repr, input_data_scale=input_data_scale)
    repaired = establish_gauge_for_mixed_domain_seams(model_or_repr, input_data_scale=input_data_scale)
    if not repaired:  # the repair certifies when it ran
        verify_boundary_currency_coherence(model_or_repr, input_data_scale=input_data_scale)
    setattr(_mapper_repr(model_or_repr), _STAMP, True)
    return repaired


def _clone(t):
    return None if t is None else torch.as_tensor(t).detach().clone()


def _gauge_state(repr_) -> dict:
    """Every slot the seam writes: the wrap slots of each host ComputeOp, each perceptron's fold and entry currencies."""
    state: dict = {"ops": [], "perceptrons": [], "input_boundary_scale": repr_.__dict__.get("input_boundary_scale"),
                   _STAMP: repr_.__dict__.get(_STAMP)}
    for node in repr_.execution_order():
        if isinstance(node, ComputeOpMapper):
            sources = None if node.per_source_scales is None else [_clone(s) for s in node.per_source_scales]
            state["ops"].append((node, node.is_wire_value_op, sources, _clone(node.output_scale)))
        p = perceptron_of(node)
        if p is not None:
            state["perceptrons"].append((p, p.per_input_scales, _clone(p.input_activation_scale)))
    return state


def _restore(repr_, state: dict) -> None:
    for node, marked, sources, out in state["ops"]:
        node.is_wire_value_op, node.per_source_scales, node.output_scale = marked, sources, out
    for p, fold, entry in state["perceptrons"]:
        p.per_input_scales = fold
        if entry is not None:
            with torch.no_grad():
                p.input_activation_scale.copy_(entry.to(p.input_activation_scale.dtype))
    for key in ("input_boundary_scale", _STAMP):
        repr_.__dict__.pop(key, None)
        if state[key] is not None:
            setattr(repr_, key, state[key])


def _flat(t) -> torch.Tensor:
    return torch.ones(1, dtype=torch.float64) if t is None else torch.as_tensor(t).detach().to(torch.float64).reshape(-1)


def _same(a, b) -> bool:
    """Equal in effect (an unstamped fold, None, folds by one); a scalar broadcasts against a vector."""
    a, b = _flat(a), _flat(b)
    if a.numel() != b.numel() and 1 not in (a.numel(), b.numel()):
        return False
    a, b = torch.broadcast_tensors(a, b)
    return bool(torch.allclose(a, b, rtol=_RTOL, atol=0.0))


def _slots_differ(before: dict, after: dict) -> list:
    changed = []
    for (node, _, src0, out0), (_, _, src1, out1) in zip(before["ops"], after["ops"]):
        if (out0 is None) != (out1 is None) or (src0 is None) != (src1 is None):
            changed.append(getattr(node, "name", None) or type(node).__name__)
        elif out0 is not None and (not _same(out0, out1) or not all(_same(a, b) for a, b in zip(src0, src1))):
            changed.append(getattr(node, "name", None) or type(node).__name__)
    for (p, fold0, entry0), (_, fold1, entry1) in zip(before["perceptrons"], after["perceptrons"]):
        if not _same(fold0, fold1) or not _same(entry0, entry1):
            changed.append(str(getattr(p, "name", None) or type(p).__name__))
    return changed


def require_established_wire_gauge(model_or_repr, *, input_data_scale: float, reader: str) -> None:
    """The twin's precondition: re-derive a stamped graph at its current thetas; pass a graph the seam would not change;
    refuse any other graph by name, leaving it as it was."""
    repr_ = _mapper_repr(model_or_repr)
    if repr_ is None:
        return  # no mapper graph: the plain cycle-accurate fallback walks no currency
    if wire_gauge_established(repr_):
        establish_wire_gauge(model_or_repr, input_data_scale=stamped_input_boundary_scale(repr_))
        return
    before = _gauge_state(repr_)
    try:
        establish_deployed_wire_gauge(model_or_repr, input_data_scale=input_data_scale)
        changed = _slots_differ(before, _gauge_state(repr_))
    finally:
        _restore(repr_, before)
    if changed:
        raise WireGaugeNotEstablishedError(
            f"{reader} walks a wire-currency graph whose gauge no stage established: the seam "
            f"(establish_deployed_wire_gauge) would re-derive {len(changed)} slot(s), first {changed[:4]}; the first stage "
            "that walks the deployed composition (the exact-QAT AQ install, LIF Adaptation's entry) must run it"
        )
