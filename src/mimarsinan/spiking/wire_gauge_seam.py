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

from typing import Any, Callable

import torch

from mimarsinan.common.refusal.toolchain_refusal import ToolchainRefusal
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.scale_propagation import arm_wrap_slots
from mimarsinan.mapping.support.activation_scales import INPUT_WIRE_GAUGE
from mimarsinan.mapping.support.negative_boundary import _is_host_node, boundary_consumers
from mimarsinan.mapping.support.value_domain import value_domain_map
from mimarsinan.spiking.scale_aware_boundaries import (
    establish_gauge_for_mixed_domain_seams,
    establish_wire_gauge,
    read_boundary_out_scales,
    stamped_input_boundary_scale,
    verify_boundary_currency_coherence,
)
from mimarsinan.spiking.segment_partition import perceptron_of

__all__ = [
    "WireGaugeNotEstablishedError",
    "arm_absolute_core_entries",
    "establish_deployed_wire_gauge",
    "require_established_wire_gauge",
    "wire_gauge_established",
]

_STAMP = "wire_gauge_established"
#: The graph-level stamps the seam writes (the input scale, the raw input's wire gauge, the seam's own).
_REPR_STAMPS = ("input_boundary_scale", INPUT_WIRE_GAUGE, _STAMP)
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


def establish_deployed_wire_gauge(model_or_repr, *, input_data_scale: float,
                                  negative_boundary: Callable[[Any], Any] | None = None) -> int:
    """Arm the wrap slots and propagate the boundary currencies (``establish_wire_gauge``), repair every mixed-domain
    join, apply the negative-value boundary policy when the caller binds one (``negative_boundary``, U-SC2: the
    calibration batch and the run's switches, ``tuning.orchestration.negative_boundary_seam``), certify one-writer
    coherence, and stamp the graph. Returns the number of joins repaired."""
    establish_wire_gauge(model_or_repr, input_data_scale=input_data_scale)
    repaired = establish_gauge_for_mixed_domain_seams(model_or_repr, input_data_scale=input_data_scale)
    if arm_absolute_core_entries(model_or_repr, input_data_scale=input_data_scale):
        establish_wire_gauge(model_or_repr, input_data_scale=input_data_scale)
    if negative_boundary is not None:
        negative_boundary(model_or_repr)
    verify_boundary_currency_coherence(model_or_repr, input_data_scale=input_data_scale)
    setattr(_mapper_repr(model_or_repr), _STAMP, True)
    return repaired


def arm_absolute_core_entries(model_or_repr, *, input_data_scale: float) -> int:
    """[U-SC2] Arm every input-rooted ABSOLUTE host op a core reads at its boundary-table gauge: wrap slots at the
    pass-through currency (raw sources decoded at one, the output at the gauge the boundary walk propagates). The
    boundary table, the core's entry currency and the walks encode such a value at that gauge (the input scale), the
    deployed program at the unit gauge of the raw buffer it folds, so at any input scale but 1 the twins disagreed (the
    first core of DenseNet-BC and PyramidNet: S1 and S4 run at 2.02); armed, the fold and the divisor follow the one
    gauge. An op at the unit gauge already agrees and is left as it is. Returns the number armed."""
    repr_ = _mapper_repr(model_or_repr)
    if repr_ is None:
        return 0
    repr_._ensure_exec_graph()
    flags, _ = value_domain_map(repr_._exec_order, repr_._deps)
    table = read_boundary_out_scales(repr_, input_data_scale=input_data_scale)
    consumers = repr_.consumer_map()
    armed = 0
    for node in repr_._exec_order:
        if not isinstance(node, ComputeOpMapper) or node.output_scale is not None or flags.get(node) is not True:
            continue
        gauge = float(table.get(node, 1.0))
        if abs(gauge - 1.0) <= _RTOL or not any(not _is_host_node(c) for c in boundary_consumers(node, consumers)):
            continue
        arm_wrap_slots(node, [1.0] * len(node._sources_list), gauge)
        armed += 1
    return armed


def _clone(t):
    return None if t is None else torch.as_tensor(t).detach().clone()


def _gauge_state(repr_) -> dict:
    """Every slot the seam writes: the wrap slots of each host ComputeOp, each perceptron's fold and entry currencies."""
    state: dict = {"ops": [], "perceptrons": [], **{key: repr_.__dict__.get(key) for key in _REPR_STAMPS}}
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
    for key in _REPR_STAMPS:
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
