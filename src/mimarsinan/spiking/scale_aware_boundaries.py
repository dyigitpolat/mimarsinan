"""Scale-aware TTFS boundaries: theta_out normalizes a block output to [0,1]; the downstream input_scale un-normalizes it."""

from __future__ import annotations

import torch

from mimarsinan.mapping.mappers.scale_propagation import arm_wrap_slots, walk_out_scales
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.support.value_domain import heterogeneous_domain_joins
from mimarsinan.spiking.segment_partition import (
    partition_spike_segments,
    perceptron_of,
)


def _as_model_repr(model_repr_or_model):
    """Accept either a ModelRepresentation or a model exposing get_mapper_repr."""
    if hasattr(model_repr_or_model, "_ensure_exec_graph"):
        return model_repr_or_model
    return model_repr_or_model.get_mapper_repr()


def establish_wire_gauge(
    model_repr_or_model, *, input_data_scale: float,
    arm_wire_value_ops: bool = True,
) -> None:
    """The gauge-establishment seam: arm every host ComputeOp's wrap slots, then
    propagate the boundary currencies (calculus §11.2/§16.6, one writer).

    Arming is what lets a HETEROGENEOUS fan-in be executed at all: each source
    decodes at its own producer gauge through the ``ScaleNormalizingWrapper``
    that IR emission installs and the twin walk runs, so both sides share one
    definition. Any stage that trains against the deployed composition must run
    this first, or its twin walks a graph whose currencies the later seams
    (LIF Affine Fold / WQ / SCM) will re-derive differently. Idempotent in
    ``activation_scales``, and inert when no gauge is non-unit.
    """
    model_repr = _as_model_repr(model_repr_or_model)
    compute_per_source_scales(model_repr, arm_wire_value_ops=arm_wire_value_ops)
    propagate_boundary_input_scales(
        model_repr_or_model, input_data_scale=input_data_scale
    )


def _arm_domain_join(node) -> None:
    """Arm a mixed fan-in the SCALE POLICY declined, at UNIT gauge.

    Unity is not a choice here, it is the only currency this fallback can ever
    see: a join reaching it has ≥2 sources that `apply_compute_op_scale_policy`
    left unarmed, which happens only when every source scale is exactly 1.0 (a
    non-unit gauge either makes the fan-in non-uniform — the legacy wrap — or
    trips the value-op wrap). ``kappa_T == kappa_S`` therefore holds trivially,
    and a producer-gauge lookup here would be decorative: it could not return
    anything but 1.0. The arming is purely STRUCTURAL — it is numerically inert
    (the wrapper multiplies and divides by one) and exists so the join HAS a
    domain, which is what lets the wire-currency twin walk the graph at all.

    The gauge choice at a join whose producers genuinely differ belongs to the
    policy, not here, and is measured there — see
    ``tests/unit/pipelining/test_streamed_mixed_seam_exactness.py::
    test_gate_catches_a_seam_decoded_at_unity`` (deployed window counts move on
    56% of neurons when the seam decodes at unity instead of producer gauge).
    """
    arm_wrap_slots(node, [1.0] * len(node._sources_list), 1.0)


def establish_gauge_for_mixed_domain_seams(
    model_repr_or_model, *, input_data_scale: float,
) -> int:
    """Precondition repair for a wire-currency twin: a graph carrying a
    heterogeneous fan-in has no domain there, so no twin can walk it.

    Establishes the gauge — the SAME arming the WQ / SCM seams re-derive, so
    the trained twin IS the deployed composition — and certifies one-writer
    coherence. Arming only ever turns a node wire, so the fixpoint is monotone
    and bounded by the node count. Returns the number of seams repaired; a graph
    whose domains already classify is left byte-identical (nothing runs).
    """
    model_repr = _as_model_repr(model_repr_or_model)
    repaired = len(heterogeneous_domain_joins(model_repr))
    if not repaired:
        return 0
    for _ in range(len(model_repr.execution_order()) + 1):
        establish_wire_gauge(model_repr_or_model, input_data_scale=input_data_scale)
        joins = heterogeneous_domain_joins(model_repr)
        if not joins:
            break
        for node in joins:
            _arm_domain_join(node)
    verify_boundary_currency_coherence(
        model_repr_or_model, input_data_scale=input_data_scale
    )
    return repaired


def read_boundary_out_scales(model_repr_or_model, input_data_scale: float) -> dict:
    """Pure (no-mutation) twin of :func:`propagate_boundary_input_scales`.

    A perceptron-bearing node yields its activation_scale (mean-collapsed);
    every other node DELEGATES to its own ``propagate_boundary_scale`` — one
    polymorphic implementation for both walks (armed buffer gauges, traffic
    lifts, residual-merge rules), so the two tables cannot drift again
    (calculus §11.2, the one-writer law).
    """
    model_repr = _as_model_repr(model_repr_or_model)
    default = float(input_data_scale)

    def visit(node, deps, out_scales):
        perceptron = getattr(node, "perceptron", None)
        if perceptron is not None:
            scale = perceptron.activation_scale
            if isinstance(scale, torch.Tensor):
                return float(scale.detach().to(torch.float64).mean())
            return float(scale)
        return node.propagate_boundary_scale(deps, out_scales, default)

    return walk_out_scales(model_repr, visit)


def propagate_boundary_input_scales(model_repr_or_model, input_data_scale: float):
    """Forward-propagate theta_out so each perceptron's ``input_activation_scale``
    equals the mean theta_out of its upstream perceptron source(s); the input
    boundary uses ``input_data_scale``.

    The scale is also stamped on the repr (``input_boundary_scale``) so pure
    re-reads (e.g. the LIF segment policy) agree with the propagated values by
    construction — one value, both walks (the NF↔SCM parity contract).
    """
    model_repr = _as_model_repr(model_repr_or_model)
    default = float(input_data_scale)
    walk_out_scales(
        model_repr,
        lambda node, deps, out_scales: node.propagate_boundary_scale(
            deps, out_scales, default
        ),
    )
    model_repr.input_boundary_scale = default


def boundary_scales_for_walk(model_repr_or_model, *, establish: bool) -> dict:
    """The out-scale table a twin walk reads, gauge FIRST when it owes one.

    A twin that decomposes hops through ``get_effective_weight`` (the
    per-event fold) reads the stamped per-source scales, so it must run the
    establishment seam before the walk; a twin that only re-encodes trains
    reads the pure table and leaves the graph untouched.
    """
    model_repr = _as_model_repr(model_repr_or_model)
    if establish:
        establish_wire_gauge(
            model_repr,
            input_data_scale=stamped_input_boundary_scale(model_repr),
        )
    return read_boundary_out_scales(
        model_repr, input_data_scale=stamped_input_boundary_scale(model_repr),
    )


def stamped_input_boundary_scale(model_repr_or_model) -> float:
    """The scale stamped by the last propagation; 1.0 (unit range) before any."""
    return float(
        getattr(_as_model_repr(model_repr_or_model), "input_boundary_scale", 1.0)
    )


_COHERENCE_RTOL = 1e-3


def verify_boundary_currency_coherence(
    model_repr_or_model, *, input_data_scale: float | None = None,
) -> None:
    """Install-seam coherence certificate (calculus §11.2): every host-fed
    segment entry's re-encode currency (the boundary table) must equal the
    consumer's trained entry currency (``input_activation_scale``); fail loud
    with the offending edges."""
    model_repr = _as_model_repr(model_repr_or_model)
    model_repr._ensure_exec_graph()
    default = (
        float(input_data_scale)
        if input_data_scale is not None
        else stamped_input_boundary_scale(model_repr)
    )
    table = read_boundary_out_scales(model_repr, input_data_scale=default)
    exec_order, deps = model_repr._exec_order, model_repr._deps
    seg_of, produces = partition_spike_segments(exec_order, deps)

    mismatches = []
    for node in exec_order:
        p = perceptron_of(node)
        if p is None or not produces.get(node, False):
            continue
        boundary_deps = [
            d for d in deps.get(node, [])
            if not (produces.get(d, False) and seg_of.get(d) == seg_of.get(node))
        ]
        if not boundary_deps:
            continue
        expected = sum(float(table.get(d, default)) for d in boundary_deps) / len(
            boundary_deps
        )
        stamped = float(
            torch.as_tensor(p.input_activation_scale).detach().to(torch.float64).mean()
        )
        if abs(stamped - expected) > _COHERENCE_RTOL * max(abs(expected), 1e-12):
            mismatches.append(
                f"{type(node).__name__}: entry currency {stamped:.6g} != "
                f"boundary table {expected:.6g}"
            )
    def _traced_armed_gauge(dep):
        """The dep's ARMED emitted gauge, traced through scale-transparent
        single-source structural nodes; None for value-domain producers
        (inputs / unarmed hosts), whose consumers correctly decode at 1."""
        seen = 0
        while dep is not None and seen < 64:
            if getattr(dep, "output_scale", None) is not None:
                return float(
                    torch.as_tensor(dep.output_scale).detach().to(torch.float64).mean()
                )
            below = deps.get(dep, [])
            if len(below) != 1 or perceptron_of(dep) is not None:
                return None
            dep = below[0]
            seen += 1
        return None

    for node in exec_order:
        ps = getattr(node, "per_source_scales", None)
        node_deps = deps.get(node, [])
        if ps is None or len(node_deps) != len(ps):
            continue
        for i, d in enumerate(node_deps):
            armed = _traced_armed_gauge(d)
            if armed is None:
                continue
            actual = float(
                torch.as_tensor(ps[i]).detach().to(torch.float64).mean()
            )
            if abs(actual - armed) > _COHERENCE_RTOL * max(abs(armed), 1e-12):
                mismatches.append(
                    f"{type(node).__name__}: per_source[{i}] {actual:.6g} != "
                    f"producer emitted gauge {armed:.6g} [calculus §16.6]"
                )
    if mismatches:
        raise RuntimeError(
            "boundary currency coherence violated (one-writer law, calculus "
            "§11.2) at: " + "; ".join(mismatches)
        )


def calibrate_scale_aware_boundaries(model, activation_scales, input_data_scale: float):
    """Set each block's ``activation_scale`` to its theta_out, then propagate so
    every input un-normalizes from [0,1]. The encoding layer is pinned to
    ``input_data_scale`` (retuning it breaks NF↔SCM deployment parity).
    """
    perceptrons = list(model.get_perceptrons())
    if len(activation_scales) != len(perceptrons):
        raise ValueError(
            f"activation_scales count {len(activation_scales)} != perceptron count "
            f"{len(perceptrons)}"
        )
    for perceptron, scale in zip(perceptrons, activation_scales):
        if getattr(perceptron, "is_encoding_layer", False):
            perceptron.set_activation_scale(float(input_data_scale))
        else:
            perceptron.set_activation_scale(float(scale))

    propagate_boundary_input_scales(model, input_data_scale=input_data_scale)
