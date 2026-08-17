"""Static on-chip-fraction pre-check: reproduce the on-chip validity gate from a native model without a pipeline run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.conv1d_mapper import Conv1DPerceptronMapper
from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.support.device_placement import single_device_of
from mimarsinan.mapping.support.host_contributors import (
    describe_host_holders,
    host_contributors_from_flow,
    host_contributors_from_ir,
)
from mimarsinan.mapping.verification.onchip_majority import (
    DEFAULT_ONCHIP_FLOOR,
    DEFAULT_ONCHIP_MAJORITY,
    ONCHIP_FLOOR_ESCAPE,
    OnchipMajorityError,
    OnchipParamBreakdown,
    compute_onchip_fraction,
    module_macs as _module_macs,
    onchip_placement_remedy,
    unwrap_scale_wrapper as _unwrap_module,
)

_PERCEPTRON_MAPPER_TYPES = (
    PerceptronMapper,
    Conv2DPerceptronMapper,
    Conv1DPerceptronMapper,
)

_VALID_METRICS = ("params", "macs", "ops")
_VALID_PLACEMENTS = ("subsume", "offload")

_DEFAULT_FLOOR = DEFAULT_ONCHIP_FLOOR
_DEFAULT_MAJORITY = DEFAULT_ONCHIP_MAJORITY

TIER_VALID = "VALID"
TIER_VALID_FLAGGED = "VALID_FLAGGED"
TIER_INVALID = "INVALID"

_UNSUPPORTED_HOST_OP_TYPES = (
    nn.MultiheadAttention,
    nn.LayerNorm,
    nn.GELU,
)


@dataclass(frozen=True)
class OnchipFractionEstimate:
    """Static host/on-chip split of a model under a given metric and placement."""

    onchip: int
    host: int
    total: int
    metric: str
    placement: str

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return self.onchip / self.total


@dataclass(frozen=True)
class ValidityVerdict:
    """Tiered deployment-validity verdict on both the param and MAC on-chip fractions.

    ``research_gap_ops`` are host ops with no on-chip SNN mapping yet;
    ``placement_fixable_ops`` are supported encoders offloadable via ``offload``.
    """

    tier: str
    param_frac: float
    mac_frac: float
    research_gap_ops: list = field(default_factory=list)
    placement_fixable_ops: list = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.tier != TIER_INVALID

    @property
    def is_flagged(self) -> bool:
        return self.tier == TIER_VALID_FLAGGED


def _is_perceptron_holder(node) -> bool:
    return isinstance(node, _PERCEPTRON_MAPPER_TYPES)


def _is_encoding_perceptron(node) -> bool:
    return _is_perceptron_holder(node) and bool(
        getattr(node.perceptron, "is_encoding_layer", False)
    )


def _host_unit(node):
    """Return the host-side ``nn.Module`` of a host mapper node, else ``None``.

    Mirrors ``count_host_params``: a ComputeOpMapper contributes its wrapped
    module; a segment-start encoding perceptron contributes itself.
    """
    if isinstance(node, ComputeOpMapper):
        return node.module
    if _is_encoding_perceptron(node):
        return node.perceptron
    return None


def _onchip_unit(node):
    """Return the on-chip ``nn.Module`` (a non-encoding perceptron), else ``None``."""
    if _is_perceptron_holder(node) and not _is_encoding_perceptron(node):
        return node.perceptron
    return None


def _perceptron_op_label(perceptron) -> str:
    layer = getattr(perceptron, "layer", None)
    return type(layer).__name__ if layer is not None else "Perceptron"


def _is_unsupported_host_op(module) -> bool:
    return isinstance(_unwrap_module(module), _UNSUPPORTED_HOST_OP_TYPES)


def _host_op_label(module) -> str:
    return type(_unwrap_module(module)).__name__


def _classify_host_node(node):
    """Classify a host-side node into a ``(category, op_label)`` pair.

    Category is ``placement`` (offloadable encoder), ``unsupported_op`` (research
    gap), or ``supported_host`` (always-host); ``(None, None)`` for non-host nodes.
    """
    if _is_encoding_perceptron(node):
        return "placement", _perceptron_op_label(node.perceptron)
    if isinstance(node, ComputeOpMapper):
        if _is_unsupported_host_op(node.module):
            return "unsupported_op", _host_op_label(node.module)
        return "supported_host", _host_op_label(node.module)
    return None, None


def _build_flow(model, input_shape, num_classes, placement):
    # Lazy: the torch_mapping package pulls the converter, which imports back
    # through mapping -> chip_simulation -> here.
    from mimarsinan.torch_mapping.encoding_layers import (
        require_resolved_encoding_placement,
    )

    if hasattr(model, "get_mapper_repr"):
        # Already a perceptron flow: its placement was resolved at flow birth,
        # and the marking it carries now IS what will deploy — including the
        # host placements the negative-boundary policy added after birth, which
        # re-marking here would erase. Measure it; never rewrite it. The
        # requirement that it BE resolved is checked, not assumed: an
        # unresolved flow is how the placement no-op stayed silent.
        # (FX re-tracing is neither possible for einops-based flows nor needed.)
        require_resolved_encoding_placement(
            model.get_mapper_repr(), placement, context="the on-chip fraction gate"
        )
        return model
    from mimarsinan.torch_mapping.converter import convert_torch_model

    return convert_torch_model(
        model,
        tuple(input_shape),
        int(num_classes),
        encoding_layer_placement=placement,
    )


def _exec_nodes(flow):
    mapper_repr = flow.get_mapper_repr()
    mapper_repr._ensure_exec_graph()
    return mapper_repr._exec_order


def _module_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _flow_device(flow) -> torch.device:
    return single_device_of(flow, what="the model under the on-chip validity gate")


def _assert_materialized(flow) -> None:
    from torch.nn.parameter import UninitializedParameter

    lazy = [
        name
        for name, p in flow.named_parameters()
        if isinstance(p, UninitializedParameter)
    ]
    if lazy:
        raise ValueError(
            "estimate_onchip_fraction: the model still carries uninitialized "
            f"lazy parameters {lazy[:4]}{'...' if len(lazy) > 4 else ''} — the "
            "Model Building warmup forward did not materialize this model "
            "(builder/device placement mismatch?). Run a real forward before "
            "the static on-chip-majority gate."
        )


def _params_breakdown(flow):
    _assert_materialized(flow)
    total = int(sum(p.numel() for p in flow.parameters()))
    seen: set[int] = set()
    host = 0
    for node in _exec_nodes(flow):
        unit = _host_unit(node)
        if unit is None or id(unit) in seen:
            continue
        seen.add(id(unit))
        host += _module_params(unit)
    return host, total


def _ops_breakdown(flow):
    """[R2] Host ComputeOp invocations vs all units — the per-invocation
    overhead multiplicand. Counted over the SAME walk as the other metrics:
    every host unit is one dispatch per inference."""
    host = 0
    total = 0
    for node in _exec_nodes(flow):
        if _host_unit(node) is not None:
            host += 1
            total += 1
        elif _onchip_unit(node) is not None:
            total += 1
    return host, total


def _macs_breakdown(flow, input_shape):
    units: dict[int, tuple[nn.Module, bool]] = {}
    for node in _exec_nodes(flow):
        host_unit = _host_unit(node)
        if host_unit is not None:
            units.setdefault(id(host_unit), (host_unit, True))
            continue
        chip_unit = _onchip_unit(node)
        if chip_unit is not None:
            units.setdefault(id(chip_unit), (chip_unit, False))

    shapes: dict[int, tuple] = {}
    handles = []

    def _make_hook(key):
        def _hook(_mod, inputs, output):
            in_t = inputs[0] if inputs else None
            in_shape = tuple(in_t.shape) if torch.is_tensor(in_t) else None
            out = output[0] if isinstance(output, (tuple, list)) and output else output
            out_shape = tuple(out.shape) if torch.is_tensor(out) else None
            shapes[key] = (in_shape, out_shape)

        return _hook

    for key, (module, _is_host) in units.items():
        handles.append(module.register_forward_hook(_make_hook(key)))

    try:
        flow.eval()
        # Probe on the flow's own device so the estimator works whether the
        # caller holds a CPU model spec (scheduler) or a GPU model mid-pipeline;
        # a flow that is on BOTH is a defect upstream and says so here.
        device = _flow_device(flow)
        with torch.no_grad():
            flow(torch.zeros(1, *tuple(input_shape), device=device))
    finally:
        for h in handles:
            h.remove()

    host = 0
    total = 0
    for key, (module, is_host) in units.items():
        in_shape, out_shape = shapes.get(key, (None, None))
        macs = _module_macs(module, in_shape, out_shape)
        total += macs
        if is_host:
            host += macs
    return host, total


def estimate_onchip_fraction(
    model,
    input_shape,
    num_classes,
    *,
    encoding_placement: str = "subsume",
    metric: str = "params",
) -> OnchipFractionEstimate:
    """Statically estimate the on-chip fraction of a native model under ``metric``.

    ``metric="params"`` reproduces ``count_host_params`` exactly; ``metric="macs"``
    reports the on-chip forward-compute fraction at the model's input shape.
    """
    _require_metric_and_placement("estimate_onchip_fraction", metric, encoding_placement)
    flow = _build_flow(model, input_shape, num_classes, encoding_placement)
    return _estimate_from_flow(flow, input_shape, encoding_placement, metric)


def estimate_onchip_fractions(
    model,
    input_shape,
    num_classes,
    *,
    encoding_placement: str = "subsume",
    metrics: Sequence[str] = ("params", "macs"),
) -> Tuple[OnchipFractionEstimate, ...]:
    """[H4] Every requested metric off ONE flow conversion.

    The conversion dominates the estimator's cost; the per-metric breakdowns
    are cheap walks over the same flow, so a caller wanting both metrics paid
    the conversion twice for nothing.
    """
    for metric in metrics:
        _require_metric_and_placement(
            "estimate_onchip_fractions", metric, encoding_placement)
    flow = _build_flow(model, input_shape, num_classes, encoding_placement)
    return tuple(
        _estimate_from_flow(flow, input_shape, encoding_placement, metric)
        for metric in metrics
    )


def _require_metric_and_placement(who: str, metric: str, encoding_placement: str) -> None:
    """Reject an unknown metric or placement before any model work happens."""
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"{who} metric must be one of {_VALID_METRICS!r}; got {metric!r}"
        )
    if encoding_placement not in _VALID_PLACEMENTS:
        raise ValueError(
            f"{who} encoding_placement must be one of {_VALID_PLACEMENTS!r}; "
            f"got {encoding_placement!r}"
        )


def _estimate_from_flow(flow, input_shape, encoding_placement, metric):
    if metric == "params":
        host, total = _params_breakdown(flow)
    elif metric == "ops":
        host, total = _ops_breakdown(flow)
    else:
        host, total = _macs_breakdown(flow, input_shape)
    return OnchipFractionEstimate(
        onchip=int(total) - int(host),
        host=int(host),
        total=int(total),
        metric=metric,
        placement=encoding_placement,
    )


def assert_onchip_majority_estimate_or_raise(
    model,
    input_shape,
    num_classes,
    *,
    encoding_placement: str = "subsume",
    metric: str = "params",
    min_fraction: float = 0.5,
) -> OnchipFractionEstimate:
    """Raise :class:`OnchipMajorityError` when the static on-chip fraction is below floor.

    The static analogue of ``assert_onchip_majority_or_raise`` for callers with a
    model spec but no mapped IR graph.
    """
    _require_metric_and_placement(
        "assert_onchip_majority_estimate_or_raise", metric, encoding_placement
    )
    flow = _build_flow(model, input_shape, num_classes, encoding_placement)
    est = _estimate_from_flow(flow, input_shape, encoding_placement, metric)
    if est.fraction < min_fraction:
        contributors = host_contributors_from_flow(flow, _host_unit)
        raise OnchipMajorityError(
            "Static on-chip parameter majority violated: only "
            f"{est.fraction:.2%} of the {est.total} {est.metric} are estimated "
            f"on chip (on-chip={est.onchip}, host={est.host}) under placement "
            f"{est.placement!r}, below the required {min_fraction:.0%} floor. "
            f"{describe_host_holders(contributors)}. "
            f"{_placement_remedy(contributors, est)} "
            f"{ONCHIP_FLOOR_ESCAPE}"
        )
    return est


def _placement_remedy(contributors, est: OnchipFractionEstimate) -> str:
    """What moving the encoding layer on chip would buy — in this model's numbers.

    ``subsume`` runs the encoding layer host-side; ``offload`` maps it on chip.
    Under ``offload`` the encoder is already on chip, so the host majority is
    something else and pointing at the knob would be a lie. Placement only; the
    refusal appends :data:`ONCHIP_FLOOR_ESCAPE` itself.
    """
    if est.placement != "subsume":
        return (
            "The encoding layer is already mapped on chip under 'offload', so "
            "this host majority is other host ops: shrink or replace them."
        )
    if est.metric != "params":
        return (
            "encoding_layer_placement='offload' maps the encoding layer on chip "
            "instead of running it host-side."
        )
    return onchip_placement_remedy(
        contributors,
        OnchipParamBreakdown(
            onchip_params=est.onchip, host_params=est.host, total_params=est.total
        ),
    )


def _collect_host_op_classes(flow):
    """Walk the flow once, returning ``(research_gap_ops, placement_fixable_ops)`` deduped by identity."""
    seen: set[int] = set()
    research_gap_ops: list[str] = []
    placement_fixable_ops: list[str] = []
    for node in _exec_nodes(flow):
        unit = _host_unit(node)
        if unit is None or id(unit) in seen:
            continue
        seen.add(id(unit))
        category, label = _classify_host_node(node)
        if label is None:
            continue
        if category == "unsupported_op":
            research_gap_ops.append(label)
        elif category == "placement":
            placement_fixable_ops.append(label)
    return research_gap_ops, placement_fixable_ops


def _tier_for(param_frac: float, mac_frac: float, floor: float, majority: float) -> str:
    worst = min(param_frac, mac_frac)
    if worst < floor:
        return TIER_INVALID
    if param_frac >= majority and mac_frac >= majority:
        return TIER_VALID
    return TIER_VALID_FLAGGED


def classify_validity(
    model,
    input_shape,
    num_classes,
    *,
    encoding_placement: str = "subsume",
    floor: float = _DEFAULT_FLOOR,
    majority: float = _DEFAULT_MAJORITY,
) -> ValidityVerdict:
    """Tiered deployment validity on both the param and MAC on-chip fractions.

    ``INVALID`` iff ``min(param_frac, mac_frac) < floor``; ``VALID`` iff both
    ``>= majority``; else ``VALID_FLAGGED``. One flow (which mutates ``model``) is reused.
    """
    if encoding_placement not in _VALID_PLACEMENTS:
        raise ValueError(
            f"classify_validity encoding_placement must be one of "
            f"{_VALID_PLACEMENTS!r}; got {encoding_placement!r}"
        )

    flow = _build_flow(model, input_shape, num_classes, encoding_placement)
    param_est = _estimate_from_flow(flow, input_shape, encoding_placement, "params")
    mac_est = _estimate_from_flow(flow, input_shape, encoding_placement, "macs")
    research_gap_ops, placement_fixable_ops = _collect_host_op_classes(flow)

    tier = _tier_for(param_est.fraction, mac_est.fraction, floor, majority)
    return ValidityVerdict(
        tier=tier,
        param_frac=param_est.fraction,
        mac_frac=mac_est.fraction,
        research_gap_ops=research_gap_ops,
        placement_fixable_ops=placement_fixable_ops,
    )


@dataclass(frozen=True)
class OnchipValidityReport:
    """Tiered validity verdict on a MAPPED IR graph: params from the graph,
    ops from the model's forward-MAC decomposition, one tier over both."""

    tier: str
    param_breakdown: OnchipParamBreakdown
    mac_estimate: OnchipFractionEstimate

    @property
    def param_frac(self) -> float:
        return self.param_breakdown.fraction

    @property
    def mac_frac(self) -> float:
        return self.mac_estimate.fraction

    @property
    def is_valid(self) -> bool:
        return self.tier != TIER_INVALID

    @property
    def is_flagged(self) -> bool:
        return self.tier == TIER_VALID_FLAGGED


def assert_onchip_validity_or_raise(
    ir_graph,
    model,
    input_shape,
    num_classes,
    *,
    encoding_placement: str = "subsume",
    floor: float = _DEFAULT_FLOOR,
    majority: float = _DEFAULT_MAJORITY,
) -> OnchipValidityReport:
    """Authoritative tiered gate on a fully-mapped IR graph, over BOTH metrics.

    The params fraction is the on-chip remainder of the mapped ``ir_graph``
    (``compute_onchip_fraction``); the ops fraction is the model's forward-MAC
    on-chip share (``estimate_onchip_fraction`` — robust across conv weight-bank
    reuse and attention, which the raw crossbar count cannot reproduce). A run is
    INVALID iff EITHER fraction is below ``floor``; the error names which metric
    and both actual fractions. Between ``floor`` and ``majority`` the run is
    VALID_FLAGGED (recorded, non-fatal).
    """
    total_params = int(sum(p.numel() for p in model.parameters()))
    param_breakdown = compute_onchip_fraction(ir_graph, total_params=total_params)
    mac_estimate = estimate_onchip_fraction(
        model,
        input_shape,
        num_classes,
        encoding_placement=encoding_placement,
        metric="macs",
    )
    param_frac = param_breakdown.fraction
    mac_frac = mac_estimate.fraction
    tier = _tier_for(param_frac, mac_frac, floor, majority)
    if tier == TIER_INVALID:
        below = []
        if param_frac < floor:
            below.append(f"params ({param_frac:.2%})")
        if mac_frac < floor:
            below.append(f"ops ({mac_frac:.2%})")
        contributors = host_contributors_from_ir(ir_graph)
        raise OnchipMajorityError(
            "On-chip validity gate: "
            f"{' and '.join(below)} below the required {floor:.0%} floor "
            f"(on-chip params {param_frac:.2%}, ops {mac_frac:.2%}). The host "
            "holds the majority of that metric, so this deployment cannot "
            "accelerate a significant fraction of the network — an erroneous "
            f"on-chip deployment. {describe_host_holders(contributors)}. "
            f"{onchip_placement_remedy(contributors, param_breakdown)} "
            f"{ONCHIP_FLOOR_ESCAPE}"
        )
    return OnchipValidityReport(
        tier=tier,
        param_breakdown=param_breakdown,
        mac_estimate=mac_estimate,
    )
