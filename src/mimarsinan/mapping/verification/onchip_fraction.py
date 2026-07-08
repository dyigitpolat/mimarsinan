"""Static on-chip-fraction pre-check: reproduce the on-chip validity gate from a native model without a pipeline run."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.conv1d_mapper import Conv1DPerceptronMapper
from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.verification.onchip_majority import (
    DEFAULT_ONCHIP_FLOOR,
    DEFAULT_ONCHIP_MAJORITY,
    OnchipMajorityError,
    OnchipParamBreakdown,
    compute_onchip_fraction,
    module_macs as _module_macs,
    unwrap_scale_wrapper as _unwrap_module,
)

_PERCEPTRON_MAPPER_TYPES = (
    PerceptronMapper,
    Conv2DPerceptronMapper,
    Conv1DPerceptronMapper,
)

_VALID_METRICS = ("params", "macs")
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
    if hasattr(model, "get_mapper_repr"):
        # Already a perceptron flow: its encoding placement is baked at build;
        # FX re-tracing is neither possible for einops-based flows nor needed.
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
    for p in flow.parameters():
        return p.device
    return torch.device("cpu")


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
        # caller holds a CPU model spec (scheduler) or a GPU model mid-pipeline.
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
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"estimate_onchip_fraction metric must be one of {_VALID_METRICS!r}; "
            f"got {metric!r}"
        )
    if encoding_placement not in _VALID_PLACEMENTS:
        raise ValueError(
            f"estimate_onchip_fraction encoding_placement must be one of "
            f"{_VALID_PLACEMENTS!r}; got {encoding_placement!r}"
        )

    flow = _build_flow(model, input_shape, num_classes, encoding_placement)
    return _estimate_from_flow(flow, input_shape, encoding_placement, metric)


def _estimate_from_flow(flow, input_shape, encoding_placement, metric):
    if metric == "params":
        host, total = _params_breakdown(flow)
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
    est = estimate_onchip_fraction(
        model,
        input_shape,
        num_classes,
        encoding_placement=encoding_placement,
        metric=metric,
    )
    if est.fraction < min_fraction:
        raise OnchipMajorityError(
            "Static on-chip parameter majority violated: only "
            f"{est.fraction:.2%} of the {est.total} {est.metric} are estimated "
            f"on chip (on-chip={est.onchip}, host={est.host}) under placement "
            f"{est.placement!r}, below the required {min_fraction:.0%} floor. The "
            "host-side ComputeOps (offloaded encoding Linear/Conv, classifier "
            "readout, attention) hold the majority, so this mapping would not be "
            "a genuine on-chip deployment."
        )
    return est


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
        raise OnchipMajorityError(
            "On-chip validity gate: "
            f"{' and '.join(below)} below the required {floor:.0%} floor "
            f"(on-chip params {param_frac:.2%}, ops {mac_frac:.2%}). The host "
            "holds the majority of that metric, so this deployment cannot "
            "accelerate a significant fraction of the network — an erroneous "
            "on-chip deployment. Set deployment_parameters.onchip_majority_gate="
            "false to deploy an intentionally host-offloaded network, or lower "
            "onchip_min_fraction."
        )
    return OnchipValidityReport(
        tier=tier,
        param_breakdown=param_breakdown,
        mac_estimate=mac_estimate,
    )
