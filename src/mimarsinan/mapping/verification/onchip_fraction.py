"""Static on-chip-fraction pre-check: reproduce the validity gate WITHOUT a run.

The authoritative gate (:mod:`onchip_majority`) needs a fully MAPPED ``IRGraph``
and a pipeline run. This estimator FX-traces / segments a native model exactly as
``convert_torch_model`` would, classifies which mapper units are HOST-side (the
segment-start encoder perceptrons under ``subsume``, plus always-host ComputeOps
like the classifier readout, attention, LayerNorm, pooling), and sums host vs.
total. The PARAMS metric reproduces ``count_host_params`` exactly; the MACS metric
reports the on-chip compute fraction (attention is param-light but MAC-heavy), the
fork the transformer validity decision needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.conv1d_mapper import Conv1DPerceptronMapper
from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.verification.onchip_majority import OnchipMajorityError

_PERCEPTRON_MAPPER_TYPES = (
    PerceptronMapper,
    Conv2DPerceptronMapper,
    Conv1DPerceptronMapper,
)

_VALID_METRICS = ("params", "macs")
_VALID_PLACEMENTS = ("subsume", "offload")

_DEFAULT_FLOOR = 0.20
_DEFAULT_MAJORITY = 0.50

TIER_VALID = "VALID"
TIER_VALID_FLAGGED = "VALID_FLAGGED"
TIER_INVALID = "INVALID"

# Host op types with NO on-chip SNN mapping yet — the research frontier.
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
    """Tiered deployment-validity verdict on BOTH the param and MAC on-chip fractions.

    ``tier`` is one of ``VALID`` / ``VALID_FLAGGED`` / ``INVALID``. ``research_gap_ops``
    names the host ops with no on-chip SNN mapping yet (the future-research frontier);
    ``placement_fixable_ops`` names supported Linear/Conv host-placed under ``subsume``
    (fixable by switching to ``offload``).
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

    Mirrors ``count_host_params``: a ``ComputeOpMapper`` contributes its wrapped
    module; a segment-start (``is_encoding_layer``) perceptron is emitted as a host
    ComputeOp whose module is the perceptron itself. Everything else is on-chip or
    structural (no host parameters / MACs).
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


def _unwrap_module(module):
    """Peel a ``ScaleNormalizingWrapper`` to reach the wrapped op module."""
    while _is_scale_wrapper(module):
        module = module.module
    return module


def _perceptron_op_label(perceptron) -> str:
    layer = getattr(perceptron, "layer", None)
    return type(layer).__name__ if layer is not None else "Perceptron"


def _is_unsupported_host_op(module) -> bool:
    return isinstance(_unwrap_module(module), _UNSUPPORTED_HOST_OP_TYPES)


def _host_op_label(module) -> str:
    return type(_unwrap_module(module)).__name__


def _classify_host_node(node):
    """Classify a host-side node into a (category, op_label) pair.

    ``placement`` — a supported encoding perceptron (Linear/Conv) host-placed under
    ``subsume``; switching to ``offload`` maps it on-chip. ``unsupported_op`` — a host
    ComputeOp wrapping a module with no on-chip SNN mapping yet (MultiheadAttention,
    LayerNorm, GELU): the research frontier. ``supported_host`` — an always-host but
    supported op (the classifier readout Linear, pooling, structural adapters); neither
    a placement fix nor a research gap. Returns ``(None, None)`` for non-host nodes.
    """
    if _is_encoding_perceptron(node):
        return "placement", _perceptron_op_label(node.perceptron)
    if isinstance(node, ComputeOpMapper):
        if _is_unsupported_host_op(node.module):
            return "unsupported_op", _host_op_label(node.module)
        return "supported_host", _host_op_label(node.module)
    return None, None


def _build_flow(model, input_shape, num_classes, placement):
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


def _params_breakdown(flow):
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


def _linear_macs(in_features: int, out_features: int, n_positions: int) -> int:
    return int(in_features) * int(out_features) * int(n_positions)


def _is_scale_wrapper(module) -> bool:
    return type(module).__name__ == "ScaleNormalizingWrapper" and hasattr(
        module, "module"
    )


def _module_macs(module: nn.Module, in_shape, out_shape) -> int:
    """Estimate forward MACs for a host/on-chip unit from its I/O tensor shapes.

    ``in_shape`` / ``out_shape`` are full tensor shapes (incl. batch). Handles the
    op families that survive conversion: Linear, Conv1d/2d, MultiheadAttention,
    and the linear ``layer`` inside a ``Perceptron``. Pure element-wise / shape ops
    (activations, norms, pooling, reshape) carry no MACs in this budget.
    """
    if _is_scale_wrapper(module):
        return _module_macs(module.module, in_shape, out_shape)

    layer = getattr(module, "layer", None)
    if isinstance(layer, nn.Linear):
        n_positions = _num_positions(in_shape, layer.in_features)
        return _linear_macs(layer.in_features, layer.out_features, n_positions)

    if isinstance(module, nn.Linear):
        n_positions = _num_positions(in_shape, module.in_features)
        return _linear_macs(module.in_features, module.out_features, n_positions)

    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        return _conv_macs(module, out_shape)

    if isinstance(module, nn.MultiheadAttention):
        return _attention_macs(module, in_shape)

    return 0


def _num_positions(in_shape, in_features) -> int:
    if in_shape is None:
        return 1
    numel = 1
    for d in in_shape:
        numel *= int(d)
    if in_features and in_features > 0 and numel % in_features == 0:
        return max(1, numel // int(in_features))
    return 1


def _conv_macs(module, out_shape) -> int:
    if out_shape is None:
        return 0
    out_spatial = 1
    for d in out_shape[2:]:
        out_spatial *= int(d)
    k = 1
    for d in module.kernel_size:
        k *= int(d)
    in_per_group = int(module.in_channels) // int(module.groups)
    return int(module.out_channels) * out_spatial * in_per_group * k


def _attention_macs(module: nn.MultiheadAttention, in_shape) -> int:
    """QKV + output projections (3·E·E + E·E per position) plus the
    scores·values double matmul (2·L·E per position) for self-attention."""
    if in_shape is None:
        return 0
    e = int(module.embed_dim)
    numel = 1
    for d in in_shape:
        numel *= int(d)
    seq = max(1, numel // e) if e > 0 else 1
    proj_macs = 4 * e * e * seq
    score_macs = 2 * seq * seq * e
    return int(proj_macs + score_macs)


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
        with torch.no_grad():
            flow(torch.zeros(1, *tuple(input_shape)))
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

    Pure-static: FX-traces and segments via ``convert_torch_model`` (no training,
    no physical core placement, no ``generated/`` run), then classifies host vs.
    on-chip. ``metric="params"`` reproduces the authoritative ``count_host_params``
    decomposition exactly; ``metric="macs"`` reports the on-chip forward-compute
    fraction at the model's input shape. ``encoding_placement`` is the converter's
    ``encoding_layer_placement`` (``subsume`` offloads segment-start encoders to the
    host; ``offload`` maps them on-chip).
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
    """Raise :class:`OnchipMajorityError` when the STATIC on-chip fraction is below floor.

    The static analogue of ``assert_onchip_majority_or_raise`` for callers (the
    campaign scheduler) that hold a model spec but no mapped IR graph.
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
    """Walk the flow once, returning ``(research_gap_ops, placement_fixable_ops)``.

    ``research_gap_ops`` are host ops with no on-chip SNN mapping (the frontier);
    ``placement_fixable_ops`` are supported encoders host-placed under ``subsume``
    (offloadable). Always-host supported ops (readout, pooling, adapters) are in
    neither list. Each host unit is classified once (deduped by identity).
    """
    seen: set[int] = set()
    research_gap_ops: list[str] = []
    placement_fixable_ops: list[str] = []
    for node in _exec_nodes(flow):
        unit = _host_unit(node)
        if unit is None or id(unit) in seen:
            continue
        seen.add(id(unit))
        category, label = _classify_host_node(node)
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
    """Tiered deployment validity on BOTH the param and MAC on-chip fractions.

    ``INVALID`` iff ``min(param_frac, mac_frac) < floor`` (the host does ~everything);
    ``VALID`` iff both fractions ``>= majority``; ``VALID_FLAGGED`` otherwise (deploys
    and counts as transferable-tuning evidence while carrying a research gap). The
    host ops keeping a flagged model below majority are classified into
    ``research_gap_ops`` (no on-chip mapping yet) vs ``placement_fixable_ops``
    (supported encoders, fixable via ``offload``). One flow is built and reused for
    both metrics and the op classification (conversion mutates ``model`` in place).
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
