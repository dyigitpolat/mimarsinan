"""Exact cross-layer per-channel scale migration through ReLU-family hops."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.leading_dim import (
    Ensure2DMapper, MergeLeadingDimsMapper, SplitLeadingDimMapper,
)
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import PermuteMapper
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.transformations.pruning.committed_masks import commit_perceptron_pruning

# Unclipped migration amplifies near-dead channels' rows/columns by 1/s_c and a
# single outlier starves NAPQ's shared symmetric grid (measured: 5-bit WQ
# 0.9513 -> 0.0892 unclipped; r=4 held 0.9493). The clip is load-bearing; see
# docs/research/findings/mixer_column_scale_pathology.md section 3.2(7).
DEFAULT_CLIP_RATIO = 4.0

# Positively homogeneous bases: sigma(lam*x) = lam*sigma(x) for lam > 0
# (LeakyGradReLU's forward IS max(0, x); only its gradient leaks). GELU fails.
_HOMOGENEOUS_ACTIVATIONS = frozenset({"ReLU", "LeakyReLU"})

# Every live scale within this band of 1.0: skip the writes entirely so a
# homogeneous vehicle stays byte-identical through the pass.
_NOOP_TOLERANCE = 1e-3

# Structural mappers that carry a channels-last axis through unchanged.
_LAST_AXIS_PASS_THROUGH = (Ensure2DMapper, MergeLeadingDimsMapper, SplitLeadingDimMapper)


@dataclass(frozen=True)
class MigratablePair:
    """One producer whose channel axis is consumed only on feature axes."""

    producer: Any
    consumer_perceptrons: tuple = ()
    consumer_modules: tuple = ()


@dataclass(frozen=True)
class MigratedHop:
    name: str
    s_min: float
    s_max: float


@dataclass(frozen=True)
class ScaleMigrationReport:
    clip_ratio: float
    migrated: tuple = ()
    skipped: tuple = ()

    def as_dict(self) -> dict:
        return {"clip_ratio": self.clip_ratio, "skipped": list(self.skipped),
                "migrated": [asdict(hop) for hop in self.migrated]}


def _producer_scalable(perceptron) -> bool:
    """Channels-last linear hop whose activation is positively homogeneous and
    whose per-channel output scale can be realized exactly (layer rows, or the
    norm affine when a normalization is attached)."""
    if (
        getattr(perceptron, "base_activation_name", None) not in _HOMOGENEOUS_ACTIVATIONS
        or not isinstance(perceptron.layer, nn.Linear)
        or int(getattr(perceptron, "output_channel_axis", 1)) != -1
        or not isinstance(perceptron.scaler, nn.Identity)
    ):
        return False
    norm = perceptron.normalization
    if isinstance(norm, nn.Identity):
        return True
    weight, bias = getattr(norm, "weight", None), getattr(norm, "bias", None)
    out_channels = int(perceptron.layer.weight.shape[0])
    return (
        torch.is_tensor(weight) and int(weight.numel()) == out_channels
        and torch.is_tensor(bias) and int(bias.numel()) == out_channels
    )


def _consumer_scalable(perceptron) -> bool:
    """Nothing may sit between the consumer's input and its weight columns."""
    return (
        isinstance(perceptron.layer, nn.Linear)
        and isinstance(perceptron.input_activation, nn.Identity)
        and getattr(perceptron, "per_input_scales", None) is None
    )


def _permute_step(node: PermuteMapper, k: int):
    dims = tuple(node.dims)
    rank = len(dims)
    pos_in = rank - int(k)
    if pos_in <= 0 or pos_in >= rank or pos_in not in dims:
        return None
    pos_out = dims.index(pos_in)
    return None if pos_out == 0 else ("pass", rank - pos_out)


def _mean_reduced_dims(adapter: ComputeAdapter):
    fallback = adapter.extra_args[0] if len(adapter.extra_args) == 1 else None
    raw = adapter.kwargs.get("dim", fallback)
    if isinstance(raw, int):
        return (raw,)
    if isinstance(raw, (tuple, list)) and all(isinstance(d, int) for d in raw):
        return tuple(raw)
    return None


def _mean_passthrough(node: ComputeOpMapper, k: int):
    """New from-end channel position through a mean over non-channel axes, or None."""
    adapter = node.module
    if not isinstance(adapter, ComputeAdapter) or adapter.fn is not torch.mean:
        return None
    if getattr(adapter, "_bound_count", 0) != 0 or len(node.get_source_mappers()) != 1:
        return None
    dims = _mean_reduced_dims(adapter)
    shapes = node.input_shapes
    if dims is None or shapes is None or shapes[0] is None:
        return None
    rank = len(shapes[0]) + 1  # input_shapes are batch-stripped
    channel_pos = rank - int(k)
    reduced = {d % rank for d in dims}
    if 0 in reduced or channel_pos in reduced:
        return None
    if bool(adapter.kwargs.get("keepdim", False)):
        return int(k)
    return int(k) - sum(1 for d in reduced if d > channel_pos)


def _step_through(node, k: int):
    """One walk transition: ('pass', new_k) | ('perceptron', p) | ('module', m) | None."""
    if isinstance(node, PerceptronMapper):
        perceptron = node.perceptron
        if k == 1 and _consumer_scalable(perceptron):
            return ("perceptron", perceptron)
        return None
    if isinstance(node, _LAST_AXIS_PASS_THROUGH):
        return ("pass", 1) if k == 1 else None
    if isinstance(node, PermuteMapper):
        return _permute_step(node, k)
    if isinstance(node, ComputeOpMapper):
        passed = _mean_passthrough(node, k)
        if passed is not None:
            return ("pass", passed)
        module = node.module
        if isinstance(module, nn.Linear) and k == 1 and len(node.get_source_mappers()) == 1:
            return ("module", module)
        return None
    return None


def _column_scale_targets(producer_node, consumers: dict):
    """All consumers of the producer's channel axis, or None when any path is
    not exactly column-scalable (fan-out closure: one bad path voids the pair).
    ``k`` is the channel position counted from the tensor's end (1 = last)."""
    frontier: list = [(producer_node, 1)]
    visited: set = set()
    perceptron_targets: dict = {}
    module_targets: dict = {}
    while frontier:
        node, k = frontier.pop()
        downstream = consumers.get(id(node), [])
        if not downstream:
            return None  # the channel axis reaches the model output unscaled
        for consumer in downstream:
            key = (id(consumer), k)
            if key in visited:
                continue
            visited.add(key)
            step = _step_through(consumer, k)
            if step is None:
                return None
            kind, value = step
            if kind == "pass":
                frontier.append((consumer, value))
            elif kind == "perceptron":
                perceptron_targets[id(value)] = value
            else:
                module_targets[id(value)] = value
    return tuple(perceptron_targets.values()), tuple(module_targets.values())


def find_migratable_pairs(mapper_repr) -> list[MigratablePair]:
    """Walk the mapper DAG for producers whose channel axis is consumed ONLY as
    linear feature axes through channelwise positively-homogeneous ops.

    Weight-shared axes (the token-mixer fc2 patch axis) fail the axis tracking
    and are honestly left alone — their exact escape is a per-channel theta on
    the scale-propagation layer, owned elsewhere."""
    consumers = mapper_repr.consumer_map()
    pairs: list[MigratablePair] = []
    for node in mapper_repr.execution_order():
        if not isinstance(node, PerceptronMapper) or not _producer_scalable(node.perceptron):
            continue
        targets = _column_scale_targets(node, consumers)
        if targets is None:
            continue
        perceptron_targets, module_targets = targets
        if perceptron_targets or module_targets:
            pairs.append(
                MigratablePair(node.perceptron, perceptron_targets, module_targets)
            )
    return pairs


def migration_scales(per_channel_q99, clip_ratio: float = DEFAULT_CLIP_RATIO):
    """s_c = q99_c / geomean(live q99), clipped to [1/r, r]; dead channels get
    s_c = 1. Returns None when no channel is live."""
    ratio = float(clip_ratio)
    if ratio < 1.0:
        raise ValueError(f"scale-migration clip ratio must be >= 1.0, got {ratio}")
    q99 = torch.as_tensor(list(per_channel_q99), dtype=torch.float64)
    if q99.numel() == 0 or bool((q99 < 0).any()):
        raise ValueError("per-channel q99 stats must be a non-empty, non-negative vector")
    live = q99 > 0
    if not bool(live.any()):
        return None
    geomean = torch.exp(torch.log(q99[live]).mean())
    scales = torch.ones_like(q99)
    scales[live] = (q99[live] / geomean).clamp(1.0 / ratio, ratio)
    return scales


def _require_consumer_width(name: str, weight: torch.Tensor, out_channels: int):
    if int(weight.shape[1]) != out_channels:
        raise ValueError(
            f"scale migration {name!r}: consumer input features {int(weight.shape[1])} "
            f"do not match the producer's {out_channels} channels"
        )


@torch.no_grad()
def apply_scale_migration(pair: MigratablePair, scales: torch.Tensor) -> bool:
    """W_A <- S^-1 W_A, b_A <- S^-1 b_A (via the norm affine when attached),
    W_B <- W_B S. Returns False (no writes) when every scale is ~1."""
    producer = pair.producer
    name = getattr(producer, "name", "<unnamed>")
    out_channels = int(producer.layer.weight.shape[0])
    if int(scales.numel()) != out_channels:
        raise ValueError(
            f"scale migration for {name!r}: {int(scales.numel())} channel scales for "
            f"{out_channels} output channels — the stats were captured on the wrong axis"
        )
    for consumer in pair.consumer_perceptrons:
        _require_consumer_width(name, consumer.layer.weight, out_channels)
    for module in pair.consumer_modules:
        _require_consumer_width(name, module.weight, out_channels)
    if bool(((scales - 1.0).abs() <= _NOOP_TOLERANCE).all()):
        return False

    weight = producer.layer.weight
    s = scales.to(device=weight.device, dtype=weight.dtype)
    if isinstance(producer.normalization, nn.Identity):
        weight.div_(s.view(-1, 1))
        if producer.layer.bias is not None:
            producer.layer.bias.div_(s)
    else:
        # Exact in both BN modes: the pre-norm batch statistics are untouched,
        # so S^-1 rides the affine (gamma <- gamma/s, beta <- beta/s).
        producer.normalization.weight.div_(s)
        producer.normalization.bias.div_(s)
    commit_perceptron_pruning(producer)

    for consumer in pair.consumer_perceptrons:
        consumer.layer.weight.mul_(s.view(1, -1))
        commit_perceptron_pruning(consumer)
    for module in pair.consumer_modules:
        module.weight.mul_(
            s.view(1, -1).to(device=module.weight.device, dtype=module.weight.dtype)
        )
    return True


def equalize_channel_scales(
    model, per_channel_q99_by_id: dict, *, clip_ratio: float = DEFAULT_CLIP_RATIO,
) -> ScaleMigrationReport:
    """Migrate every exactly-migratable pair of ``model``'s mapper DAG using the
    supplied producer-channel q99 stats (keyed by ``id(perceptron)``)."""
    migrated: list[MigratedHop] = []
    skipped: list[str] = []
    for pair in find_migratable_pairs(model.get_mapper_repr()):
        name = str(getattr(pair.producer, "name", "<unnamed>"))
        q99 = per_channel_q99_by_id.get(id(pair.producer))
        if q99 is None:
            raise ValueError(
                f"scale migration: no channel stats captured for migratable producer {name!r}"
            )
        scales = migration_scales(q99, clip_ratio)
        if scales is None or not apply_scale_migration(pair, scales):
            skipped.append(name)
            continue
        migrated.append(MigratedHop(name, float(scales.min()), float(scales.max())))
    return ScaleMigrationReport(float(clip_ratio), tuple(migrated), tuple(skipped))
