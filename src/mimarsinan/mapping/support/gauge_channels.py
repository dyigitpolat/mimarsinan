"""The channel axis a gauge vector runs along, and a node's channel count on it (U-SC2, D-USC-1).

The scale walk (``compute_per_source_scales``) carries one gauge per CHANNEL of a node's output: a perceptron emits its
theta per output channel along its declared ``output_channel_axis``, the input root its unit gauge along its leading
axis, and every other node keeps the channel axis of its first source (a host op computes per channel along it, a layout
node moves whole channels). A host op that changes the channel count (a convolution, a channel pad) must therefore carry
its gauge over its OWN output channels, and a concatenation along the channel axis joins its parts' aligned vectors, or
the vector a reader decodes is misaligned with the channels it reads (DenseNet-BC's ``cat_1`` held 19 entries for 12
channels and its first reader decoded every channel at their mean).

Duck-typed on the mapper attributes (``perceptron``, ``get_source_mappers``, ``output_shape``, ``input_shapes``,
``dim``/``rank``), so the support layer imports no mapper class.
"""

from __future__ import annotations

import torch

from mimarsinan.mapping.support.scale_broadcast import spread_scalar

__all__ = [
    "LAST",
    "LEADING",
    "align_gauge",
    "concat_joins_channels",
    "gauge_axis",
    "gauge_channels",
    "gauge_view",
    "same_gauge",
    "wire_gauge",
    "wrap_channel_axis",
]

#: Per-sample axes a gauge vector runs along: the leading axis (a channel-first layout, ``(C, H, W)``, or a flat
#: ``(F,)``) or the last (a channels-last layout, ``(N, F)``).
LEADING = 0
LAST = -1


def _sources(node) -> list:
    get = getattr(node, "get_source_mappers", None)
    return [s for s in get() if s is not None] if get is not None else []


def gauge_axis(node) -> int | None:
    """The per-sample axis ``node``'s gauge vector runs along: a perceptron's declared output channel axis (1, the conv
    mappers', is the leading per-sample axis), the input root's leading axis, and through any other node its first
    source's. ``None`` past a node that moves axes (``moves_gauge_axis``: a permute or a rearrangement)."""
    seen: set[int] = set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        perceptron = getattr(node, "perceptron", None)
        if perceptron is not None and hasattr(perceptron, "output_channels"):
            return LEADING if int(getattr(perceptron, "output_channel_axis", -1)) == 1 else LAST
        if getattr(node, "moves_gauge_axis", False):
            return None
        sources = _sources(node)
        if not sources:
            return LEADING
        node = sources[0]
    return None


def _per_sample_shape(node):
    shape = getattr(node, "output_shape", None)
    return tuple(int(d) for d in shape) if shape else None


def gauge_channels(node) -> int | None:
    """The channel count of ``node``'s output on its gauge axis, where it is known: a perceptron's output channels, and a
    host op's recorded output size on the axis unless its recorded first input has another rank (a reshape or a
    reduction that drops the axis leaves the vector as its source carried it; an op recorded without input shapes is a
    per-instance module, which maps the last axis and keeps the rank). ``None`` otherwise."""
    perceptron = getattr(node, "perceptron", None)
    if perceptron is not None and hasattr(perceptron, "output_channels"):
        return int(perceptron.output_channels)
    out = _per_sample_shape(node)
    shapes = getattr(node, "input_shapes", None)
    first = shapes[0] if shapes else None
    if out is None or (first is not None and len(out) != len(first)):
        return None
    axis = gauge_axis(node)
    return None if axis is None else int(out[axis])


def align_gauge(vector: torch.Tensor, n: int | None) -> torch.Tensor:
    """``vector`` as one gauge per channel of an ``n``-channel output: unchanged when it already is (or ``n`` is
    unknown); a uniform gauge re-spread over the ``n`` channels, element for element the same value; any other vector
    (a per-channel gauge on a channel-changing node, which carries no channel correspondence) at its mean."""
    if n is None:
        return vector
    flat = vector.reshape(-1)
    if flat.numel() == n:
        return vector
    if bool(torch.all(flat == flat[0])):
        return flat[:1].expand(n).clone()
    return spread_scalar(flat.mean(), n)


def concat_joins_channels(node, axis: int | None) -> bool:
    """Whether a concatenation (``dim``/``rank`` over the batched tensor) joins along the gauge ``axis``."""
    if axis is None:
        return False
    dim, rank = int(getattr(node, "dim", 1)), getattr(node, "rank", None)
    if axis == LEADING:
        return dim == 1
    if rank is None:
        return dim == -1
    return dim == -1 or dim == int(rank) - 1


def gauge_view(vector, like: torch.Tensor, axis: int | None) -> torch.Tensor | None:
    """A per-channel gauge ``vector`` broadcastable against a batched value ``like`` (U-SC2: the per-element scale a
    walk re-encodes a wire value at, the one the deployed core folds): along axis 1 of a channel-first value on the
    leading gauge axis, along the last axis otherwise, repeated over the leading groups of a flattened channel-first
    value as the fold repeats it; [U-TGAUGE3] a gauge per element of a sample (a layout's, placed by its elements'
    sources) over the sample's elements. ``None`` for a scalar (uniform) gauge, which every reader takes from the
    boundary table, or a vector the value's layout does not hold."""
    if vector is None:
        return None
    flat = torch.as_tensor(vector).detach().reshape(-1)
    if flat.numel() <= 1 or bool(torch.all(flat == flat[0])):
        return None
    flat = flat.to(device=like.device, dtype=like.dtype)
    dim = like.dim()
    ax = 1 if (axis == LEADING and dim >= 3) else dim - 1
    n = int(like.shape[ax])
    if n == flat.numel():
        shape = [1] * dim
        shape[ax] = n
        return flat.view(*shape)
    if dim >= 2 and flat.numel() == like[0].numel():
        return flat.view(1, *like.shape[1:])
    if ax == dim - 1 and n % flat.numel() == 0:
        return flat.repeat_interleave(n // flat.numel())
    return None


def wire_gauge(node, vector, scalar, like: torch.Tensor):
    """[U-TGAUGE4] The gauge ``node``'s wire value ``like`` holds each element at: the source walk's placed ``vector``
    per element where it is no one value (``gauge_view``); its one value where ``node`` is a layout whose elements all
    sit at it while ``scalar`` (the boundary table's, a layout's own propagation: the mean of its sources) is another
    (a gather that holds one producer of a join at differing gauges; ``same_gauge``); else ``scalar``."""
    view = gauge_view(vector, like, gauge_axis(node))
    if view is not None or vector is None or not getattr(node, "moves_elements_only", False):
        return scalar if view is None else view
    flat = torch.as_tensor(vector).detach().reshape(-1)
    if flat.numel() and not same_gauge(float(flat[0]), float(scalar)):
        return float(flat[0])
    return scalar


#: A gauge read through two float32 tables (the boundary walk's float64 mean written into a float32 parameter, the
#: source walk's float32 combination) differs by a few float32 units in the last place; more is another producer's.
GAUGE_READINGS_ULPS = 4


def same_gauge(a: float, b: float) -> bool:
    """[U-TGAUGE4] Whether two readings of a gauge are one gauge (within ``GAUGE_READINGS_ULPS`` float32 units)."""
    return abs(float(a) - float(b)) <= GAUGE_READINGS_ULPS * float(torch.finfo(torch.float32).eps) * max(
        abs(float(a)), abs(float(b)))


def wrap_channel_axis(node, slots) -> int | None:
    """The batched axis a host op's ScaleNormalizingWrapper decodes a per-channel slot along: 1 for a channel-first input
    (per-sample rank 2 or more on the leading gauge axis) when some slot is per-channel; ``None`` (the last axis, as a
    flat or channels-last input already is) otherwise."""
    shapes = getattr(node, "input_shapes", None)
    first = shapes[0] if shapes else None
    if first is None or len(first) < 2 or gauge_axis(node) != LEADING:
        return None
    for slot in slots:
        flat = torch.as_tensor(slot).reshape(-1)
        if flat.numel() > 1 and not bool(torch.all(flat == flat[0])):
            return 1
    return None
