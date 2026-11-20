"""[U-TGAUGE3] The placement rule: each element of a value at its producer's gauge, read at its source index.

A twin holds every element of a value at the gauge of the producer that emitted it: a core at its threshold (per neuron
where it carries a vector), a host op at its emitted gauge, the raw input at one. A layout node (a gather, a
concatenation, a flatten: ``moves_elements_only``) only moves elements, so its gauge per element is no property of the
layout; it is each element's source's. The table that holds every element's source is the IR's: its producer node and
its output index (``IRSource``), as ``map_to_ir`` builds it and as the deployed program holds its outputs
(``output_sources``). ``place_gauges`` is the one placement every reader of a layout applies: every element at its
producer's held gauge read at its source index (the raw input, an off and an always-on source at one). The deployed
readers read the program's table (``output_decode.program_output_gauge``); the walks and the source walk read the
identity program's (``map_model_to_ir.mapper_source_table``), the mapping the program is built from.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator, Optional

import numpy as np
import torch

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.support.gauge_channels import LEADING, gauge_channels
from mimarsinan.mapping.support.scale_broadcast import spread_scalar
from mimarsinan.transformations.perceptron.position_fold import SLOT_SOURCE, PositionFoldError, reads_channels_first

__all__ = ["GaugePlacementError", "channel_form", "fold_per_input", "holds_channels", "place_gauges",
           "reshape_gauge", "sample_shapes", "source_index"]

#: [U-PFOLD] The stamping walk's reader of a node's per-sample IR source shape (``sample_shapes``), else None.
_SAMPLE_SHAPE: "ContextVar[Optional[Callable[[Any], tuple]]]" = ContextVar("fold_sample_shape", default=None)


class GaugePlacementError(ValueError):
    """A layout element whose producer's gauge cannot be read at its source index (a per-channel gauge its producer
    does not hold along its own channel axis)."""


def source_index(sources: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(node ids, output indices)`` of an IR source array in C order (the order a value holds a sample in); an entry
    that is no ``IRSource`` reads as an off source."""
    flat = np.asarray(sources, dtype=object).reshape(-1)
    ids = np.fromiter((int(s.node_id) if isinstance(s, IRSource) else -1 for s in flat), dtype=np.int64,
                      count=flat.size)
    idx = np.fromiter((int(s.index) if isinstance(s, IRSource) else 0 for s in flat), dtype=np.int64,
                      count=flat.size)
    return ids, idx


def place_gauges(ids: np.ndarray, idx: np.ndarray, held: Callable[[int], Any], *,
                 dtype: Any = np.float64) -> np.ndarray:
    """The gauge of every element whose source is node ``ids[e]``'s output ``idx[e]``: ``held(node_id)`` (a scalar, or
    one gauge per output index of that node) read at the element's index; a negative id (the raw input, an off or an
    always-on source) at one."""
    gauge = np.ones(ids.size, dtype=dtype)
    for nid in np.unique(ids).tolist():
        if nid < 0:
            continue
        at = ids == nid
        held_gauge = np.asarray(held(int(nid)), dtype=dtype).reshape(-1)
        gauge[at] = held_gauge[0] if held_gauge.size == 1 else held_gauge[idx[at]]
    return gauge


@contextmanager
def sample_shapes(read: Callable[[Any], tuple]) -> Iterator[None]:
    """[U-PFOLD] Within a stamping walk, ``read(node)`` is ``node``'s per-sample IR source shape (the identity program's
    table), from which a convolution's fold per position names each element its patches read."""
    token = _SAMPLE_SHAPE.set(read)
    try:
        yield
    finally:
        _SAMPLE_SHAPE.reset(token)


def fold_per_input(perceptron: Any, anchored: torch.Tensor, in_features: int, reader: Any = None,
                   source: Any = None) -> torch.Tensor:
    """[U-TGAUGE4, U-PFOLD] A source vector that is no one gauge per input of ``perceptron`` (one per element of a
    larger sample, a layout's placed gauge), read in the reader's own input order: a convolution's (``reader`` declares
    ``in_channels``; its perceptron reads channels first) along its leading channel axis, a linear's along its last.
    One value where it is one; one gauge per input where every element of an input channel shares it (a
    convolution's spread over its kernel slots); else one row per core position (``position_fold``): a linear's
    ``(positions, in_features)``, a convolution's ``(output positions, in_features)``, each row its patch's gauges
    (a padding slot at one). A vector that is no whole number of rows of the reader's input refuses by name."""
    flat = anchored.reshape(-1)
    if _every(flat == flat[0]):
        return spread_scalar(flat.mean(), in_features)
    channels = getattr(reader, "in_channels", None) if reads_channels_first(perceptron) else None
    if channels is not None and flat.numel() % int(channels) == 0:
        grid = flat.reshape(int(channels), -1)
        if _every(grid == grid[:, :1]):
            return grid[:, 0].repeat_interleave(int(in_features) // int(channels))
        return _patch_rows(perceptron, flat, reader, source)
    if flat.numel() % in_features == 0:
        leading = perceptron.layer.weight.dim() > 2
        grid = flat.reshape(in_features, -1) if leading else flat.reshape(-1, in_features).t()
        if _every(grid == grid[:, :1]):
            return grid[:, 0].clone()
        if not leading:
            return flat.reshape(-1, in_features).clone()
    raise GaugePlacementError(
        f"{getattr(perceptron, 'name', None)!r}: its sources hold {flat.numel()} gauges over {in_features} inputs, no "
        f"whole number of rows of its input, so no fold holds each element at its producer's gauge")


def _patch_rows(perceptron: Any, flat: torch.Tensor, reader: Any, source: Any) -> torch.Tensor:
    """A convolution's fold per output position: its per-element input gauge gathered by its own patch table (the
    mapping's ``patch_gather``), a padding slot at one; the table is kept as ``fold_slot_source``."""
    read = _SAMPLE_SHAPE.get()
    shape = None if read is None or source is None else tuple(int(d) for d in read(source))
    if shape is None or int(np.prod(shape)) != flat.numel() or not hasattr(reader, "patch_gather"):
        raise PositionFoldError(
            f"{getattr(perceptron, 'name', None)!r}: a convolution's fold per element of its input is read in that "
            f"input's own grid, which only the stamping walk names (its sources' table)")
    table = np.asarray(reader.patch_gather(shape).slot_source_index, dtype=np.int64)
    padded = torch.cat([flat, flat.new_ones(1)])
    perceptron.__dict__[SLOT_SOURCE] = (table, int(flat.numel()))
    return padded[torch.as_tensor(table, device=flat.device)].clone()


def _every(mask: torch.Tensor) -> bool:
    """Whether every entry of ``mask`` holds. A tensor on the meta device holds shapes and devices but no values, so
    none of its entries can differ: the fold then takes its one-value form, on the anchored vector's device."""
    return mask.is_meta or bool(torch.all(mask))


def reshape_gauge(vector: torch.Tensor, axis: int, output_shape: Any) -> torch.Tensor:
    """[U-TGAUGE3] A per-channel gauge across a C-order reshape (its output's element ``k`` is its source's element
    ``k``, so element ``k``'s source index is ``k``): kept per channel where the reshape keeps the channel axis with
    its channels (a reshape that declares no output shape moves leading axes only; a channel-first source flattened
    holds each channel's block contiguous, as readers lay it), else one gauge per element of the output, a
    channels-last source's channels cycling fastest."""
    if output_shape is None:
        return vector
    shape = tuple(int(d) for d in output_shape)
    flat = vector.reshape(-1)
    channels, elements = flat.numel(), int(np.prod(shape, dtype=np.int64))
    if (shape[0] if axis == LEADING else shape[-1]) == channels or (axis == LEADING and len(shape) == 1):
        return vector
    if elements % channels:
        raise GaugePlacementError(
            f"a reshape to {shape} of a {channels}-channel gauge: {elements} elements hold no whole number of "
            f"channels, so no element's source channel can be read")
    return flat.repeat_interleave(elements // channels) if axis == LEADING else flat.repeat(elements // channels)


def channel_form(placed: np.ndarray, shape: tuple, source: tuple, axis: Any, *, joins: bool) -> np.ndarray:
    """[U-TGAUGE4] A layout's placed gauge (one per element of its sample ``shape``) in the walk's per-channel form: one
    gauge per channel along ``axis`` where the layout keeps its first source's (``source``) channel axis (a join of
    one rank, or a reshape that keeps the axis's size) and every channel's elements share one gauge; else per
    element (a rank-one sample's elements are its channels)."""
    if axis is None or len(shape) < 2:
        return placed
    at = 0 if axis == LEADING else len(shape) - 1
    same_rank = len(shape) == len(source)
    reshaped = int(np.prod(shape)) == int(np.prod(source)) and shape[at] == source[0 if axis == LEADING else -1]
    if not ((joins and same_rank) or (same_rank and shape[at] == source[at]) or reshaped):
        return placed
    grid = np.moveaxis(placed.reshape(shape), at, 0).reshape(shape[at], -1)
    return grid[:, 0].copy() if bool(np.all(grid == grid[:, :1])) else placed


def holds_channels(node: Any, scale: torch.Tensor) -> bool:
    """Whether a node that is no layout emits one gauge per channel of its output: a perceptron (its threshold per
    output channel), the input root (one per leading channel), a host op whose vector is aligned to its own channels."""
    if getattr(getattr(node, "perceptron", None), "output_channels", None) is not None:
        return True
    if not getattr(node, "get_source_mappers", lambda: [])():
        return True
    channels = gauge_channels(node)
    return channels is not None and int(scale.numel()) == int(channels)
