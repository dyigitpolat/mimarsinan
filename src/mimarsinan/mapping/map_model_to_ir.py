"""Convenience wrapper to map a model representation to an IRGraph; [U-DOOR] and the NF twin's comb keys read from that
mapping (``mapper_phase_keys``); [U-TGAUGE3] and every element's IR source, the table the twins place a layout's gauges
by (``mapper_source_table``, ``walk_layout_gauge``, ``LayoutGaugePlacement``)."""

from __future__ import annotations

import weakref
from typing import Any, Callable, Dict

import numpy as np
import torch

from mimarsinan.mapping.ir import IRGraph
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.phase_channels import node_phase_channels, source_channels
from mimarsinan.mapping.support.gauge_channels import LEADING, concat_joins_channels, gauge_axis, gauge_view
from mimarsinan.mapping.support.gauge_placement import (
    GaugePlacementError, channel_form, holds_channels, place_gauges, reshape_gauge, source_index)
from mimarsinan.mapping.support.scale_broadcast import broadcast_scale_along

def map_model_to_ir(
    model_representation,
    q_max: float = 1.0,
    firing_mode: str = "Default",
    max_axons: int | None = None,
    max_neurons: int | None = None,
    allow_coalescing: bool = False,
    hardware_bias: bool = False,
) -> IRGraph:
    """Convenience wrapper around ``IRMapping.map``."""
    return IRMapping(
        q_max=q_max,
        firing_mode=firing_mode,
        max_axons=max_axons,
        max_neurons=max_neurons,
        allow_coalescing=allow_coalescing,
        hardware_bias=hardware_bias,
    ).map(model_representation)


#: Per mapper graph: (its structure, its nodes' keys); kept while the graph lives (a key walk maps the whole graph).
_KEYS: "weakref.WeakKeyDictionary[Any, tuple]" = weakref.WeakKeyDictionary()


def _structure(model_representation: Any) -> tuple:
    """What decides the IR's node partition: the nodes in execution order and which perceptrons are encoding layers
    (a host node rather than a core)."""
    return tuple((id(n), type(n).__name__, bool(getattr(getattr(n, "perceptron", None), "is_encoding_layer", False)))
                 for n in model_representation.execution_order())


def mapper_phase_keys(model_representation: Any) -> Dict[int, np.ndarray]:
    """[U-DOOR] ``id(node) -> channels``: every mapper node's channel at unbounded geometry per element of one sample
    of its value, the NF twin's comb keys: the node's IR sources (``map_to_ir``) through
    ``phase_channels.node_phase_channels``, the table the deployed program keys its stage inputs by
    (``phase_channels.source_channels``), so both twins key from one rule. The graph is mapped as the identity program
    maps it (unbounded cores, no coalescing: E1's channels do not move with a cost-only decision), with the perceptron
    indices Soft Core Mapping assigns (an idempotent walk); its walk memos are cleared after. Kept per graph while its
    structure holds. A graph the mapping refuses refuses here by the mapping's own name."""
    structure = _structure(model_representation)
    kept = _KEYS.get(model_representation)
    if kept is not None and kept[0] == structure:
        return kept[1]
    try:
        with torch.no_grad():
            model_representation.assign_perceptron_indices()
            table = node_phase_channels(map_model_to_ir(model_representation))
            keys = {id(n): source_channels(table, sources) for n in model_representation.execution_order()
                    if (sources := getattr(n, "_ir_sources", None)) is not None}
    finally:
        model_representation.clear_ir_caches()
    _KEYS[model_representation] = (structure, keys)
    return keys


#: Per mapper graph: (its structure, its source table); kept while the graph lives (the table maps the whole graph).
_SOURCES: "weakref.WeakKeyDictionary[Any, tuple]" = weakref.WeakKeyDictionary()


class SourceTable:
    """[U-TGAUGE3] Every mapper node's value, element by element, as the IR sources of the identity program hold it:
    ``sources[id(node)]`` its ``(node ids, output indices)`` in C order and ``shapes[id(node)]`` its source array's
    shape; ``owners[nid]`` the mapper node that emits IR node ``nid`` (the first node in execution order that is no
    layout and holds it) with the position, in that node's value, of each of ``nid``'s outputs."""

    def __init__(self, sources: dict, shapes: dict, owners: dict):
        self.sources, self.shapes, self.owners = sources, shapes, owners


def mapper_source_table(model_representation: Any) -> SourceTable:
    """[U-TGAUGE3] The source table of a mapper graph, mapped as ``mapper_phase_keys`` maps it (the identity program,
    the perceptron indices Soft Core Mapping assigns, the walk memos cleared after), kept per graph while its
    structure holds. A graph the mapping refuses refuses here by the mapping's own name."""
    structure = _structure(model_representation)
    kept = _SOURCES.get(model_representation)
    if kept is not None and kept[0] == structure:
        return kept[1]
    sources: dict = {}
    shapes: dict = {}
    owners: dict = {}
    try:
        with torch.no_grad():
            model_representation.assign_perceptron_indices()
            map_model_to_ir(model_representation)
            for node in model_representation.execution_order():
                array = getattr(node, "_ir_sources", None)
                if array is None:
                    continue
                ids, idx = source_index(array)
                sources[id(node)], shapes[id(node)] = (ids, idx), tuple(np.shape(array))
                if getattr(node, "moves_elements_only", False):
                    continue
                for nid in np.unique(ids[ids >= 0]).tolist():
                    if nid not in owners:
                        at = np.flatnonzero(ids == nid)
                        positions = np.full(int(idx[at].max()) + 1, -1, dtype=np.int64)
                        positions[idx[at]] = at
                        owners[nid] = (id(node), positions)
    finally:
        model_representation.clear_ir_caches()
    table = SourceTable(sources, shapes, owners)
    _SOURCES[model_representation] = (structure, table)
    return table


def sample_shape(shape: tuple) -> tuple:
    """A source array's shape as one sample of the value holds it (its leading unit axes dropped: a flat value's
    array is ``(1, F)`` or ``(F,)``)."""
    dims = [int(d) for d in shape] or [1]
    while len(dims) > 1 and dims[0] == 1:
        dims.pop(0)
    return tuple(dims)


def walk_layout_gauge(model_representation: Any, node: Any, own_gauge: Callable[[Any, tuple], Any]) -> np.ndarray:
    """[U-TGAUGE3] ``node``'s gauge per element of one sample (C order) by the placement rule: each element at its
    producer's gauge, ``own_gauge(producer, shape)`` (the producer's own gauge over its own sample shape, flat, or a
    scalar), read at the element's position there, from the identity program's IR sources (``place_gauges``)."""
    table = mapper_source_table(model_representation)
    by_id = {id(n): n for n in model_representation.execution_order()}
    ids, idx = table.sources[id(node)]

    def held(nid: int):
        owner = table.owners.get(nid)
        if owner is None:
            raise GaugePlacementError(
                f"{type(node).__name__} {getattr(node, 'name', None)!r}: IR node {nid} has no producer in the mapper "
                f"graph, so its elements' gauge has no table to be read from")
        gauge = np.asarray(own_gauge(by_id[owner[0]], sample_shape(table.shapes[owner[0]])), dtype=np.float64)
        return gauge.reshape(-1) if gauge.size == 1 else gauge.reshape(-1)[owner[1]]

    return place_gauges(ids, idx, held)


def own_gauge_of(vector: Any, producer: Any, shape: tuple) -> Any:
    """A producer's own gauge over its own sample shape (flat), the gauge it emits each element at: its per-channel
    vector along its own channel axis, or its one value; a host op that normalizes its output (its wrapper armed) at
    its output scale as that wrapper broadcasts it over its output (the deployed reader's ``_emitted_gauge``: a
    reduction's scale over its input's elements at their mean). Refused by name where its value does not hold the
    vector along that axis."""
    if vector is None:
        return 1.0
    flat = torch.as_tensor(vector).detach().to(torch.float64).reshape(-1)
    if flat.numel() == 1 or bool(torch.all(flat == flat[0])):
        return float(flat[0])
    like = torch.zeros((1, *shape), dtype=torch.float64)
    view = gauge_view(flat, like, gauge_axis(producer))
    if view is None and getattr(producer, "per_source_scales", None) is not None and getattr(
            producer, "output_scale", None) is not None:
        view = broadcast_scale_along(flat, like, getattr(producer, "wrap_channel_axis", None))
    if view is None:
        raise GaugePlacementError(
            f"{type(producer).__name__} {getattr(producer, 'name', None)!r}: its per-channel gauge ({flat.numel()} "
            f"values) is not held along its own channel axis of a {shape} sample, so no element's gauge can be read")
    return view.expand_as(like).reshape(-1).cpu().numpy()


def one_value(gauge: Any) -> bool:
    """Whether a gauge (a scalar, a vector, or None: one) holds one value."""
    flat = torch.as_tensor(1.0 if gauge is None else gauge).detach().reshape(-1)
    return flat.numel() <= 1 or bool(torch.all(flat == flat[0]))


class LayoutGaugePlacement:
    """[U-TGAUGE3, U-TGAUGE4] The source walk's placement (``walk_out_scales(..., place=)``): every element of a layout
    node's gauge at its source's, read at its source index, before any consumer reads it.

    A closed form keeps the propagated vector only where its premise holds, read from the sources, never from the
    layout's own propagated scale: a gauge of one value where every source holds that same value (every element at
    it); a join along the gauge axis (``concat_joins_channels``), a residual lane join (``joins_equal_parts``) and a
    C-order reshape (``keeps_element_order``, ``reshape_gauge``) where every source's vector is one gauge per channel
    of that source, which holds by construction for a perceptron's, the input's, a host op's aligned to its own
    channels and a closed form's over such sources (``_per_channel``, read only where a closed form asks). Any other
    layout is placed from the identity program's IR table (``walk_layout_gauge``), held one gauge per channel along its
    gauge axis where the table shows every channel's elements at one gauge (``channel_form``), else one per element. U-TGAUGE3 kept the closed forms
    on the propagated vector alone: a stack held its first source's one value, a join over an axis move kept a
    misaligned vector, and a reshape of such a vector refused a layout the table places."""

    def __init__(self, model_representation: Any):
        self.repr = model_representation
        self._walk: Any = None
        self._channels: dict = {}

    def __call__(self, node: Any, scale: Any, out_scales: dict) -> Any:
        if not isinstance(scale, torch.Tensor) or not getattr(node, "moves_elements_only", False):
            return scale
        if self._walk is not out_scales:  # whether each node's vector is one gauge per channel, for this walk
            self._walk, self._channels = out_scales, {}
        self.repr._ensure_exec_graph()
        sources = [d for d in self.repr._deps.get(node, []) if d is not None]
        if one_value(scale) and all(one_value(out_scales.get(d)) and _one(out_scales.get(d)) == _one(scale)
                                    for d in sources if d in out_scales):
            return scale
        kept = self._closed_form(node, scale, out_scales, sources) if all(
            self._per_channel(d, out_scales) for d in sources) else None
        if kept is not None:
            self._channels[node] = kept is scale or len(_sample(node, kept)) <= 1
            return kept
        placed = walk_layout_gauge(self.repr, node, lambda p, shape: own_gauge_of(out_scales.get(p), p, shape))
        table = mapper_source_table(self.repr)
        shapes = [sample_shape(table.shapes[id(n)]) for n in (node, *sources[:1])]
        held = channel_form(placed, shapes[0], shapes[-1], gauge_axis(node), joins=len(sources) > 1)
        self._channels[node] = held.size != placed.size or len(shapes[0]) <= 1
        return torch.as_tensor(held, dtype=scale.dtype, device=scale.device)

    def _per_channel(self, node: Any, out_scales: dict) -> bool:
        """Whether ``node``'s vector is one gauge per channel of its output (read only where a layout's closed form
        asks, memoized for the walk): a node that is no layout by ``holds_channels``; a layout the walk passed by
        the one-value form where its sources are and its closed form keeps the vector."""
        if node in self._channels:
            return self._channels[node]
        scale = out_scales.get(node)
        if not isinstance(scale, torch.Tensor):
            held = False
        elif not getattr(node, "moves_elements_only", False):
            held = holds_channels(node, scale)
        else:
            sources = [d for d in self.repr._deps.get(node, []) if d is not None]
            held = all(self._per_channel(d, out_scales) for d in sources) and (
                self._closed_form(node, scale, out_scales, sources) is scale)
        self._channels[node] = held
        return held

    @staticmethod
    def _closed_form(node: Any, scale: torch.Tensor, out_scales: dict, sources: list) -> Any:
        """U-TGAUGE3's closed forms over sources whose vectors are one gauge per channel; None where none applies."""
        if len(sources) > 1 and hasattr(node, "dim") and concat_joins_channels(node, gauge_axis(node)):
            return scale
        if getattr(node, "joins_equal_parts", False) and (gauge_axis(node) == LEADING or all(
                one_value(out_scales.get(d)) for d in sources)):
            return scale
        axis = gauge_axis(sources[0]) if len(sources) == 1 else None
        if axis is not None and getattr(node, "keeps_element_order", False):
            try:
                return reshape_gauge(scale, axis, getattr(node, "output_shape", None))
            except GaugePlacementError:
                return None
        return None


def _sample(node: Any, vector: torch.Tensor) -> tuple:
    shape = getattr(node, "output_shape", None)
    return tuple(int(d) for d in shape) if shape else tuple(vector.shape)


def _one(gauge: Any) -> float:
    return float(torch.as_tensor(1.0 if gauge is None else gauge).detach().reshape(-1)[0])


def as_gauge_like(gauge: Any, like: torch.Tensor) -> Any:
    """A gauge (a scalar, or one per element of a sample, flat) broadcastable against the batched value ``like``."""
    if np.ndim(gauge) == 0:
        return float(gauge)
    return torch.as_tensor(gauge, dtype=like.dtype, device=like.device).view(1, *like.shape[1:])
