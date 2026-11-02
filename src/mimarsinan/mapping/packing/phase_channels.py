"""[E1] Every stage-input channel's channel at unbounded geometry: the id the phase-dithered Uniform encode rotates its
comb by (``spike_modes.channel_phase_offsets``).

A stage input holds, per slice, a prefix of one IR node's outputs (``hybrid_segment_helpers``'s layout) in the packer's
compacted index space. Output ``k`` of a perceptron's node is the perceptron's output channel
``perceptron_output_slice[0] + c``, where ``c`` is the column output ``k`` held before compaction (IR pruning's
``pre_pruning_col_mask``, then the packer's ``reindex_maps``); a host op's and the raw input's channel is the output
index (neither is tiled). That channel is the neuron's index in its IR node at unbounded geometry, and the index the NF
twin rotates by (it encodes each producer's own output), so no cost-only decision (core geometry, the IR's node ids and
order, packing, pass cuts) moves a dithered train. Before E1 the rotation followed the channel's position in the
executed unit's stage input, whose layout orders external nodes by IR node id. A node with no perceptron identity (a
relay, an inserted core) has no channel at unbounded geometry: a dithered encode of it refuses by name.

[U-DOOR] The NF twin reads its comb keys from the same table (``map_model_to_ir.mapper_phase_keys`` through
``source_channels``, read per sample by ``element_channels``): every element of a mapper node's
value is an output of one IR node (``map_to_ir``'s sources: its node and output index), and which node is the
mapping's decision alone (a token-wise encoding is one host node per token, a host add over the same token value one
node), so no rule by the value's shape is the deployed one."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import ComputeOp, IRGraph, NeuralCore

#: A node's channels: ``base`` (output ``k`` is channel ``base + k``) or one channel per output index.
NodeChannels = Union[int, np.ndarray]
#: The program attribute that holds every keyed node's channels (``HybridHardCoreMapping.node_phase_channels``).
ATTRIBUTE = "node_phase_channels"
RAW_INPUT = -2
#: The key ``source_channels`` gives an output of a node the table holds no channel for (read through
#: ``element_channels``, which refuses it by name).
UNKEYED = -1
#: An off and an always-on source (``IRSource.is_off``, ``is_always_on``): a constant axon, no stage input's position.
_CONSTANT_SOURCES = (-1, -3)


class PhaseChannelRefusal(ToolchainRefusal, ValueError):
    """A phase-dithered encode of a stage-input channel that has no channel at unbounded geometry."""


def _columns(node: NeuralCore) -> Optional[np.ndarray]:
    """The pre-compaction column of each current column of an IR-compacted node, None for an uncompacted node, and an
    empty array when the mask's survivors are not the node's columns (the node then has no channels). A bank-backed
    node's columns are counted as an owned matrix's (``get_output_count``: its bank column slice)."""
    mask = getattr(node, "pre_pruning_col_mask", None)
    if mask is None:
        return None
    kept = np.flatnonzero(~np.asarray(mask, dtype=bool))
    if len(kept) != int(node.get_output_count()):
        return np.zeros(0, dtype=np.int64)
    return kept


def _packed_sources(remap: Optional[Mapping[int, int]]) -> Optional[np.ndarray]:
    """Per packed output index the IR output index it holds (the inverse of the packer's reindex map), None for an
    identity map."""
    if not remap or all(int(o) == int(n) for o, n in remap.items()):
        return None
    out = np.empty(len(remap), dtype=np.int64)
    for old, new in remap.items():
        out[int(new)] = int(old)
    return out


def node_phase_channels(ir_graph: IRGraph,
                        reindex_maps: Optional[Mapping[int, Mapping[int, int]]] = None) -> Dict[int, NodeChannels]:
    """Every keyed node's channels in the program's index space (``reindex_maps``: the packer's compaction, keyed by IR
    node id); the raw input and every host op keyed by their output index; an unkeyed node absent."""
    out: Dict[int, NodeChannels] = {RAW_INPUT: 0}
    for node in ir_graph.nodes:
        nid = int(node.id)
        if isinstance(node, ComputeOp):
            out[nid] = 0
            continue
        if not isinstance(node, NeuralCore) or node.perceptron_index is None or node.perceptron_output_slice is None:
            continue
        base, columns = int(node.perceptron_output_slice[0]), _columns(node)
        packed = _packed_sources((reindex_maps or {}).get(nid))
        if columns is None:
            out[nid] = base if packed is None else base + packed
        elif len(columns) and (packed is None or int(packed.max(initial=0)) < len(columns)):
            out[nid] = base + (columns if packed is None else columns[packed])
    return out


def channels_of(mapping: Any, node_id: int, size: int, *, where: str = "") -> torch.Tensor:
    """The channels at unbounded geometry of outputs ``0 .. size - 1`` of ``node_id`` in ``mapping``; refuses by name
    when the program holds none for it (no perceptron identity, or a program built without the table)."""
    entry = (getattr(mapping, ATTRIBUTE, None) or {}).get(int(node_id))
    if entry is not None and not isinstance(entry, (int, np.integer)) and len(entry) < int(size):
        entry = None
    if entry is None:
        raise PhaseChannelRefusal(
            f"{where or 'stage input'}: node {int(node_id)} has no channel at unbounded geometry (no perceptron "
            f"identity, or a program built without its channel table): a phase-dithered encode keyed on where a "
            f"cost-only decision puts it would move with that decision")
    return torch.as_tensor(_channels_at(entry, np.arange(int(size), dtype=np.int64)))


def _channels_at(entry: Union[NodeChannels, np.integer], outputs: np.ndarray) -> np.ndarray:
    """The channels of a node's ``outputs`` (output indices) from its table entry: ``base + k`` or the ``k``-th
    channel; the one rule a stage-input slice (``channels_of``) and the NF twin (``source_channels``) read."""
    if isinstance(entry, (int, np.integer)):
        return int(entry) + outputs
    return np.asarray(entry, dtype=np.int64)[outputs]


def source_channels(table: Mapping[int, NodeChannels], sources: Any) -> np.ndarray:
    """[U-DOOR] The channel at unbounded geometry of every element of an IR source array (``IRSource``: its node and
    output index), shaped as ``sources``: output ``k`` of node ``n`` reads ``table[n]`` at ``k``, as a stage-input
    slice of ``n`` does. A constant source takes 0 (count 0 or T: no rotation moves it); an output of a node the table
    holds no channel for takes ``UNKEYED``."""
    arr = np.asarray(sources, dtype=object)
    flat = arr.reshape(-1)
    nodes = np.fromiter((int(s.node_id) for s in flat), dtype=np.int64, count=flat.size)
    outputs = np.fromiter((int(s.index) for s in flat), dtype=np.int64, count=flat.size)
    out = np.zeros(flat.size, dtype=np.int64)
    for nid in np.unique(nodes).tolist():
        if nid in _CONSTANT_SOURCES:
            continue
        at, entry = nodes == nid, table.get(int(nid))
        if entry is None or (not isinstance(entry, (int, np.integer)) and len(entry) <= int(outputs[at].max())):
            out[at] = UNKEYED
        else:
            out[at] = _channels_at(entry, outputs[at])
    return out.reshape(arr.shape)


def element_channels(keys: Mapping[int, np.ndarray], node: Any, sample_shape: Sequence[int]) -> torch.Tensor:
    """[U-DOOR] The comb keys of one sample of ``node``'s value (``map_model_to_ir.mapper_phase_keys``), shaped
    ``sample_shape`` (its IR sources in C order, the order a host buffer holds a sample in); refuses by name a value its
    node's IR sources do not hold element for element, and an element no channel keys (as ``channels_of`` refuses its
    stage-input slice)."""
    entry, shape = keys.get(id(node)), tuple(int(d) for d in sample_shape)
    name = getattr(node, "name", None) or type(node).__name__
    if entry is None or entry.size != int(np.prod(shape, dtype=np.int64)):
        raise PhaseChannelRefusal(
            f"the NF twin's phase-dithered encode of {name!r}: its value's sample {shape} is not held element for "
            f"element by its IR sources ({None if entry is None else entry.shape}), so no element has the channel the "
            f"deployed program keys it by. Remedy: declare spike_phase_dither false")
    if (entry < 0).any():
        raise PhaseChannelRefusal(
            f"the NF twin's phase-dithered encode of {name!r}: an element's IR node has no channel at unbounded "
            f"geometry (no perceptron identity), so the deployed program refuses its stage input by name too. Remedy: "
            f"declare spike_phase_dither false")
    return torch.as_tensor(entry.reshape(shape))


def stage_input_channels(mapping: Any, stage: Any, width: int, *, cached=()) -> torch.Tensor:
    """``(width,)`` the channel at unbounded geometry of every position of ``stage``'s input; a ``cached`` node's slice
    (its train is read verbatim, never encoded) keeps its position."""
    channels = torch.arange(int(width), dtype=torch.long)
    for s in stage.input_map:
        if int(s.node_id) in cached:
            continue
        channels[s.offset:s.offset + s.size] = channels_of(mapping, s.node_id, s.size,
                                                           where=f"stage {getattr(stage, 'name', '?')!r}")
    return channels
