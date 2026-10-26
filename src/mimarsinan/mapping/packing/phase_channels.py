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
relay, an inserted core) has no channel at unbounded geometry: a dithered encode of it refuses by name."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import ComputeOp, IRGraph, NeuralCore

#: A node's channels: ``base`` (output ``k`` is channel ``base + k``) or one channel per output index.
NodeChannels = Union[int, np.ndarray]
#: The program attribute that holds every keyed node's channels (``HybridHardCoreMapping.node_phase_channels``).
ATTRIBUTE = "node_phase_channels"
RAW_INPUT = -2


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
    if isinstance(entry, (int, np.integer)):
        return torch.arange(int(entry), int(entry) + int(size), dtype=torch.long)
    return torch.as_tensor(np.asarray(entry[:int(size)], dtype=np.int64))


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
