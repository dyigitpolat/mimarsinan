"""Deployed per-neuron survival: which output neurons of each perceptron survive pruning, by the program's own rule.

[U-IN A-2, F-SURV] A core's deployed outputs are the columns both compactions the program applies keep: IR pruning's
compaction of an OWNED core (its ``pre_pruning_col_mask`` over the tile it held before), then Soft Core Mapping's
compaction of the core's current masks (``compaction.compaction_keep_indices`` over ``pruned_col_mask``, where a
bank-backed core keeps its pruned channels until then; ``compacted_core_extent`` is that width). One function derives
them per core (``core_deployed_channels``). A perceptron's survivors are keyed by the IR's provenance, (column,
channel): a convolution's or a token-wise perceptron's cores share one channel range at every position, so a channel
alone does not name a neuron. A record is projected through its perceptron's declared ``output_channel_axis``
(``record_layouts``); a core or a record the rule cannot derive refuses by name (``SurvivalRefusal``), never as a shape
mismatch.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

import numpy as np

from mimarsinan.common.refusal.toolchain_refusal import ToolchainRefusal
from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.packing.softcore.compaction import compacted_core_extent, compaction_keep_indices

__all__ = [
    "DeployedNeuronSurvival",
    "SurvivalRefusal",
    "core_deployed_channels",
    "derive_deployed_neuron_survival",
    "record_layouts",
]

#: A record's per-sample channel axis: leading (channel-first, ``(C, H, W)``) or last (channels-last, ``(N, C)``).
LEADING, LAST = 0, -1


class SurvivalRefusal(ToolchainRefusal, AssertionError):
    """A core's deployed outputs, or a record's projection onto them, cannot be derived by the program's kept-column
    rule (an assertion too: a record inconsistent with its deployment fails loud, as it always did)."""


def record_layouts(perceptrons) -> Dict[int, Tuple[int, int]]:
    """``{perceptron index: (per-sample channel axis, channels)}`` from each perceptron's declared
    ``output_channel_axis`` (1, a convolution's, is the leading per-sample axis; -1 the last); a perceptron declaring
    another axis has no layout here."""
    layouts: Dict[int, Tuple[int, int]] = {}
    for index, perceptron in enumerate(perceptrons):
        channels = getattr(perceptron, "output_channels", None)
        axis = {1: LEADING, -1: LAST}.get(int(getattr(perceptron, "output_channel_axis", -1)))
        if channels is not None and axis is not None:
            layouts[index] = (axis, int(channels))
    return layouts


def core_deployed_channels(node: NeuralCore) -> np.ndarray:
    """The perceptron output channels ``node``'s deployed columns hold, in their deployed order: the tile columns IR
    pruning's compaction kept (``pre_pruning_col_mask``), then those the packer's compaction keeps of the core's current
    columns (``compaction_keep_indices``), offset by the tile's first channel."""
    out_slice = node.perceptron_output_slice
    try:
        width = int(node.get_output_count())
    except ValueError as exc:
        raise SurvivalRefusal(f"core {node.name}: no output count to derive its deployed outputs from ({exc})") from exc
    mask = node.pre_pruning_col_mask
    if mask is None:
        held = np.arange(width, dtype=np.int64)
    else:
        tile = int(out_slice[1]) - int(out_slice[0]) if out_slice is not None else len(mask)
        if len(mask) != tile:
            raise SurvivalRefusal(f"core {node.name} of perceptron {node.perceptron_index}: a pre-pruning mask over "
                                  f"{len(mask)} columns on an output tile of {tile}")
        held = np.flatnonzero(~np.asarray(mask, dtype=bool)).astype(np.int64)
        if len(held) != width:
            raise SurvivalRefusal(f"survivor count {len(held)} != deployed output width {width} for perceptron "
                                  f"{node.perceptron_index} core {node.name}: IR pruning's compaction kept {len(held)} "
                                  f"of its tile, the core holds {width}")
    sources = node.input_sources.flatten()
    last = sources[-1] if len(sources) else None
    _rows, keep = compaction_keep_indices(
        n_axons=int(len(sources)), n_neurons=width, pruned_row_mask=node.pruned_row_mask,
        pruned_col_mask=node.pruned_col_mask, keep_last_axon=bool(last is not None and last.is_always_on()))
    if len(keep) != compacted_core_extent(node)[1]:
        raise SurvivalRefusal(f"core {node.name}: {len(keep)} kept columns against the packer's extent "
                              f"{compacted_core_extent(node)[1]}")
    base = int(out_slice[0]) if out_slice is not None else 0
    return base + held[np.asarray(keep, dtype=np.int64)]


@dataclass(frozen=True)
class DeployedNeuronSurvival:
    """Per perceptron, the surviving neurons of the deployed mapping: ``survivors`` their channels, ``columns`` their
    columns (absent: every survivor at column 0), ordered by (column, channel). A perceptron absent from the map, or a
    record already at deployed width, is passed through unchanged."""

    survivors: Mapping[int, np.ndarray]
    columns: Mapping[int, np.ndarray] = field(default_factory=dict)

    def project(self, records: Dict[int, np.ndarray], layouts: Mapping[int, Tuple[int, int]] | None = None,
                cycles: int = 1) -> Dict[int, np.ndarray]:
        """Select each per-perceptron record's surviving ORIGINAL neurons; ``(batch, N * cycles)`` ->
        ``(batch, M * cycles)``, ``cycles`` values per neuron (a neuron-major raster), ``layouts`` as
        ``record_layouts`` gives them (a multi-column perceptron's record needs its own).

        Index-based by construction: a pruned neuron's record value is irrelevant (streamed LIF membrane init or a
        folded half-step bias can make a zero-input neuron emit, and records may carry negative values)."""
        projected: Dict[int, np.ndarray] = {}
        for pi, vals in records.items():
            channels = self.survivors.get(pi)
            if channels is None or vals.shape[1] == len(channels) * int(cycles):
                projected[pi] = vals
                continue
            neurons = self._neuron_index(pi, vals.shape[1], int(cycles), (layouts or {}).get(pi))
            index = (neurons[:, None] * int(cycles) + np.arange(int(cycles), dtype=np.int64)).reshape(-1)
            projected[pi] = vals[:, index]
        return projected

    def _neuron_index(self, pi: int, width: int, cycles: int, layout) -> np.ndarray:
        channels = np.asarray(self.survivors[pi], dtype=np.int64)
        columns = np.asarray(self.columns.get(pi, np.zeros(len(channels), dtype=np.int64)), dtype=np.int64)
        neurons, rest = divmod(int(width), cycles)

        def refuse(why: str) -> SurvivalRefusal:
            return SurvivalRefusal(f"perceptron {pi}: a record of width {width} ({cycles} value(s) per neuron) cannot "
                                   f"be projected onto its {len(channels)} deployed neuron(s): {why}")

        if rest or not len(channels) or neurons <= len(channels):
            raise refuse("it is neither the deployed width nor wider than it")
        if layout is not None:
            axis, n_channels = int(layout[0]), int(layout[1])
            if n_channels <= 0 or neurons % n_channels:
                raise refuse(f"its width is not a multiple of the perceptron's {n_channels} channels")
            positions = neurons // n_channels
            if int(channels.max()) >= n_channels or int(columns.max()) >= positions:
                raise refuse(f"a survivor lies outside its {n_channels} channels x {positions} columns")
            index = channels * positions + columns if axis == LEADING else columns * n_channels + channels
        elif not columns.any():
            index = channels  # one column: the channel is the neuron, whatever the channel axis
        else:
            raise refuse("its perceptron has several columns and no declared channel axis (record_layouts)")
        if int(index.max()) >= neurons or len(np.unique(index)) != len(index):
            raise refuse("the survivors do not index distinct neurons of it")
        return np.sort(index)


def derive_deployed_neuron_survival(ir_graph: IRGraph) -> DeployedNeuronSurvival:
    """Each perceptron's surviving neurons in the pruned ``ir_graph`` (the deployment authority), by the program's own
    kept-column rule per core (``core_deployed_channels``); an unpruned graph yields full survival, so the projection is
    the identity there."""
    channels: Dict[int, list] = defaultdict(list)
    columns: Dict[int, list] = defaultdict(list)
    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore) or node.psum_role not in (None, "accum"):
            continue
        if node.perceptron_index is None or node.perceptron_index < 0:
            continue
        kept = core_deployed_channels(node)
        channels[node.perceptron_index].extend(kept.tolist())
        columns[node.perceptron_index].extend([int(node.perceptron_output_column or 0)] * len(kept))
    survivors, by_column = {}, {}
    for pi in channels:
        ch, col = np.asarray(channels[pi], dtype=np.int64), np.asarray(columns[pi], dtype=np.int64)
        order = np.lexsort((ch, col))
        survivors[pi], by_column[pi] = ch[order], col[order]
    return DeployedNeuronSurvival(survivors=survivors, columns=by_column)
