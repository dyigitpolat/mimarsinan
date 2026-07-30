"""[W6c] The STRUCTURAL identity of a mapped layer — the table's row key.

A mapper emits one :class:`NeuralCore` per (source layer x position x output
tile), and the paper table wants ROWS PER MAPPED LAYER. W6b recovered that
identity by string-munging the node names, which is not workload-agnostic: a
converter-built ``nn.Sequential`` names its cores ``_0_pos4_7_g0`` / ``_4_col0``
/ ``_6_col0``, every one of which peels down to the empty string, so the whole
program degenerated into ONE BLANK ROW. Conv vehicles are the headline case, so
the identity must come from the IR, not from the name.

**The rule.** Two mapped softcore instances belong to the same mapped layer iff
they realize the SAME SOURCE PERCEPTRON. The IR records that provenance
directly, and this module reads it off a ladder of decreasing authority:

1. ``NeuralCore.perceptron_index`` — assigned by
   ``MapperRepr.assign_perceptron_indices()`` in forward-topological order and
   threaded through every mapper into ``add_neural_core`` /
   ``add_shared_neural_core``. When present this IS the mapped-layer identity:
   ``blocks_0_fc1``'s 65 column instances all carry ``perceptron_index=1``, the
   64 conv positions all carry ``perceptron_index=0``, and the ``_col{i}`` /
   ``_tile_{s}_{e}`` / ``_pos{h}_{w}_g{g}`` suffixes a mapper appends are
   irrelevant to it. Negative indices mean "no provenance" (the convention
   ``deployed_neuron_survival`` already uses) and fall through.
2. ``WeightBank.perceptron_index`` — ``register_weight_bank`` stamps the same
   provenance on shared storage, so a bank-backed core whose own field was
   never set still resolves to its source layer.
3. ``(weight_bank_id, weight_row_slice)`` — WEIGHT IDENTITY. Instances sharing
   a bank and the same column window read literally the same weight matrix,
   hence the same source layer. The window is part of the key because a
   weight-stationary tiled layer registers ONE BANK PER OUTPUT TILE
   (``layout_ir_mapping_fc._map_fc_weight_stationary_tiled``), so distinct
   windows inside one bank would be distinct matrices; keeping it can only
   refuse to merge, never merge two unrelated layers.
4. ``node.id`` — an owned crossbar with no provenance and no shared storage is
   its own layer. One row, never blank, never merged with anything else.

Every level is a property of the mapping, so a layer named ``0`` and a layer
named ``blocks.3.mlp.fc1`` are keyed identically well. Names are used only for
the DISPLAY label (see :mod:`...labels`), and a label failure can no longer
collapse or shatter a row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MappedLayerKey:
    """One mapped layer, identified structurally. Hashable and orderable."""

    #: ``"perceptron"`` | ``"bank"`` | ``"core"`` — which rung of the ladder.
    kind: str
    #: perceptron index / weight-bank id / node id, per ``kind``.
    index: int
    #: the bank column window, set only for ``kind == "bank"``.
    window: tuple[int, int] | None = None

    @property
    def sort_key(self) -> tuple[str, int, int, int]:
        start, end = self.window if self.window is not None else (-1, -1)
        return (self.kind, self.index, start, end)

    def __str__(self) -> str:
        if self.window is None:
            return f"{self.kind}:{self.index}"
        return f"{self.kind}:{self.index}[{self.window[0]}:{self.window[1]}]"


def _provenance(value: Any) -> int | None:
    """A perceptron index, or None when it is absent / the "unset" sentinel."""
    if value is None:
        return None
    index = int(value)
    return index if index >= 0 else None


def mapped_layer_key(node: Any, graph: Any = None) -> MappedLayerKey:
    """The structural mapped-layer key of one softcore instance.

    ``graph`` is optional: it is only consulted to read a shared bank's own
    perceptron provenance (rung 2), and its absence merely drops the key one
    rung, never changes which instances share a key within one graph.
    """
    index = _provenance(getattr(node, "perceptron_index", None))
    if index is not None:
        return MappedLayerKey("perceptron", index)

    bank_id = getattr(node, "weight_bank_id", None)
    if bank_id is not None:
        bank = _bank(graph, bank_id)
        if bank is not None:
            index = _provenance(getattr(bank, "perceptron_index", None))
            if index is not None:
                return MappedLayerKey("perceptron", index)
        window = getattr(node, "weight_row_slice", None)
        return MappedLayerKey(
            "bank",
            int(bank_id),
            None if window is None else (int(window[0]), int(window[1])),
        )

    return MappedLayerKey("core", int(node.id))


def _bank(graph: Any, bank_id: Any) -> Any:
    banks = getattr(graph, "weight_banks", None) if graph is not None else None
    if not banks:
        return None
    return banks.get(bank_id)
