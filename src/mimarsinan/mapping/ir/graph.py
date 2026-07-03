"""IRGraph container and graph-level operations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import numpy as np


def _new_build_token() -> str:
    return uuid.uuid4().hex

from mimarsinan.mapping.ir.types import (
    ComputeOp,
    IRNode,
    IRSource,
    NeuralCore,
    WeightBank,
)


@dataclass
class IRGraph:
    """IR graph of NeuralCores and ComputeOps with shared WeightBanks."""
    nodes: List[IRNode]
    output_sources: np.ndarray
    weight_banks: Dict[int, WeightBank] = field(default_factory=dict)
    layout_softcores: List[Any] = field(default_factory=list)
    # Unique per construction; derived artifacts record it to detect stale cached copies across resumes.
    # None only for legacy unpickled graphs that predate the field (see __getattr__).
    build_token: str | None = field(default_factory=_new_build_token)

    def __getattr__(self, name: str):
        if name == "weight_banks":
            self.weight_banks = {}
            return self.weight_banks
        if name == "layout_softcores":
            self.layout_softcores = []
            return self.layout_softcores
        if name == "build_token":
            self.build_token = None
            return self.build_token
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")

    def get_neural_cores(self) -> List[NeuralCore]:
        """Return all neural core nodes."""
        return [n for n in self.nodes if isinstance(n, NeuralCore)]

    def get_compute_ops(self) -> List[ComputeOp]:
        """Return all compute operation nodes."""
        return [n for n in self.nodes if isinstance(n, ComputeOp)]

    def get_node_by_id(self, node_id: int) -> IRNode | None:
        """Look up a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_weight_bank(self, bank_id: int) -> WeightBank | None:
        """Look up a weight bank by its ID."""
        return self.weight_banks.get(bank_id)

    def resolve_core_matrix(self, core: NeuralCore) -> np.ndarray:
        """Convenience: resolve the effective core matrix for any NeuralCore."""
        return core.get_core_matrix(self)

    def remove_nodes(self, node_ids: Iterable[int]) -> None:
        """Delete nodes and rewire dangling references to OFF (the canonical whole-node deletion API).

        Enforces atomic psum/coalescing-group removal and a non-empty output; raises ValueError on violation.
        Orphaned weight banks are dropped; output_sources shape is preserved (entries rewritten, not removed).
        """
        ids_to_remove = {int(nid) for nid in node_ids}
        if not ids_to_remove:
            return

        existing_ids = {n.id for n in self.nodes}
        unknown = ids_to_remove - existing_ids
        if unknown:
            raise ValueError(
                f"IRGraph.remove_nodes: unknown node ids: {sorted(unknown)} "
                f"(not in self.nodes)"
            )

        self._enforce_atomic_group_removal(
            ids_to_remove, attr="psum_group_id", label="psum",
        )
        self._enforce_atomic_group_removal(
            ids_to_remove, attr="coalescing_group_id", label="coalescing",
        )

        if self.output_sources is not None and self.output_sources.size:
            survives = [
                isinstance(s, IRSource)
                and (s.node_id < 0 or s.node_id not in ids_to_remove)
                for s in self.output_sources.flatten()
            ]
            if not any(survives):
                raise ValueError(
                    "IRGraph.remove_nodes: would empty output_sources "
                    "(every live output target is in the removal set)"
                )

        off_source = IRSource(node_id=-1, index=0)

        def _rewire(arr: np.ndarray) -> np.ndarray:
            flat = arr.flatten()
            for i, src in enumerate(flat):
                if (
                    isinstance(src, IRSource)
                    and src.node_id >= 0
                    and src.node_id in ids_to_remove
                ):
                    flat[i] = IRSource(node_id=-1, index=0)
            return flat.reshape(arr.shape)

        self.nodes = [n for n in self.nodes if n.id not in ids_to_remove]

        for node in self.nodes:
            if hasattr(node, "input_sources") and node.input_sources is not None:
                node.input_sources = _rewire(node.input_sources)

        if self.output_sources is not None and self.output_sources.size:
            self.output_sources = _rewire(self.output_sources)

        self._cleanup_orphan_weight_banks()
        _ = off_source

    def _enforce_atomic_group_removal(
        self,
        ids_to_remove: "set[int]",
        *,
        attr: str,
        label: str,
    ) -> None:
        """Reject removals that split a non-trivial group (>=2 nodes sharing a non-None attr); the whole group must be removed together."""
        groups: Dict[int, List[int]] = {}
        for n in self.nodes:
            gid = getattr(n, attr, None)
            if gid is None:
                continue
            groups.setdefault(int(gid), []).append(n.id)
        for gid, members in groups.items():
            if len(members) < 2:
                continue
            in_set = [m for m in members if m in ids_to_remove]
            if 0 < len(in_set) < len(members):
                missing = sorted(set(members) - set(in_set))
                raise ValueError(
                    f"IRGraph.remove_nodes: cannot split {label} group "
                    f"id={gid}; members {sorted(members)} are atomic. "
                    f"Removing {sorted(in_set)} would leave {missing} "
                    "without their group partners."
                )

    def _cleanup_orphan_weight_banks(self) -> None:
        """Drop weight banks no longer referenced by any NeuralCore."""
        banks = getattr(self, "weight_banks", None)
        if not banks:
            return
        referenced = {
            n.weight_bank_id
            for n in self.nodes
            if isinstance(n, NeuralCore) and n.weight_bank_id is not None
        }
        for bank_id in list(banks.keys()):
            if bank_id not in referenced:
                banks.pop(bank_id, None)

    def validate(self) -> List[str]:
        """Return validation errors (empty if valid)."""
        errors = []
        node_ids = {n.id for n in self.nodes}

        for node in self.nodes:
            for src in node.input_sources.flatten():
                if not isinstance(src, IRSource):
                    continue
                if src.node_id >= 0 and src.node_id not in node_ids:
                    errors.append(
                        f"Node {node.id} ({node.name}) references non-existent node {src.node_id}"
                    )

        for src in self.output_sources.flatten():
            if not isinstance(src, IRSource):
                continue
            if src.node_id >= 0 and src.node_id not in node_ids:
                errors.append(f"Output references non-existent node {src.node_id}")

        for node in self.nodes:
            if isinstance(node, NeuralCore) and node.weight_bank_id is not None:
                if node.weight_bank_id not in self.weight_banks:
                    errors.append(
                        f"NeuralCore {node.id} ({node.name}) references "
                        f"weight_bank_id={node.weight_bank_id} which does not exist"
                    )

        return errors

