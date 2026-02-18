from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec


@dataclass
class LayoutIRMapping:
    """
    Mapping backend that collects *shape-only* neural cores (LayoutSoftCoreSpec) while
    running the model's Mapper graph via map_to_ir().

    It returns IRSource arrays purely as placeholders so downstream mappers can
    reshape/rearrange sources; connectivity is irrelevant for layout metrics.

    After the full graph is built, latency tags and threshold groups are computed
    and assigned to each softcore spec.
    """

    max_axons: int
    max_neurons: int
    allow_axon_tiling: bool = False
    threshold_groups: int = 1
    threshold_seed: int = 0

    def __post_init__(self):
        self.max_axons = int(self.max_axons)
        self.max_neurons = int(self.max_neurons)
        self.allow_axon_tiling = bool(self.allow_axon_tiling)
        self.threshold_groups = max(1, int(self.threshold_groups))
        self.threshold_seed = int(self.threshold_seed)

        self._next_node_id = 0
        self.layout_softcores: List[LayoutSoftCoreSpec] = []

        # Dependency tracking for latency computation
        self._node_input_node_ids: Dict[int, Set[int]] = {}
        self._node_id_to_softcore_idx: Dict[int, int] = {}

    def _alloc_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def _extract_input_node_ids(self, input_sources: Optional[np.ndarray]) -> Set[int]:
        """Extract upstream node IDs from an array of IRSource."""
        ids: Set[int] = set()
        if input_sources is not None:
            for src in np.array(input_sources, dtype=object).flatten():
                if isinstance(src, IRSource) and src.node_id >= 0:
                    ids.add(src.node_id)
        return ids

    def _emit_core(
        self,
        *,
        input_count: int,
        output_count: int,
        name: Optional[str],
        input_sources: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        node_id = self._alloc_node_id()

        # Track dependencies for latency computation
        self._node_input_node_ids[node_id] = self._extract_input_node_ids(input_sources)

        sc_idx = len(self.layout_softcores)
        self._node_id_to_softcore_idx[node_id] = sc_idx

        # Create with placeholder threshold_group_id and latency_tag.
        # These are finalized in _finalize_softcores() after the full graph is known.
        self.layout_softcores.append(
            LayoutSoftCoreSpec(
                input_count=int(input_count),
                output_count=int(output_count),
                threshold_group_id=0,
                latency_tag=None,
                name=name,
            )
        )
        return np.array(
            [IRSource(node_id=node_id, index=i) for i in range(int(output_count))],
            dtype=object,
        )

    # ------------------------------------------------------------------
    # Latency computation
    # ------------------------------------------------------------------

    def _compute_latencies(self) -> Dict[int, int]:
        """
        Compute latency (depth in hops through neural cores) for each node.

        Similar to IRLatency but operates on the layout dependency graph.
        A node's latency is ``max(latency of each dependency) + 1``, or 0 if it
        has no upstream core dependencies.
        """
        memo: Dict[int, int] = {}

        def _get(node_id: int) -> int:
            if node_id in memo:
                return memo[node_id]
            deps = self._node_input_node_ids.get(node_id)
            if not deps:
                memo[node_id] = 0
                return 0
            result = max(_get(d) for d in deps) + 1
            memo[node_id] = result
            return result

        for node_id in self._node_input_node_ids:
            _get(node_id)

        return memo

    def _finalize_softcores(self) -> None:
        """
        Compute latencies and assign threshold groups, then rebuild each
        LayoutSoftCoreSpec with the correct ``latency_tag`` and
        ``threshold_group_id``.

        Threshold groups are assigned via latency-stratified random
        assignment: softcores sharing the same latency are randomly (but
        deterministically) placed into one of N groups.
        """
        latencies = self._compute_latencies()
        rng = random.Random(self.threshold_seed)

        for node_id, sc_idx in self._node_id_to_softcore_idx.items():
            latency = latencies.get(node_id, 0)
            tg = rng.randint(0, max(self.threshold_groups - 1, 0))

            old = self.layout_softcores[sc_idx]
            self.layout_softcores[sc_idx] = LayoutSoftCoreSpec(
                input_count=old.input_count,
                output_count=old.output_count,
                threshold_group_id=int(tg),
                latency_tag=int(latency),
                name=old.name,
            )

    # ------------------------------------------------------------------
    # Public mapping interface (called by Mapper.map_to_ir)
    # ------------------------------------------------------------------

    def add_compute_op(self, *args, **kwargs):
        raise NotImplementedError(
            "LayoutIRMapping does not support ComputeOp yet (layout-only MNIST PoC)."
        )

    def add_neural_core(
        self,
        *,
        input_sources: np.ndarray,
        weights: Any,
        biases: Any = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        out_features = int(getattr(weights, "shape", [0])[0])
        in_count = int(np.array(input_sources, dtype=object).flatten().shape[0])
        if biases is not None:
            in_count += 1  # bias axon (always-on)
        return self._emit_core(
            input_count=in_count,
            output_count=out_features,
            name=name,
            input_sources=input_sources,
        )

    def map_fc(
        self,
        input_tensor_sources: np.ndarray,
        output_shape: np.ndarray,
        fc_weights: Any,
        fc_biases: Any = None,
        activation_scale=None,
        parameter_scale=None,
        input_activation_scale=None,
        name: Optional[str] = None,
    ) -> np.ndarray:
        # Only shapes matter
        out_features = int(getattr(fc_weights, "shape", [0, 0])[0])
        in_features = int(getattr(fc_weights, "shape", [0, 0])[1])

        src_arr = np.array(input_tensor_sources, dtype=object)

        # Support batched FC applications: (in_features, core_count)
        if src_arr.ndim == 2:
            if src_arr.shape[0] != in_features and src_arr.shape[1] == in_features:
                src_arr = src_arr.T
            if src_arr.shape[0] != in_features:
                raise ValueError(
                    f"LayoutIRMapping.map_fc: input sources first dim must match in_features "
                    f"({src_arr.shape} vs in_features={in_features})"
                )

            core_count = int(src_arr.shape[1])
            outs = []
            for i in range(core_count):
                col_sources = np.array(src_arr[:, i], dtype=object).flatten()
                outs.append(
                    self.map_fc(
                        col_sources,
                        np.array([out_features, 1]),
                        fc_weights,
                        fc_biases,
                        activation_scale,
                        parameter_scale,
                        input_activation_scale,
                        name=(f"{name}_col{i}" if name else None),
                    ).flatten()
                )
            out = np.stack(outs, axis=1)
            return out.reshape(tuple(output_shape))

        # Axon tiling check (bias uses 1 axon)
        if self.max_axons is not None and in_features > self.max_axons - 1:
            if not self.allow_axon_tiling:
                raise ValueError(
                    f"FC requires {in_features} axons but max is {self.max_axons - 1} (enable allow_axon_tiling)"
                )
            # For MNIST PoC we keep this unimplemented to avoid producing invalid accumulator cores.
            raise NotImplementedError(
                "allow_axon_tiling layout mapping not supported in LayoutIRMapping (yet)"
            )

        # Output tiling
        if self.max_neurons is not None and out_features > self.max_neurons:
            chunks = []
            start = 0
            while start < out_features:
                end = min(start + self.max_neurons, out_features)
                out_block = end - start
                in_count = int(src_arr.flatten().shape[0]) + (1 if fc_biases is not None else 0)
                chunks.append(
                    self._emit_core(
                        input_count=in_count,
                        output_count=out_block,
                        name=name,
                        input_sources=src_arr,
                    )
                )
                start = end
            return np.concatenate(chunks).reshape(tuple(output_shape))

        # Simple single core
        in_count = int(src_arr.flatten().shape[0]) + (1 if fc_biases is not None else 0)
        return self._emit_core(
            input_count=in_count,
            output_count=out_features,
            name=name,
            input_sources=src_arr,
        ).reshape(tuple(output_shape))

    def collect_layout_softcores(self, model_representation) -> List[LayoutSoftCoreSpec]:
        """
        Execute map_to_ir over the mapper graph, collecting LayoutSoftCoreSpec instances.

        After the graph is fully built the latency tags and threshold groups are
        computed and assigned.
        """
        _ = model_representation.map_to_ir(self)
        self._finalize_softcores()
        return list(self.layout_softcores)
