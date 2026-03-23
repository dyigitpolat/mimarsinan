from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

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
    threshold_groups: int = 1
    threshold_seed: int = 0
    pruning_fraction: float = 0.0
    allow_core_coalescing: bool = False
    hardware_bias: bool = False

    def __post_init__(self):
        self.max_axons = int(self.max_axons)
        self.max_neurons = int(self.max_neurons)
        self.threshold_groups = max(1, int(self.threshold_groups))
        self.threshold_seed = int(self.threshold_seed)

        self._next_node_id = 0
        self.layout_softcores: List[LayoutSoftCoreSpec] = []
        self.host_side_segment_count: int = 0
        self.layout_preview: Dict[str, Any] | None = None

        # Dependency tracking for latency computation
        self._node_input_node_ids: Dict[int, Set[int]] = {}
        self._node_id_to_softcore_idx: Dict[int, int] = {}
        self._node_is_neural: Dict[int, bool] = {}

        # Weight bank registry for conv-style shared-weight cores
        self._next_bank_id: int = 0
        # bank_id -> (in_features_with_bias, out_features)
        self._layout_weight_banks: Dict[int, Tuple[int, int]] = {}

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
        self._node_is_neural[node_id] = True

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
        Compute latency (depth in hops through neural segments) for each node.

        Similar to IRLatency but operates on the layout dependency graph.
        Neural nodes increase latency by 1 over the deepest dependency.
        ComputeOp nodes preserve the upstream latency, so host-side barriers
        do not collapse distinct neural segments into one depth.
        """
        memo: Dict[int, int] = {}

        def _get(node_id: int) -> int:
            if node_id in memo:
                return memo[node_id]
            deps = self._node_input_node_ids.get(node_id)
            if not deps:
                memo[node_id] = 0
                return 0
            max_upstream = max(_get(d) for d in deps)
            result = max_upstream + (1 if self._node_is_neural.get(node_id, False) else 0)
            memo[node_id] = result
            return result

        for node_id in self._node_input_node_ids:
            _get(node_id)

        return memo

    def _compute_segment_ids(self) -> Dict[int, int]:
        """
        Compute neural-segment IDs for each node.

        A neural segment is a maximal run of neural nodes connected without an
        intervening ComputeOp. ComputeOps carry forward the latest neural
        segment index, and the next downstream neural node starts a new segment.
        """
        memo: Dict[int, int] = {}

        def _get(node_id: int) -> int:
            if node_id in memo:
                return memo[node_id]

            deps = self._node_input_node_ids.get(node_id)
            is_neural = self._node_is_neural.get(node_id, False)
            if not deps:
                memo[node_id] = 0 if is_neural else -1
                return memo[node_id]

            upstream_segments = [_get(d) for d in deps]
            if is_neural:
                has_compute_dep = any(not self._node_is_neural.get(d, False) for d in deps)
                memo[node_id] = max(upstream_segments) + 1 if has_compute_dep else max(upstream_segments)
            else:
                memo[node_id] = max(upstream_segments)
            return memo[node_id]

        for node_id in self._node_input_node_ids:
            _get(node_id)

        return memo

    def _compute_host_side_segment_count(self, segment_ids: Dict[int, int]) -> int:
        """Count maximal host-side (compute-only) segments in the layout graph."""
        host_segments = {
            int(segment_ids[node_id] + 1)
            for node_id, is_neural in self._node_is_neural.items()
            if not is_neural and node_id in segment_ids
        }
        return len(host_segments)

    def _build_layout_preview(
        self,
        segment_ids: Dict[int, int],
        latencies: Dict[int, int],
    ) -> Dict[str, Any]:
        """Build a compact layout-only flow summary for the wizard miniview."""
        neural_latency_tags = sorted({
            int(sc.latency_tag) for sc in self.layout_softcores if sc.latency_tag is not None
        })
        latency_to_index = {lat: idx for idx, lat in enumerate(neural_latency_tags)}
        min_neural_latency = neural_latency_tags[0] if neural_latency_tags else 0

        host_counts: Dict[int, int] = {}
        for node_id, is_neural in self._node_is_neural.items():
            if is_neural or node_id not in latencies:
                continue
            slot = max(0, int(latencies[node_id] - min_neural_latency + 1))
            host_counts[slot] = host_counts.get(slot, 0) + 1

        neural_summary: Dict[int, Dict[str, Any]] = {}
        for sc in self.layout_softcores:
            if sc.latency_tag is None:
                continue
            lat_tag = int(sc.latency_tag)
            info = neural_summary.setdefault(lat_tag, {
                "latency_tag": lat_tag,
                "latency_group_index": latency_to_index.get(lat_tag, 0),
                "softcore_count": 0,
                "segment_ids": set(),
            })
            info["softcore_count"] += 1
            if sc.segment_id is not None:
                info["segment_ids"].add(int(sc.segment_id))

        for info in neural_summary.values():
            seg_ids = sorted(info.pop("segment_ids"))
            info["segment_count"] = len(seg_ids)
            info["segment_ids"] = seg_ids

        neural_groups = [neural_summary[k] for k in sorted(neural_summary)]
        host_segments = [
            {"slot": slot, "compute_op_count": host_counts[slot]}
            for slot in sorted(host_counts)
        ]

        flow: List[Dict[str, Any]] = [{"kind": "input"}]
        max_slot = max(
            [len(neural_groups)] + list(host_counts.keys())
        ) if (neural_groups or host_counts) else 0
        for slot in range(max_slot + 1):
            if slot in host_counts:
                flow.append({
                    "kind": "host",
                    "slot": slot,
                    "compute_op_count": host_counts[slot],
                })
            if slot < len(neural_groups):
                group = next((g for g in neural_groups if g["latency_group_index"] == slot), None)
                if group is not None:
                    flow.append({
                        "kind": "neural",
                        "latency_group_index": group["latency_group_index"],
                        "latency_tag": group["latency_tag"],
                        "softcore_count": group["softcore_count"],
                        "segment_count": group["segment_count"],
                    })
        flow.append({"kind": "output"})

        return {
            "neural_segments": [
                {
                    "segment_id": int(seg_id),
                    "softcore_count": sum(1 for sc in self.layout_softcores if sc.segment_id == seg_id),
                    "latency_group_count": len({
                        int(sc.latency_tag) for sc in self.layout_softcores
                        if sc.segment_id == seg_id and sc.latency_tag is not None
                    }),
                }
                for seg_id in sorted({
                    int(sc.segment_id) for sc in self.layout_softcores if sc.segment_id is not None
                })
            ],
            "latency_groups": neural_groups,
            "host_segments": host_segments,
            "flow": flow,
        }

    def _finalize_softcores(self) -> None:
        """
        Compute latencies / segment IDs and assign threshold groups, then rebuild
        each LayoutSoftCoreSpec with the correct ``latency_tag``, ``segment_id``, and
        ``threshold_group_id``.

        Threshold groups are assigned via latency-stratified random
        assignment: softcores sharing the same latency are randomly (but
        deterministically) placed into one of N groups.
        """
        latencies = self._compute_latencies()
        segment_ids = self._compute_segment_ids()
        self.host_side_segment_count = self._compute_host_side_segment_count(segment_ids)
        rng = random.Random(self.threshold_seed)

        for node_id, sc_idx in self._node_id_to_softcore_idx.items():
            latency = latencies.get(node_id, 0)
            segment_id = segment_ids.get(node_id, 0)
            tg = rng.randint(0, max(self.threshold_groups - 1, 0))

            old = self.layout_softcores[sc_idx]
            self.layout_softcores[sc_idx] = LayoutSoftCoreSpec(
                input_count=old.input_count,
                output_count=old.output_count,
                threshold_group_id=int(tg),
                latency_tag=int(latency),
                segment_id=int(segment_id),
                name=old.name,
            )

        self.layout_preview = self._build_layout_preview(segment_ids, latencies)

    # ------------------------------------------------------------------
    # Public mapping interface (called by Mapper.map_to_ir)
    # ------------------------------------------------------------------

    def add_compute_op(
        self,
        input_sources: np.ndarray,
        op_type: str,
        params=None,
        input_shape=None,
        output_shape=None,
        name: str | None = None,
    ) -> np.ndarray:
        """
        ComputeOps don't produce layout softcores (they execute on the host).
        We still need to return correctly-shaped placeholder sources and preserve
        dependency metadata so downstream NeuralCore mappers can be
        layout-estimated with the correct segment structure.
        """
        if output_shape is not None:
            output_size = 1
            for d in output_shape:
                output_size *= d
        else:
            output_size = int(np.array(input_sources, dtype=object).flatten().shape[0])

        node_id = self._alloc_node_id()
        self._node_input_node_ids[node_id] = self._extract_input_node_ids(input_sources)
        self._node_is_neural[node_id] = False
        output_sources = np.array([
            IRSource(node_id=node_id, index=i) for i in range(output_size)
        ])

        if output_shape is not None:
            output_sources = output_sources.reshape(output_shape)

        return output_sources

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
        normalization_type: Optional[str] = None,
        activation_type: Optional[str] = None,
        **kwargs: Any,
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

        # Axon tiling: mirror IRMapping.map_fc behaviour for wide layers.
        _bias_slots = 1 if fc_biases is not None else 0
        if self.max_axons is not None and in_features + _bias_slots > self.max_axons:
            if self.allow_core_coalescing:
                return self._map_fc_coalescing_layout(
                    src_arr, output_shape, in_features, out_features,
                    _bias_slots, name,
                )
            else:
                return self._map_fc_psum_layout(
                    src_arr, output_shape, in_features, out_features,
                    _bias_slots, name,
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
        )

    def _map_fc_coalescing_layout(
        self,
        src_arr: np.ndarray,
        output_shape: np.ndarray,
        in_features: int,
        out_features: int,
        bias_slots: int,
        name: Optional[str],
    ) -> np.ndarray:
        """Layout-level coalescing for wide FC: one wide softcore per output tile.

        Mirrors IRMapping.map_fc coalescing path — the core keeps its full
        input width and hardware packing fuses physical cores as needed.
        """
        in_count = in_features + bias_slots
        if self.max_neurons is not None and out_features > self.max_neurons:
            chunks = []
            start = 0
            while start < out_features:
                end = min(start + self.max_neurons, out_features)
                chunks.append(self._emit_core(
                    input_count=in_count,
                    output_count=end - start,
                    name=name,
                    input_sources=src_arr,
                ))
                start = end
            return np.concatenate(chunks).reshape(tuple(output_shape))
        return self._emit_core(
            input_count=in_count,
            output_count=out_features,
            name=name,
            input_sources=src_arr,
        )

    def _map_fc_psum_layout(
        self,
        src_arr: np.ndarray,
        output_shape: np.ndarray,
        in_features: int,
        out_features: int,
        bias_slots: int,
        name: Optional[str],
    ) -> np.ndarray:
        """Layout-level psum decomposition for wide FC.

        Emits the same number and shape of softcores as IRMapping._map_fc_with_psum
        (2 * tile_count partials + 1 accumulator per output block).
        """
        max_axons = int(self.max_axons)
        max_neurons = int(self.max_neurons) if self.max_neurons is not None else out_features

        tile_count = math.ceil(in_features / max_axons)
        accum_bias_axons = bias_slots if not self.hardware_bias else 0
        max_out_by_accum = (max_axons - accum_bias_axons) // (2 * tile_count)
        if max_out_by_accum <= 0:
            raise ValueError(
                f"Cannot build psum accumulator: tile_count={tile_count} "
                f"requires at least {2 * tile_count + accum_bias_axons} axons, "
                f"but max_axons={max_axons}."
            )
        out_block_size = min(max_neurons, max_out_by_accum)

        all_output_sources: list[np.ndarray] = []
        a = 0
        while a < out_features:
            b = min(out_features, a + out_block_size)
            block = b - a

            for t_idx in range(tile_count):
                t_start = t_idx * max_axons
                t_end = min(in_features, t_start + max_axons)
                tile_width = t_end - t_start
                tile_src = src_arr.flatten()[t_start:t_end] if src_arr.size > t_start else src_arr.flatten()
                # pos partial
                self._emit_core(
                    input_count=tile_width,
                    output_count=block,
                    name=f"{name}_psum_pos_t{t_idx}_o{a}" if name else None,
                    input_sources=tile_src,
                )
                # neg partial
                self._emit_core(
                    input_count=tile_width,
                    output_count=block,
                    name=f"{name}_psum_neg_t{t_idx}_o{a}" if name else None,
                    input_sources=tile_src,
                )

            accum_axons = 2 * tile_count * block + accum_bias_axons
            accum_out = self._emit_core(
                input_count=accum_axons,
                output_count=block,
                name=f"{name}_psum_accum_o{a}" if name else None,
                input_sources=src_arr,
            )
            all_output_sources.append(accum_out)
            a = b

        return np.concatenate(all_output_sources).reshape(tuple(output_shape))

    def register_weight_bank(
        self,
        weights: Any,
        biases: Any = None,
        **kwargs,
    ) -> int:
        """Register a shared weight bank (shape-only) and return its bank ID.

        Used by Conv2DPerceptronMapper._map_to_ir so that layout estimation
        works for convolutional models without requiring actual weight tensors.
        """
        bank_id = self._next_bank_id
        self._next_bank_id += 1

        w_shape = getattr(weights, "shape", None)
        if w_shape is not None:
            out_features = int(w_shape[0])
            in_features = int(w_shape[1])
        else:
            out_features = 1
            in_features = 1

        has_bias = biases is not None
        in_features_with_bias = in_features + (1 if has_bias else 0)

        self._layout_weight_banks[bank_id] = (in_features_with_bias, out_features)
        return bank_id

    def add_shared_neural_core(
        self,
        *,
        input_sources: Any,
        weight_bank_id: int,
        has_bias: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """Add a shared-weight neural core (shape-only) for conv positions.

        Each call corresponds to one spatial position in a conv layer.
        """
        bank_shape = self._layout_weight_banks.get(weight_bank_id)
        if bank_shape is None:
            raise ValueError(f"Unknown weight_bank_id={weight_bank_id}")
        in_features_with_bias, out_features = bank_shape

        src_arr = np.array(input_sources, dtype=object).flatten()
        in_count = int(src_arr.shape[0]) + (1 if has_bias else 0)

        return self._emit_core(
            input_count=in_count,
            output_count=out_features,
            name=name,
            input_sources=src_arr,
        )

    def collect_layout_softcores(self, model_representation) -> List[LayoutSoftCoreSpec]:
        """
        Execute map_to_ir over the mapper graph, collecting LayoutSoftCoreSpec instances.

        After the graph is fully built the latency tags and threshold groups are
        computed and assigned.

        If ``pruning_fraction > 0``, applies 80% of the user-provided fraction
        as a random dimension reduction to each softcore to estimate post-pruning
        core sizes (overestimating actual sizes for safety).
        """
        _ = model_representation.map_to_ir(self)
        self._finalize_softcores()

        softcores = list(self.layout_softcores)

        # Apply pruning estimation
        if self.pruning_fraction > 0:
            effective = self.pruning_fraction * 0.8
            rng = random.Random(self.threshold_seed + 7919)  # distinct from threshold RNG
            pruned = []
            for sc in softcores:
                in_reduce = int(math.floor(sc.input_count * effective))
                out_reduce = int(math.floor(sc.output_count * effective))
                new_in = max(1, sc.input_count - in_reduce)
                new_out = max(1, sc.output_count - out_reduce)
                pruned.append(LayoutSoftCoreSpec(
                    input_count=new_in,
                    output_count=new_out,
                    threshold_group_id=sc.threshold_group_id,
                    latency_tag=sc.latency_tag,
                    name=sc.name,
                ))
            softcores = pruned

        return softcores
