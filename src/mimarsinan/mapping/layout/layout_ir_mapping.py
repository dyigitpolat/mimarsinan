from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.mapping_structure import (
    compute_core_input_count,
    compute_fc_tiling_mode,
    compute_psum_params,
)


@dataclass
class LayoutIRMapping:
    """Shape-only mapping backend — the single source of truth for all
    softcore emission decisions (tiling mode, psum decomposition, coalescing,
    bias-axon counting, shared-bank wiring).

    Both the wizard/architecture-search flow and the real ``IRMapping`` build
    on top of this class.  Shape-only callers use it directly; ``IRMapping``
    subclasses it and overrides the emission hooks (``add_neural_core``,
    ``add_shared_neural_core``, ``register_weight_bank``, ``add_compute_op``)
    to additionally attach weight material and build an ``IRGraph``.

    Threshold groups are assigned from ``perceptron_index``: all softcores
    produced by a single perceptron (output-tiling, psum fragments, conv
    positions) share one integer group id, matching the real packer's
    behavior for shared-weight cores (they end up with identical quantization
    scales and thus identical float thresholds within 10% tolerance).

    Softcores without a ``perceptron_index`` (e.g. synthesised accumulator
    cores) fall back to a unique per-node id so they are never misjudged as
    "shareable" with unrelated cores.
    """

    max_axons: Optional[int]
    max_neurons: Optional[int]
    allow_coalescing: bool = False
    hardware_bias: bool = False

    def __post_init__(self):
        self.max_axons = int(self.max_axons) if self.max_axons is not None else None
        self.max_neurons = int(self.max_neurons) if self.max_neurons is not None else None
        self.allow_coalescing = bool(self.allow_coalescing)
        self.hardware_bias = bool(self.hardware_bias)

        self._next_node_id = 0
        self._coalescing_group_counter = 0
        self._psum_group_counter = 0
        self._next_bank_id = 0

        self.layout_softcores: List[LayoutSoftCoreSpec] = []
        self.output_sources: np.ndarray = np.array([])
        self.host_side_segment_count: int = 0
        self.layout_preview: Dict[str, Any] | None = None

        # Dependency tracking for latency / segment computation
        self._node_input_node_ids: Dict[int, Set[int]] = {}
        self._node_id_to_softcore_idx: Dict[int, int] = {}
        self._node_is_neural: Dict[int, bool] = {}
        self._sc_idx_to_perceptron_index: Dict[int, Optional[int]] = {}

        # Shared weight-bank shape registry
        # bank_id -> (in_features_with_bias, out_features)
        self._layout_weight_banks: Dict[int, Tuple[int, int]] = {}
        self._sc_idx_to_bank_id: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Node id allocation
    # ------------------------------------------------------------------

    def _alloc_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    # ------------------------------------------------------------------
    # Dependency / shape helpers
    # ------------------------------------------------------------------

    def _extract_input_node_ids(self, input_sources: Optional[np.ndarray]) -> Set[int]:
        ids: Set[int] = set()
        if input_sources is not None:
            for src in np.array(input_sources, dtype=object).flatten():
                if isinstance(src, IRSource) and src.node_id >= 0:
                    ids.add(src.node_id)
        return ids

    def _emit_softcore_record(
        self,
        *,
        node_id: int,
        input_count: int,
        output_count: int,
        name: Optional[str],
        input_sources: Optional[np.ndarray],
        perceptron_index: Optional[int],
    ) -> np.ndarray:
        """Record a softcore shape under ``node_id`` and return output
        ``IRSource``s (1-D)."""
        self._node_input_node_ids[node_id] = self._extract_input_node_ids(input_sources)
        self._node_is_neural[node_id] = True

        sc_idx = len(self.layout_softcores)
        self._node_id_to_softcore_idx[node_id] = sc_idx
        self._sc_idx_to_perceptron_index[sc_idx] = (
            int(perceptron_index) if perceptron_index is not None else None
        )

        # threshold_group_id / latency_tag / segment_id are finalised later
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
    # Public mapping protocol (called by Mapper.map_to_ir)
    # ------------------------------------------------------------------

    def map(self, model_representation) -> np.ndarray:
        """Run ``map_to_ir`` over the mapper graph and finalise metadata.

        Returns the graph's output sources.  ``IRMapping`` subclasses this to
        wrap the result in an ``IRGraph``.
        """
        output_sources = model_representation.map_to_ir(self)
        self.output_sources = output_sources
        self._finalize_softcores()
        return output_sources

    def collect_layout_softcores(self, model_representation) -> List[LayoutSoftCoreSpec]:
        """Back-compat entry point for shape-only callers (wizard / search)."""
        self.map(model_representation)
        return list(self.layout_softcores)

    def add_compute_op(
        self,
        input_sources: np.ndarray,
        op_type: str,
        params=None,
        input_shape=None,
        output_shape=None,
        name: Optional[str] = None,
    ) -> np.ndarray:
        """Record a compute-op's dependency structure and return placeholder
        output sources.  Shape-only: no ComputeOp node is built."""
        if output_shape is not None:
            output_size = 1
            for d in output_shape:
                output_size *= d
        else:
            output_size = int(np.array(input_sources, dtype=object).flatten().shape[0])

        node_id = self._alloc_node_id()
        self._node_input_node_ids[node_id] = self._extract_input_node_ids(input_sources)
        self._node_is_neural[node_id] = False

        output = np.array(
            [IRSource(node_id=node_id, index=i) for i in range(output_size)],
            dtype=object,
        )
        if output_shape is not None:
            output = output.reshape(output_shape)
        return output

    def register_weight_bank(
        self,
        weights: Any,
        biases: Any = None,
        activation_scale=None,
        parameter_scale=None,
        input_activation_scale=None,
        perceptron_index: Optional[int] = None,
    ) -> int:
        """Register a shared weight bank (shape only) and return its ID."""
        bank_id = self._next_bank_id
        self._next_bank_id += 1

        w_shape = getattr(weights, "shape", None)
        if w_shape is not None:
            out_features = int(w_shape[0])
            in_features = int(w_shape[1]) if len(w_shape) > 1 else 1
        else:
            out_features = 1
            in_features = 1

        has_bias = biases is not None
        in_features_with_bias = compute_core_input_count(
            in_features, has_bias, self.hardware_bias
        )
        self._layout_weight_banks[bank_id] = (in_features_with_bias, out_features)
        return bank_id

    def add_neural_core(
        self,
        *,
        input_sources: np.ndarray,
        weights: Any,
        biases: Any = None,
        activation_scale: Any = None,
        parameter_scale: Any = None,
        input_activation_scale: Any = None,
        name: Optional[str] = None,
        normalization_type: Optional[str] = None,
        activation_type: Optional[str] = None,
        perceptron_index: Optional[int] = None,
        perceptron_input_slice: Optional[Tuple[int, int]] = None,
        perceptron_output_slice: Optional[Tuple[int, int]] = None,
        psum_group_id: Optional[int] = None,
        psum_role: Optional[str] = None,
        coalescing_group_id: Optional[int] = None,
        coalescing_role: Optional[str] = None,
    ) -> np.ndarray:
        """Emit a single neural softcore (owned weights).

        Base implementation records shape only.  ``IRMapping`` overrides to
        additionally construct a concrete ``NeuralCore`` node.
        """
        src_arr = np.array(input_sources, dtype=object).flatten()
        in_features = int(src_arr.shape[0])
        out_features = int(getattr(weights, "shape", [0])[0])
        has_bias = biases is not None
        in_count = compute_core_input_count(
            in_features, has_bias, self.hardware_bias
        )

        node_id = self._alloc_node_id()
        return self._emit_softcore_record(
            node_id=node_id,
            input_count=in_count,
            output_count=out_features,
            name=name,
            input_sources=src_arr,
            perceptron_index=perceptron_index,
        )

    def add_shared_neural_core(
        self,
        *,
        input_sources: Any,
        weight_bank_id: int,
        has_bias: bool = True,
        weight_row_slice: Optional[Tuple[int, int]] = None,
        name: Optional[str] = None,
        normalization_type: Optional[str] = None,
        activation_type: Optional[str] = None,
        perceptron_index: Optional[int] = None,
        psum_group_id: Optional[int] = None,
        psum_role: Optional[str] = None,
        coalescing_group_id: Optional[int] = None,
        coalescing_role: Optional[str] = None,
    ) -> np.ndarray:
        """Emit a bank-backed neural softcore (one conv position).

        Base implementation records shape only.  ``IRMapping`` overrides to
        also build the concrete ``NeuralCore`` referencing the bank.
        """
        bank_shape = self._layout_weight_banks.get(weight_bank_id)
        if bank_shape is None:
            raise ValueError(f"Unknown weight_bank_id={weight_bank_id}")
        _in_features_with_bias, bank_out_features = bank_shape

        src_arr = np.array(input_sources, dtype=object).flatten()
        in_count = compute_core_input_count(
            int(src_arr.shape[0]), has_bias, self.hardware_bias
        )

        if weight_row_slice is not None:
            out_features = weight_row_slice[1] - weight_row_slice[0]
        else:
            out_features = bank_out_features

        node_id = self._alloc_node_id()
        sc_idx = len(self.layout_softcores)
        result = self._emit_softcore_record(
            node_id=node_id,
            input_count=in_count,
            output_count=out_features,
            name=name,
            input_sources=src_arr,
            perceptron_index=perceptron_index,
        )
        self._sc_idx_to_bank_id[sc_idx] = weight_bank_id
        return result

    # ------------------------------------------------------------------
    # Shared FC tiling dispatch
    # ------------------------------------------------------------------

    def map_fc(
        self,
        input_tensor_sources: np.ndarray,
        output_shape: np.ndarray,
        fc_weights: Any,
        fc_biases: Any = None,
        activation_scale: Any = None,
        parameter_scale: Any = None,
        input_activation_scale: Any = None,
        name: Optional[str] = None,
        normalization_type: Optional[str] = None,
        activation_type: Optional[str] = None,
        perceptron_index: Optional[int] = None,
        psum_group_id: Optional[int] = None,
        psum_role: Optional[str] = None,
        coalescing_group_id: Optional[int] = None,
        coalescing_role: Optional[str] = None,
    ) -> np.ndarray:
        """Decide tiling mode and dispatch to ``add_neural_core`` (or the
        psum / output-tiled helpers).  All structural decisions live here."""
        out_features = int(getattr(fc_weights, "shape", [0, 0])[0])
        in_features = int(getattr(fc_weights, "shape", [0, 0])[1])

        src_arr = np.array(input_tensor_sources, dtype=object)

        # Batched FC: (in_features, core_count) -> dispatch column-by-column.
        if src_arr.ndim == 2:
            if src_arr.shape[0] != in_features and src_arr.shape[1] == in_features:
                src_arr = src_arr.T
            if src_arr.shape[0] != in_features:
                raise ValueError(
                    f"map_fc: input sources first dim must match in_features "
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
                        normalization_type=normalization_type,
                        activation_type=activation_type,
                        perceptron_index=perceptron_index,
                        psum_group_id=psum_group_id,
                        psum_role=psum_role,
                        coalescing_group_id=coalescing_group_id,
                        coalescing_role=coalescing_role,
                    ).flatten()
                )
            out = np.stack(outs, axis=1)
            return out.reshape(tuple(output_shape))

        has_bias = fc_biases is not None
        mode = compute_fc_tiling_mode(
            in_features, out_features,
            self.max_axons, self.max_neurons,
            has_bias, self.hardware_bias, self.allow_coalescing,
        )

        if mode == "psum":
            return self._map_fc_with_psum(
                input_sources=src_arr.flatten(),
                fc_weights=fc_weights,
                fc_biases=fc_biases,
                activation_scale=activation_scale,
                parameter_scale=parameter_scale,
                input_activation_scale=input_activation_scale,
                name=name,
                normalization_type=normalization_type,
                activation_type=activation_type,
                perceptron_index=perceptron_index,
            )

        if mode == "coalescing" and coalescing_group_id is None:
            coalescing_group_id = self._coalescing_group_counter
            self._coalescing_group_counter += 1
            coalescing_role = "master"

        wide_and_output_tiled = (
            mode == "coalescing"
            and self.max_neurons is not None
            and out_features > self.max_neurons
        )
        if mode == "output_tiled" or wide_and_output_tiled:
            return self._map_fc_output_tiled(
                src_arr=src_arr,
                fc_weights=fc_weights,
                fc_biases=fc_biases,
                activation_scale=activation_scale,
                parameter_scale=parameter_scale,
                input_activation_scale=input_activation_scale,
                name=name,
                normalization_type=normalization_type,
                activation_type=activation_type,
                perceptron_index=perceptron_index,
                psum_group_id=psum_group_id,
                psum_role=psum_role,
                coalescing_group_id=coalescing_group_id,
                coalescing_role=coalescing_role,
            )

        # "single" or non-output-tiled "coalescing"
        fc_input = src_arr.T if src_arr.ndim > 1 else src_arr
        return self.add_neural_core(
            input_sources=fc_input,
            weights=fc_weights,
            biases=fc_biases,
            activation_scale=activation_scale,
            parameter_scale=parameter_scale,
            input_activation_scale=input_activation_scale,
            name=name,
            normalization_type=normalization_type,
            activation_type=activation_type,
            perceptron_index=perceptron_index,
            psum_group_id=psum_group_id,
            psum_role=psum_role,
            coalescing_group_id=coalescing_group_id,
            coalescing_role=coalescing_role,
        )

    def _map_fc_output_tiled(
        self,
        *,
        src_arr: np.ndarray,
        fc_weights: Any,
        fc_biases: Any,
        activation_scale: Any,
        parameter_scale: Any,
        input_activation_scale: Any,
        name: Optional[str],
        normalization_type: Optional[str],
        activation_type: Optional[str],
        perceptron_index: Optional[int],
        psum_group_id: Optional[int],
        psum_role: Optional[str],
        coalescing_group_id: Optional[int],
        coalescing_role: Optional[str],
    ) -> np.ndarray:
        out_features = int(getattr(fc_weights, "shape", [0, 0])[0])
        chunk_size = int(self.max_neurons)

        output_sources_list = []
        start = 0
        while start < out_features:
            end = min(start + chunk_size, out_features)
            tile_weights = fc_weights[start:end, :] if fc_weights is not None else None
            tile_biases = fc_biases[start:end] if fc_biases is not None else None

            tile_sources = self.add_neural_core(
                input_sources=src_arr.flatten(),
                weights=tile_weights,
                biases=tile_biases,
                activation_scale=activation_scale,
                parameter_scale=parameter_scale,
                input_activation_scale=input_activation_scale,
                name=(f"{name}_tile_{start}_{end}" if name else None),
                normalization_type=normalization_type,
                activation_type=activation_type,
                perceptron_index=perceptron_index,
                perceptron_output_slice=(start, end),
                psum_group_id=psum_group_id,
                psum_role=psum_role,
                coalescing_group_id=coalescing_group_id,
                coalescing_role=coalescing_role,
            )
            output_sources_list.append(tile_sources)
            start = end
        return np.concatenate(output_sources_list)

    def _map_fc_with_psum(
        self,
        *,
        input_sources: np.ndarray,
        fc_weights: Any,
        fc_biases: Any,
        activation_scale: Any,
        parameter_scale: Any,
        input_activation_scale: Any,
        name: Optional[str],
        normalization_type: Optional[str],
        activation_type: Optional[str],
        perceptron_index: Optional[int],
    ) -> np.ndarray:
        out_features = int(getattr(fc_weights, "shape", [0, 0])[0])
        in_features = int(getattr(fc_weights, "shape", [0, 0])[1])

        pp = compute_psum_params(
            in_features, out_features,
            int(self.max_axons), self.max_neurons,
            fc_biases is not None, self.hardware_bias,
        )

        src_arr = np.array(input_sources, dtype=object).flatten()
        group_id = self._psum_group_counter
        self._psum_group_counter += 1

        all_output_sources: list[np.ndarray] = []
        a = 0
        while a < out_features:
            b = min(out_features, a + pp.out_block_size)
            block = b - a
            # NB: un-clamped slice.  Subclasses that need real weights
            # reconstruct pos/neg via psum_role at emission time.
            w_block = fc_weights[a:b, :] if fc_weights is not None else None
            b_block = fc_biases[a:b] if fc_biases is not None else None

            partial_pos_sources: list[np.ndarray] = []
            partial_neg_sources: list[np.ndarray] = []
            for t_idx, (ta, tb) in enumerate(pp.tile_slices):
                w_tile = w_block[:, ta:tb] if w_block is not None else None
                tile_src = src_arr[ta:tb]

                pos_out = self.add_neural_core(
                    input_sources=tile_src,
                    weights=w_tile,
                    biases=None,
                    activation_scale=activation_scale,
                    parameter_scale=parameter_scale,
                    input_activation_scale=input_activation_scale,
                    name=(f"{name}_psum_pos_g{group_id}_t{t_idx}_o{a}_{b}"
                          if name else None),
                    normalization_type=normalization_type,
                    activation_type=activation_type,
                    perceptron_index=perceptron_index,
                    perceptron_input_slice=(ta, tb),
                    perceptron_output_slice=(a, b),
                    psum_group_id=group_id,
                    psum_role="partial_pos",
                )
                neg_out = self.add_neural_core(
                    input_sources=tile_src,
                    weights=w_tile,
                    biases=None,
                    activation_scale=activation_scale,
                    parameter_scale=parameter_scale,
                    input_activation_scale=input_activation_scale,
                    name=(f"{name}_psum_neg_g{group_id}_t{t_idx}_o{a}_{b}"
                          if name else None),
                    normalization_type=normalization_type,
                    activation_type=activation_type,
                    perceptron_index=perceptron_index,
                    perceptron_input_slice=(ta, tb),
                    perceptron_output_slice=(a, b),
                    psum_group_id=group_id,
                    psum_role="partial_neg",
                )
                partial_pos_sources.append(pos_out)
                partial_neg_sources.append(neg_out)

            # Accumulator weight matrix is small and purely structural
            # (identity-like).  Build it at the base level so subclasses
            # can feed it directly into a real NeuralCore.
            acc_input_list: list[IRSource] = []
            for t_idx in range(pp.tile_count):
                for n in range(block):
                    acc_input_list.append(partial_pos_sources[t_idx][n])
            for t_idx in range(pp.tile_count):
                for n in range(block):
                    acc_input_list.append(partial_neg_sources[t_idx][n])

            ps_val = (
                parameter_scale.item()
                if hasattr(parameter_scale, "item")
                else float(parameter_scale)
                if parameter_scale is not None
                else 1.0
            )
            unit = 1.0 / float(ps_val) if ps_val else 1.0
            acc_axons = 2 * pp.tile_count * block
            acc_w = np.zeros((block, acc_axons), dtype=float)
            pos_off = 0
            neg_off = pp.tile_count * block
            for t_idx in range(pp.tile_count):
                for n in range(block):
                    acc_w[n, pos_off + t_idx * block + n] = unit
                    acc_w[n, neg_off + t_idx * block + n] = -unit

            acc_out = self.add_neural_core(
                input_sources=np.array(acc_input_list, dtype=object),
                weights=acc_w,
                biases=b_block,
                activation_scale=activation_scale,
                parameter_scale=parameter_scale,
                input_activation_scale=input_activation_scale,
                name=(f"{name}_psum_accum_g{group_id}_o{a}_{b}" if name else None),
                normalization_type=normalization_type,
                activation_type=activation_type,
                perceptron_index=perceptron_index,
                perceptron_output_slice=(a, b),
                psum_group_id=group_id,
                psum_role="accum",
            )
            all_output_sources.append(acc_out)
            a = b

        return np.concatenate(all_output_sources)

    # ------------------------------------------------------------------
    # Finalisation: latencies, segments, threshold groups
    # ------------------------------------------------------------------

    def _compute_latencies(self) -> Dict[int, int]:
        memo: Dict[int, int] = {}

        def _get(node_id: int) -> int:
            if node_id in memo:
                return memo[node_id]
            deps = self._node_input_node_ids.get(node_id)
            if not deps:
                memo[node_id] = 0
                return 0
            max_upstream = max(_get(d) for d in deps)
            result = max_upstream + (
                1 if self._node_is_neural.get(node_id, False) else 0
            )
            memo[node_id] = result
            return result

        for node_id in self._node_input_node_ids:
            _get(node_id)
        return memo

    def _compute_segment_ids(self) -> Dict[int, int]:
        memo: Dict[int, int] = {}

        def _get(node_id: int) -> int:
            if node_id in memo:
                return memo[node_id]
            deps = self._node_input_node_ids.get(node_id)
            is_neural = self._node_is_neural.get(node_id, False)
            if not deps:
                memo[node_id] = 0 if is_neural else -1
                return memo[node_id]
            upstream = [_get(d) for d in deps]
            if is_neural:
                has_compute_dep = any(
                    not self._node_is_neural.get(d, False) for d in deps
                )
                memo[node_id] = max(upstream) + 1 if has_compute_dep else max(upstream)
            else:
                memo[node_id] = max(upstream)
            return memo[node_id]

        for node_id in self._node_input_node_ids:
            _get(node_id)
        return memo

    def _compute_host_side_segment_count(
        self, segment_ids: Dict[int, int]
    ) -> int:
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
        neural_latency_tags = sorted({
            int(sc.latency_tag) for sc in self.layout_softcores
            if sc.latency_tag is not None
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
                group = next(
                    (g for g in neural_groups if g["latency_group_index"] == slot),
                    None,
                )
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
                    "softcore_count": sum(
                        1 for sc in self.layout_softcores
                        if sc.segment_id == seg_id
                    ),
                    "latency_group_count": len({
                        int(sc.latency_tag) for sc in self.layout_softcores
                        if sc.segment_id == seg_id and sc.latency_tag is not None
                    }),
                }
                for seg_id in sorted({
                    int(sc.segment_id) for sc in self.layout_softcores
                    if sc.segment_id is not None
                })
            ],
            "latency_groups": neural_groups,
            "host_segments": host_segments,
            "flow": flow,
        }

    def _finalize_softcores(self) -> None:
        """Compute latencies / segment ids and rewrite each softcore with
        its finalised ``latency_tag``, ``segment_id``, and
        ``threshold_group_id = perceptron_index`` (falling back to a unique
        id when ``perceptron_index`` is ``None``).
        """
        latencies = self._compute_latencies()
        segment_ids = self._compute_segment_ids()
        self.host_side_segment_count = self._compute_host_side_segment_count(segment_ids)

        # Unique, stable, negative ids for non-perceptron cores so they
        # never collide with perceptron indices (which are >= 0).
        for node_id, sc_idx in self._node_id_to_softcore_idx.items():
            latency = latencies.get(node_id, 0)
            segment_id = segment_ids.get(node_id, 0)

            pi = self._sc_idx_to_perceptron_index.get(sc_idx)
            tg = int(pi) if pi is not None else -(sc_idx + 1)

            old = self.layout_softcores[sc_idx]
            self.layout_softcores[sc_idx] = LayoutSoftCoreSpec(
                input_count=old.input_count,
                output_count=old.output_count,
                threshold_group_id=tg,
                latency_tag=int(latency),
                segment_id=int(segment_id),
                name=old.name,
            )

        self.layout_preview = self._build_layout_preview(segment_ids, latencies)
