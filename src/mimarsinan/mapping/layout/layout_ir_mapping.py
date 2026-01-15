from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec


def _canonical_threshold_key(name: Optional[str]) -> str:
    if not name:
        return "unnamed"

    # Strip common tiling suffixes to keep groups stable per \"softcore category\".
    for sep in ("_tile_", "_psum_", "_col", "_pos"):
        if sep in name:
            name = name.split(sep)[0]
    return name


@dataclass
class LayoutIRMapping:
    """
    Mapping backend that collects *shape-only* neural cores (LayoutSoftCoreSpec) while
    running the model's Mapper graph via map_to_ir().

    It returns IRSource arrays purely as placeholders so downstream mappers can
    reshape/rearrange sources; connectivity is irrelevant for layout metrics.
    """

    max_axons: int
    max_neurons: int
    allow_axon_tiling: bool = False
    threshold_groups: int = 1

    def __post_init__(self):
        self.max_axons = int(self.max_axons)
        self.max_neurons = int(self.max_neurons)
        self.allow_axon_tiling = bool(self.allow_axon_tiling)
        self.threshold_groups = max(1, int(self.threshold_groups))

        self._next_node_id = 0
        self.layout_softcores: List[LayoutSoftCoreSpec] = []

    def _alloc_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def _threshold_group(self, name: Optional[str]) -> int:
        key = _canonical_threshold_key(name)
        h = zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF
        return int(h % self.threshold_groups)

    def _emit_core(self, *, input_count: int, output_count: int, name: Optional[str]) -> np.ndarray:
        node_id = self._alloc_node_id()
        tg = self._threshold_group(name)
        self.layout_softcores.append(
            LayoutSoftCoreSpec(
                input_count=int(input_count),
                output_count=int(output_count),
                threshold_group_id=int(tg),
                latency_tag=None,
                name=name,
            )
        )
        return np.array([IRSource(node_id=node_id, index=i) for i in range(int(output_count))], dtype=object)

    def add_compute_op(self, *args, **kwargs):
        raise NotImplementedError("LayoutIRMapping does not support ComputeOp yet (layout-only MNIST PoC).")

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
        return self._emit_core(input_count=in_count, output_count=out_features, name=name)

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
            raise NotImplementedError("allow_axon_tiling layout mapping not supported in LayoutIRMapping (yet)")

        # Output tiling
        if self.max_neurons is not None and out_features > self.max_neurons:
            chunks = []
            start = 0
            while start < out_features:
                end = min(start + self.max_neurons, out_features)
                out_block = end - start
                in_count = int(src_arr.flatten().shape[0]) + (1 if fc_biases is not None else 0)
                chunks.append(self._emit_core(input_count=in_count, output_count=out_block, name=name))
                start = end
            return np.concatenate(chunks).reshape(tuple(output_shape))

        # Simple single core
        in_count = int(src_arr.flatten().shape[0]) + (1 if fc_biases is not None else 0)
        return self._emit_core(input_count=in_count, output_count=out_features, name=name).reshape(tuple(output_shape))

    def collect_layout_softcores(self, model_representation) -> List[LayoutSoftCoreSpec]:
        """
        Execute map_to_ir over the mapper graph, collecting LayoutSoftCoreSpec instances.
        """
        _ = model_representation.map_to_ir(self)
        return list(self.layout_softcores)


