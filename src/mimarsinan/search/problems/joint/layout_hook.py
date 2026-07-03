"""Model build, layout collection, and HW objective helpers for joint search."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.parameter import UninitializedParameter

from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.platform.coalescing import normalize_coalescing_config
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats
from mimarsinan.torch_mapping.converter import convert_torch_model

from .types import HwOnlyCache


class JointLayoutMixin:
    """Layout mapping and HW metric computation for :class:`JointArchHwProblem`."""

    def _build_raw_model(self, model_config: Dict, pcfg: Dict):
        """Build and warm up a raw model. Returns (model, total_params) or raises."""
        builder = self.builder_factory(
            self.device,
            self.input_shape,
            self.num_classes,
            {**pcfg, "target_tq": int(self.target_tq)},
        )
        model = builder.build(model_config).to(self.device)

        model.eval()
        with torch.no_grad():
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                model_device = self.device
            dummy = torch.zeros((1, *tuple(self.input_shape)), device=model_device)
            _ = model(dummy)

        if any(isinstance(p, UninitializedParameter) for p in model.parameters()):
            raise RuntimeError("Model has uninitialised parameters after forward pass")

        total_params = float(sum(int(p.numel()) for p in model.parameters()))
        return model, total_params

    def _ensure_mapper_repr(self, model):
        """Convert via torch mapping if the model lacks ``get_mapper_repr``."""
        if hasattr(model, "get_mapper_repr"):
            return model
        return convert_torch_model(
            model,
            input_shape=tuple(self.input_shape),
            num_classes=self.num_classes,
            device=self.device,
            Tq=self.target_tq,
        )

    def _build_model(self, model_config: Dict, pcfg: Dict):
        """Build, warm up, and convert a model. Returns (model, total_params)."""
        model, total_params = self._build_raw_model(model_config, pcfg)
        model = self._ensure_mapper_repr(model)
        return model, total_params

    def _collect_softcores(
        self,
        model,
        pcfg: Dict,
    ) -> Tuple[List[LayoutSoftCoreSpec], int]:
        """Collect layout softcores and host-side segment count from model."""
        cores = pcfg["cores"]
        pmap = resolve_platform_mapping_params(
            cores,
            allow_coalescing=bool(pcfg.get("allow_coalescing", False)),
        )
        layout_mapper = LayoutIRMapping(
            max_axons=pmap.effective_max_axons,
            max_neurons=pmap.effective_max_neurons,
            allow_coalescing=pmap.allow_coalescing,
            hardware_bias=pmap.hardware_bias,
        )
        mapper_repr = model.get_mapper_repr()
        if hasattr(mapper_repr, "assign_perceptron_indices"):
            mapper_repr.assign_perceptron_indices()
        softcores = layout_mapper.collect_layout_softcores(mapper_repr)
        host_segments = getattr(layout_mapper, "host_side_segment_count", 0)
        return softcores, host_segments

    def _ensure_hw_only_cache(self) -> HwOnlyCache:
        """Build model once and cache softcores for HW-only search."""
        if self._hw_only_cache is not None:
            return self._hw_only_cache

        mc = self.fixed_model_config or {}
        pcfg = dict(self.fixed_platform_constraints or {})
        normalize_coalescing_config(pcfg)

        torch.manual_seed(int(self.accuracy_seed))
        np.random.seed(int(self.accuracy_seed))

        model, total_params = self._build_model(mc, pcfg)
        softcores, host_segments = self._collect_softcores(model, pcfg)

        self._hw_only_cache = HwOnlyCache(
            softcores=softcores,
            total_params=total_params,
            host_side_segment_count=host_segments,
        )
        return self._hw_only_cache

    @staticmethod
    def _make_core_types(pcfg: Dict) -> List[LayoutHardCoreType]:
        return [
            LayoutHardCoreType(
                max_axons=int(ct["max_axons"]),
                max_neurons=int(ct["max_neurons"]),
                count=int(ct["count"]),
            )
            for ct in pcfg["cores"]
        ]

    @staticmethod
    def _compute_chip_capacity(pcfg: Dict) -> float:
        return float(
            sum(
                int(ct["max_axons"]) * int(ct["max_neurons"]) * int(ct["count"])
                for ct in pcfg["cores"]
            )
        )

    def _penalty_objectives(self) -> Dict[str, float]:
        """Return penalty values for all objectives (infeasible candidate)."""
        large = 1e18
        obj: Dict[str, float] = {}
        for spec in self.objectives:
            obj[spec.name] = 0.0 if spec.goal == "max" else large
        return obj

    def _compute_hw_objectives(
        self,
        softcores: List[LayoutSoftCoreSpec],
        pcfg: Dict,
        total_params: float,
        host_side_segment_count: int,
    ) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        """Run bin-packing and compute all non-accuracy objectives."""
        core_types = self._make_core_types(pcfg)
        stats, error = compute_mapping_stats(
            softcores=softcores,
            core_types=core_types,
            **ChipCapabilities.from_platform_constraints(pcfg).permission_kwargs(),
        )

        if not stats.feasible:
            total_hw_capacity = sum(
                ct.max_axons * ct.max_neurons * ct.count for ct in core_types
            )
            full_error = error or "HW bin-packing infeasible"
            full_error += (
                f" | softcores={len(softcores)}"
                f", total_hw_capacity={total_hw_capacity}"
            )
            return None, full_error

        chip_capacity = self._compute_chip_capacity(pcfg)

        return {
            "total_params": total_params,
            "total_param_capacity": chip_capacity,
            "total_sync_barriers": float(
                host_side_segment_count + stats.schedule_sync_count
            ),
            "param_utilization_pct": stats.mapped_params_pct,
            "neuron_wastage_pct": stats.total_wasted_neurons_pct,
            "axon_wastage_pct": stats.total_wasted_axons_pct,
            "fragmentation_pct": stats.fragmentation_pct,
        }, None
