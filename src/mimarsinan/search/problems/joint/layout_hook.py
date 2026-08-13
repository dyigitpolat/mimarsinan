"""Model build, layout collection, and the candidate view for joint search."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.parameter import UninitializedParameter

from mimarsinan.deployment_record.objectives import (
    CandidateStaticView,
    candidate_probe_without,
    declared_core_capacity,
)
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats
from mimarsinan.mapping.verification.layout_verification_types import (
    LayoutVerificationStats,
)
from mimarsinan.models.builders import build_model
from mimarsinan.torch_mapping.converter import convert_torch_model

from .types import (
    HW_PACKING_PHASE,
    CandidateFailure,
    HwOnlyCache,
    JointHostContract,
)


class JointLayoutMixin(JointHostContract):
    """Layout mapping and candidate-view construction for :class:`JointArchHwProblem`."""

    def _build_raw_model(self, model_config: Dict, pcfg: Dict, placement: str):
        """Build and warm up a raw model. Returns (model, total_params) or raises."""
        builder = self.builder_factory(
            self.device,
            self.input_shape,
            self.num_classes,
            {**pcfg, "target_tq": int(self.target_tq)},
        )
        model = build_model(
            builder, model_config, encoding_placement=placement
        ).to(self.device)

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

    def _convert_to_mapper_repr(self, model, placement: str):
        """Convert via torch mapping if the model lacks ``get_mapper_repr``.

        A native builder's flow already had its placement resolved by
        ``build_model``; a torch module's flow is born here and resolves the
        same one, so a candidate's core count is the deployed model's.
        """
        if hasattr(model, "get_mapper_repr"):
            return model
        return convert_torch_model(
            model,
            input_shape=tuple(self.input_shape),
            num_classes=self.num_classes,
            device=self.device,
            Tq=self.target_tq,
            encoding_layer_placement=placement,
        )

    def _ensure_mapper_repr(self, model, placement: str):
        """The model in mapper form, converted once per model rather than per candidate.

        Only the LAYOUT depends on the candidate chip; lowering the model into
        mapper form depends on the model alone. A hardware-only search reuses
        one model across every candidate, so its representation is memoized on
        that fixture — a model-bearing search builds a new model per candidate
        and converts it exactly once anyway.
        """
        cache = self._hw_only_cache.get(placement)
        if cache is None or model is not cache.model:
            return self._convert_to_mapper_repr(model, placement)
        if cache.mapper_repr is None:
            cache.mapper_repr = self._convert_to_mapper_repr(model, placement)
        return cache.mapper_repr

    def _candidate_model(
        self, mc: Dict, pcfg: Dict, placement: str,
    ) -> Tuple[Any, float]:
        """The model this candidate is scored with, and its parameter census.

        A model-bearing search builds the candidate's own; a hardware-only
        search reuses the run's fixed model, which no candidate influences.
        Seeding belongs here, next to the build it makes reproducible.
        """
        torch.manual_seed(int(self.accuracy_seed))
        np.random.seed(int(self.accuracy_seed))
        if self._searches_model:
            return self._build_raw_model(mc, pcfg, placement)
        cache = self._ensure_hw_only_cache(placement)
        return cache.model, cache.total_params

    def _build_model(self, model_config: Dict, pcfg: Dict, placement: str):
        """Build, warm up, and convert a model. Returns (model, total_params)."""
        model, total_params = self._build_raw_model(model_config, pcfg, placement)
        model = self._ensure_mapper_repr(model, placement)
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

    def _ensure_hw_only_cache(self, placement: str) -> HwOnlyCache:
        """Build the candidate-independent model once for a hardware-only search.

        Only the MODEL is reused: every candidate re-derives its own layout,
        because softcore tiling is a function of the candidate's core geometry.
        Keyed by PLACEMENT: the encoder's side of the NeuralOps/ComputeOps
        boundary is baked at flow birth, so two placements are two fixtures.
        """
        cached = self._hw_only_cache.get(placement)
        if cached is not None:
            return cached

        base = self.fixed_platform_constraints
        if not base or "cores" not in base:
            raise ValueError(
                "hardware-only search requires a platform_resolver whose resolved "
                "base declares 'cores'"
            )
        mc = self.fixed_model_config or {}

        torch.manual_seed(int(self.accuracy_seed))
        np.random.seed(int(self.accuracy_seed))

        model, total_params = self._build_raw_model(mc, dict(base), placement)
        cache = HwOnlyCache(model=model, total_params=total_params)
        self._hw_only_cache[placement] = cache
        return cache

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

    def _requires_fragment(self, fragment: str) -> bool:
        """Does any ACTIVE objective need this candidate fragment? The registry answers."""
        probe = candidate_probe_without(fragment)
        return any(not spec.available(probe) for spec in self.active_specs)

    def _penalty_objectives(self) -> Dict[str, float]:
        """Return penalty values for all objectives (infeasible candidate)."""
        large = 1e18
        obj: Dict[str, float] = {}
        for spec in self.objectives:
            obj[spec.name] = 0.0 if spec.goal == "max" else large
        return obj

    def _pack_candidate(
        self, softcores: List[LayoutSoftCoreSpec], pcfg: Dict,
    ) -> Tuple[LayoutVerificationStats, Optional[str]]:
        """Pack the candidate's softcores onto the chip it declares.

        The capability declaration is forwarded WHOLE (``layout_kwargs``): the
        pass structure is not a property of the permission bits alone, so a
        census computed without the platform's ``schedule_policy`` would score
        the candidate against a program its chip never runs.
        """
        return compute_mapping_stats(
            softcores=softcores,
            core_types=self._make_core_types(pcfg),
            **ChipCapabilities.from_platform_constraints(pcfg).layout_kwargs(),
        )

    def _packing_failure(
        self,
        stats: LayoutVerificationStats,
        error: Optional[str],
        softcores: List[LayoutSoftCoreSpec],
        pcfg: Dict,
    ) -> CandidateFailure:
        """Why the candidate does not fit, with the census that shows how badly."""
        total_hw_capacity = sum(
            ct.max_axons * ct.max_neurons * ct.count
            for ct in self._make_core_types(pcfg)
        )
        message = error or "HW bin-packing infeasible"
        message += (
            f" | softcores={len(softcores)}"
            f", total_hw_capacity={total_hw_capacity}"
        )
        return CandidateFailure(phase=HW_PACKING_PHASE, message=message)

    def _static_view(
        self,
        stats: LayoutVerificationStats,
        pcfg: Dict,
        total_params: float,
        host_side_segment_count: int,
    ) -> CandidateStaticView:
        """The static facts of a packed candidate — what every objective reads."""
        return CandidateStaticView(
            layout=stats,
            chip_param_capacity=declared_core_capacity(pcfg),
            total_params=total_params,
            host_side_segment_count=host_side_segment_count,
        )

    def _layoutless_view(self, pcfg: Dict, total_params: float) -> CandidateStaticView:
        """The view of a candidate no active objective needs a layout for."""
        return CandidateStaticView(
            layout=None,
            chip_param_capacity=declared_core_capacity(pcfg),
            total_params=total_params,
            host_side_segment_count=None,
        )
