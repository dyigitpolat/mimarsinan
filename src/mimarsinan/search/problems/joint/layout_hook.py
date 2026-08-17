"""Model build, layout collection, and the candidate view for joint search."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from mimarsinan.deployment_record.objectives import (
    CandidateStaticView,
    candidate_probe_without,
    declared_core_capacity,
)
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.noc import census_of_walk
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats
from mimarsinan.mapping.verification.layout_verification_types import (
    LayoutVerificationStats,
)
from .candidate_fragments import (
    StageSemantics, candidate_fragments, compute_onchip_census,
    make_core_types, stage_semantics_of,
)
from .model_build import build_raw_model, convert_to_mapper_repr
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
        return build_raw_model(
            builder_factory=self.builder_factory, device=self.device,
            input_shape=tuple(self.input_shape), num_classes=self.num_classes,
            target_tq=int(self.target_tq), model_config=model_config,
            pcfg=pcfg, placement=placement,
            prune_sparsity=float(self.prune_sparsity), pruning=bool(self.pruning),
            prune_criterion=str(self.prune_criterion),
            pruning_fraction=float(self.pruning_fraction),
            firing_mode=str(self.firing_mode),
        )

    def _convert_to_mapper_repr(self, model, placement: str):
        """The mapper-form model — see :func:`convert_to_mapper_repr`."""
        return convert_to_mapper_repr(
            model,
            input_shape=tuple(self.input_shape),
            num_classes=self.num_classes,
            device=self.device,
            target_tq=self.target_tq,
            placement=placement,
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
        *,
        collect_census: bool = False,
    ) -> Tuple[List[LayoutSoftCoreSpec], int, Optional[Any]]:
        """Collect layout softcores, host segments, and (opt-in — collection
        materialises the deferred views) the walk's wire census."""
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
            collect_wire_census=collect_census,
        )
        mapper_repr = model.get_mapper_repr()
        if hasattr(mapper_repr, "assign_perceptron_indices"):
            mapper_repr.assign_perceptron_indices()
        softcores = layout_mapper.collect_layout_softcores(mapper_repr)
        host_segments = getattr(layout_mapper, "host_side_segment_count", 0)
        census = census_of_walk(layout_mapper) if collect_census else None
        return softcores, host_segments, census

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
        return make_core_types(pcfg)

    def _requires_fragment(self, fragment: str) -> bool:
        """Does any ACTIVE objective need this candidate fragment? The registry
        answers — once per problem [H4]: the active set is fixed at
        construction, so the probe walk was pure per-candidate waste."""
        cached = self._fragment_needs_cache.get(fragment)
        if cached is None:
            probe = candidate_probe_without(fragment)
            cached = any(not spec.available(probe) for spec in self.active_specs)
            self._fragment_needs_cache[fragment] = cached
        return cached

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
        census computed without the platform's pass budget would score the
        candidate against a program its chip never runs.
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

    def _onchip_census(self, model, placement: str):
        """Host/on-chip param+MAC counts when an active axis prices them —
        two flow walks, so registry-gated and memoized on the hw-only fixture."""
        if not self._requires_fragment("quantity_context"):
            return None
        if not self._searches_model:
            cache = self._hw_only_cache.get(placement)
            if cache is not None and cache.onchip_census is not None:
                return cache.onchip_census
        census = compute_onchip_census(
            model, tuple(self.input_shape), int(self.num_classes), placement,
        )
        if not self._searches_model:
            cache = self._hw_only_cache.get(placement)
            if cache is not None:
                cache.onchip_census = census
        return census

    @property
    def stage_semantics(self) -> StageSemantics:
        """[E1] The firing semantics the executed-window rule branches on."""
        return stage_semantics_of(
            self.spiking_mode, self.ttfs_cycle_schedule,
            retimed=bool(self.per_hop_retiming),
        )

    def _static_view(
        self,
        stats: LayoutVerificationStats,
        pcfg: Dict,
        total_params: float,
        host_side_segment_count: int,
        census=None,
        noc=None,
        program=None,
    ) -> CandidateStaticView:
        """The static facts of a packed candidate — what every objective reads."""
        physics, context = candidate_fragments(pcfg, census, program)
        return CandidateStaticView(
            layout=stats,
            chip_param_capacity=declared_core_capacity(pcfg),
            total_params=total_params,
            host_side_segment_count=host_side_segment_count,
            physics=physics,
            quantity_context=context,
            noc_fragments=noc,
        )

    def _layoutless_view(self, pcfg: Dict, total_params: float) -> CandidateStaticView:
        """The view of a candidate no active objective needs a layout for."""
        physics, context = candidate_fragments(pcfg)
        return CandidateStaticView(
            layout=None,
            chip_param_capacity=declared_core_capacity(pcfg),
            total_params=total_params,
            host_side_segment_count=None,
            physics=physics,
            quantity_context=context,
        )
