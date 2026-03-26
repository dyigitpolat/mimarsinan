"""
Generic joint architecture + hardware search problem.

Works with *any* model that can be expressed through the mimarsinan IR.
Model-specific behaviour is injected via constructor parameters:

- ``arch_options``           – list of (key, values) pairs defining the
                               architecture search space.
- ``model_config_assembler`` – callable that post-processes the decoded
                               architecture dict into a final model_config.
- ``validate_fn``            – optional callable for fast feasibility checks.
- ``constraint_fn``          – optional callable returning a continuous
                               constraint-violation score.
- ``builder_factory``        – callable that creates a model builder given
                               resolved platform constraints.

Search mode controls which dimensions are optimised:

    "model"    – architecture indices only; hardware is fixed.
    "hardware" – core dimensions / threshold groups only; model is fixed.
                 Model is built *once* and softcores are cached; subsequent
                 evaluations only run bin-packing (very fast).
    "joint"    – both architecture and hardware are searched.

Objectives are selected from the canonical catalogue in ``search.results``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.layout_verification_stats import build_stats_from_packing_result
from mimarsinan.search.evaluators.fast_accuracy_evaluator import FastAccuracyEvaluator
from mimarsinan.search.evaluators.extrapolating_accuracy_evaluator import ExtrapolatingAccuracyEvaluator
from mimarsinan.search.problems.encoded_problem import EncodedProblem
from mimarsinan.search.results import (
    ACCURACY_OBJECTIVE_NAME,
    ObjectiveSpec,
    resolve_active_objectives,
)


def _json_key(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _clip_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(float(v))))))


# ---------------------------------------------------------------------------
# Type aliases for the injected callables
# ---------------------------------------------------------------------------
BuilderFactory = Callable[..., Any]
ModelConfigAssembler = Callable[[Dict[str, Any]], Dict[str, Any]]
ValidateFn = Callable[[Dict[str, Any], Dict[str, Any], Tuple[int, ...]], bool]
ConstraintFn = Callable[[Dict[str, Any], Dict[str, Any], Tuple[int, ...]], float]


# ---------------------------------------------------------------------------
# Cache for HW-only search (model built once)
# ---------------------------------------------------------------------------
@dataclass
class _HwOnlyCache:
    softcores: List[LayoutSoftCoreSpec]
    total_params: float
    host_side_segment_count: int


# ---------------------------------------------------------------------------
# The problem
# ---------------------------------------------------------------------------
@dataclass
class JointArchHwProblem(EncodedProblem[Dict[str, Any]]):
    """
    Model-agnostic search problem with variable geometry.

    The decision vector contains only variables for the searched dimensions:
    - ``"model"``:    architecture indices
    - ``"hardware"``: threshold_groups + core dims per type
    - ``"joint"``:    architecture indices + threshold_groups + core dims
    """

    data_provider_factory: Any
    device: torch.device
    input_shape: Tuple[int, ...]
    num_classes: int
    target_tq: int
    lr: float

    # --- Search mode ---
    search_mode: str = "joint"  # "model", "hardware", "joint"

    # --- Pluggable, model-specific pieces ---
    builder_factory: BuilderFactory = None  # type: ignore[assignment]
    arch_options: Sequence[Tuple[str, Sequence[Any]]] = ()
    model_config_assembler: ModelConfigAssembler = None  # type: ignore[assignment]
    validate_fn: Optional[ValidateFn] = None
    constraint_fn: Optional[ConstraintFn] = None

    # --- Fixed values for non-searched dimensions ---
    fixed_model_config: Optional[Dict[str, Any]] = None
    fixed_platform_constraints: Optional[Dict[str, Any]] = None

    # --- Active objectives (resolved ObjectiveSpec names) ---
    active_objective_names: Sequence[str] = ()

    # --- Hardware search space ---
    num_core_types: int = 1
    core_type_counts: Sequence[int] = (100,)
    core_axons_bounds: Tuple[int, int] = (64, 2048)
    core_neurons_bounds: Tuple[int, int] = (64, 2048)
    max_threshold_groups: int = 4

    # --- Evaluation knobs ---
    accuracy_seed: int = 0
    warmup_fraction: float = 0.10
    training_batch_size: Optional[int] = None
    accuracy_evaluator: str = "extrapolating"
    extrapolation_num_train_epochs: int = 1
    extrapolation_num_checkpoints: int = 5
    extrapolation_target_epochs: int = 10
    pruning_fraction: float = 0.0

    # --- Internal ---
    _cache: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False)
    _hw_only_cache: Optional[_HwOnlyCache] = field(default=None, init=False)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @property
    def _searches_model(self) -> bool:
        return self.search_mode in ("model", "joint")

    @property
    def _searches_hw(self) -> bool:
        return self.search_mode in ("hardware", "joint")

    @property
    def _n_arch_vars(self) -> int:
        return len(self.arch_options) if self._searches_model else 0

    @property
    def _n_hw_vars(self) -> int:
        return (1 + 2 * int(self.num_core_types)) if self._searches_hw else 0

    # ------------------------------------------------------------------ #
    # EncodedProblem interface
    # ------------------------------------------------------------------ #

    @property
    def objectives(self) -> Sequence[ObjectiveSpec]:
        if self.active_objective_names:
            return resolve_active_objectives(self.search_mode, self.active_objective_names)
        return resolve_active_objectives(self.search_mode)

    @property
    def n_var(self) -> int:
        return self._n_arch_vars + self._n_hw_vars

    @property
    def xl(self) -> np.ndarray:
        xl: List[float] = []
        if self._searches_model:
            xl.extend([0.0] * len(self.arch_options))
        if self._searches_hw:
            xl.append(1.0)  # threshold_groups lower bound
            for _ in range(int(self.num_core_types)):
                xl.extend([float(self.core_axons_bounds[0]), float(self.core_neurons_bounds[0])])
        return np.array(xl, dtype=float)

    @property
    def xu(self) -> np.ndarray:
        xu: List[float] = []
        if self._searches_model:
            xu.extend([float(len(opts) - 1) for _, opts in self.arch_options])
        if self._searches_hw:
            xu.append(float(self.max_threshold_groups))
            for _ in range(int(self.num_core_types)):
                xu.extend([float(self.core_axons_bounds[1]), float(self.core_neurons_bounds[1])])
        return np.array(xu, dtype=float)

    # ------------------------------------------------------------------ #
    # Decode
    # ------------------------------------------------------------------ #

    def _decode_arch(self, x: np.ndarray, offset: int) -> Dict[str, Any]:
        """Decode architecture indices from x starting at *offset*."""
        raw_arch: Dict[str, Any] = {}
        for i, (key, options) in enumerate(self.arch_options):
            idx = _clip_int(x[offset + i], 0, len(options) - 1)
            raw_arch[key] = options[idx]
        return self.model_config_assembler(raw_arch)

    def _decode_hw(self, x: np.ndarray, offset: int) -> Tuple[Dict[str, Any], int]:
        """Decode hardware variables from x starting at *offset*.

        Returns (platform_constraints_dict, threshold_groups).
        """
        threshold_groups = _clip_int(x[offset], 1, int(self.max_threshold_groups))

        core_types: List[Dict[str, int]] = []
        idx = offset + 1
        for t in range(int(self.num_core_types)):
            ax = _clip_int(x[idx], int(self.core_axons_bounds[0]), int(self.core_axons_bounds[1]))
            neu = _clip_int(x[idx + 1], int(self.core_neurons_bounds[0]), int(self.core_neurons_bounds[1]))
            idx += 2
            ax = max(8, int(round(ax / 8)) * 8)
            neu = max(8, int(round(neu / 8)) * 8)
            core_types.append({
                "max_axons": ax,
                "max_neurons": neu,
                "count": int(self.core_type_counts[t]),
            })

        min_axons = min(int(ct["max_axons"]) for ct in core_types)
        min_neurons = min(int(ct["max_neurons"]) for ct in core_types)

        platform_constraints = {
            "cores": core_types,
            "max_axons": int(min_axons),
            "max_neurons": int(min_neurons),
            "target_tq": int(self.target_tq),
            "weight_bits": 8,
        }
        return platform_constraints, int(threshold_groups)

    def decode(self, x: np.ndarray) -> Dict[str, Any]:
        x = np.array(x, dtype=float).flatten()
        if x.shape[0] != self.n_var:
            raise ValueError(f"Expected x of length {self.n_var}, got {x.shape}")

        offset = 0

        # Architecture
        if self._searches_model:
            model_config = self._decode_arch(x, offset)
            offset += self._n_arch_vars
        else:
            model_config = dict(self.fixed_model_config or {})

        # Hardware
        if self._searches_hw:
            platform_constraints, threshold_groups = self._decode_hw(x, offset)
        else:
            platform_constraints = dict(self.fixed_platform_constraints or {})
            threshold_groups = int(platform_constraints.get("threshold_groups", 1))

        return {
            "model_config": model_config,
            "platform_constraints": platform_constraints,
            "threshold_groups": threshold_groups,
        }

    # ------------------------------------------------------------------ #
    # Validation / constraint
    # ------------------------------------------------------------------ #

    def validate(self, configuration: Dict[str, Any]) -> bool:
        try:
            if self.validate_fn is not None:
                return bool(self.validate_fn(
                    configuration["model_config"],
                    configuration["platform_constraints"],
                    self.input_shape,
                ))
            return True
        except Exception:
            return False

    def constraint_violation(self, configuration: Dict[str, Any]) -> float:
        try:
            if self.constraint_fn is not None:
                return float(self.constraint_fn(
                    configuration["model_config"],
                    configuration["platform_constraints"],
                    self.input_shape,
                ))
            return 0.0 if self.validate(configuration) else 1.0
        except Exception:
            return 1e6

    # ------------------------------------------------------------------ #
    # Model building & layout helpers
    # ------------------------------------------------------------------ #

    def _build_model(self, model_config: Dict, pcfg: Dict):
        """Build a model and initialise lazy modules. Returns (model, total_params) or raises."""
        builder = self.builder_factory(
            self.device,
            self.input_shape,
            self.num_classes,
            int(pcfg["max_axons"]),
            int(pcfg["max_neurons"]),
            {**pcfg, "target_tq": int(self.target_tq)},
        )
        model = builder.build(model_config).to(self.device)

        from torch.nn.parameter import UninitializedParameter

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

        if not hasattr(model, "get_mapper_repr"):
            from mimarsinan.torch_mapping.converter import convert_torch_model
            model = convert_torch_model(
                model,
                input_shape=tuple(self.input_shape),
                num_classes=self.num_classes,
                device=self.device,
                Tq=self.target_tq,
            )

        return model, total_params

    def _collect_softcores(
        self,
        model,
        pcfg: Dict,
        threshold_groups: int,
    ) -> Tuple[List[LayoutSoftCoreSpec], int]:
        """Collect layout softcores and host-side segment count from model."""
        layout_mapper = LayoutIRMapping(
            max_axons=int(pcfg["max_axons"]),
            max_neurons=int(pcfg["max_neurons"]),
            threshold_groups=threshold_groups,
            threshold_seed=int(self.accuracy_seed),
            pruning_fraction=float(self.pruning_fraction),
            allow_core_coalescing=bool(pcfg.get("allow_core_coalescing", False)),
            hardware_bias=bool(pcfg.get("has_bias", False)),
        )
        softcores = layout_mapper.collect_layout_softcores(model.get_mapper_repr())
        host_segments = getattr(layout_mapper, "host_side_segment_count", 0)
        return softcores, host_segments

    def _ensure_hw_only_cache(self) -> _HwOnlyCache:
        """Build model once and cache softcores for HW-only search."""
        if self._hw_only_cache is not None:
            return self._hw_only_cache

        mc = self.fixed_model_config or {}
        pcfg = self.fixed_platform_constraints or {}

        torch.manual_seed(int(self.accuracy_seed))
        np.random.seed(int(self.accuracy_seed))

        model, total_params = self._build_model(mc, pcfg)
        softcores, host_segments = self._collect_softcores(
            model, pcfg, int(pcfg.get("threshold_groups", 1)),
        )

        self._hw_only_cache = _HwOnlyCache(
            softcores=softcores,
            total_params=total_params,
            host_side_segment_count=host_segments,
        )
        return self._hw_only_cache

    # ------------------------------------------------------------------ #
    # Objective computation helpers
    # ------------------------------------------------------------------ #

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
        return float(sum(
            int(ct["max_axons"]) * int(ct["max_neurons"]) * int(ct["count"])
            for ct in pcfg["cores"]
        ))

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
    ) -> Optional[Dict[str, float]]:
        """Run bin-packing and compute all non-accuracy objectives.

        Returns None if packing is infeasible.
        """
        core_types = self._make_core_types(pcfg)
        pack = pack_layout(softcores=softcores, core_types=core_types)

        if not pack.feasible:
            return None

        stats = build_stats_from_packing_result(
            pack,
            num_original_softcores=len(softcores),
            softcores=softcores,
            core_types=core_types,
        )

        chip_capacity = self._compute_chip_capacity(pcfg)
        schedule_passes = stats.schedule_pass_count

        return {
            "total_params": total_params,
            "total_param_capacity": chip_capacity,
            "total_sync_barriers": float(host_side_segment_count + schedule_passes),
            "param_utilization_pct": stats.mapped_params_pct,
            "neuron_wastage_pct": stats.total_wasted_neurons_pct,
            "axon_wastage_pct": stats.total_wasted_axons_pct,
        }

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(self, configuration: Dict[str, Any]) -> Dict[str, float]:
        key = _json_key(configuration)
        if key in self._cache:
            return self._cache[key]

        if not self.validate(configuration):
            obj = self._penalty_objectives()
            self._cache[key] = obj
            return obj

        mc = configuration["model_config"]
        pcfg = configuration["platform_constraints"]
        threshold_groups = int(configuration["threshold_groups"])

        torch.manual_seed(int(self.accuracy_seed))
        np.random.seed(int(self.accuracy_seed))

        try:
            obj = self._evaluate_inner(mc, pcfg, threshold_groups)
        except Exception:
            obj = self._penalty_objectives()

        self._cache[key] = obj
        return obj

    def _evaluate_inner(
        self,
        mc: Dict[str, Any],
        pcfg: Dict[str, Any],
        threshold_groups: int,
    ) -> Dict[str, float]:
        active_names = {spec.name for spec in self.objectives}
        needs_accuracy = ACCURACY_OBJECTIVE_NAME in active_names

        # --- HW-only search: use cached softcores ---
        if self.search_mode == "hardware":
            cache = self._ensure_hw_only_cache()
            hw_obj = self._compute_hw_objectives(
                cache.softcores, pcfg, cache.total_params, cache.host_side_segment_count,
            )
            if hw_obj is None:
                return self._penalty_objectives()
            return {k: v for k, v in hw_obj.items() if k in active_names}

        # --- Model or joint search: build model fresh ---
        model, total_params = self._build_model(mc, pcfg)
        softcores, host_segments = self._collect_softcores(model, pcfg, threshold_groups)

        hw_obj = self._compute_hw_objectives(softcores, pcfg, total_params, host_segments)
        if hw_obj is None:
            return self._penalty_objectives()

        obj = {k: v for k, v in hw_obj.items() if k in active_names}

        # Accuracy evaluation (expensive)
        if needs_accuracy:
            accuracy = self._evaluate_accuracy(model)
            obj[ACCURACY_OBJECTIVE_NAME] = accuracy

        return obj

    def _evaluate_accuracy(self, model) -> float:
        if self.accuracy_evaluator == "extrapolating":
            acc_eval = ExtrapolatingAccuracyEvaluator(
                data_provider_factory=self.data_provider_factory,
                device=self.device,
                lr=float(self.lr),
                num_train_epochs=int(self.extrapolation_num_train_epochs),
                num_checkpoints=int(self.extrapolation_num_checkpoints),
                target_epochs=int(self.extrapolation_target_epochs),
                warmup_fraction=float(self.warmup_fraction),
                num_workers=0,
                training_batch_size=self.training_batch_size,
                seed=int(self.accuracy_seed),
            )
        else:
            acc_eval = FastAccuracyEvaluator(
                data_provider_factory=self.data_provider_factory,
                device=self.device,
                lr=float(self.lr),
                warmup_fraction=float(self.warmup_fraction),
                num_workers=0,
                training_batch_size=self.training_batch_size,
                seed=int(self.accuracy_seed),
            )
        return float(acc_eval.evaluate(model))
