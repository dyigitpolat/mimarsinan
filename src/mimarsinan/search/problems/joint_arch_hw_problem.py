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

Hardware search variables (core dimensions, threshold groups) are always
appended to the architecture variables automatically.

Objectives:
    accuracy    (max)
    wasted_area (min) — unused silicon area after bin-packing
    total_params (min)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.search.evaluators.fast_accuracy_evaluator import FastAccuracyEvaluator
from mimarsinan.search.evaluators.extrapolating_accuracy_evaluator import ExtrapolatingAccuracyEvaluator
from mimarsinan.search.problems.encoded_problem import EncodedProblem
from mimarsinan.search.results import ObjectiveSpec


def _json_key(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _clip_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(float(v))))))


# ---------------------------------------------------------------------------
# Type aliases for the injected callables
# ---------------------------------------------------------------------------
# builder_factory(device, input_shape, num_classes, max_axons, max_neurons, platform_cfg) -> builder
BuilderFactory = Callable[..., Any]

# model_config_assembler(decoded_arch_dict) -> model_config dict
ModelConfigAssembler = Callable[[Dict[str, Any]], Dict[str, Any]]

# validate_fn(model_config, platform_constraints, input_shape, allow_axon_tiling) -> bool
ValidateFn = Callable[[Dict[str, Any], Dict[str, Any], Tuple[int, ...], bool], bool]

# constraint_fn(model_config, platform_constraints, input_shape, allow_axon_tiling) -> float
ConstraintFn = Callable[[Dict[str, Any], Dict[str, Any], Tuple[int, ...], bool], float]


@dataclass
class JointArchHwProblem(EncodedProblem[Dict[str, Any]]):
    """
    Model-agnostic joint architecture + hardware search problem.

    The architecture search space is defined by ``arch_options``: a list of
    ``(config_key, options_list)`` tuples.  Each entry contributes one integer
    decision variable (an index into ``options_list``).

    Hardware variables (core dimensions × num_core_types, threshold groups) are
    appended automatically.
    """

    data_provider_factory: Any
    device: torch.device
    input_shape: Tuple[int, ...]
    num_classes: int
    target_tq: int
    lr: float

    # --- Pluggable, model-specific pieces ---
    builder_factory: BuilderFactory
    arch_options: Sequence[Tuple[str, Sequence[Any]]]
    model_config_assembler: ModelConfigAssembler
    validate_fn: Optional[ValidateFn] = None
    constraint_fn: Optional[ConstraintFn] = None

    # --- Hardware search space (shared across all model types) ---
    num_core_types: int = 1
    core_type_counts: Sequence[int] = (100,)
    core_axons_bounds: Tuple[int, int] = (64, 2048)
    core_neurons_bounds: Tuple[int, int] = (64, 2048)
    max_threshold_groups: int = 4
    allow_axon_tiling: bool = False

    # --- Evaluation knobs ---
    accuracy_seed: int = 0
    warmup_fraction: float = 0.10
    training_batch_size: Optional[int] = None
    accuracy_evaluator: str = "extrapolating"
    extrapolation_num_train_epochs: int = 1
    extrapolation_num_checkpoints: int = 5
    extrapolation_target_epochs: int = 10

    _cache: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------ #
    # EncodedProblem interface
    # ------------------------------------------------------------------ #

    @property
    def objectives(self) -> Sequence[ObjectiveSpec]:
        return (
            ObjectiveSpec("accuracy", "max"),
            ObjectiveSpec("wasted_area", "min"),
            ObjectiveSpec("total_params", "min"),
        )

    @property
    def n_var(self) -> int:
        return len(self.arch_options) + 1 + 2 * int(self.num_core_types)

    @property
    def xl(self) -> np.ndarray:
        xl: List[float] = [0.0] * len(self.arch_options)  # indices
        xl.append(1.0)  # threshold_groups lower bound
        for _ in range(int(self.num_core_types)):
            xl.extend([float(self.core_axons_bounds[0]), float(self.core_neurons_bounds[0])])
        return np.array(xl, dtype=float)

    @property
    def xu(self) -> np.ndarray:
        xu: List[float] = [float(len(opts) - 1) for _, opts in self.arch_options]
        xu.append(float(self.max_threshold_groups))
        for _ in range(int(self.num_core_types)):
            xu.extend([float(self.core_axons_bounds[1]), float(self.core_neurons_bounds[1])])
        return np.array(xu, dtype=float)

    # ------------------------------------------------------------------ #
    # Decode
    # ------------------------------------------------------------------ #

    def decode(self, x: np.ndarray) -> Dict[str, Any]:
        x = np.array(x, dtype=float).flatten()
        if x.shape[0] != self.n_var:
            raise ValueError(f"Expected x of length {self.n_var}, got {x.shape}")

        n_arch = len(self.arch_options)

        # Decode architecture indices
        raw_arch: Dict[str, Any] = {}
        for i, (key, options) in enumerate(self.arch_options):
            idx = _clip_int(x[i], 0, len(options) - 1)
            raw_arch[key] = options[idx]

        model_config = self.model_config_assembler(raw_arch)

        # Threshold groups
        threshold_groups = _clip_int(x[n_arch], 1, int(self.max_threshold_groups))

        # Core types
        core_types: List[Dict[str, int]] = []
        idx = n_arch + 1
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
            "allow_axon_tiling": bool(self.allow_axon_tiling),
        }

        return {
            "model_config": model_config,
            "platform_constraints": platform_constraints,
            "threshold_groups": int(threshold_groups),
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
                    self.allow_axon_tiling,
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
                    self.allow_axon_tiling,
                ))
            return 0.0 if self.validate(configuration) else 1.0
        except Exception:
            return 1e6

    # ------------------------------------------------------------------ #
    # Evaluation (model-agnostic)
    # ------------------------------------------------------------------ #

    def evaluate(self, configuration: Dict[str, Any]) -> Dict[str, float]:
        key = _json_key(configuration)
        if key in self._cache:
            return self._cache[key]

        large = 1e18

        if not self.validate(configuration):
            obj = {"accuracy": 0.0, "wasted_area": large, "total_params": large}
            self._cache[key] = obj
            return obj

        mc = configuration["model_config"]
        pcfg = configuration["platform_constraints"]
        threshold_groups = int(configuration["threshold_groups"])

        torch.manual_seed(int(self.accuracy_seed))
        np.random.seed(int(self.accuracy_seed))

        try:
            builder = self.builder_factory(
                self.device,
                self.input_shape,
                self.num_classes,
                int(pcfg["max_axons"]),
                int(pcfg["max_neurons"]),
                {**pcfg, "target_tq": int(self.target_tq)},
            )
            model = builder.build(mc).to(self.device)

            # Initialise any Lazy modules before inspecting parameters.
            from torch.nn.parameter import UninitializedParameter

            init_ok = True
            try:
                model.eval()
                with torch.no_grad():
                    try:
                        model_device = next(model.parameters()).device
                    except StopIteration:
                        model_device = self.device
                    dummy = torch.zeros((1, *tuple(self.input_shape)), device=model_device)
                    _ = model(dummy)
            except Exception:
                init_ok = False

            if any(isinstance(p, UninitializedParameter) for p in model.parameters()):
                init_ok = False

            if not init_ok:
                obj = {"accuracy": 0.0, "wasted_area": large, "total_params": large}
                self._cache[key] = obj
                return obj

            total_params = float(sum(int(p.numel()) for p in model.parameters()))

            # Layout-only mapping
            layout_mapper = LayoutIRMapping(
                max_axons=int(pcfg["max_axons"]),
                max_neurons=int(pcfg["max_neurons"]),
                allow_axon_tiling=bool(pcfg.get("allow_axon_tiling", False)),
                threshold_groups=threshold_groups,
                threshold_seed=int(self.accuracy_seed),
            )
            softcores = layout_mapper.collect_layout_softcores(model.get_mapper_repr())

            core_types = [
                LayoutHardCoreType(
                    max_axons=int(ct["max_axons"]),
                    max_neurons=int(ct["max_neurons"]),
                    count=int(ct["count"]),
                )
                for ct in pcfg["cores"]
            ]
            pack = pack_layout(softcores=softcores, core_types=core_types)

            if not pack.feasible:
                obj = {"accuracy": 0.0, "wasted_area": large, "total_params": total_params}
                self._cache[key] = obj
                return obj

            # Accuracy objective
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
            accuracy = float(acc_eval.evaluate(model))

            obj = {
                "accuracy": float(accuracy),
                "wasted_area": float(pack.unused_area_total),
                "total_params": float(total_params),
            }
            self._cache[key] = obj
            return obj
        except Exception:
            obj = {"accuracy": 0.0, "wasted_area": large, "total_params": large}
            self._cache[key] = obj
            return obj
