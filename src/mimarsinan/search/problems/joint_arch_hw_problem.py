from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.models.builders.perceptron_mixer_builder import PerceptronMixerBuilder
from mimarsinan.search.evaluators.fast_accuracy_evaluator import FastAccuracyEvaluator
from mimarsinan.search.evaluators.extrapolating_accuracy_evaluator import ExtrapolatingAccuracyEvaluator
from mimarsinan.search.problems.encoded_problem import EncodedProblem
from mimarsinan.search.results import ObjectiveSpec


def _json_key(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass
class JointPerceptronMixerArchHwProblem(EncodedProblem[Dict[str, Any]]):
    """
    Joint search over:
    - PerceptronMixer architecture parameters
    - hardware core-type dimensions + threshold grouping approximation

    Produces objectives:
    - hard_cores_used (min)
    - avg_unused_area_per_core (min)
    - total_params (min)
    - accuracy (max)
    """

    data_provider_factory: Any
    device: torch.device
    input_shape: Tuple[int, ...]
    num_classes: int
    target_tq: int
    lr: float

    # Search space
    patch_rows_options: Sequence[int]
    patch_cols_options: Sequence[int]
    patch_channels_options: Sequence[int]
    fc_w1_options: Sequence[int]
    fc_w2_options: Sequence[int]

    num_core_types: int
    core_type_counts: Sequence[int]
    core_axons_bounds: Tuple[int, int] = (64, 2048)
    core_neurons_bounds: Tuple[int, int] = (64, 2048)

    max_threshold_groups: int = 4
    allow_axon_tiling: bool = False

    # Evaluation knobs
    accuracy_seed: int = 0
    warmup_fraction: float = 0.10
    training_batch_size: Optional[int] = None

    # Accuracy evaluator selection: "fast" (1-epoch) or "extrapolating" (curve-fit)
    accuracy_evaluator: str = "extrapolating"
    # Extra knobs for the extrapolating evaluator
    extrapolation_num_train_epochs: int = 1
    extrapolation_num_checkpoints: int = 5
    extrapolation_target_epochs: int = 10

    # Internal cache: key -> objectives dict
    _cache: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False)

    @property
    def objectives(self) -> Sequence[ObjectiveSpec]:
        return (
            ObjectiveSpec("hard_cores_used", "min"),
            ObjectiveSpec("avg_unused_area_per_core", "min"),
            ObjectiveSpec("total_params", "min"),
            ObjectiveSpec("accuracy", "max"),
        )

    @property
    def n_var(self) -> int:
        return 6 + 2 * int(self.num_core_types)

    @property
    def xl(self) -> np.ndarray:
        xl = []
        xl += [0, 0, 0, 0, 0]  # indices into option lists
        xl += [1]  # threshold_groups
        for _ in range(int(self.num_core_types)):
            xl += [int(self.core_axons_bounds[0]), int(self.core_neurons_bounds[0])]
        return np.array(xl, dtype=float)

    @property
    def xu(self) -> np.ndarray:
        xu = []
        xu += [
            len(self.patch_rows_options) - 1,
            len(self.patch_cols_options) - 1,
            len(self.patch_channels_options) - 1,
            len(self.fc_w1_options) - 1,
            len(self.fc_w2_options) - 1,
        ]
        xu += [int(self.max_threshold_groups)]
        for _ in range(int(self.num_core_types)):
            xu += [int(self.core_axons_bounds[1]), int(self.core_neurons_bounds[1])]
        return np.array(xu, dtype=float)

    def decode(self, x: np.ndarray) -> Dict[str, Any]:
        x = np.array(x, dtype=float).flatten()
        if x.shape[0] != self.n_var:
            raise ValueError(f"Expected x of length {self.n_var}, got {x.shape}")

        def _clip_int(v: float, lo: int, hi: int) -> int:
            return int(max(lo, min(hi, int(round(float(v))))))

        # architecture indices
        pr = self.patch_rows_options[_clip_int(x[0], 0, len(self.patch_rows_options) - 1)]
        pc = self.patch_cols_options[_clip_int(x[1], 0, len(self.patch_cols_options) - 1)]
        pch = self.patch_channels_options[_clip_int(x[2], 0, len(self.patch_channels_options) - 1)]
        fw1 = self.fc_w1_options[_clip_int(x[3], 0, len(self.fc_w1_options) - 1)]
        fw2 = self.fc_w2_options[_clip_int(x[4], 0, len(self.fc_w2_options) - 1)]

        threshold_groups = _clip_int(x[5], 1, int(self.max_threshold_groups))

        # core types
        core_types = []
        idx = 6
        for t in range(int(self.num_core_types)):
            ax = _clip_int(x[idx], int(self.core_axons_bounds[0]), int(self.core_axons_bounds[1]))
            neu = _clip_int(x[idx + 1], int(self.core_neurons_bounds[0]), int(self.core_neurons_bounds[1]))
            idx += 2

            # Quantize to multiples of 8 for more stable search (keeps values reasonable)
            ax = max(8, int(round(ax / 8)) * 8)
            neu = max(8, int(round(neu / 8)) * 8)

            core_types.append({"max_axons": ax, "max_neurons": neu, "count": int(self.core_type_counts[t])})

        # Tile to the SMALLEST core type so softcores can pack into ANY core,
        # maximising utilisation of heterogeneous hardware.
        min_axons = min(int(ct["max_axons"]) for ct in core_types)
        min_neurons = min(int(ct["max_neurons"]) for ct in core_types)

        model_config = {
            "base_activation": "LeakyReLU",
            "patch_n_1": int(pr),
            "patch_m_1": int(pc),
            "patch_c_1": int(pch),
            "fc_w_1": int(fw1),
            "fc_w_2": int(fw2),
        }

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

    def _quick_validate(self, cfg: Dict[str, Any]) -> bool:
        mc = cfg["model_config"]
        pc = int(mc["patch_m_1"])
        pr = int(mc["patch_n_1"])

        # Divisibility constraints for einops patching
        h = int(self.input_shape[-2])
        w = int(self.input_shape[-1])
        if h % pr != 0 or w % pc != 0:
            return False

        patch_h = h // pr
        patch_w = w // pc
        in_ch = int(self.input_shape[-3])
        patch_size = patch_h * patch_w * in_ch
        patch_count = pr * pc

        patch_channels = int(mc["patch_c_1"])

        # Most restrictive axon requirement is the classifier: patch_count * patch_channels (+bias).
        max_in_features = max(
            patch_size,
            patch_count,
            int(mc["fc_w_1"]),
            patch_channels,
            int(mc["fc_w_2"]),
            patch_count * patch_channels,
        )

        max_axons = int(cfg["platform_constraints"]["max_axons"])
        if not self.allow_axon_tiling and max_in_features > max_axons - 1:
            return False

        return True

    def validate(self, configuration: Dict[str, Any]) -> bool:
        try:
            return bool(self._quick_validate(configuration))
        except Exception:
            return False

    def constraint_violation(self, configuration: Dict[str, Any]) -> float:
        """Continuous constraint violation for pymoo's constraint-domination.

        Returns 0.0 when feasible, a positive value proportional to how far
        the configuration is from satisfying the axon/neuron constraints.
        """
        try:
            mc = configuration["model_config"]
            pc_m = int(mc["patch_m_1"])
            pr = int(mc["patch_n_1"])

            h = int(self.input_shape[-2])
            w = int(self.input_shape[-1])

            # Divisibility — hard to quantify, use a large step penalty
            if h % pr != 0 or w % pc_m != 0:
                return 1e6

            patch_h = h // pr
            patch_w = w // pc_m
            in_ch = int(self.input_shape[-3])
            patch_size = patch_h * patch_w * in_ch
            patch_count = pr * pc_m
            patch_channels = int(mc["patch_c_1"])

            max_in_features = max(
                patch_size,
                patch_count,
                int(mc["fc_w_1"]),
                patch_channels,
                int(mc["fc_w_2"]),
                patch_count * patch_channels,
            )

            max_axons = int(configuration["platform_constraints"]["max_axons"])
            violation = float(max_in_features - (max_axons - 1))
            return max(0.0, violation)
        except Exception:
            return 1e6

    def evaluate(self, configuration: Dict[str, Any]) -> Dict[str, float]:
        key = _json_key(configuration)
        if key in self._cache:
            return self._cache[key]

        # Penalties for invalid configs
        large = 1e18

        if not self.validate(configuration):
            obj = {
                "hard_cores_used": large,
                "avg_unused_area_per_core": large,
                "total_params": large,
                "accuracy": 0.0,
            }
            self._cache[key] = obj
            return obj

        mc = configuration["model_config"]
        pcfg = configuration["platform_constraints"]
        threshold_groups = int(configuration["threshold_groups"])

        # Deterministic init
        torch.manual_seed(int(self.accuracy_seed))
        np.random.seed(int(self.accuracy_seed))

        try:
            # Build model (untuned, 1-epoch eval)
            builder = PerceptronMixerBuilder(
                self.device,
                self.input_shape,
                self.num_classes,
                int(pcfg["max_axons"]),
                int(pcfg["max_neurons"]),
                {**pcfg, "target_tq": int(self.target_tq)},
            )
            model = builder.build(mc).to(self.device)

            # Initialize any Lazy modules (e.g., LazyBatchNorm1d) before inspecting parameters.
            from torch.nn.parameter import UninitializedParameter

            init_ok = True
            try:
                model.eval()
                with torch.no_grad():
                    # Ensure dummy is on the same device as the model parameters.
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
                obj = {
                    "hard_cores_used": large,
                    "avg_unused_area_per_core": large,
                    "total_params": large,
                    "accuracy": 0.0,
                }
                self._cache[key] = obj
                return obj

            total_params = float(sum(int(p.numel()) for p in model.parameters()))

            # Layout-only mapping: collect softcores (shape-only) using the mapper graph
            layout_mapper = LayoutIRMapping(
                max_axons=int(pcfg["max_axons"]),
                max_neurons=int(pcfg["max_neurons"]),
                allow_axon_tiling=bool(pcfg.get("allow_axon_tiling", False)),
                threshold_groups=threshold_groups,
                threshold_seed=int(self.accuracy_seed),
            )
            softcores = layout_mapper.collect_layout_softcores(model.get_mapper_repr())

            # Hardware core types for packing
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
                obj = {
                    "hard_cores_used": large,
                    "avg_unused_area_per_core": large,
                    "total_params": total_params,
                    "accuracy": 0.0,
                }
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
                "hard_cores_used": float(pack.cores_used),
                "avg_unused_area_per_core": float(pack.avg_unused_area_per_core),
                "total_params": float(total_params),
                "accuracy": float(accuracy),
            }
            self._cache[key] = obj
            return obj
        except Exception:
            obj = {
                "hard_cores_used": large,
                "avg_unused_area_per_core": large,
                "total_params": large,
                "accuracy": 0.0,
            }
            self._cache[key] = obj
            return obj


