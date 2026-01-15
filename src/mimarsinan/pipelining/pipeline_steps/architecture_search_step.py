from __future__ import annotations

import os
from typing import Any, Dict, List

from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.search.optimizers.nsga2_optimizer import NSGA2Optimizer
from mimarsinan.search.problems.joint_arch_hw_problem import JointPerceptronMixerArchHwProblem

from mimarsinan.models.builders import (
    PerceptronMixerBuilder,
    SimpleConvBuilder,
    SimpleMLPBuilder,
    VGG16Builder,
)
from mimarsinan.visualization.search_visualization import (
    create_interactive_search_report,
    write_final_population_json,
)


def _divisors(n: int) -> List[int]:
    n = int(n)
    out = []
    for d in range(1, n + 1):
        if n % d == 0:
            out.append(d)
    return out


def _search_result_to_jsonable(result) -> Dict[str, Any]:
    def cand_to_dict(c):
        return {
            "configuration": c.configuration,
            "objectives": c.objectives,
            "metadata": c.metadata,
        }

    return {
        "objectives": [{"name": o.name, "goal": o.goal} for o in result.objectives],
        "best": cand_to_dict(result.best),
        "pareto_front": [cand_to_dict(c) for c in result.pareto_front],
        "all_candidates": [cand_to_dict(c) for c in result.all_candidates],
        "history": result.history,
    }


class ArchitectureSearchStep(PipelineStep):
    """
    Produces model configuration + resolved platform constraints.

    Modes:
    - user: passthrough (uses pipeline.config['model_config'] and existing platform constraints)
    - nas: runs NSGA-II joint search for PerceptronMixer + hardware layout constraints
    """

    def __init__(self, pipeline):
        requires = []
        promises = ["model_config", "model_builder", "platform_constraints_resolved", "architecture_search_result"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def validate(self):
        # This step does not change the trained model directly; keep target metric unchanged.
        return self.pipeline.get_target_metric()

    def process(self):
        model_type = self.pipeline.config["model_type"]
        configuration_mode = self.pipeline.config.get("configuration_mode", "user")

        # Builders need current (possibly searched) max_axons/max_neurons values.
        def _make_builder():
            builders = {
                "mlp_mixer": PerceptronMixerBuilder(
                    self.pipeline.config["device"],
                    self.pipeline.config["input_shape"],
                    self.pipeline.config["num_classes"],
                    self.pipeline.config["max_axons"],
                    self.pipeline.config["max_neurons"],
                    self.pipeline.config,
                ),
                "simple_mlp": SimpleMLPBuilder(
                    self.pipeline.config["device"],
                    self.pipeline.config["input_shape"],
                    self.pipeline.config["num_classes"],
                    self.pipeline.config["max_axons"],
                    self.pipeline.config["max_neurons"],
                    self.pipeline.config,
                ),
                "simple_conv": SimpleConvBuilder(
                    self.pipeline.config["device"],
                    self.pipeline.config["input_shape"],
                    self.pipeline.config["num_classes"],
                    self.pipeline.config["max_axons"],
                    self.pipeline.config["max_neurons"],
                    self.pipeline.config,
                ),
                "vgg16": VGG16Builder(
                    self.pipeline.config["device"],
                    self.pipeline.config["input_shape"],
                    self.pipeline.config["num_classes"],
                    self.pipeline.config["max_axons"],
                    self.pipeline.config["max_neurons"],
                    self.pipeline.config,
                ),
            }
            return builders[model_type]

        if configuration_mode == "user":
            model_config = self.pipeline.config["model_config"]
            builder = _make_builder()
            self.add_entry("model_builder", builder, "pickle")
            self.add_entry("model_config", model_config)
            self.add_entry(
                "platform_constraints_resolved",
                {
                    "cores": self.pipeline.config.get("cores", []),
                    "max_axons": self.pipeline.config.get("max_axons"),
                    "max_neurons": self.pipeline.config.get("max_neurons"),
                    "allow_axon_tiling": self.pipeline.config.get("allow_axon_tiling", False),
                    "target_tq": self.pipeline.config.get("target_tq"),
                    "simulation_steps": self.pipeline.config.get("simulation_steps"),
                    "weight_bits": self.pipeline.config.get("weight_bits"),
                },
            )
            self.add_entry("architecture_search_result", {"mode": "user"})
            return

        if configuration_mode != "nas":
            raise ValueError(f"Invalid configuration_mode: {configuration_mode}")

        if model_type != "mlp_mixer":
            raise NotImplementedError("ArchitectureSearchStep NAS currently supports only model_type='mlp_mixer'")

        # --- NSGA-II joint search (PerceptronMixer + layout/hardware constraints) ---
        arch_cfg = self.pipeline.config.get("arch_search", {})

        pop_size = int(arch_cfg.get("pop_size", 12))
        generations = int(arch_cfg.get("generations", 5))
        seed = int(arch_cfg.get("seed", 0))

        num_core_types = int(arch_cfg.get("num_core_types", len(self.pipeline.config.get("cores", [])) or 1))
        # counts are fixed by config (as requested)
        default_counts = [int(ct.get("count", 100)) for ct in self.pipeline.config.get("cores", [])]
        if len(default_counts) < num_core_types:
            default_counts = (default_counts + [100] * num_core_types)[:num_core_types]
        core_type_counts = arch_cfg.get("core_type_counts", default_counts)

        # Search space: patch rows/cols must divide input H/W
        h = int(self.pipeline.config["input_shape"][-2])
        w = int(self.pipeline.config["input_shape"][-1])

        patch_rows_options = arch_cfg.get("patch_rows_options", _divisors(h))
        patch_cols_options = arch_cfg.get("patch_cols_options", _divisors(w))
        patch_channels_options = arch_cfg.get(
            "patch_channels_options",
            [8, 16, 24, 32, 48, 64, 96, 128, 192, 256],
        )
        fc_w1_options = arch_cfg.get("fc_w1_options", [16, 32, 48, 64, 96, 128, 192, 256])
        fc_w2_options = arch_cfg.get("fc_w2_options", [16, 32, 48, 64, 96, 128, 192, 256])

        core_axons_bounds = tuple(arch_cfg.get("core_axons_bounds", [64, 2048]))
        core_neurons_bounds = tuple(arch_cfg.get("core_neurons_bounds", [64, 2048]))
        max_threshold_groups = int(arch_cfg.get("max_threshold_groups", 4))

        warmup_fraction = float(arch_cfg.get("warmup_fraction", 0.10))
        training_batch_size = arch_cfg.get("training_batch_size")  # optional

        problem = JointPerceptronMixerArchHwProblem(
            data_provider_factory=self.pipeline.data_provider_factory,
            device=self.pipeline.config["device"],
            input_shape=tuple(self.pipeline.config["input_shape"]),
            num_classes=int(self.pipeline.config["num_classes"]),
            target_tq=int(self.pipeline.config["target_tq"]),
            lr=float(self.pipeline.config["lr"]),
            patch_rows_options=patch_rows_options,
            patch_cols_options=patch_cols_options,
            patch_channels_options=patch_channels_options,
            fc_w1_options=fc_w1_options,
            fc_w2_options=fc_w2_options,
            num_core_types=num_core_types,
            core_type_counts=core_type_counts,
            core_axons_bounds=(int(core_axons_bounds[0]), int(core_axons_bounds[1])),
            core_neurons_bounds=(int(core_neurons_bounds[0]), int(core_neurons_bounds[1])),
            max_threshold_groups=max_threshold_groups,
            allow_axon_tiling=bool(self.pipeline.config.get("allow_axon_tiling", False)),
            accuracy_seed=seed,
            warmup_fraction=warmup_fraction,
            training_batch_size=(int(training_batch_size) if training_batch_size is not None else None),
        )

        optimizer = NSGA2Optimizer(
            pop_size=pop_size,
            generations=generations,
            seed=seed,
            eliminate_duplicates=True,
            verbose=True,
        )

        result = optimizer.optimize(problem)
        result_json = _search_result_to_jsonable(result)

        # Write MNIST PoC artifacts to working directory
        try:
            out_dir = self.pipeline.working_directory
            write_final_population_json(result_json, os.path.join(out_dir, "final_population.json"))
            # Single-page interactive HTML report
            report_html = os.path.join(out_dir, "search_report.html")
            create_interactive_search_report(result_json, report_html)

            # Best-effort cleanup of legacy reports
            for legacy in ["search_report.pdf", "search_report.png"]:
                legacy_path = os.path.join(out_dir, legacy)
                if os.path.exists(legacy_path):
                    try:
                        os.remove(legacy_path)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[ArchitectureSearchStep] Visualization failed (non-fatal): {e}")

        best_cfg = result.best.configuration
        model_config = best_cfg["model_config"]
        platform_constraints = best_cfg["platform_constraints"]

        # Apply selected platform constraints to pipeline config so downstream mapping uses them
        self.pipeline.config["cores"] = platform_constraints["cores"]
        self.pipeline.config["max_axons"] = int(platform_constraints["max_axons"])
        self.pipeline.config["max_neurons"] = int(platform_constraints["max_neurons"])
        self.pipeline.config["allow_axon_tiling"] = bool(platform_constraints.get("allow_axon_tiling", False))

        builder = _make_builder()

        self.add_entry("model_builder", builder, "pickle")
        self.add_entry("model_config", model_config)
        self.add_entry("platform_constraints_resolved", platform_constraints)
        self.add_entry("architecture_search_result", result_json)


