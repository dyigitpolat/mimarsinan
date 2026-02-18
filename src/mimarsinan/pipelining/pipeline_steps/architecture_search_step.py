from __future__ import annotations

import os
from typing import Any, Dict, List, Literal

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


# Type alias for optimizer selection
OptimizerType = Literal["nsga2", "kedi"]


def _build_kedi_config_schema(
    arch_cfg: Dict[str, Any],
    input_shape: tuple,
    target_tq: int,
) -> Dict[str, Any]:
    """Build configuration schema for Kedi optimizer LLM."""
    # Extract search space options
    patch_rows = arch_cfg.get("patch_rows_options", [1, 2, 4, 7, 14, 28])
    patch_cols = arch_cfg.get("patch_cols_options", [1, 2, 4, 7, 14, 28])
    patch_channels = arch_cfg.get("patch_channels_options", [16, 32, 48, 64, 96, 128])
    fc_w1 = arch_cfg.get("fc_w1_options", [32, 64, 96, 128])
    fc_w2 = arch_cfg.get("fc_w2_options", [32, 64, 96, 128])
    num_core_types = arch_cfg.get("num_core_types", 2)
    core_type_counts = arch_cfg.get("core_type_counts", [200, 200])
    core_axons_bounds = arch_cfg.get("core_axons_bounds", [64, 1024])
    core_neurons_bounds = arch_cfg.get("core_neurons_bounds", [64, 1024])
    max_threshold_groups = arch_cfg.get("max_threshold_groups", 3)

    return {
        "model_config": {
            "base_activation": "LeakyReLU (fixed)",
            "patch_n_1": f"integer from {patch_rows} (must divide input height {input_shape[-2]})",
            "patch_m_1": f"integer from {patch_cols} (must divide input width {input_shape[-1]})",
            "patch_c_1": f"integer from {patch_channels}",
            "fc_w_1": f"integer from {fc_w1}",
            "fc_w_2": f"integer from {fc_w2}",
        },
        "platform_constraints": {
            "cores": (
                f"list of {num_core_types} objects, each with max_axons "
                f"(int {core_axons_bounds[0]}-{core_axons_bounds[1]}, multiple of 8), "
                f"max_neurons (int {core_neurons_bounds[0]}-{core_neurons_bounds[1]}, "
                f"multiple of 8), count (fixed: {core_type_counts}). "
                f"Different core types may have different sizes (heterogeneous)."
            ),
            "max_axons": (
                "MIN of all cores' max_axons (computed automatically). "
                "This is the tiling limit -- softcores are sized to fit the smallest core type."
            ),
            "max_neurons": (
                "MIN of all cores' max_neurons (computed automatically). "
                "This is the tiling limit -- softcores are sized to fit the smallest core type."
            ),
            "target_tq": f"{target_tq} (fixed)",
            "weight_bits": "8 (fixed)",
            "allow_axon_tiling": "false (fixed)",
        },
        "threshold_groups": f"integer from 1 to {max_threshold_groups}",
    }


def _build_kedi_example_config(
    arch_cfg: Dict[str, Any],
    input_shape: tuple,
    target_tq: int,
) -> Dict[str, Any]:
    """Build an example configuration for Kedi optimizer LLM.

    IMPORTANT: The example must be VALID, satisfying all constraints.
    Key constraint: patch_n_1 * patch_m_1 * patch_c_1 <= max_axons - 1
    where max_axons = min of all cores' max_axons (tiling limit).
    """
    num_core_types = arch_cfg.get("num_core_types", 2)
    core_type_counts = arch_cfg.get("core_type_counts", [200, 200])

    # Example with heterogeneous cores: a small type and a large type.
    # patch_n_1=2, patch_m_1=2, patch_c_1=64 gives 2*2*64=256
    # min_axons must be >= 257, we use 512 for the small type.
    patch_n = 2  # divides 28
    patch_m = 2  # divides 28
    patch_c = 64

    # Heterogeneous: small cores (512) and large cores (1024)
    small_axons, small_neurons = 512, 512
    large_axons, large_neurons = 1024, 1024

    core_dims = [
        (small_axons, small_neurons),
        (large_axons, large_neurons),
    ]

    cores = []
    for i in range(num_core_types):
        ax, neu = core_dims[i % len(core_dims)]
        cores.append({"max_axons": ax, "max_neurons": neu, "count": core_type_counts[i]})

    # Tiling limit = min across core types
    min_axons = min(c["max_axons"] for c in cores)
    min_neurons = min(c["max_neurons"] for c in cores)

    return {
        "model_config": {
            "base_activation": "LeakyReLU",
            "patch_n_1": patch_n,
            "patch_m_1": patch_m,
            "patch_c_1": patch_c,
            "fc_w_1": 64,
            "fc_w_2": 64,
        },
        "platform_constraints": {
            "cores": cores,
            "max_axons": min_axons,
            "max_neurons": min_neurons,
            "target_tq": target_tq,
            "weight_bits": 8,
            "allow_axon_tiling": False,
        },
        "threshold_groups": 2,
    }


def _create_optimizer(
    optimizer_type: OptimizerType,
    arch_cfg: Dict[str, Any],
    seed: int,
    pop_size: int,
    generations: int,
    input_shape: tuple = None,
    target_tq: int = 16,
):
    """
    Create the appropriate optimizer based on configuration.
    
    Args:
        optimizer_type: "nsga2" or "kedi"
        arch_cfg: Architecture search configuration dict
        seed: Random seed
        pop_size: Population size
        generations: Number of generations
        input_shape: Input tensor shape (for Kedi schema)
        target_tq: Target TQ (for Kedi schema)
        
    Returns:
        Configured optimizer instance
    """
    if optimizer_type == "kedi":
        try:
            from mimarsinan.search.optimizers.kedi_optimizer import KediOptimizer
            
            # Kedi-specific configuration
            kedi_model = arch_cfg.get("kedi_model", "openai:gpt-4o")
            kedi_adapter = arch_cfg.get("kedi_adapter", "pydantic")
            candidates_per_batch = arch_cfg.get("candidates_per_batch", 5)
            max_regen_rounds = arch_cfg.get("max_regen_rounds", 10)
            max_failed_examples = arch_cfg.get("max_failed_examples", 5)
            llm_retries = arch_cfg.get("llm_retries", 3)
            
            # Build config schema and example for LLM
            config_schema = None
            example_config = None
            if input_shape is not None:
                config_schema = _build_kedi_config_schema(arch_cfg, input_shape, target_tq)
                example_config = _build_kedi_example_config(arch_cfg, input_shape, target_tq)
            
            # Build constraint description for LLM with explicit calculations
            h = int(input_shape[-2]) if input_shape else 28
            w = int(input_shape[-1]) if input_shape else 28
            constraints_desc = arch_cfg.get("constraints_description") or f"""
CRITICAL CONSTRAINTS:

1. PATCH DIVISIBILITY: patch_n_1 must divide {h}, patch_m_1 must divide {w}
   - Valid patch_n_1 options: {_divisors(h)}
   - Valid patch_m_1 options: {_divisors(w)}

2. TILING CONSTRAINT (most important!):
   Softcores are tiled to fit the SMALLEST core type. The tiling limit is
   max_axons = min(cores[*].max_axons) and max_neurons = min(cores[*].max_neurons).
   The largest layer input must fit in max_axons - 1.

   Calculate: patch_count = patch_n_1 * patch_m_1
   The LARGEST input is: patch_count * patch_c_1

   RULE: (patch_n_1 * patch_m_1 * patch_c_1) <= max_axons - 1

   Examples:
   - patch_n_1=7, patch_m_1=7, patch_c_1=16: 7*7*16 = 784 -> needs max_axons >= 785
   - patch_n_1=4, patch_m_1=4, patch_c_1=32: 4*4*32 = 512 -> needs max_axons >= 513
   - patch_n_1=2, patch_m_1=2, patch_c_1=64: 2*2*64 = 256 -> needs max_axons >= 257

3. HETEROGENEOUS CORES:
   Different core types may have different max_axons/max_neurons.
   Softcores are sized for the smallest core, so they can pack into ANY core type.
   Larger cores simply hold more softcores, improving utilization.
   The max_axons/max_neurons in platform_constraints must equal the MIN across all cores.

4. Core dimensions: each core's max_axons and max_neurons must be multiples of 8, between 64 and 1024.

5. threshold_groups: integer from 1 to 3.
"""
            
            return KediOptimizer(
                pop_size=pop_size,
                generations=generations,
                candidates_per_batch=candidates_per_batch,
                max_regen_rounds=max_regen_rounds,
                max_failed_examples=max_failed_examples,
                model=kedi_model,
                adapter_type=kedi_adapter,
                llm_retries=llm_retries,
                config_schema=config_schema,
                example_config=example_config,
                constraints_description=constraints_desc,
                verbose=True,
            )
        except ImportError as e:
            print(f"[ArchitectureSearchStep] Kedi optimizer not available: {e}")
            print("[ArchitectureSearchStep] Falling back to NSGA2")
            optimizer_type = "nsga2"
    
    # Default: NSGA2
    return NSGA2Optimizer(
        pop_size=pop_size,
        generations=generations,
        seed=seed,
        eliminate_duplicates=True,
        verbose=True,
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
        promises = [
            "model_config",
            "model_builder",
            "platform_constraints_resolved",
            "architecture_search_result",
            "scaled_simulation_length",
        ]
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
            sim_steps = int(round(self.pipeline.config.get("simulation_steps", 32)))
            self.add_entry("scaled_simulation_length", sim_steps)
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

        # Accuracy evaluator: "extrapolating" (curve-fit, default) or "fast" (1-epoch)
        accuracy_evaluator = str(arch_cfg.get("accuracy_evaluator", "extrapolating"))
        extrapolation_num_train_epochs = int(arch_cfg.get("extrapolation_num_train_epochs", 1))
        extrapolation_num_checkpoints = int(arch_cfg.get("extrapolation_num_checkpoints", 5))
        extrapolation_target_epochs = int(arch_cfg.get("extrapolation_target_epochs", 10))
        
        # Optimizer selection: "nsga2" (default) or "kedi" (LLM-based)
        optimizer_type: OptimizerType = arch_cfg.get("optimizer", "nsga2")

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
            accuracy_evaluator=accuracy_evaluator,
            extrapolation_num_train_epochs=extrapolation_num_train_epochs,
            extrapolation_num_checkpoints=extrapolation_num_checkpoints,
            extrapolation_target_epochs=extrapolation_target_epochs,
        )

        # Create optimizer based on configuration
        optimizer = _create_optimizer(
            optimizer_type=optimizer_type,
            arch_cfg=arch_cfg,
            seed=seed,
            pop_size=pop_size,
            generations=generations,
            input_shape=tuple(self.pipeline.config["input_shape"]),
            target_tq=int(self.pipeline.config["target_tq"]),
        )
        
        print(f"[ArchitectureSearchStep] Using {optimizer_type} optimizer")

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
        if not best_cfg:
            raise RuntimeError(
                "[ArchitectureSearchStep] Architecture search produced no candidates. "
                "Consider increasing pop_size or generations, or relaxing the search bounds."
            )

        if not problem.validate(best_cfg):
            cv = problem.constraint_violation(best_cfg)
            raise RuntimeError(
                f"[ArchitectureSearchStep] Architecture search failed to find a feasible "
                f"configuration (constraint violation = {cv:.1f}).  "
                f"Best candidate: {best_cfg}.  "
                f"Consider increasing pop_size/generations, widening core_axons_bounds, "
                f"or reducing the number of core types."
            )

        model_config = best_cfg["model_config"]
        platform_constraints = best_cfg["platform_constraints"]

        # Build the model builder using the search-resolved constraints (no
        # side-effect writes to pipeline.config -- downstream steps read from
        # the cached platform_constraints_resolved entry instead).
        builder = PerceptronMixerBuilder(
            self.pipeline.config["device"],
            self.pipeline.config["input_shape"],
            self.pipeline.config["num_classes"],
            int(platform_constraints["max_axons"]),
            int(platform_constraints["max_neurons"]),
            {**self.pipeline.config, **platform_constraints},
        )

        self.add_entry("model_builder", builder, "pickle")
        self.add_entry("model_config", model_config)
        self.add_entry("platform_constraints_resolved", platform_constraints)
        self.add_entry("architecture_search_result", result_json)
        sim_steps = int(round(self.pipeline.config.get("simulation_steps", 32)))
        self.add_entry("scaled_simulation_length", sim_steps)


