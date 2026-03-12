from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Sequence, Tuple

from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.pipelining.model_registry import ModelRegistry
from mimarsinan.search.optimizers.nsga2_optimizer import NSGA2Optimizer
from mimarsinan.search.problems.joint_arch_hw_problem import JointArchHwProblem
from mimarsinan.visualization.search_visualization import (
    create_interactive_search_report,
    write_final_population_json,
)


OptimizerType = Literal["nsga2", "kedi"]


# ====================================================================== #
# Generic Kedi optimizer helpers
# ====================================================================== #


def _build_kedi_config_schema_generic(
    arch_options: List[Tuple[str, List[Any]]],
    arch_cfg: Dict[str, Any],
    target_tq: int,
) -> Dict[str, Any]:
    num_core_types = arch_cfg.get("num_core_types", 2)
    core_type_counts = arch_cfg.get("core_type_counts", [200, 200])
    core_axons_bounds = arch_cfg.get("core_axons_bounds", [64, 1024])
    core_neurons_bounds = arch_cfg.get("core_neurons_bounds", [64, 1024])
    max_threshold_groups = arch_cfg.get("max_threshold_groups", 3)

    model_config_desc = {key: f"one of {values}" for key, values in arch_options}

    return {
        "model_config": model_config_desc,
        "platform_constraints": {
            "cores": (
                f"list of {num_core_types} objects, each with max_axons "
                f"(int {core_axons_bounds[0]}-{core_axons_bounds[1]}, multiple of 8), "
                f"max_neurons (int {core_neurons_bounds[0]}-{core_neurons_bounds[1]}, "
                f"multiple of 8), count (fixed: {core_type_counts}). "
                f"Different core types may have different sizes (heterogeneous)."
            ),
            "max_axons": "MIN of all cores' max_axons (computed automatically).",
            "max_neurons": "MIN of all cores' max_neurons (computed automatically).",
            "target_tq": f"{target_tq} (fixed)",
            "weight_bits": "8 (fixed)",
            "allow_core_coalescing": "false (fixed)",
        },
        "threshold_groups": f"integer from 1 to {max_threshold_groups}",
    }


def _build_kedi_example_config_generic(
    arch_options: List[Tuple[str, List[Any]]],
    arch_cfg: Dict[str, Any],
    target_tq: int,
) -> Dict[str, Any]:
    num_core_types = arch_cfg.get("num_core_types", 2)
    core_type_counts = arch_cfg.get("core_type_counts", [200, 200])

    model_config = {key: values[len(values) // 2] for key, values in arch_options}

    small_axons, small_neurons = 512, 512
    large_axons, large_neurons = 1024, 1024
    core_dims = [(small_axons, small_neurons), (large_axons, large_neurons)]
    cores = []
    for i in range(num_core_types):
        ax, neu = core_dims[i % len(core_dims)]
        cores.append({"max_axons": ax, "max_neurons": neu, "count": core_type_counts[i]})

    return {
        "model_config": model_config,
        "platform_constraints": {
            "cores": cores,
            "max_axons": min(c["max_axons"] for c in cores),
            "max_neurons": min(c["max_neurons"] for c in cores),
            "target_tq": target_tq,
            "weight_bits": 8,
            "allow_core_coalescing": False,
        },
        "threshold_groups": 2,
    }


def _build_kedi_constraints_desc(
    arch_options: List[Tuple[str, List[Any]]],
    arch_cfg: Dict[str, Any],
) -> str:
    core_axons_bounds = arch_cfg.get("core_axons_bounds", [64, 1024])
    core_neurons_bounds = arch_cfg.get("core_neurons_bounds", [64, 1024])
    max_threshold_groups = arch_cfg.get("max_threshold_groups", 3)

    option_lines = "\n".join(
        f"   - {key}: must be one of {values}" for key, values in arch_options
    )

    return f"""
CRITICAL CONSTRAINTS:

1. MODEL CONFIGURATION OPTIONS:
{option_lines}

2. TILING CONSTRAINT (most important!):
   Softcores are tiled to fit the SMALLEST core type. The tiling limit is
   max_axons = min(cores[*].max_axons) and max_neurons = min(cores[*].max_neurons).
   The largest layer input must fit in max_axons - 1.

3. HETEROGENEOUS CORES:
   Different core types may have different max_axons/max_neurons.
   Softcores are sized for the smallest core, so they can pack into ANY core type.
   The max_axons/max_neurons in platform_constraints must equal the MIN across all cores.

4. Core dimensions: each core's max_axons and max_neurons must be multiples of 8,
   between {core_axons_bounds[0]} and {core_axons_bounds[1]}.
   Core neurons must be in range {core_neurons_bounds[0]}-{core_neurons_bounds[1]}.

5. threshold_groups: integer from 1 to {max_threshold_groups}.
"""


def _create_optimizer(
    optimizer_type: OptimizerType,
    arch_cfg: Dict[str, Any],
    arch_options: List[Tuple[str, List[Any]]],
    seed: int,
    pop_size: int,
    generations: int,
    target_tq: int = 16,
):
    if optimizer_type == "kedi":
        try:
            from mimarsinan.search.optimizers.kedi_optimizer import KediOptimizer

            kedi_model = arch_cfg.get("kedi_model", "openai:gpt-4o")
            kedi_adapter = arch_cfg.get("kedi_adapter", "pydantic")
            candidates_per_batch = arch_cfg.get("candidates_per_batch", 5)
            max_regen_rounds = arch_cfg.get("max_regen_rounds", 10)
            max_failed_examples = arch_cfg.get("max_failed_examples", 5)
            llm_retries = arch_cfg.get("llm_retries", 3)

            config_schema = _build_kedi_config_schema_generic(arch_options, arch_cfg, target_tq)
            example_config = _build_kedi_example_config_generic(arch_options, arch_cfg, target_tq)
            constraints_desc = arch_cfg.get("constraints_description") or \
                _build_kedi_constraints_desc(arch_options, arch_cfg)

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

    return NSGA2Optimizer(
        pop_size=pop_size,
        generations=generations,
        seed=seed,
        eliminate_duplicates=True,
        verbose=True,
    )


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


def _derive_arch_options(
    builder_cls: type,
    arch_cfg: Dict[str, Any],
    input_shape: tuple,
) -> Tuple[List[Tuple[str, List[Any]]], Dict[str, Any]]:
    """
    Derive (arch_options, schema_map) from a builder's schema and optional
    get_nas_search_options() classmethod.

    arch_options is a list of (key, values) pairs for all searchable config keys:
    - "select" fields with multiple options
    - numeric fields listed in get_nas_search_options()

    schema_map maps key -> field descriptor for assembler type coercion.
    """
    schema = getattr(builder_cls, "get_config_schema", lambda: [])()
    schema_map = {f["key"]: f for f in schema}

    nas_opts_fn = getattr(builder_cls, "get_nas_search_options", None)
    builder_options: Dict[str, List[Any]] = (
        nas_opts_fn(input_shape=input_shape) if nas_opts_fn else {}
    )

    arch_options: List[Tuple[str, List[Any]]] = []
    for field_desc in schema:
        key = field_desc["key"]
        field_type = field_desc.get("type")
        if field_type == "select" and "options" in field_desc:
            values = arch_cfg.get(f"{key}_options", field_desc["options"])
            if len(values) > 1:
                arch_options.append((key, list(values)))
        elif key in builder_options:
            values = arch_cfg.get(f"{key}_options", builder_options[key])
            if len(values) > 1:
                arch_options.append((key, list(values)))

    # Also include builder_options keys not covered by schema
    schema_keys = {f["key"] for f in schema}
    for key, default_values in builder_options.items():
        if key not in schema_keys:
            values = arch_cfg.get(f"{key}_options", default_values)
            if len(values) > 1:
                arch_options.append((key, list(values)))

    return arch_options, schema_map


def _make_assembler(schema: List[Dict[str, Any]], schema_map: Dict[str, Any]):
    """
    Return a model_config assembler that:
    - Fills in schema defaults for all fields
    - Coerces searched values to their declared types
    """
    def assembler(raw: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for field_desc in schema:
            if "default" in field_desc:
                result[field_desc["key"]] = field_desc["default"]
        for k, v in raw.items():
            field_info = schema_map.get(k, {})
            if field_info.get("type") == "number":
                default = field_info.get("default", 0)
                result[k] = float(v) if isinstance(default, float) else int(v)
            else:
                result[k] = v
        return result
    return assembler


# ====================================================================== #
# The pipeline step
# ====================================================================== #

class ArchitectureSearchStep(PipelineStep):
    """
    Produces model configuration + resolved platform constraints.

    Modes:
    - user: passthrough (uses pipeline.config['model_config'] and existing platform constraints)
    - nas:  runs NSGA-II / Kedi joint search via the generic JointArchHwProblem

    NAS search-space definitions are derived generically from each builder's
    get_config_schema() and optional get_nas_search_options() classmethods.
    No per-model NAS provider classes are needed.
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
        m = getattr(self, "_last_metric", None)
        return m if m is not None else self.pipeline.get_target_metric()

    def process(self):
        model_type = self.pipeline.config["model_type"]
        configuration_mode = self.pipeline.config.get("configuration_mode", "user")

        builder_cls = ModelRegistry.get_builder_cls(model_type)

        def _make_builder(max_axons=None, max_neurons=None, extra_cfg=None):
            cfg = self.pipeline.config if extra_cfg is None else extra_cfg
            return builder_cls(
                self.pipeline.config["device"],
                self.pipeline.config["input_shape"],
                self.pipeline.config["num_classes"],
                max_axons if max_axons is not None else self.pipeline.config["max_axons"],
                max_neurons if max_neurons is not None else self.pipeline.config["max_neurons"],
                cfg,
            )

        if configuration_mode == "user":
            model_config = self.pipeline.config["model_config"]
            builder = _make_builder()
            self.add_entry("model_builder", builder, "pickle")
            self.add_entry("model_config", model_config)

            cores = self.pipeline.config.get("cores", [])
            if cores:
                effective_max_axons = max(ct["max_axons"] for ct in cores)
                effective_max_neurons = max(ct["max_neurons"] for ct in cores)
            else:
                effective_max_axons = self.pipeline.config.get("max_axons")
                effective_max_neurons = self.pipeline.config.get("max_neurons")

            self.add_entry(
                "platform_constraints_resolved",
                {
                    "cores": cores,
                    "max_axons": effective_max_axons,
                    "max_neurons": effective_max_neurons,
                    "allow_core_coalescing": self.pipeline.config.get("allow_core_coalescing", False),
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

        # ---- Derive search space generically from builder schema ---- #
        arch_cfg = self.pipeline.config.get("arch_search", {})
        input_shape = tuple(self.pipeline.config["input_shape"])

        arch_options, schema_map = _derive_arch_options(builder_cls, arch_cfg, input_shape)

        if not arch_options:
            schema = getattr(builder_cls, "get_config_schema", lambda: [])()
            raise NotImplementedError(
                f"No NAS search space defined for model_type='{model_type}'. "
                f"Add get_nas_search_options() or 'select' fields with multiple options "
                f"to {builder_cls.__name__}.get_config_schema(). "
                f"Current schema keys: {[f['key'] for f in schema]}"
            )

        schema = getattr(builder_cls, "get_config_schema", lambda: [])()
        assembler = _make_assembler(schema, schema_map)

        # Generic validate_fn and constraint_fn derived from builder.validate_config
        validate_config_fn = getattr(builder_cls, "validate_config", None)

        def validate_fn(model_config, platform_constraints, inp_shape):
            if validate_config_fn is not None:
                return bool(validate_config_fn(
                    model_config, platform_constraints, inp_shape
                ))
            return True

        def constraint_fn(model_config, platform_constraints, inp_shape):
            if validate_config_fn is not None:
                if not validate_config_fn(
                    model_config, platform_constraints, inp_shape
                ):
                    return 1.0
            return 0.0

        # ---- Common NAS parameters ---- #
        pop_size = int(arch_cfg.get("pop_size", 12))
        generations = int(arch_cfg.get("generations", 5))
        seed = int(arch_cfg.get("seed", 0))

        num_core_types = int(arch_cfg.get(
            "num_core_types", len(self.pipeline.config.get("cores", [])) or 1
        ))
        default_counts = [int(ct.get("count", 100)) for ct in self.pipeline.config.get("cores", [])]
        if len(default_counts) < num_core_types:
            default_counts = (default_counts + [100] * num_core_types)[:num_core_types]
        core_type_counts = arch_cfg.get("core_type_counts", default_counts)

        core_axons_bounds = tuple(arch_cfg.get("core_axons_bounds", [64, 2048]))
        core_neurons_bounds = tuple(arch_cfg.get("core_neurons_bounds", [64, 2048]))
        max_threshold_groups = int(arch_cfg.get("max_threshold_groups", 4))

        warmup_fraction = float(arch_cfg.get("warmup_fraction", 0.10))
        training_batch_size = arch_cfg.get("training_batch_size")

        accuracy_evaluator = str(arch_cfg.get("accuracy_evaluator", "extrapolating"))
        extrapolation_num_train_epochs = int(arch_cfg.get("extrapolation_num_train_epochs", 1))
        extrapolation_num_checkpoints = int(arch_cfg.get("extrapolation_num_checkpoints", 5))
        extrapolation_target_epochs = int(arch_cfg.get("extrapolation_target_epochs", 10))

        optimizer_type: OptimizerType = arch_cfg.get("optimizer", "nsga2")

        # ---- Build the single generic problem ---- #
        problem = JointArchHwProblem(
            data_provider_factory=self.pipeline.data_provider_factory,
            device=self.pipeline.config["device"],
            input_shape=input_shape,
            num_classes=int(self.pipeline.config["num_classes"]),
            target_tq=int(self.pipeline.config["target_tq"]),
            lr=float(self.pipeline.config["lr"]),
            builder_factory=builder_cls,
            arch_options=arch_options,
            model_config_assembler=assembler,
            validate_fn=validate_fn,
            constraint_fn=constraint_fn,
            num_core_types=num_core_types,
            core_type_counts=core_type_counts,
            core_axons_bounds=(int(core_axons_bounds[0]), int(core_axons_bounds[1])),
            core_neurons_bounds=(int(core_neurons_bounds[0]), int(core_neurons_bounds[1])),
            max_threshold_groups=max_threshold_groups,
            accuracy_seed=seed,
            warmup_fraction=warmup_fraction,
            training_batch_size=(int(training_batch_size) if training_batch_size is not None else None),
            accuracy_evaluator=accuracy_evaluator,
            extrapolation_num_train_epochs=extrapolation_num_train_epochs,
            extrapolation_num_checkpoints=extrapolation_num_checkpoints,
            extrapolation_target_epochs=extrapolation_target_epochs,
            pruning_fraction=float(self.pipeline.config.get("pruning_fraction", 0.0)),
        )

        optimizer = _create_optimizer(
            optimizer_type=optimizer_type,
            arch_cfg=arch_cfg,
            arch_options=arch_options,
            seed=seed,
            pop_size=pop_size,
            generations=generations,
            target_tq=int(self.pipeline.config["target_tq"]),
        )

        print(f"[ArchitectureSearchStep] model_type='{model_type}' | optimizer={optimizer_type} "
              f"| search space: {[(k, len(v)) for k, v in arch_options]}")

        _reporter = getattr(self.pipeline, "reporter", None)
        _report_fn = getattr(_reporter, "report", None) if _reporter else None
        result = optimizer.optimize(problem, reporter=_report_fn)
        result_json = _search_result_to_jsonable(result)
        acc = None
        if result.best and result.best.objectives:
            acc = result.best.objectives.get("accuracy")
        if acc is not None:
            self._last_metric = float(acc)

        try:
            out_dir = self.pipeline.working_directory
            write_final_population_json(result_json, os.path.join(out_dir, "final_population.json"))
            report_html = os.path.join(out_dir, "search_report.html")
            create_interactive_search_report(result_json, report_html)

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

        # Apply global has_bias (not searchable) to every core from pipeline config
        global_has_bias = self.pipeline.config.get("platform_constraints", {}).get("has_bias", True)
        for c in platform_constraints.get("cores", []):
            c["has_bias"] = global_has_bias

        merged_config = {**self.pipeline.config, **platform_constraints}
        builder = builder_cls(
            self.pipeline.config["device"],
            input_shape,
            self.pipeline.config["num_classes"],
            int(platform_constraints["max_axons"]),
            int(platform_constraints["max_neurons"]),
            merged_config,
        )

        self.add_entry("model_builder", builder, "pickle")
        self.add_entry("model_config", model_config)
        platform_constraints["allow_core_coalescing"] = bool(self.pipeline.config.get("allow_core_coalescing", False))
        self.add_entry("platform_constraints_resolved", platform_constraints)
        self.add_entry("architecture_search_result", result_json)
        sim_steps = int(round(self.pipeline.config.get("simulation_steps", 32)))
        self.add_entry("scaled_simulation_length", sim_steps)
