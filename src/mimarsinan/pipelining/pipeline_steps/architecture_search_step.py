from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Sequence, Tuple

from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.pipelining.model_registry import ModelRegistry
from mimarsinan.pipelining.search_mode import derive_search_mode
from mimarsinan.search.optimizers.nsga2_optimizer import NSGA2Optimizer
from mimarsinan.search.problems.joint_arch_hw_problem import JointArchHwProblem
from mimarsinan.search.results import (
    ACCURACY_OBJECTIVE_NAME,
    resolve_active_objectives,
)
from mimarsinan.visualization.search_visualization import (
    create_interactive_search_report,
    write_final_population_json,
)


OptimizerType = Literal["nsga2", "agent_evolve"]


# ====================================================================== #
# Agentic Evolution config schema builders — mode-aware
# ====================================================================== #


def _build_agent_evolve_config_schema(
    search_mode: str,
    arch_options: List[Tuple[str, List[Any]]],
    arch_cfg: Dict[str, Any],
    target_tq: int,
) -> Dict[str, Any]:
    """Build the configuration schema description for Agentic Evolution LLM prompts."""
    schema: Dict[str, Any] = {}

    if search_mode in ("model", "joint"):
        schema["model_config"] = {key: f"one of {values}" for key, values in arch_options}

    if search_mode in ("hardware", "joint"):
        num_core_types = arch_cfg.get("num_core_types", 2)
        core_axons_bounds = arch_cfg.get("core_axons_bounds", [64, 1024])
        core_neurons_bounds = arch_cfg.get("core_neurons_bounds", [64, 1024])
        core_count_bounds = arch_cfg.get("core_count_bounds", [50, 500])

        schema["platform_constraints"] = {
            "cores": (
                f"list of {num_core_types} objects, each with max_axons "
                f"(int {core_axons_bounds[0]}-{core_axons_bounds[1]}, multiple of 8), "
                f"max_neurons (int {core_neurons_bounds[0]}-{core_neurons_bounds[1]}, "
                f"multiple of 8), count (int {core_count_bounds[0]}-{core_count_bounds[1]}). "
                f"Different core types may have different sizes (heterogeneous)."
            ),
            "target_tq": f"{target_tq} (fixed)",
            "weight_bits": "8 (fixed)",
        }

    return schema


def _build_agent_evolve_example_config(
    search_mode: str,
    arch_options: List[Tuple[str, List[Any]]],
    arch_cfg: Dict[str, Any],
    target_tq: int,
) -> Dict[str, Any]:
    """Build an example configuration for Agentic Evolution LLM prompts."""
    example: Dict[str, Any] = {}

    if search_mode in ("model", "joint"):
        example["model_config"] = {key: values[len(values) // 2] for key, values in arch_options}

    if search_mode in ("hardware", "joint"):
        num_core_types = arch_cfg.get("num_core_types", 2)
        core_dims = [(512, 512), (1024, 1024)]
        cores = []
        for i in range(num_core_types):
            ax, neu = core_dims[i % len(core_dims)]
            cores.append({"max_axons": ax, "max_neurons": neu, "count": 200})

        example["platform_constraints"] = {
            "cores": cores,
            "target_tq": target_tq,
            "weight_bits": 8,
        }

    return example


def _build_agent_evolve_constraints_desc(
    search_mode: str,
    arch_options: List[Tuple[str, List[Any]]],
    arch_cfg: Dict[str, Any],
) -> str:
    """Build constraint description for Agentic Evolution LLM prompts."""
    parts: List[str] = ["\nCRITICAL CONSTRAINTS:\n"]

    if search_mode in ("model", "joint") and arch_options:
        option_lines = "\n".join(f"   - {key}: must be one of {values}" for key, values in arch_options)
        parts.append(f"1. MODEL CONFIGURATION OPTIONS:\n{option_lines}\n")

    if search_mode in ("hardware", "joint"):
        core_axons_bounds = arch_cfg.get("core_axons_bounds", [64, 1024])
        core_neurons_bounds = arch_cfg.get("core_neurons_bounds", [64, 1024])
        core_count_bounds = arch_cfg.get("core_count_bounds", [50, 500])

        parts.append(
            f"2. HETEROGENEOUS CORES:\n"
            f"   Different core types may have different max_axons/max_neurons.\n"
            f"   Softcores are tiled for the LARGEST core type and packed by the bin-packer.\n"
        )
        parts.append(
            f"3. Core dimensions: max_axons and max_neurons must be multiples of 8,\n"
            f"   axons in [{core_axons_bounds[0]}, {core_axons_bounds[1]}],\n"
            f"   neurons in [{core_neurons_bounds[0]}, {core_neurons_bounds[1]}].\n"
            f"   Core count in [{core_count_bounds[0]}, {core_count_bounds[1]}].\n"
        )

    return "\n".join(parts)


# ====================================================================== #
# Optimizer factory
# ====================================================================== #


def _create_optimizer(
    optimizer_type: OptimizerType,
    arch_cfg: Dict[str, Any],
    search_mode: str,
    arch_options: List[Tuple[str, List[Any]]],
    seed: int,
    pop_size: int,
    generations: int,
    target_tq: int = 16,
):
    if optimizer_type == "agent_evolve":
        try:
            from mimarsinan.search.optimizers.agent_evolve_optimizer import AgentEvolveOptimizer

            agent_model = arch_cfg.get("agent_model", "openai:gpt-4o")
            candidates_per_batch = arch_cfg.get("candidates_per_batch", 5)
            max_regen_rounds = arch_cfg.get("max_regen_rounds", 10)
            max_failed_examples = arch_cfg.get("max_failed_examples", 5)
            llm_retries = arch_cfg.get("llm_retries", 3)

            config_schema = _build_agent_evolve_config_schema(search_mode, arch_options, arch_cfg, target_tq)
            example_config = _build_agent_evolve_example_config(search_mode, arch_options, arch_cfg, target_tq)
            constraints_desc = arch_cfg.get("constraints_description") or \
                _build_agent_evolve_constraints_desc(search_mode, arch_options, arch_cfg)

            return AgentEvolveOptimizer(
                pop_size=pop_size,
                generations=generations,
                candidates_per_batch=candidates_per_batch,
                max_regen_rounds=max_regen_rounds,
                max_failed_examples=max_failed_examples,
                model=agent_model,
                llm_retries=llm_retries,
                config_schema=config_schema,
                example_config=example_config,
                constraints_description=constraints_desc,
                verbose=True,
            )
        except ImportError as e:
            print(f"[ArchitectureSearchStep] Agentic Evolution optimizer not available: {e}")
            print("[ArchitectureSearchStep] Falling back to NSGA2")

    return NSGA2Optimizer(
        pop_size=pop_size,
        generations=generations,
        seed=seed,
        eliminate_duplicates=True,
        verbose=True,
    )


# ====================================================================== #
# Helpers
# ====================================================================== #


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
    """Derive (arch_options, schema_map) from builder's schema + get_nas_search_options()."""
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

    schema_keys = {f["key"] for f in schema}
    for key, default_values in builder_options.items():
        if key not in schema_keys:
            values = arch_cfg.get(f"{key}_options", default_values)
            if len(values) > 1:
                arch_options.append((key, list(values)))

    return arch_options, schema_map


def _make_assembler(schema: List[Dict[str, Any]], schema_map: Dict[str, Any]):
    """Return a model_config assembler with type coercion and defaults."""
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


def _build_fixed_platform_constraints(pipeline_config: Dict) -> Dict[str, Any]:
    """Build a platform_constraints dict from user-provided pipeline config.

    No global max_axons/max_neurons — only ``cores`` list.
    """
    cores = list(pipeline_config.get("cores", []))
    if not cores:
        cores = [{"max_axons": 256, "max_neurons": 256, "count": 1000}]

    return {
        "cores": cores,
        "target_tq": pipeline_config.get("target_tq", 32),
        "weight_bits": pipeline_config.get("weight_bits", 8),
        "allow_core_coalescing": bool(pipeline_config.get("allow_core_coalescing", False)),
        "allow_scheduling": bool(pipeline_config.get("allow_scheduling", False)),
    }


# ====================================================================== #
# The pipeline step
# ====================================================================== #

class ArchitectureSearchStep(PipelineStep):
    """
    Produces model configuration + resolved platform constraints.

    Modes (derived from ``search_mode``):
    - fixed:    passthrough (uses pipeline config directly)
    - model:    searches NN architecture, HW config fixed from user
    - hardware: searches HW config, NN architecture fixed from user
    - joint:    searches both NN architecture and HW config
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
        search_mode = derive_search_mode(self.pipeline.config)

        if search_mode == "fixed":
            self._process_fixed()
        else:
            self._process_search(search_mode)

    # ------------------------------------------------------------------ #
    # Fixed mode — passthrough
    # ------------------------------------------------------------------ #

    def _process_fixed(self):
        model_type = self.pipeline.config["model_type"]
        builder_cls = ModelRegistry.get_builder_cls(model_type)

        model_config = self.pipeline.config["model_config"]
        builder = builder_cls(
            self.pipeline.config["device"],
            self.pipeline.config["input_shape"],
            self.pipeline.config["num_classes"],
            self.pipeline.config,
        )

        self.add_entry("model_builder", builder, "pickle")
        self.add_entry("model_config", model_config)

        pcfg = _build_fixed_platform_constraints(self.pipeline.config)

        # Propagate has_bias to every core type
        global_has_bias = self.pipeline.config.get("platform_constraints", {}).get("has_bias", True)
        for c in pcfg.get("cores", []):
            c.setdefault("has_bias", global_has_bias)

        self.add_entry("platform_constraints_resolved", pcfg)
        self.add_entry("architecture_search_result", {"search_mode": "fixed"})
        sim_steps = int(round(self.pipeline.config.get("simulation_steps", 32)))
        self.add_entry("scaled_simulation_length", sim_steps)

    # ------------------------------------------------------------------ #
    # Search mode — run optimisation
    # ------------------------------------------------------------------ #

    def _process_search(self, search_mode: str):
        model_type = self.pipeline.config["model_type"]
        builder_cls = ModelRegistry.get_builder_cls(model_type)
        arch_cfg = self.pipeline.config.get("arch_search", {})
        input_shape = tuple(self.pipeline.config["input_shape"])

        # ---- Derive architecture search space (if searching model) ----
        arch_options: List[Tuple[str, List[Any]]] = []
        assembler = None

        if search_mode in ("model", "joint"):
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

        if assembler is None:
            assembler = lambda raw: dict(raw)

        # ---- Fixed values for non-searched dimensions ----
        fixed_model_config = None
        fixed_platform_constraints = None

        if search_mode == "hardware":
            fixed_model_config = dict(self.pipeline.config.get("model_config", {}))

        if search_mode == "model":
            fixed_platform_constraints = _build_fixed_platform_constraints(self.pipeline.config)

        # ---- Validate and constraint functions ----
        validate_config_fn = getattr(builder_cls, "validate_config", None)

        def validate_fn(model_config, platform_constraints, inp_shape):
            if validate_config_fn is not None:
                return bool(validate_config_fn(model_config, platform_constraints, inp_shape))
            return True

        def constraint_fn(model_config, platform_constraints, inp_shape):
            if validate_config_fn is not None:
                if not validate_config_fn(model_config, platform_constraints, inp_shape):
                    return 1.0
            return 0.0

        # ---- Common NAS parameters ----
        pop_size = int(arch_cfg.get("pop_size", 12))
        generations = int(arch_cfg.get("generations", 5))
        seed = int(arch_cfg.get("seed", 0))

        num_core_types = int(arch_cfg.get(
            "num_core_types", len(self.pipeline.config.get("cores", [])) or 1
        ))

        core_axons_bounds = tuple(arch_cfg.get("core_axons_bounds", [64, 2048]))
        core_neurons_bounds = tuple(arch_cfg.get("core_neurons_bounds", [64, 2048]))
        core_count_bounds = tuple(arch_cfg.get("core_count_bounds", [50, 500]))

        warmup_fraction = float(arch_cfg.get("warmup_fraction", 0.10))
        training_batch_size = arch_cfg.get("training_batch_size") or None

        accuracy_evaluator = str(arch_cfg.get("accuracy_evaluator", "extrapolating"))
        extrapolation_num_train_epochs = int(arch_cfg.get("extrapolation_num_train_epochs", 1))
        extrapolation_num_checkpoints = int(arch_cfg.get("extrapolation_num_checkpoints", 5))
        extrapolation_target_epochs = int(arch_cfg.get("extrapolation_target_epochs", 10))

        optimizer_type: OptimizerType = arch_cfg.get("optimizer", "nsga2")

        # ---- Resolve active objectives ----
        user_objectives = arch_cfg.get("objectives")
        active_objectives = resolve_active_objectives(search_mode, user_objectives)
        active_objective_names = [o.name for o in active_objectives]

        # ---- Build the problem ----
        problem = JointArchHwProblem(
            data_provider_factory=self.pipeline.data_provider_factory,
            device=self.pipeline.config["device"],
            input_shape=input_shape,
            num_classes=int(self.pipeline.config["num_classes"]),
            target_tq=int(self.pipeline.config["target_tq"]),
            lr=float(self.pipeline.config["lr"]),
            search_mode=search_mode,
            builder_factory=builder_cls,
            arch_options=arch_options,
            model_config_assembler=assembler,
            validate_fn=validate_fn,
            constraint_fn=constraint_fn,
            fixed_model_config=fixed_model_config,
            fixed_platform_constraints=fixed_platform_constraints,
            active_objective_names=active_objective_names,
            num_core_types=num_core_types,
            core_axons_bounds=(int(core_axons_bounds[0]), int(core_axons_bounds[1])),
            core_neurons_bounds=(int(core_neurons_bounds[0]), int(core_neurons_bounds[1])),
            core_count_bounds=(int(core_count_bounds[0]), int(core_count_bounds[1])),
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
            search_mode=search_mode,
            arch_options=arch_options,
            seed=seed,
            pop_size=pop_size,
            generations=generations,
            target_tq=int(self.pipeline.config["target_tq"]),
        )

        print(f"[ArchitectureSearchStep] model_type='{model_type}' | search_mode={search_mode} "
              f"| optimizer={optimizer_type} | objectives={active_objective_names} "
              f"| arch vars: {[(k, len(v)) for k, v in arch_options]}")

        _reporter = getattr(self.pipeline, "reporter", None)
        _report_fn = getattr(_reporter, "report", None) if _reporter else None
        result = optimizer.optimize(problem, reporter=_report_fn)
        result_json = _search_result_to_jsonable(result)

        acc = None
        if result.best and result.best.objectives:
            acc = result.best.objectives.get(ACCURACY_OBJECTIVE_NAME)
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
                f"Consider increasing pop_size/generations or widening core bounds."
            )

        model_config = best_cfg["model_config"]
        platform_constraints = best_cfg["platform_constraints"]

        # Apply global has_bias (not searchable)
        global_has_bias = self.pipeline.config.get("platform_constraints", {}).get("has_bias", True)
        for c in platform_constraints.get("cores", []):
            c["has_bias"] = global_has_bias

        # For model-only search, merge fixed platform constraints
        if search_mode == "model" and fixed_platform_constraints:
            platform_constraints = {**fixed_platform_constraints, **platform_constraints}

        # Build the builder with resolved constraints
        merged_config = {**self.pipeline.config, **platform_constraints}
        builder = builder_cls(
            self.pipeline.config["device"],
            input_shape,
            self.pipeline.config["num_classes"],
            merged_config,
        )

        # Write discovered parameters
        discovered = {
            "search_mode_used": search_mode,
            "discovered_model_config": model_config if search_mode in ("model", "joint") else None,
            "discovered_platform_constraints": platform_constraints if search_mode in ("hardware", "joint") else None,
            "active_objectives": active_objective_names,
            "best_objectives": result.best.objectives,
        }

        self.add_entry("model_builder", builder, "pickle")
        self.add_entry("model_config", model_config)
        platform_constraints["allow_core_coalescing"] = bool(self.pipeline.config.get("allow_core_coalescing", False))
        platform_constraints["allow_scheduling"] = bool(self.pipeline.config.get("allow_scheduling", False))
        self.add_entry("platform_constraints_resolved", platform_constraints)
        self.add_entry("architecture_search_result", {**result_json, **discovered})
        sim_steps = int(round(self.pipeline.config.get("simulation_steps", 32)))
        self.add_entry("scaled_simulation_length", sim_steps)
