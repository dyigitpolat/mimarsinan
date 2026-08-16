"""Architecture / platform search or fixed-configuration passthrough."""

from __future__ import annotations

from typing import List

from mimarsinan.pipelining.core.steps.pipeline_step import (
    METRIC_CARRIED,
    METRIC_MEASURED,
    PipelineStep,
)
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.determinism import isolated_rng_stream
from mimarsinan.pipelining.core.model_config_emit import emit_model_config_entries
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
from mimarsinan.pipelining.core.search_mode import derive_search_mode
from mimarsinan.search.problems.joint import JointArchHwProblem
from mimarsinan.deployment_record.platform_physics.resolve import (
    resolve_platform_physics,
)
from mimarsinan.config_schema.registry import effective_value as _effective
from mimarsinan.search.option_axes import build_option_axes
from mimarsinan.search.results import (
    ACCURACY_OBJECTIVE_NAME,
    resolve_active_objectives,
)
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    OptimizerType,
    build_fixed_platform_constraints,
    create_optimizer,
    derive_arch_options,
    make_assembler,
    make_platform_resolver,
    search_result_to_jsonable,
    write_search_visualizations,
)


class ArchitectureSearchStep(PipelineStep):
    """Resolve model_config and platform_constraints (search or fixed passthrough)."""

    PROMISES = (
        "model_config",
        "model_builder",
        "platform_constraints_resolved",
        "architecture_search_result",
    )

    @classmethod
    def applies_to(cls, plan):
        return plan.search_mode != "fixed"

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def validate(self):
        m = getattr(self, "_last_metric", None)
        return m if m is not None else self.pipeline.get_target_metric()

    def validate_metric_kind(self) -> str:
        m = getattr(self, "_last_metric", None)
        return METRIC_MEASURED if m is not None else METRIC_CARRIED

    def process(self):
        search_mode = derive_search_mode(self.pipeline.config)

        if search_mode == "fixed":
            self._process_fixed()
        else:
            # A search CHOOSES a configuration; it must not move the stream the
            # deployment draws from. Candidate scoring seeds the world to its
            # own scoring seed, so without this the weights a run deploys would
            # depend on how many candidates the search looked at — the same
            # config would train differently with the search ON than with the
            # winning chip declared by hand, which is exactly the difference a
            # searched-hardware guard cell must NOT have from its fixed anchor.
            with isolated_rng_stream():
                self._process_search(search_mode)

    def _process_fixed(self):
        emit_model_config_entries(self, self.pipeline.config)
        pcfg = build_fixed_platform_constraints(self.pipeline.config)
        self.add_entry("platform_constraints_resolved", pcfg)
        self.add_entry("architecture_search_result", {"search_mode": "fixed"})

    def _process_search(self, search_mode: str):
        model_type = self.pipeline.config["model_type"]
        builder_cls = ModelRegistry.get_builder_cls(model_type)
        arch_cfg = self.pipeline.config.get("arch_search", {})
        input_shape = tuple(self.pipeline.config["input_shape"])

        arch_options: List = []
        assembler = None

        if search_mode in ("model", "joint"):
            arch_options, schema_map = derive_arch_options(builder_cls, arch_cfg, input_shape)
            if not arch_options:
                schema = getattr(builder_cls, "get_config_schema", lambda: [])()
                raise NotImplementedError(
                    f"No NAS search space defined for model_type='{model_type}'. "
                    f"Add get_nas_search_options() or 'select' fields with multiple options "
                    f"to {builder_cls.__name__}.get_config_schema(). "
                    f"Current schema keys: {[f['key'] for f in schema]}"
                )
            schema = getattr(builder_cls, "get_config_schema", lambda: [])()
            assembler = make_assembler(schema, schema_map)

        if assembler is None:
            assembler = lambda raw: dict(raw)

        fixed_model_config = None
        # Every candidate platform is this run's DEPLOYMENT resolution with the
        # searched dimensions overlaid — searched chip and deployed chip are the
        # same chip by construction, in every mode.
        platform_resolver = make_platform_resolver(self.pipeline.config)

        if search_mode == "hardware":
            fixed_model_config = dict(self.pipeline.config.get("model_config", {}))

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

        user_objectives = arch_cfg.get("objectives")
        # THIS run's physics, from the same SSOT the deployment resolver uses: a
        # vendor-priced axis is refused by name here, before any candidate is built.
        run_physics = resolve_platform_physics(
            str(self.pipeline.config.get("platform_physics_profile", "") or ""),
            self.pipeline.config.get("platform_physics_overrides") or {},
        )
        active_objectives = resolve_active_objectives(
            search_mode, user_objectives, physics=run_physics,
            activity_factor=float(
                self.pipeline.config.get("activity_factor", 0.0) or 0.0
            ),
        )
        active_objective_names = [o.name for o in active_objectives]

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
            platform_resolver=platform_resolver,
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
            pruning_fraction=DeploymentPlan.of(self.pipeline).pruning_fraction,
            encoding_placement=str(
                self.pipeline.config.get("encoding_layer_placement", "subsume")
            ),
            # Deployment options the run promoted to search axes, and the floor
            # that shapes the feasible region they move through.
            option_axes=build_option_axes(arch_cfg.get("option_axes")),
            onchip_min_fraction=(
                float(_effective(self.pipeline.config, "onchip_min_fraction"))
                if bool(_effective(self.pipeline.config, "onchip_majority_gate"))
                else 0.0
            ),
        )

        optimizer = create_optimizer(
            optimizer_type=optimizer_type,
            arch_cfg=arch_cfg,
            search_mode=search_mode,
            arch_options=arch_options,
            seed=seed,
            pop_size=pop_size,
            generations=generations,
            target_tq=int(self.pipeline.config["target_tq"]),
            active_objective_names=active_objective_names,
        )

        print(f"[ArchitectureSearchStep] model_type='{model_type}' | search_mode={search_mode} "
              f"| optimizer={optimizer_type} | objectives={active_objective_names} "
              f"| arch vars: {[(k, len(v)) for k, v in arch_options]}")

        _reporter = getattr(self.pipeline, "reporter", None)
        _report_fn = getattr(_reporter, "report", None) if _reporter else None
        result = optimizer.optimize(problem, reporter=_report_fn)
        result_json = search_result_to_jsonable(result)

        acc = None
        if result.best and result.best.objectives:
            acc = result.best.objectives.get(ACCURACY_OBJECTIVE_NAME)
        if acc is not None:
            self._last_metric = float(acc)

        write_search_visualizations(result_json, self.pipeline.working_directory)

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
        # The winning candidate's platform IS the deployed platform: it came out
        # of the same resolver this step would run on it, so there is nothing
        # left to merge, patch, or re-stamp here.
        platform_constraints = problem.resolve_candidate_platform(
            best_cfg["platform_constraints"]
        )

        merged_config = {**self.pipeline.config, **platform_constraints}
        builder = builder_cls(
            self.pipeline.config["device"],
            input_shape,
            self.pipeline.config["num_classes"],
            merged_config,
        )

        # The winner's OPTIONS are part of what deploys: the run must execute
        # under the options the winner was scored with, or the deployed thing is
        # not the thing the search chose.
        searched_options = dict(best_cfg.get("deployment_options") or {})
        for key, value in searched_options.items():
            self.pipeline.config[key] = value

        discovered = {
            "search_mode_used": search_mode,
            "discovered_deployment_options": searched_options,
            "constraint_census": problem.constraint_census(),
            "discovered_model_config": model_config if search_mode in ("model", "joint") else None,
            "discovered_platform_constraints": platform_constraints if search_mode in ("hardware", "joint") else None,
            "active_objectives": active_objective_names,
            "best_objectives": result.best.objectives,
        }

        self.add_entry("model_builder", builder, "pickle")
        self.add_entry("model_config", model_config)
        self.add_entry("platform_constraints_resolved", platform_constraints)
        self.add_entry("architecture_search_result", {**result_json, **discovered})
