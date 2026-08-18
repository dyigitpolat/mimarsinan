"""Architecture / platform search or fixed-configuration passthrough."""

from __future__ import annotations

from mimarsinan.pipelining.core.steps.pipeline_step import (
    METRIC_CARRIED,
    METRIC_MEASURED,
    PipelineStep,
)
from mimarsinan.pipelining.determinism import isolated_rng_stream
from mimarsinan.pipelining.core.model_config_emit import emit_model_config_entries
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
from mimarsinan.pipelining.core.search_mode import derive_search_mode
from mimarsinan.search.results import ACCURACY_OBJECTIVE_NAME
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    OptimizerType,
    build_fixed_platform_constraints,
    create_optimizer,
    resolve_arch_options,
    search_result_to_jsonable,
    write_search_visualizations,
)
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_problem import (
    build_search_problem,
    no_candidate_failure,
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

        arch_options, assembler = resolve_arch_options(
            builder_cls, arch_cfg, input_shape,
            searches_model=search_mode in ("model", "joint"),
            model_type=model_type,
        )

        pop_size = int(arch_cfg.get("pop_size", 12))
        generations = int(arch_cfg.get("generations", 5))
        seed = int(arch_cfg.get("seed", 0))
        optimizer_type: OptimizerType = arch_cfg.get("optimizer", "nsga2")

        problem, active_objective_names = build_search_problem(
            self.pipeline,
            search_mode=search_mode,
            builder_cls=builder_cls,
            arch_options=arch_options,
            model_config_assembler=assembler,
            seed=seed,
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
            # A search that rejected everything must SAY WHY, in the refusals
            # and the constraint census the problem itself recorded.
            raise RuntimeError(no_candidate_failure(problem))

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
