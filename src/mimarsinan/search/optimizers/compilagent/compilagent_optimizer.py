"""``CompilagentOptimizer`` — drives a compilagent ``OptimizationSession``.

The optimizer is the third option ``ArchitectureSearchStep`` exposes,
alongside ``NSGA2Optimizer`` and ``AgentEvolveOptimizer``. It:

1. Builds a per-run ``WorkloadSpec`` and registers the live
   ``JointArchHwProblem`` so :class:`MimarsinanLayoutBackend` can
   delegate into it.
2. Constructs an ``OptimizationSession`` wired to a
   :class:`MultiObjectiveSink` so every ``OBJECTIVES_RECORDED`` event is
   captured for Pareto-front reconstruction.
3. Drives the session through ``run_session(...)`` against a configurable
   harness (default: ``pydantic_ai``).
4. Drains the sink, converts each captured candidate into a
   :class:`CandidateResult`, and reuses
   :mod:`mimarsinan.search.optimizers.agent_evolve_support` (Pareto +
   minimax) to build the final :class:`SearchResult`.
5. Streams the same ``search_event`` JSON shape the AgentEvolve optimizer
   emits (so the live GUI works without GUI-side changes for the basic
   per-generation summary).

The optimizer never re-implements Pareto-front computation, candidate
scoring, or `SearchResult` shape — those live in `agent_evolve_support`
and `search.results`. We only translate compilagent's per-candidate
record into the shared `CandidateResult` envelope.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from compilagent import (
    HarnessRunRequest,
    NullSink,
    OptimizationSession,
    OptimizationWorkspace,
    harness_registry,
    register_workload_safely,
    run_session,
)

from mimarsinan.search.optimizers.agent_evolve_support import (
    CandidateResult,
    compute_pareto_front,
    result_to_candidate,
    select_best_candidate_minimax,
    sort_pareto_results_minimax_first,
)
from mimarsinan.search.optimizers.base import SearchOptimizer
from mimarsinan.search.results import (
    ACCURACY_OBJECTIVE_NAME,
    Candidate,
    ObjectiveSpec,
    SearchResult,
)
from mimarsinan.search.search_space_description import SearchSpaceDescription

from .backend import MimarsinanLayoutBackend
from .guided_toolset import GuidedToolset
from .sink import CandidateRecord, MultiObjectiveSink  # noqa: F401  (re-exported via type hint)
from .workload import (
    build_workload_spec,
    register_problem,
    unregister_problem,
)


_DEFAULT_USER_PROMPT = (
    "Co-search a neural-architecture + hardware mapping for spiking "
    "deployment. This is a MULTI-OBJECTIVE problem — every active "
    "objective is equally weighted and you are building a Pareto front, "
    "not maximising one number.\n\n"
    "Required workflow (do not skip steps):\n"
    "  1. INSPECT FIRST. Call `inspect_workload` once — its response "
    "now includes a `[BASELINE FOOTPRINT]` block with the default "
    "model's softcore count and largest dimensions; that is your "
    "starting point for sizing the chip. Then call `list_objectives` "
    "(goal directions), `inspect_search_space` (lever ranges), and "
    "`inspect_layer_breakdown` with `candidate_id='baseline'` to see "
    "how layers map.\n"
    "  2. PROPOSE A DELIBERATELY DIVERSE BATCH. Use `propose_candidates` "
    "with 3-5 plans that span the extremes of the design space — "
    "small/medium/large core counts, narrow/wide FC widths, "
    "small/large patches. Do NOT cluster all proposals in one region.\n"
    "  3. RUN THE BATCH. Call `run_candidates(candidate_ids=...)`. "
    "Each result now includes a `[GUIDANCE]` block showing the live "
    "Pareto front, per-metric leaders, under-explored axes, and "
    "next-direction suggestions — READ IT before doing anything else.\n"
    "  4. REFLECT. Call `pareto_front` and `metric_summary` to see the "
    "global multi-objective state. Use `query_top_candidates(metric=...)` "
    "to find the best on each axis. Use `inspect_layout_stats` to "
    "understand why winners win.\n"
    "  5. NEXT BATCH targets the axes the GUIDANCE block flagged as "
    "under-explored. Do not repeat candidates the agent already ran.\n\n"
    "Stop when proposing more candidates would not change the Pareto "
    "front. Conclude with `synthesize_findings` + `compare_runs`."
)


def _default_system_instructions(objective_names: Sequence[str]) -> str:
    obj_list = ", ".join(objective_names) if objective_names else "(none)"
    return (
        "You are a neuromorphic-compiler researcher driving a "
        "multi-objective architecture + hardware co-search.\n\n"
        f"Active objectives (ALL equally weighted; no axis is "
        f"'the primary'): {obj_list}.\n"
        "Use `list_objectives` to see the goal direction (max/min) and "
        "unit for each axis.\n\n"
        "Proposal vocabulary — every intervention is one of:\n"
        "  - `target.kind='arch'`, `selector=<model_config key>` "
        "(activation, patch sizes, FC widths, etc.).\n"
        "  - `target.kind='hw.core'`, `selector='<core_index>.<dim>'` "
        "where `<dim>` ∈ {`max_axons`, `max_neurons`, `count`}.\n\n"
        "Hardware levers are OPEN INTEGER RANGES, not fixed enums: "
        "`max_axons` and `max_neurons` accept any positive multiple of "
        "8 (typical 8-8192); `count` accepts any positive integer "
        "(typical 1-4096). You decide the values — the backend validates "
        "them.\n\n"
        "Model-size vs chip-capacity intuition: `param_utilization_pct` "
        "= used_softcore_area / nominal_chip_capacity. If utilization "
        "stays single-digit, your chip is much bigger than the model — "
        "shrink `count` toward the baseline softcore count (visible in "
        "the `[BASELINE FOOTPRINT]` block) and shrink the per-core dims "
        "to bracket the largest softcore. Conversely, if compiles fail "
        "with 'No more hard cores available', the chip is too small "
        "for the proposed model — grow `count` or widen the cores.\n\n"
        "Multi-objective tools (call these often!):\n"
        "  - `pareto_front()` — current non-dominated set.\n"
        "  - `metric_summary()` — best / worst / median per axis.\n"
        "  - `query_top_candidates(metric=...)` — top-k by one axis.\n"
        "  - `compare_candidates(candidate_ids=...)` — side-by-side.\n"
        "  - `compare_runs()` — full leaderboard, all objectives.\n\n"
        "Backend introspection (for understanding why a candidate wins):\n"
        "  - `inspect_softcores(candidate_id)` — per-softcore shape.\n"
        "  - `inspect_layer_breakdown(candidate_id)` — per-layer counts.\n"
        "  - `inspect_layout_stats(candidate_id)` — full LayoutVerificationStats.\n\n"
        "Each `run_candidate` / `run_candidates` response also carries "
        "a `[GUIDANCE]` block (Pareto front + leaders + under-explored "
        "axes + next-direction suggestions) and the full `objectives` "
        "dict — read both before proposing the next batch."
    )


@dataclass
class CompilagentOptimizer(SearchOptimizer):
    """Search backend that drives mimarsinan via compilagent's session loop."""

    pop_size: int = 8
    description: Optional[SearchSpaceDescription] = None

    # LLM / harness configuration
    model: str = "openai:gpt-4o"
    harness_id: str = "pydantic_ai"
    max_candidates: int = 8
    max_continuations: int = 4
    system_prompt_extra: str = ""
    user_prompt: str = _DEFAULT_USER_PROMPT

    # Result / objective configuration
    active_objective_names: Sequence[str] = ()

    # Misc
    workspace_dir: Optional[str] = None
    invalid_penalty: float = 1e18
    verbose: bool = True

    # Internal — populated during optimize()
    _trace_reporter: Any = field(default=None, init=False, repr=False)
    _trace_seq: int = field(default=0, init=False, repr=False)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def optimize(self, problem: Any, reporter: Any = None) -> SearchResult:
        """Run one compilagent session against ``problem`` and return ``SearchResult``."""

        objectives: List[ObjectiveSpec] = list(problem.objectives)
        if not objectives:
            raise ValueError("SearchProblem.objectives must not be empty")

        self._trace_reporter = reporter
        self._trace_seq = 0
        try:
            return self._run(problem, objectives)
        finally:
            self._trace_reporter = None

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def _run(
        self,
        problem: Any,
        objectives: List[ObjectiveSpec],
    ) -> SearchResult:
        workload_id = f"mimarsinan_layout_{uuid.uuid4().hex[:10]}"
        spec = build_workload_spec(workload_id)
        # Register the workload builder so `OptimizationSession` can
        # resolve it; the build returns a stub `WorkloadInstance` since
        # the backend never invokes a `forward` callable.
        register_workload_safely(spec)(_build_workload_instance)
        register_problem(workload_id, problem)

        workspace_root = Path(self.workspace_dir) if self.workspace_dir else Path.cwd()
        workspace = OptimizationWorkspace(session_cwd=workspace_root)
        backend = MimarsinanLayoutBackend()
        sink = MultiObjectiveSink(NullSink(), live_reporter=self._trace_reporter)

        active_objective_names = (
            tuple(self.active_objective_names)
            or tuple(o.name for o in objectives)
        )

        # Session header — the live monitor uses this to populate the
        # COMPILAGENT title bar (model, harness, objective catalogue).
        sink.emit_session_start(
            workload_id=workload_id,
            backend_id="mimarsinan_layout",
            model=str(self.model),
            harness=str(self.harness_id),
            max_candidates=int(self.max_candidates),
            max_continuations=int(self.max_continuations),
            objectives=[{"name": s.name, "goal": s.goal} for s in objectives],
        )

        started = time.perf_counter()
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        session = OptimizationSession(
            workload_id=workload_id,
            run_id=run_id,
            workspace=workspace,
            sink=sink,
            backend=backend,
            max_candidates=int(self.max_candidates),
        )

        # Wrap the canonical session toolset so `run_candidate` /
        # `run_candidates` results carry a `[GUIDANCE]` block (live
        # Pareto front, per-metric leaders, under-explored axes,
        # next-batch suggestions) and the first `inspect_workload`
        # response carries a `[BASELINE FOOTPRINT]` block (softcore
        # counts, max dimensions, baseline objective values). Everything
        # else passes through unchanged.
        guided_toolset = GuidedToolset(
            session.toolset,
            sink=sink,
            backend=backend,
            objectives=objectives,
        )

        request = HarnessRunRequest(
            toolset=guided_toolset,
            system_instructions=self._build_system_instructions(active_objective_names),
            user_prompt=self.user_prompt,
            model_id=self.model,
        )

        try:
            harness = harness_registry.get(self.harness_id)
        except KeyError as exc:
            self._log(f"[CompilagentOptimizer] harness `{self.harness_id}` not registered: {exc}")
            unregister_problem(workload_id)
            raise

        try:
            asyncio.run(
                run_session(
                    session=session,
                    harness=harness,
                    request=request,
                    max_continuations=int(self.max_continuations),
                )
            )
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            try:
                session.finalize()
            except Exception as exc:  # noqa: BLE001
                self._log(f"[CompilagentOptimizer] session.finalize() failed: {exc!r}")
            unregister_problem(workload_id)

        return self._build_result(
            sink=sink,
            objectives=objectives,
            elapsed_ms=elapsed,
            backend=backend,
        )

    # ------------------------------------------------------------------ #
    # Result construction
    # ------------------------------------------------------------------ #

    def _build_result(
        self,
        *,
        sink: MultiObjectiveSink,
        objectives: List[ObjectiveSpec],
        elapsed_ms: float,
        backend: MimarsinanLayoutBackend,
    ) -> SearchResult:
        valid_results: List[CandidateResult] = []
        all_candidates: List[Candidate] = []
        active_names = {o.name for o in objectives}

        penalty_objectives = self._penalty_objectives(objectives)

        for rec in sink.records():
            # Pull the actual `(model_config, platform_constraints)` from
            # the backend's per-candidate payload — populated during
            # `compile()`. Falls back to the sink-attached config or to a
            # description-only stub when the candidate failed before
            # ``compile`` produced a payload.
            cfg = self._configuration_for(backend, rec)
            layout_md = self._layout_metadata_for(backend, rec.candidate_id)
            if rec.rejected or not rec.objectives:
                cr = CandidateResult(
                    configuration=dict(cfg),
                    objectives=dict(penalty_objectives),
                    is_valid=False,
                    error_message=rec.reject_reason or "compile or evaluation failed",
                    failure_phase="compile",
                )
                metadata = {"is_pareto": False, "valid": False}
                if layout_md:
                    metadata["layout"] = layout_md
                all_candidates.append(result_to_candidate(cr, metadata))
                continue

            objs: Dict[str, float] = {
                name: float(rec.objectives.get(name, 0.0)) for name in active_names
            }
            cr = CandidateResult(
                configuration=dict(cfg), objectives=objs, is_valid=True,
            )
            valid_results.append(cr)
            metadata = {
                "is_pareto": False,
                "candidate_id": rec.candidate_id,
            }
            if layout_md:
                metadata["layout"] = layout_md
            all_candidates.append(result_to_candidate(cr, metadata))

        pareto = compute_pareto_front(valid_results, objectives)
        pareto_sorted = sort_pareto_results_minimax_first(pareto, objectives)
        best_result = select_best_candidate_minimax(pareto, objectives)
        if best_result is not None:
            best = result_to_candidate(best_result, {"is_pareto": True})
        else:
            best = Candidate(configuration={}, objectives={}, metadata={})

        pareto_candidates = [
            result_to_candidate(r, {"is_pareto": True}) for r in pareto_sorted
        ]

        # Mark the corresponding entries in all_candidates as on the front.
        pareto_keys = {
            json.dumps(c.configuration, sort_keys=True, default=str)
            for c in pareto_candidates
        }
        for c in all_candidates:
            key = json.dumps(c.configuration, sort_keys=True, default=str)
            if key in pareto_keys:
                c.metadata["is_pareto"] = True

        history = [{
            "gen": 1,
            "valid_count": len(valid_results),
            "failed_count": len(sink.failed_records()),
            "pareto_size": len(pareto),
            "elapsed_ms": elapsed_ms,
        }]

        # Emit the final Pareto-front + session-complete events on the
        # compilagent live channel. Note we never emit AgentEvolve-shaped
        # events (`generation_*`, `search_complete`) — the compilagent
        # live monitor is dedicated and the AgentEvolve UI never sees
        # compilagent traffic.
        sink.emit_pareto_update(
            pareto_front=[
                {
                    "candidate_id": (c.metadata or {}).get("candidate_id"),
                    "configuration": c.configuration,
                    "objectives": dict(c.objectives),
                }
                for c in pareto_candidates
            ]
        )
        sink.emit_session_complete(
            total_valid=len(valid_results),
            total_failed=len(sink.failed_records()),
            pareto_size=len(pareto),
            best_objectives=dict(best.objectives) if best.objectives else None,
            elapsed_ms=float(elapsed_ms),
        )

        return SearchResult(
            objectives=objectives,
            best=best,
            pareto_front=pareto_candidates,
            all_candidates=all_candidates,
            history=history,
        )

    @staticmethod
    def _configuration_for(
        backend: MimarsinanLayoutBackend, rec: CandidateRecord,
    ) -> Dict[str, Any]:
        """Return the full ``(model_config, platform_constraints)`` config.

        Falls back to the sink-attached configuration (set by callers
        that pre-decode the plan), then to a description-only stub.
        """

        try:
            payload = backend.get_candidate_payload(rec.candidate_id)
        except KeyError:
            payload = None
        if payload and isinstance(payload.get("config"), dict):
            return dict(payload["config"])
        if rec.configuration:
            return dict(rec.configuration)
        return {"description": rec.description}

    @staticmethod
    def _layout_metadata_for(
        backend: MimarsinanLayoutBackend, candidate_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return a slim layout summary for the search-tab / static report.

        The full per-softcore list lives in the backend's payload but is
        too verbose for the snapshot; we surface the per-layer aggregate
        and the headline ``LayoutVerificationStats`` numbers so the
        post-run UI can render a "Layout details" panel without reading
        artifact JSON files.
        """

        try:
            payload = backend.get_candidate_payload(candidate_id)
        except KeyError:
            return None
        layout_stats = payload.get("layout_stats", {}) or {}
        return {
            "softcore_count": len(payload.get("softcores", [])),
            "per_layer": payload.get("per_layer", []),
            "summary": {
                "total_cores": layout_stats.get("total_cores"),
                "total_softcores": layout_stats.get("total_softcores"),
                "neural_segment_count": layout_stats.get("neural_segment_count"),
                "threshold_group_count": layout_stats.get("threshold_group_count"),
                "fragmentation_pct": layout_stats.get("fragmentation_pct"),
                "mapped_params_pct": layout_stats.get("mapped_params_pct"),
                "schedule_pass_count": layout_stats.get("schedule_pass_count"),
                "schedule_sync_count": layout_stats.get("schedule_sync_count"),
            },
        }

    def _penalty_objectives(self, objectives: Sequence[ObjectiveSpec]) -> Dict[str, float]:
        return {
            spec.name: (0.0 if spec.goal == "max" else float(self.invalid_penalty))
            for spec in objectives
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_system_instructions(self, objective_names: Sequence[str]) -> str:
        base = _default_system_instructions(objective_names)
        if self.system_prompt_extra:
            return f"{base}\n\n{self.system_prompt_extra.strip()}"
        return base


def _build_workload_instance(spec):
    """Workload builder: returns a no-op WorkloadInstance.

    The mimarsinan backend never calls ``forward`` (it always works
    through ``JointArchHwProblem.evaluate`` instead), so a placeholder
    callable is sufficient and keeps the registry happy.
    """

    from compilagent import WorkloadInstance

    return WorkloadInstance(
        spec=spec, forward=lambda: None, example_inputs=(),
        metadata={"workload_id": spec.id},
    )


__all__ = ["CompilagentOptimizer"]
