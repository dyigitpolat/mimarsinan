"""CompilagentOptimizer — drives a compilagent OptimizationSession."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

from compilagent import (
    HarnessRunRequest,
    NullSink,
    OptimizationSession,
    OptimizationWorkspace,
    harness_registry,
    register_workload_safely,
    run_session,
)

from mimarsinan.search.optimizers.base import SearchOptimizer
from mimarsinan.search.results import ObjectiveSpec, SearchResult
from mimarsinan.search.search_space_description import SearchSpaceDescription

from .backend import MimarsinanLayoutBackend
from .guided_toolset import GuidedToolset
from .optimizer_result import build_search_result, build_workload_instance
from .sink import MultiObjectiveSink
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

    model: str = "openai:gpt-4o"
    harness_id: str = "pydantic_ai"
    max_candidates: int = 8
    max_continuations: int = 4
    system_prompt_extra: str = ""
    user_prompt: str = _DEFAULT_USER_PROMPT

    active_objective_names: Sequence[str] = ()

    workspace_dir: Optional[str] = None
    invalid_penalty: float = 1e18
    verbose: bool = True

    _trace_reporter: Any = field(default=None, init=False, repr=False)
    _trace_seq: int = field(default=0, init=False, repr=False)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def optimize(self, problem: Any, reporter: Any = None) -> SearchResult:
        objectives: List[ObjectiveSpec] = list(problem.objectives)
        if not objectives:
            raise ValueError("SearchProblem.objectives must not be empty")

        self._trace_reporter = reporter
        self._trace_seq = 0
        try:
            return self._run(problem, objectives)
        finally:
            self._trace_reporter = None

    def _run(
        self,
        problem: Any,
        objectives: List[ObjectiveSpec],
    ) -> SearchResult:
        workload_id = f"mimarsinan_layout_{uuid.uuid4().hex[:10]}"
        spec = build_workload_spec(workload_id)
        register_workload_safely(spec)(build_workload_instance)
        register_problem(workload_id, problem)

        workspace_root = Path(self.workspace_dir) if self.workspace_dir else Path.cwd()
        workspace = OptimizationWorkspace(session_cwd=workspace_root)
        backend = MimarsinanLayoutBackend()
        sink = MultiObjectiveSink(NullSink(), live_reporter=self._trace_reporter)

        active_objective_names = (
            tuple(self.active_objective_names)
            or tuple(o.name for o in objectives)
        )

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

        guided_toolset = GuidedToolset(
            session.toolset,
            sink=sink,
            backend=backend,
            objectives=objectives,
        )

        request = HarnessRunRequest(
            toolset=guided_toolset,  # pyright: ignore[reportArgumentType] — deliberate duck-typed drop-in for the frozen-slots Toolset
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

        return build_search_result(
            sink=sink,
            objectives=objectives,
            elapsed_ms=elapsed,
            backend=backend,
            invalid_penalty=self.invalid_penalty,
        )

    def _build_system_instructions(self, objective_names: Sequence[str]) -> str:
        base = _default_system_instructions(objective_names)
        if self.system_prompt_extra:
            return f"{base}\n\n{self.system_prompt_extra.strip()}"
        return base


__all__ = ["CompilagentOptimizer"]
