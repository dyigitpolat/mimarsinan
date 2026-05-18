"""Tests for the ``GuidedToolset`` middleware.

We do not exercise a real ``OptimizationSession`` here — that path is
covered by ``test_optimizer.py``. The point of this suite is to pin the
shape of the augmented tool results: ``inspect_workload`` gains a
``[BASELINE FOOTPRINT]`` block (once), and ``run_candidate`` /
``run_candidates`` results gain a ``[GUIDANCE]`` block built from the
live sink + backend payload cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compilagent import Plan, ToleranceConfig, Toolset, WorkloadKind, WorkloadSpec

from mimarsinan.search.optimizers.compilagent.backend import MimarsinanLayoutBackend
from mimarsinan.search.optimizers.compilagent.guided_toolset import GuidedToolset
from mimarsinan.search.optimizers.compilagent.sink import (
    CandidateRecord,
    MultiObjectiveSink,
)
from mimarsinan.search.optimizers.compilagent.workload import (
    register_problem,
    unregister_problem,
)
from mimarsinan.search.results import ObjectiveSpec

# Re-use the synthetic problem fixture from test_backend.py
from .test_backend import _make_problem, _make_workload  # noqa: E402


def _build_real_toolset() -> Toolset:
    """A minimal Toolset whose two relevant tools just echo a JSON payload."""
    from compilagent import ToolDecl
    from pydantic import BaseModel

    class _NoArgs(BaseModel):
        pass

    def inspect_workload() -> str:
        return json.dumps({"workload": "fake-summary"})

    def run_candidate(*, candidate_id: str) -> str:
        return json.dumps({"candidate_id": candidate_id, "compile_ok": True})

    class _RunArgs(BaseModel):
        candidate_id: str

    return Toolset(tools=(
        ToolDecl(
            name="inspect_workload",
            description="echo",
            args_schema=_NoArgs.model_json_schema(),
            handler=inspect_workload,
            args_model=_NoArgs,
            read_only=True,
        ),
        ToolDecl(
            name="run_candidate",
            description="echo",
            args_schema=_RunArgs.model_json_schema(),
            handler=run_candidate,
            args_model=_RunArgs,
            read_only=False,
        ),
    ))


@pytest.fixture
def populated_state(tmp_path: Path):
    """Backend with baseline payload cached + sink with one scored candidate."""

    workload_id = "guided_toolset_test"
    problem = _make_problem()
    register_problem(workload_id, problem)
    backend = MimarsinanLayoutBackend()
    workload = _make_workload(workload_id)
    # The "baseline" payload key the GuidedToolset reads from is written
    # whenever `compile()`'s artifact_dir is named "baseline".
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    compile_result = backend.compile(workload, Plan(), artifact_dir=baseline_dir)
    assert compile_result.ok
    sink = MultiObjectiveSink(__import__(
        "compilagent.observation.sink", fromlist=["NullSink"]
    ).NullSink())
    # Inject one scored candidate record so the GUIDANCE block has content.
    rec = CandidateRecord(candidate_id="cand-fixture")
    rec.objectives = {
        "param_utilization_pct": 7.5,
        "fragmentation_pct": 21.4,
        "estimated_accuracy": 0.62,
    }
    rec.objective_metadata = {
        "param_utilization_pct": {"name": "param_utilization_pct", "value": 7.5, "goal": "max", "unit": "%"},
        "fragmentation_pct": {"name": "fragmentation_pct", "value": 21.4, "goal": "min", "unit": "%"},
        "estimated_accuracy": {"name": "estimated_accuracy", "value": 0.62, "goal": "max", "unit": ""},
    }
    rec.configuration = {
        "model_config": {"patch_n_1": 4},
        "platform_constraints": {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 100}]},
    }
    sink._records["cand-fixture"] = rec
    sink._candidate_order["cand-fixture"] = 0
    try:
        yield backend, sink
    finally:
        unregister_problem(workload_id)


class TestBaselineFootprintInjection:
    def test_first_inspect_workload_call_appends_baseline_block(self, populated_state):
        backend, sink = populated_state
        base = _build_real_toolset()
        objectives = [
            ObjectiveSpec("param_utilization_pct", "max"),
            ObjectiveSpec("fragmentation_pct", "min"),
        ]
        guided = GuidedToolset(base, sink=sink, backend=backend, objectives=objectives)
        decl = guided.by_name("inspect_workload")
        result = decl.invoke({})
        assert "fake-summary" in result
        assert "[BASELINE FOOTPRINT]" in result
        # The block references the synthetic backend's softcore data
        assert "softcores emitted" in result
        assert "Active objectives" in result
        assert "param_utilization_pct (max)" in result

    def test_second_inspect_workload_call_does_not_re_append(self, populated_state):
        backend, sink = populated_state
        base = _build_real_toolset()
        guided = GuidedToolset(base, sink=sink, backend=backend, objectives=[])
        decl = guided.by_name("inspect_workload")
        first = decl.invoke({})
        second = decl.invoke({})
        assert first.count("[BASELINE FOOTPRINT]") == 1
        assert second.count("[BASELINE FOOTPRINT]") == 0


class TestRunResultGuidance:
    def test_run_candidate_result_carries_guidance_block(self, populated_state):
        backend, sink = populated_state
        base = _build_real_toolset()
        objectives = [
            ObjectiveSpec("param_utilization_pct", "max"),
            ObjectiveSpec("fragmentation_pct", "min"),
            ObjectiveSpec("estimated_accuracy", "max"),
        ]
        guided = GuidedToolset(base, sink=sink, backend=backend, objectives=objectives)
        decl = guided.by_name("run_candidate")
        result = decl.invoke({"candidate_id": "cand-fixture"})
        assert "[GUIDANCE]" in result
        assert "Pareto front" in result
        assert "Per-metric leaders" in result
        # The seeded record has util=7.5% which is < 10% → suggestion fires
        assert "Shrink the chip aggressively" in result

    def test_run_result_with_no_records_skips_guidance(self):
        """If the sink is empty, the wrapper passes the raw result through."""
        from compilagent.observation.sink import NullSink

        base = _build_real_toolset()
        sink = MultiObjectiveSink(NullSink())
        backend = MimarsinanLayoutBackend()
        guided = GuidedToolset(base, sink=sink, backend=backend, objectives=[])
        decl = guided.by_name("run_candidate")
        result = decl.invoke({"candidate_id": "x"})
        assert "[GUIDANCE]" not in result


class TestPassThrough:
    def test_unrelated_tool_passes_through_unchanged(self, populated_state):
        backend, sink = populated_state
        from compilagent import ToolDecl
        from pydantic import BaseModel

        class _Args(BaseModel):
            pass

        called = []
        def _echo() -> str:
            called.append(1)
            return "raw"

        base = Toolset(tools=(
            ToolDecl(
                name="custom_tool",
                description="x",
                args_schema=_Args.model_json_schema(),
                handler=_echo,
                args_model=_Args,
                read_only=True,
            ),
        ))
        guided = GuidedToolset(base, sink=sink, backend=backend, objectives=[])
        decl = guided.by_name("custom_tool")
        assert decl.invoke({}) == "raw"
        assert called == [1]


class TestToolsProperty:
    def test_tools_returns_wrapped_canonical_tools(self, populated_state):
        backend, sink = populated_state
        base = _build_real_toolset()
        guided = GuidedToolset(base, sink=sink, backend=backend, objectives=[])
        tool_names = [t.name for t in guided.tools]
        assert set(tool_names) == {"inspect_workload", "run_candidate"}
        # Calling via the materialized tuple still appends GUIDANCE
        run_decl = next(t for t in guided.tools if t.name == "run_candidate")
        result = run_decl.invoke({"candidate_id": "cand-fixture"})
        assert "[GUIDANCE]" in result


class TestSignaturePreservation:
    """Regression: pydantic-ai's tool adapter walks ``__wrapped__`` to
    introspect the typed parameters of each handler. If we replace
    ``handler`` with a ``**kwargs`` lambda, the model sees an empty
    schema and never passes the required arguments. ``functools.wraps``
    must keep the original signature visible."""

    def test_wrapped_handler_preserves_original_signature(self, populated_state):
        import inspect

        backend, sink = populated_state
        base = _build_real_toolset()
        guided = GuidedToolset(base, sink=sink, backend=backend, objectives=[])
        wrapped = guided.by_name("run_candidate")
        sig = inspect.signature(wrapped.handler)
        # `run_candidate(*, candidate_id: str)` — the augmented handler
        # must show the same parameters.
        assert "candidate_id" in sig.parameters
        # functools.wraps sets __wrapped__ so inspect.signature follows it
        assert hasattr(wrapped.handler, "__wrapped__")
