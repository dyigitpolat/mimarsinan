"""The ``ToolDecl``s ``MimarsinanLayoutBackend`` advertises.

Every inspect tool is declared FROM the introspection registry, so what the
agent can ask is the registry's declared surface: one tool per payload the
candidate view can answer, each response carrying the payload's name and
version. Renaming an existing tool would break saved traces, so the two
agent-facing aliases are pinned here too.

The backend under them is driven against the REAL ``JointArchHwProblem`` (see
``real_problem.py``): what an inspect tool returns is that problem's own
``candidate_layout``, served — so every response is cross-checked against the
problem rather than against a hand-written stand-in for it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compilagent import Plan

from mimarsinan.deployment_record.introspection import (
    CANDIDATE_LAYOUT,
    INTROSPECTION_REGISTRY,
)
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.search.optimizers.compilagent.backend import MimarsinanLayoutBackend
from mimarsinan.search.optimizers.compilagent.tools import (
    build_introspection_tools,
    tool_name_for,
)
from mimarsinan.search.optimizers.compilagent.workload import (
    register_problem,
    unregister_problem,
)

# The REAL problem the backend adapts — see real_problem.py.
from .real_problem import HW_OBJECTIVES, make_problem, make_workload


@pytest.fixture
def compiled_backend(tmp_path: Path):
    workload_id = "real_layout_tools_test"
    problem = make_problem("hardware")
    register_problem(workload_id, problem)
    backend = MimarsinanLayoutBackend()
    workload = make_workload(workload_id)
    cdir = tmp_path / "cand-abc"
    cdir.mkdir()
    result = backend.compile(workload, Plan(), artifact_dir=cdir)
    assert result.ok, result.diagnostics
    candidate_id = cdir.name
    try:
        yield backend, candidate_id, problem
    finally:
        unregister_problem(workload_id)


def _by_name(decls, name):
    return next(d for d in decls if d.name == name)


def _invoke(backend, name, candidate_id):
    decl = _by_name(build_introspection_tools(backend), name)
    return json.loads(decl.invoke({"candidate_id": candidate_id}))


def _candidate_layout(backend, candidate_id, problem):
    """The layout the compiled candidate was served from — the problem's own."""
    return problem.candidate_layout(
        backend.get_candidate_payload(candidate_id)["config"]
    )


class TestSurfaceShape:
    def test_the_tools_are_the_candidate_answerable_payloads(self, compiled_backend):
        backend, _, _problem = compiled_backend
        names = sorted(d.name for d in build_introspection_tools(backend))
        assert names == [
            "inspect_capabilities",
            "inspect_layer_breakdown",
            "inspect_layout_stats",
            "inspect_schedule",
            "inspect_softcores",
            "inspect_weight_banks",
            "list_objectives",
        ]

    def test_every_candidate_payload_has_exactly_one_tool(self, compiled_backend):
        backend, _, _problem = compiled_backend
        declared = {d.name for d in build_introspection_tools(backend)}
        expected = {
            tool_name_for(spec.name)
            for spec in INTROSPECTION_REGISTRY.all()
            if CANDIDATE_LAYOUT in spec.builders
        }
        assert expected <= declared
        # A record-only payload is never advertised to a candidate-scoped agent.
        assert "inspect_placement" not in declared

    def test_the_legacy_tool_names_did_not_move(self):
        assert tool_name_for("softcores") == "inspect_softcores"
        assert tool_name_for("layer_rollup") == "inspect_layer_breakdown"
        assert tool_name_for("layout_stats") == "inspect_layout_stats"

    def test_all_tools_are_read_only(self, compiled_backend):
        backend, _, _problem = compiled_backend
        assert all(d.read_only for d in build_introspection_tools(backend))

    def test_descriptions_state_the_payload_version(self, compiled_backend):
        backend, _, _problem = compiled_backend
        decl = _by_name(build_introspection_tools(backend), "inspect_softcores")
        assert "`softcores` v1" in decl.description


class TestInspectSoftcores:
    def test_returns_the_versioned_payload_of_the_candidates_own_softcores(
        self, compiled_backend,
    ):
        backend, candidate_id, problem = compiled_backend
        result = _invoke(backend, "inspect_softcores", candidate_id)
        expected = _candidate_layout(backend, candidate_id, problem).softcores
        assert result["candidate_id"] == candidate_id
        assert result["payload"] == "softcores"
        assert result["payload_version"] == 1
        assert result["softcores_count"] == len(expected) > 0
        assert [sc["input_count"] for sc in result["softcores"]] == [
            sc.input_count for sc in expected
        ]
        assert [sc["output_count"] for sc in result["softcores"]] == [
            sc.output_count for sc in expected
        ]
        assert [sc["perceptron_index"] for sc in result["softcores"]] == [
            sc.perceptron_index for sc in expected
        ]

    def test_unknown_candidate_raises(self, compiled_backend):
        backend, _, _problem = compiled_backend
        decl = _by_name(build_introspection_tools(backend), "inspect_softcores")
        with pytest.raises(ValueError, match="unknown candidate"):
            decl.invoke({"candidate_id": "nope"})


class TestInspectLayerBreakdown:
    def test_rows_are_keyed_on_the_source_layer(self, compiled_backend):
        backend, candidate_id, problem = compiled_backend
        result = _invoke(backend, "inspect_layer_breakdown", candidate_id)
        softcores = _candidate_layout(backend, candidate_id, problem).softcores
        assert result["payload"] == "layer_rollup"
        assert result["layers_count"] == len(result["layers"]) > 0
        assert [row["perceptron_index"] for row in result["layers"]] == sorted(
            {sc.perceptron_index for sc in softcores}
        )
        assert sum(row["softcore_count"] for row in result["layers"]) == len(softcores)
        assert sum(row["total_area"] for row in result["layers"]) == sum(
            sc.area for sc in softcores
        )


class TestInspectLayoutStats:
    def test_returns_layout_stats_and_objectives(self, compiled_backend):
        backend, candidate_id, problem = compiled_backend
        result = _invoke(backend, "inspect_layout_stats", candidate_id)
        expected = _candidate_layout(backend, candidate_id, problem)
        assert result["payload"] == "layout_stats"
        assert result["stats"] == expected.stats.to_dict()
        assert result["stats"]["feasible"] is True
        assert set(result["hw_objectives"]) >= set(HW_OBJECTIVES)
        scored = problem.evaluate(
            backend.get_candidate_payload(candidate_id)["config"]
        )
        for name, value in scored.items():
            assert result["hw_objectives"][name] == value


class TestTheThreeAdditiveTools:
    def test_weight_banks_state_who_owns_their_weights(self, compiled_backend):
        # The MLP fixture shares no bank; the payload says so rather than
        # serving an empty table that reads like "not measured". The
        # sharing-degree rows are pinned in the registry's own suite.
        backend, candidate_id, problem = compiled_backend
        result = _invoke(backend, "inspect_weight_banks", candidate_id)
        softcores = _candidate_layout(backend, candidate_id, problem).softcores
        assert result["payload"] == "bank_composition"
        assert result["banks"] == []
        assert result["unbanked_softcore_count"] == len(softcores)

    def test_schedule_names_the_policy_the_platform_declares(self, compiled_backend):
        backend, candidate_id, problem = compiled_backend
        result = _invoke(backend, "inspect_schedule", candidate_id)
        layout = _candidate_layout(backend, candidate_id, problem)
        capabilities = ChipCapabilities.from_platform_constraints(layout.platform)
        assert result["payload"] == "schedule"
        assert result["max_schedule_passes"] == capabilities.max_schedule_passes == 8
        assert result["segments_count"] == len(
            {sc.segment_id for sc in layout.softcores}
        )
        assert result["sync_count"] == int(layout.stats.schedule_sync_count)

    def test_capabilities_serve_every_declared_bit(self, compiled_backend):
        backend, candidate_id, problem = compiled_backend
        result = _invoke(backend, "inspect_capabilities", candidate_id)
        layout = _candidate_layout(backend, candidate_id, problem)
        assert result["payload"] == "capabilities"
        assert {
            "allow_coalescing", "allow_neuron_splitting", "allow_scheduling",
            "allow_per_layer_s", "max_schedule_passes",
            "hardware_bias", "max_axons", "max_neurons",
        } == set(result["bits"])
        assert result["bits"] == ChipCapabilities.from_platform_constraints(
            layout.platform
        ).capability_bits()


class TestListObjectives:
    def test_returns_objective_catalog(self, compiled_backend):
        backend, _, _problem = compiled_backend
        decl = _by_name(build_introspection_tools(backend), "list_objectives")
        result = json.loads(decl.invoke({}))
        names = {entry["name"] for entry in result["objectives"]}
        assert names >= {
            "total_param_capacity", "param_utilization_pct", "fragmentation_pct",
        }
        for entry in result["objectives"]:
            assert entry["goal"] in ("min", "max")

    def test_returns_empty_when_no_candidate_compiled(self):
        backend = MimarsinanLayoutBackend()
        decl = _by_name(build_introspection_tools(backend), "list_objectives")
        result = json.loads(decl.invoke({}))
        assert result == {"objectives": []}


class TestArgsValidation:
    def test_missing_candidate_id_is_validation_error(self, compiled_backend):
        backend, _, _problem = compiled_backend
        decl = _by_name(build_introspection_tools(backend), "inspect_softcores")
        with pytest.raises(ValueError):
            decl.invoke({})  # candidate_id is required

    def test_an_unserved_payload_says_so_instead_of_looking_empty(
        self, compiled_backend
    ):
        backend, candidate_id, _problem = compiled_backend
        backend._candidate_payloads[candidate_id]["introspection"] = {}
        result = _invoke(backend, "inspect_softcores", candidate_id)
        assert "unavailable" in result
        assert "softcores" not in result
