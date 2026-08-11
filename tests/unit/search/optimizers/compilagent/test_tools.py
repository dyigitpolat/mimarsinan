"""The ``ToolDecl``s ``MimarsinanLayoutBackend`` advertises.

Every inspect tool is declared FROM the introspection registry, so what the
agent can ask is the registry's declared surface: one tool per payload the
candidate view can answer, each response carrying the payload's name and
version. Renaming an existing tool would break saved traces, so the two
agent-facing aliases are pinned here too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compilagent import Plan, ToleranceConfig, WorkloadKind, WorkloadSpec

from mimarsinan.deployment_record.introspection import (
    CANDIDATE_LAYOUT,
    INTROSPECTION_REGISTRY,
)
from mimarsinan.search.optimizers.compilagent.backend import MimarsinanLayoutBackend
from mimarsinan.search.optimizers.compilagent.tools import (
    build_introspection_tools,
    tool_name_for,
)
from mimarsinan.search.optimizers.compilagent.workload import (
    register_problem,
    unregister_problem,
)

# Re-use the fakes from the backend tests
from .test_backend import _make_problem, _make_workload  # noqa: E402


@pytest.fixture
def compiled_backend(tmp_path: Path):
    workload_id = "fake_layout_tools_test"
    problem = _make_problem()
    register_problem(workload_id, problem)
    backend = MimarsinanLayoutBackend()
    workload = _make_workload(workload_id)
    cdir = tmp_path / "cand-abc"
    cdir.mkdir()
    result = backend.compile(workload, Plan(), artifact_dir=cdir)
    assert result.ok, result.diagnostics
    candidate_id = cdir.name
    try:
        yield backend, candidate_id
    finally:
        unregister_problem(workload_id)


def _by_name(decls, name):
    return next(d for d in decls if d.name == name)


def _invoke(backend, name, candidate_id):
    decl = _by_name(build_introspection_tools(backend), name)
    return json.loads(decl.invoke({"candidate_id": candidate_id}))


class TestSurfaceShape:
    def test_the_tools_are_the_candidate_answerable_payloads(self, compiled_backend):
        backend, _ = compiled_backend
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
        backend, _ = compiled_backend
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
        backend, _ = compiled_backend
        assert all(d.read_only for d in build_introspection_tools(backend))

    def test_descriptions_state_the_payload_version(self, compiled_backend):
        backend, _ = compiled_backend
        decl = _by_name(build_introspection_tools(backend), "inspect_softcores")
        assert "`softcores` v1" in decl.description


class TestInspectSoftcores:
    def test_returns_the_versioned_payload_with_both_identities(self, compiled_backend):
        backend, candidate_id = compiled_backend
        result = _invoke(backend, "inspect_softcores", candidate_id)
        assert result["candidate_id"] == candidate_id
        assert result["payload"] == "softcores"
        assert result["payload_version"] == 1
        assert result["softcores_count"] == 3
        assert {sc["name"] for sc in result["softcores"]} >= {
            "conv1_pos0_0", "conv1_pos1_0", "fc1_tile_0_64",
        }
        assert {sc["perceptron_index"] for sc in result["softcores"]} == {0, 1}
        assert {sc["bank_id"] for sc in result["softcores"]} == {0, None}

    def test_unknown_candidate_raises(self, compiled_backend):
        backend, _ = compiled_backend
        decl = _by_name(build_introspection_tools(backend), "inspect_softcores")
        with pytest.raises(ValueError, match="unknown candidate"):
            decl.invoke({"candidate_id": "nope"})


class TestInspectLayerBreakdown:
    def test_rows_are_keyed_on_the_source_layer(self, compiled_backend):
        backend, candidate_id = compiled_backend
        result = _invoke(backend, "inspect_layer_breakdown", candidate_id)
        assert result["layers_count"] == 2
        assert [row["perceptron_index"] for row in result["layers"]] == [0, 1]
        assert result["layers"][0]["softcore_count"] == 2


class TestInspectLayoutStats:
    def test_returns_layout_stats_and_objectives(self, compiled_backend):
        backend, candidate_id = compiled_backend
        result = _invoke(backend, "inspect_layout_stats", candidate_id)
        assert result["stats"]
        assert "hw_objectives" in result
        assert "total_param_capacity" in result["hw_objectives"]


class TestTheThreeAdditiveTools:
    def test_weight_banks_expose_the_sharing_degree(self, compiled_backend):
        backend, candidate_id = compiled_backend
        result = _invoke(backend, "inspect_weight_banks", candidate_id)
        assert result["payload"] == "bank_composition"
        assert result["banks"][0]["bank_id"] == 0
        assert result["banks"][0]["softcore_count"] == 2
        assert result["unbanked_softcore_count"] == 1

    def test_schedule_names_the_policy_the_platform_declares(self, compiled_backend):
        backend, candidate_id = compiled_backend
        result = _invoke(backend, "inspect_schedule", candidate_id)
        assert result["payload"] == "schedule"
        assert result["schedule_policy"] == "pool"
        assert result["max_schedule_passes"] == 8
        assert result["segments_count"] == 1

    def test_capabilities_serve_every_declared_bit(self, compiled_backend):
        backend, candidate_id = compiled_backend
        result = _invoke(backend, "inspect_capabilities", candidate_id)
        assert result["payload"] == "capabilities"
        assert {
            "allow_coalescing", "allow_neuron_splitting", "allow_scheduling",
            "allow_per_layer_s", "schedule_policy", "max_schedule_passes",
            "hardware_bias", "max_axons", "max_neurons",
        } == set(result["bits"])


class TestListObjectives:
    def test_returns_objective_catalog(self, compiled_backend):
        backend, _ = compiled_backend
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
        backend, _ = compiled_backend
        decl = _by_name(build_introspection_tools(backend), "inspect_softcores")
        with pytest.raises(ValueError):
            decl.invoke({})  # candidate_id is required

    def test_an_unserved_payload_says_so_instead_of_looking_empty(
        self, compiled_backend
    ):
        backend, candidate_id = compiled_backend
        backend._candidate_payloads[candidate_id]["introspection"] = {}
        result = _invoke(backend, "inspect_softcores", candidate_id)
        assert "unavailable" in result
        assert "softcores" not in result
