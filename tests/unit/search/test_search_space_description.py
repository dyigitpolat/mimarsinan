"""Tests for `SearchSpaceDescription`.

The description object is the single source of truth shared by the
AgentEvolve LLM prompt rendering and the compilagent backend's
`derive_search_space(...)` implementation. These tests pin down the
contract so a future change to one renderer cannot silently desynchronise
the other.
"""

from __future__ import annotations

import pytest

from mimarsinan.search.search_space_description import (
    CORE_DIM_GRANULARITY,
    SearchSpaceDescription,
)


def _description(search_mode: str = "joint") -> SearchSpaceDescription:
    return SearchSpaceDescription.from_arch_search(
        search_mode=search_mode,
        arch_options=(
            ("activation", ("ReLU", "LeakyReLU", "GELU")),
            ("fc_w_1", (32, 64, 128)),
        ),
        arch_cfg={
            "num_core_types": 2,
            "core_axons_bounds": [64, 1024],
            "core_neurons_bounds": [64, 1024],
            "core_count_bounds": [50, 500],
        },
        target_tq=32,
    )


class TestAgentEvolveRenderers:
    def test_joint_schema_has_both_sections(self):
        d = _description(search_mode="joint")
        schema = d.to_agent_evolve_schema()
        assert set(schema) == {"model_config", "platform_constraints"}
        assert "activation" in schema["model_config"]
        assert "cores" in schema["platform_constraints"]

    def test_hardware_only_schema_omits_model(self):
        d = _description(search_mode="hardware")
        schema = d.to_agent_evolve_schema()
        assert "model_config" not in schema
        assert "platform_constraints" in schema

    def test_model_only_schema_omits_platform(self):
        d = _description(search_mode="model")
        schema = d.to_agent_evolve_schema()
        assert "model_config" in schema
        assert "platform_constraints" not in schema

    def test_example_matches_schema_keys(self):
        d = _description()
        example = d.to_agent_evolve_example()
        schema = d.to_agent_evolve_schema()
        assert set(example) == set(schema)
        # Example respects num_core_types
        assert len(example["platform_constraints"]["cores"]) == d.num_core_types

    def test_constraints_text_mentions_bounds(self):
        d = _description()
        text = d.to_agent_evolve_constraints()
        assert "64" in text and "1024" in text
        assert "50" in text and "500" in text
        assert "multiples of 8" in text


class TestCompilagentLevers:
    def test_one_lever_per_searchable_variable(self):
        d = _description()
        levers = d.to_compilagent_levers(workload_id="w", backend_id="mimarsinan_layout")
        # 2 arch options + 3 dims * 2 core types = 8
        assert len(levers) == len(d.arch_options) + 3 * d.num_core_types

    def test_arch_levers_are_enums(self):
        from compilagent import EnumChoice

        d = _description()
        levers = d.to_compilagent_levers(workload_id="w", backend_id="b")
        arch_levers = [lv for lv in levers if lv.target_kind == "arch"]
        assert len(arch_levers) == 2
        for lv in arch_levers:
            assert isinstance(lv.range, EnumChoice)
            assert lv.range.candidates  # non-empty

    def test_hw_levers_are_open_ranges_with_wide_bounds(self):
        """Compilagent levers are intentionally open ranges (``IntFreeform``)
        so the agent freely chooses scale based on the workload's
        footprint — JSON bounds from ``core_*_bounds`` are ignored on
        purpose. The wide defaults cover every realistic neuromorphic
        crossbar geometry."""
        from compilagent import IntFreeform

        d = _description()
        levers = d.to_compilagent_levers(workload_id="w", backend_id="b")
        hw_levers = [lv for lv in levers if lv.target_kind == "hw.core"]
        assert len(hw_levers) == 6  # 2 core types * (max_axons, max_neurons, count)
        for lv in hw_levers:
            assert isinstance(lv.range, IntFreeform)
            assert lv.range.min >= 1
            assert lv.range.max >= lv.range.min
            # The compilagent-specific wide defaults are *much* wider
            # than the JSON `core_*_bounds`, regardless of what the
            # user provided.
            if lv.id.endswith("max_axons"):
                assert (lv.range.min, lv.range.max) == d.COMPILAGENT_AXON_BOUNDS
            elif lv.id.endswith("max_neurons"):
                assert (lv.range.min, lv.range.max) == d.COMPILAGENT_NEURON_BOUNDS
            else:
                assert (lv.range.min, lv.range.max) == d.COMPILAGENT_COUNT_BOUNDS

    def test_hw_dim_levers_have_step_granularity(self):
        """``max_axons`` and ``max_neurons`` snap to ``CORE_DIM_GRANULARITY``
        via the ``IntFreeform.step`` field; ``count`` accepts any integer
        (step=1)."""
        d = _description()
        levers = d.to_compilagent_levers(workload_id="w", backend_id="b")
        hw_levers = {lv.id: lv for lv in levers if lv.target_kind == "hw.core"}
        for lid, lv in hw_levers.items():
            if lid.endswith("count"):
                assert lv.range.step == 1
            else:
                assert lv.range.step == CORE_DIM_GRANULARITY

    def test_jsonsearch_bounds_are_ignored_by_compilagent_renderer(self):
        """If a user passes silly-tight JSON bounds, the compilagent lever
        surface still uses the wide defaults. The JSON bounds only flow
        through to NSGA2 / AgentEvolve (which need them for their
        encoded variable space)."""
        d = SearchSpaceDescription.from_arch_search(
            search_mode="hardware",
            arch_options=(),
            arch_cfg={
                "num_core_types": 1,
                "core_axons_bounds": [256, 256],
                "core_neurons_bounds": [256, 256],
                "core_count_bounds": [10, 10],
            },
            target_tq=16,
        )
        levers = d.to_compilagent_levers(workload_id="w", backend_id="b")
        for lv in levers:
            if lv.id.endswith("max_axons"):
                assert lv.range.min == d.COMPILAGENT_AXON_BOUNDS[0]
                assert lv.range.max == d.COMPILAGENT_AXON_BOUNDS[1]
                # The JSON bound 256 was discarded
                assert lv.range.min != 256

    def test_levers_carry_evidence_and_backend_id(self):
        d = _description()
        levers = d.to_compilagent_levers(
            workload_id="w", backend_id="mimarsinan_layout"
        )
        for lv in levers:
            assert lv.backend_id == "mimarsinan_layout"
            assert lv.evidence.signal  # non-empty derivation evidence

    def test_hardware_only_skips_arch_levers(self):
        d = _description(search_mode="hardware")
        levers = d.to_compilagent_levers(workload_id="w", backend_id="b")
        assert all(lv.target_kind == "hw.core" for lv in levers)

    def test_model_only_skips_hw_levers(self):
        d = _description(search_mode="model")
        levers = d.to_compilagent_levers(workload_id="w", backend_id="b")
        assert all(lv.target_kind == "arch" for lv in levers)


class TestEdgeCases:
    def test_collapsed_json_bounds_do_not_collapse_compilagent_levers(self):
        """Tight JSON bounds are ignored by ``to_compilagent_levers`` —
        the open-range surface stays wide so the agent can right-size."""
        d = SearchSpaceDescription.from_arch_search(
            search_mode="hardware",
            arch_options=(),
            arch_cfg={
                "num_core_types": 1,
                "core_axons_bounds": [128, 128],
                "core_neurons_bounds": [128, 128],
                "core_count_bounds": [10, 10],
            },
            target_tq=16,
        )
        levers = d.to_compilagent_levers(workload_id="w", backend_id="b")
        assert len(levers) == 3
        for lv in levers:
            assert lv.range.max > lv.range.min
