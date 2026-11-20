"""The selection key ``residency_plan``: served only when declared, judged by name, refused where no cut census is."""

from __future__ import annotations

import inspect
from dataclasses import fields
from types import SimpleNamespace

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.layout.layout_plan import build_layout_plan
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
    ResidencyPlanError,
)
from mimarsinan.mapping.support.schedule.residency_composition import ResidencySelection
from mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats
from mimarsinan.mapping.verification.verifier import verify_hardware_config

from . import vehicles as V

PLATFORM = {
    "cores": [{"max_axons": 64, "max_neurons": 32, "count": 4, "has_bias": True}],
    "allow_scheduling": True, "allow_coalescing": False, "allow_neuron_splitting": True,
    "max_schedule_passes": 6,
}
#: The base served dicts of PLATFORM (the Wave 0 tree), byte for byte.
BASE_BITS = {"max_axons": 64, "max_neurons": 32, "hardware_bias": True, "allow_coalescing": False,
             "allow_neuron_splitting": True, "allow_scheduling": True, "allow_per_layer_s": False,
             "max_schedule_passes": 6}
BASE_KWARGS = {"allow_neuron_splitting": True, "allow_coalescing": False, "allow_scheduling": True,
               "max_schedule_passes": 6}


class TestTheKeyIsDeclaredOnly:
    def test_absent_every_served_dict_equals_the_base(self):
        caps = ChipCapabilities.from_platform_constraints(PLATFORM)
        assert caps.capability_bits() == BASE_BITS
        assert caps.layout_kwargs() == BASE_KWARGS
        assert list(caps.capability_bits()) == list(BASE_BITS)          # key order too

    def test_declared_it_rides_both_dicts_verbatim(self):
        plan = {"step": "index", "index": 2}
        caps = ChipCapabilities.from_platform_constraints({**PLATFORM, "residency_plan": plan})
        assert caps.capability_bits() == {**BASE_BITS, "residency_plan": plan}
        assert caps.layout_kwargs() == {**BASE_KWARGS, "residency_plan": plan}
        assert MappingStrategy.resolve(caps).layout_kwargs() == caps.layout_kwargs()
        hash(caps)                                                        # a dict value keeps the type hashable

    @pytest.mark.parametrize("value", ["floor", ["step", "floor"], 3, {"step": "floor", "extra": 1}, {},
                                       {"step": "index"}, {"step": "greedy"}])
    def test_a_malformed_declaration_refuses_by_name_where_it_is_read(self, value):
        """Review round 3: the capability layer judges the declared value by the one rule
        (``residency_declaration``), so a malformed value never escapes as a bare ValueError or TypeError and is
        never accepted to fail later at planning time."""
        for make in (lambda: ChipCapabilities.from_platform_constraints({**PLATFORM, "residency_plan": value}),
                     lambda: ChipCapabilities(allow_scheduling=True, residency_plan=value)):
            with pytest.raises(ResidencyPlanError, match="residency_plan") as caught:
                make()
            assert isinstance(caught.value, ToolchainRefusal)

    def test_declared_without_scheduling_it_refuses_by_name(self):
        with pytest.raises(ResidencyPlanError) as caught:
            ChipCapabilities.from_platform_constraints(
                {**PLATFORM, "allow_scheduling": False, "residency_plan": {"step": "floor"}})
        assert isinstance(caught.value, ToolchainRefusal) and isinstance(caught.value, ValueError)

    def test_one_metadata_rule_with_no_branch_on_a_key_name(self):
        declared_only = [f.name for f in fields(ChipCapabilities) if f.metadata.get("declared_only")]
        # [U-F4] pipelined_multiplexing joins the declared-only fields under the same one metadata rule; [thru:k] so do
        # the layer copies' two keys
        assert declared_only == ["residency_plan", "position_multiplexing", "pipelined_multiplexing",
                                 "layer_replication", "wavefront_group_cores"]
        for method in (ChipCapabilities.declared_fields, ChipCapabilities.capability_bits,
                       ChipCapabilities.layout_kwargs):
            assert "residency" not in inspect.getsource(method)


class TestTheDeclaredValueIsJudged:
    @pytest.mark.parametrize("value", [
        {"step": "greedy"}, {"step": "index"}, {"step": "index", "index": -1}, {"step": "floor", "index": 0},
        {"step": "index", "index": True}, {"step": "floor", "extra": 1}, "floor",
    ])
    def test_an_illegal_value_refuses_by_name(self, value):
        with pytest.raises(ResidencyPlanError):
            ResidencySelection.from_declaration(value)

    @pytest.mark.parametrize("value, want", [
        ({"step": "floor"}, ("floor", None)), ({"step": "stateless"}, ("stateless", None)),
        ({"step": "index", "index": 3}, ("index", 3)),
    ])
    def test_a_legal_value_parses(self, value, want):
        selection = ResidencySelection.from_declaration(value)
        assert (selection.rule, selection.index) == want


class TestEntryPointsWithoutACutCensusRefuse:
    """The stats route and the wizard preview hold no cut census: they refuse the key, never plan U1 silently."""

    SPECS = [LayoutSoftCoreSpec(8, 8)]
    PLAN = {"step": "floor"}

    def test_the_stats_route(self):
        with pytest.raises(ResidencyPlanError):
            compute_mapping_stats(self.SPECS, [LayoutHardCoreType(16, 16, 2)], allow_scheduling=True,
                                  residency_plan=self.PLAN)

    def test_the_wizard_preview(self):
        with pytest.raises(ResidencyPlanError):
            verify_hardware_config(self.SPECS, [{"max_axons": 16, "max_neurons": 16, "count": 2}],
                                   allow_scheduling=True, residency_plan=self.PLAN)
        with pytest.raises(ResidencyPlanError):
            build_layout_plan(SimpleNamespace(softcores=self.SPECS),
                              [{"max_axons": 16, "max_neurons": 16, "count": 2}], allow_scheduling=True,
                              residency_plan=self.PLAN)

    def test_fragments_without_the_wire_census(self):
        specs, _census, program = V.layout_plane(V.chain_repr())
        with pytest.raises(ResidencyPlanError, match="cut census"):
            V.fragments(specs, None, 5, self.PLAN, program=program)

    def test_fragments_without_scheduling(self):
        specs, census, _program = V.layout_plane(V.chain_repr())
        with pytest.raises(ResidencyPlanError):
            collect_noc_fragments(softcores=specs, core_types=[LayoutHardCoreType(V.MAX_AXONS, V.MAX_NEURONS, 5)],
                                  census=census, allow_scheduling=False, allow_neuron_splitting=False,
                                  allow_coalescing=False, max_schedule_passes=8, residency_plan=self.PLAN)

    def test_absent_every_entry_point_answers_as_before(self):
        stats, error = compute_mapping_stats(self.SPECS, [LayoutHardCoreType(16, 16, 2)], allow_scheduling=True)
        assert error is None and stats.feasible
