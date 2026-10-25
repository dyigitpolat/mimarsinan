"""W08: the lane block a lane chip prices, in every value-domain bundle, and nowhere else."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.platform_physics.lane_blocks import (
    LANE_ROWS,
    LaneBlockError,
    TargetNode,
    lane_block_names,
    lane_block_rows,
    load_lane_block,
)
from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics
from mimarsinan.mapping.lanes.declaration import LANE_KEY, declared_lane_block, lane_declaration
from mimarsinan.mapping.lanes.output_stage import declared_lanes_per_tile
from mimarsinan.mapping.platform.target_bundles import target_bundles

BLOCK = "ng45_os_lane"


class TestLaneBlock:
    def test_the_shipped_block_declares_the_g3_lane_rows_as_estimates(self):
        assert BLOCK in lane_block_names()
        block = load_lane_block(BLOCK)
        assert set(block.constants) == set(LANE_ROWS) and block.classes == ("L", "E", "P")
        assert all(c["evidence_kind"] == "estimated" and c["note"] for c in block.constants.values())
        assert block.columns_per_lane == 64

    def test_an_unknown_block_refuses_by_name(self):
        with pytest.raises(LaneBlockError, match="no shipped lane block"):
            load_lane_block("missing_block")

    def test_the_block_adds_only_the_rows_the_physics_lacks(self):
        rows = lane_block_rows(BLOCK, {"e_lane_op": object()}, TargetNode(technology_node_nm=45.0, supply_v=1.1))
        assert set(rows) == {"area_per_lane", "t_lane_op"}
        assert all(row["note"].startswith(f"[lane block {BLOCK}]") for row in rows.values())
        assert lane_block_rows(None, {}) == {}


class TestResolution:
    def test_a_profile_gains_the_lane_rows_only_with_the_block(self):
        plain = resolve_platform_physics("generic_estimated_22nm", {})
        lanes = resolve_platform_physics("generic_estimated_22nm", {}, lane_block=BLOCK)
        assert plain is not None and lanes is not None
        assert not set(LANE_ROWS) & set(plain.constants)
        assert set(LANE_ROWS) <= set(lanes.constants)
        assert {k: v for k, v in lanes.constants.items() if k not in LANE_ROWS} == dict(plain.constants)
        assert not any(lanes.constants[k].overridden for k in LANE_ROWS)

    def test_an_override_wins_over_the_block(self):
        override = {"e_lane_op": {"nominal": 1.0, "note": "operator value"}}
        physics = resolve_platform_physics("generic_estimated_22nm", override, lane_block=BLOCK)
        assert physics is not None and physics.constants["e_lane_op"].nominal == 1.0
        assert physics.constants["e_lane_op"].overridden

    def test_a_block_alone_makes_no_chip_priceable(self):
        assert resolve_platform_physics("", {}, lane_block=BLOCK) is None

    def test_the_declaration_names_the_block_only_with_lane_classes(self):
        assert declared_lane_block({LANE_KEY: {"classes": ["L"], "lane_block": BLOCK}}) == BLOCK
        assert declared_lane_block({LANE_KEY: {"classes": []}}) is None
        assert declared_lane_block({}) is None

    def test_lanes_per_tile_is_lanes_per_core_times_cores_per_tile(self):
        platform = {LANE_KEY: {"classes": ["L", "E", "P"], "lanes_per_core": 4}, "cores_per_tile_resolved": 3}
        assert declared_lanes_per_tile(platform) == 12
        assert declared_lanes_per_tile({LANE_KEY: {"classes": []}}) == 0
        assert declared_lanes_per_tile({}) is None


#: [integration of W08 with CL2] Claim classes of a design-space value chip, whose bundle adds the estimated lane
#: block; a bundle of a published chip's page-sourced geometry (headline-eligible, quarantined) declares what its
#: sources state, and they state no output-stage lane block.
DESIGN_SPACE_CLAIMS = frozenset({"synthetic", "curve-only"})


class TestBundles:
    def test_every_value_domain_bundle_declares_lanes_with_a_block(self):
        bundles = target_bundles()
        value = [b for b in bundles.values() if b.values.get("core_semantics") == "mvm"
                 and b.claim_class in DESIGN_SPACE_CLAIMS]
        assert {b.name for b in value} >= {"generic_estimated_22nm", "isaac_like"}
        for bundle in value:
            declaration = lane_declaration(bundle.values)
            assert declaration is not None and declaration.classes == frozenset("LEP"), bundle.name
            assert declaration.lane_block in lane_block_names() and declaration.lanes_per_core, bundle.name
            assert "output-stage lanes estimated" in bundle.faithfulness.named, bundle.name
            assert any("per-array ADC not built" in label for label in bundle.faithfulness.named), bundle.name

    def test_a_published_value_chip_declares_no_lane_block(self):
        """The CL2 chips (HERMES-64, NeuRRAM, the SRAM-CIM macro) are bundled as their pages state them: no lanes, so
        their programs keep the output stage's ops on the host and price no lane block."""
        published = [b for b in target_bundles().values() if b.values.get("core_semantics") == "mvm"
                     and b.claim_class not in DESIGN_SPACE_CLAIMS]
        assert {b.name for b in published} >= {"hermes_64", "neurram", "sram_cim_macro"}
        for bundle in published:
            assert lane_declaration(bundle.values) is None and LANE_KEY not in bundle.values, bundle.name

    def test_no_other_bundle_declares_lanes(self):
        for bundle in target_bundles().values():
            if bundle.values.get("core_semantics") != "mvm":
                assert LANE_KEY not in bundle.values, bundle.name

    @pytest.mark.parametrize("name,lanes", [("generic_estimated_22nm", 4), ("isaac_like", 2)])
    def test_one_lane_per_64_output_columns(self, name, lanes):
        bundle = target_bundles()[name]
        columns = max(int(core["max_neurons"]) for core in bundle.values["cores"])
        assert lane_declaration(bundle.values).lanes_per_core == lanes == columns // 64
