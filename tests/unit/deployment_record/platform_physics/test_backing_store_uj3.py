"""[U-J3 R-1] Where a chip's backing store lives, declared: the chip's own DRAM through its memory PHY, or the host's
memory over the host-chip link.

A chip description states it (``backing_store``: ``{"chip": {<source>}}`` or ``{"host": {<source>}}``); a run may
restate it as a declared departure (a ``backing_store`` entry of a named override block). A declaration stating both
stores, none, an unknown field or no source refuses by name. The chip's area components agree with it where the chip
declares them: a memory PHY exactly when the store is the chip's, a host-owned store reached over a link PHY to the
host; a host-owned restatement drops the chip's memory PHYs, so its fixed area is the sum of the rest. A host attach
row (a chip channel fed over the host link) on a chip-owned store refuses by name. The study chips declare their store
chip-owned, with the study's source."""

from __future__ import annotations

import copy
import json

import pytest

from mimarsinan.deployment_record.platform_physics import PlatformPhysics, get_platform_physics
from mimarsinan.deployment_record.platform_physics import area_components as AC
from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    host_attach_block, host_description_block,
)
from mimarsinan.deployment_record.platform_physics.loader import profile_from_dict
from mimarsinan.deployment_record.platform_physics.registry import profiles_dir
from mimarsinan.deployment_record.platform_physics.resolve import (
    OverrideBlockConflictError, merge_override_blocks, resolve_platform_physics,
)

STUDY = ("study_compass", "study_vr_like", "study_isaac_like")
HOST = "jetson_orin_nano_8gb"
CORNERS = ("low", "nominal", "high")


def _profile_json(name):
    return json.loads((profiles_dir() / f"{name}.json").read_text())


def _restated(store, note="a declared variant (test)"):
    return ST.store_block(store, note)


class TestTheDeclaration:
    def test_one_store_with_its_source_reads(self):
        chip = ST.store_from_declaration({"chip": {"citation": "a source", "note": "its own DRAM"}}, "test")
        host = ST.store_from_declaration({"host": {"derivation": "over the link"}}, "test")
        assert (chip.store, chip.citation, chip.note, chip.overridden) == (ST.CHIP, "a source", "its own DRAM", False)
        assert host.store == ST.HOST and host.derivation == "over the link"

    def test_both_stores_refuse_by_name(self):
        with pytest.raises(ST.BackingStoreError, match="both"):
            ST.store_from_declaration({"chip": {"citation": "a"}, "host": {"citation": "b"}}, "test")

    @pytest.mark.parametrize("data", [{}, {"note": "[a block]"}])
    def test_no_store_refuses_by_name(self, data):
        with pytest.raises(ST.BackingStoreError, match="no store"):
            ST.store_from_declaration(data, "test")

    @pytest.mark.parametrize("data", [{"chip": {"citation": "a"}, "where": "x"}, {"chip": {"bandwidth": 1.0}},
                                      {"chip": "its DRAM"}])
    def test_a_field_outside_the_closed_set_refuses_by_name(self, data):
        with pytest.raises(ST.BackingStoreError):
            ST.store_from_declaration(data, "test")

    def test_a_store_without_its_source_refuses_by_name(self):
        with pytest.raises(ST.BackingStoreError, match="source"):
            ST.store_from_declaration({"chip": {}}, "test")

    def test_it_round_trips_and_a_chip_without_one_serializes_as_before(self):
        physics = get_platform_physics("study_compass")
        assert PlatformPhysics.from_dict(physics.to_dict()).backing_store == physics.backing_store
        generic = get_platform_physics("generic_estimated_22nm")
        assert generic.backing_store is None and "backing_store" not in generic.to_dict()


class TestTheAreaComponentsAgree:
    def test_a_memory_phy_beside_a_host_owned_store_refuses_by_name(self):
        data = _profile_json("study_compass")
        data["backing_store"] = {"host": {"note": "test"}}
        with pytest.raises(ST.BackingStoreError, match="memory PHY"):
            profile_from_dict(data)

    def test_a_memory_phy_without_a_declared_store_refuses_by_name(self):
        data = _profile_json("study_compass")
        data.pop("backing_store", None)
        with pytest.raises(ST.BackingStoreError, match="memory PHY"):
            profile_from_dict(data)

    def test_a_chip_owned_store_needs_its_memory_phy(self):
        data = _profile_json("isaac_like")
        data["backing_store"] = {"chip": {"note": "test"}}
        with pytest.raises(ST.BackingStoreError, match="memory PHY"):
            profile_from_dict(data)

    def test_a_host_owned_store_needs_a_link_phy_to_the_host(self):
        data = _profile_json("isaac_like")  # its HyperTransport links reach a peer node, not a host
        data["backing_store"] = {"host": {"note": "test"}}
        with pytest.raises(ST.BackingStoreError, match="link PHY"):
            profile_from_dict(data)

    def test_a_chip_without_components_declares_either_store(self):
        data = _profile_json("generic_estimated_22nm")
        for store in (ST.CHIP, ST.HOST):
            data["backing_store"] = {store: {"note": "test"}}
            assert profile_from_dict(data).backing_store.store == store


@pytest.mark.parametrize("name", STUDY)
class TestTheStudyChips:
    def test_each_declares_its_store_chip_owned_with_the_studys_source(self, name):
        store = get_platform_physics(name).backing_store
        assert store is not None and store.store == ST.CHIP and not store.overridden
        source = " ".join((store.citation, store.derivation, store.note))
        assert "cd41_chip.CTX" in source and "wang2020shuhai" in source and "36 GB/s" in source

    def test_the_declaration_moves_no_area(self, name):
        data = _profile_json(name)
        data.pop("backing_store")
        decls = AC.chip_components_of(data["chip_components"])
        expected = AC.fixed_area_declaration(decls, data["validity"]["technology_node_nm"])
        fixed = get_platform_physics(name).constants["area_global_fixed"]
        assert (fixed.low, fixed.nominal, fixed.high) == pytest.approx(
            (expected["low"], expected["nominal"], expected["high"]), rel=1e-12)


class TestARunRestatesTheStore:
    def test_a_host_owned_restatement_drops_the_memory_phy_and_its_area(self):
        chip = get_platform_physics("study_compass")
        restated = resolve_platform_physics("study_compass", merge_override_blocks({"store": _restated(ST.HOST)}))
        assert restated.backing_store.store == ST.HOST and restated.backing_store.overridden
        assert "a declared variant (test)" in restated.backing_store.note
        kept = [d for d in chip.chip_components if AC.load_component(d.component).kind != "memory_phy"]
        assert restated.chip_components == tuple(kept) and len(kept) == len(chip.chip_components) - 1
        expected = AC.fixed_area_declaration(kept, chip.validity.technology_node_nm)
        fixed = restated.constants["area_global_fixed"]
        assert (fixed.low, fixed.nominal, fixed.high) == pytest.approx(
            (expected["low"], expected["nominal"], expected["high"]), rel=1e-12)
        assert fixed.nominal < chip.constants["area_global_fixed"].nominal - 19.0  # 19.8 mm2 of DDR PHY (mm^2)
        assert fixed.overridden

    def test_the_restatement_round_trips_marked_as_a_departure(self):
        restated = resolve_platform_physics("study_compass", merge_override_blocks({"store": _restated(ST.HOST)}))
        again = PlatformPhysics.from_dict(json.loads(json.dumps(restated.to_dict())))
        assert again.backing_store == restated.backing_store and again.chip_components == restated.chip_components

    def test_a_chip_owned_restatement_on_a_chip_without_a_memory_phy_refuses_by_name(self):
        with pytest.raises(ST.BackingStoreError, match="memory PHY"):
            resolve_platform_physics("isaac_like", merge_override_blocks({"store": _restated(ST.CHIP)}))

    def test_a_fixed_area_declared_beside_a_restatement_that_moves_it_refuses_by_name(self):
        area = {"area_global_fixed": {"nominal": 5.0, "unit": "mm^2", "evidence_kind": "derived",
                                      "derivation": "a test", "note": "test"}}
        with pytest.raises(ST.BackingStoreError, match="area_global_fixed"):
            resolve_platform_physics("study_compass", merge_override_blocks(
                {"store": _restated(ST.HOST), "area": area}))

    def test_two_blocks_restating_the_store_refuse_by_name(self):
        with pytest.raises(OverrideBlockConflictError, match="backing_store"):
            merge_override_blocks({"a": _restated(ST.HOST), "b": _restated(ST.CHIP)})

    def test_a_restatement_stating_both_stores_refuses_by_name(self):
        block = {"backing_store": {"chip": {"note": "a"}, "host": {"note": "b"}}}
        with pytest.raises(ST.BackingStoreError, match="both"):
            resolve_platform_physics("generic_estimated_22nm", merge_override_blocks({"store": block}))


class TestTheHostAttachRow:
    def test_on_a_chip_owned_store_it_refuses_by_name(self):
        blocks = {"host": host_description_block(HOST), "host_attach": host_attach_block(HOST)}
        with pytest.raises(ST.BackingStoreError, match="chip-owned"):
            resolve_platform_physics("study_compass", merge_override_blocks(blocks))

    def test_on_a_host_owned_store_it_composes_as_before(self):
        blocks = {"host": host_description_block(HOST), "host_attach": host_attach_block(HOST),
                  "store": _restated(ST.HOST)}
        physics = resolve_platform_physics("study_compass", merge_override_blocks(blocks))
        chip = get_platform_physics("study_compass")
        for corner in CORNERS:
            assert getattr(physics.band("t_program_per_byte"), corner) == pytest.approx(max(
                getattr(chip.band("t_program_per_byte"), corner), getattr(physics.band("t_link_per_byte"), corner)))

    def test_the_host_block_alone_resolves_on_every_store(self):
        block = {"host": host_description_block(HOST)}
        assert resolve_platform_physics("study_compass", merge_override_blocks(block)).backing_store.store == ST.CHIP
        assert resolve_platform_physics("generic_estimated_22nm", merge_override_blocks(block)).backing_store is None


def test_the_store_block_is_a_named_block_of_one_entry():
    block = ST.store_block(ST.HOST, "why", citation="where")
    assert set(block) == {ST.STORE_FIELD}
    merged = merge_override_blocks({"variant": copy.deepcopy(block)})
    assert ST.store_from_declaration(merged[ST.STORE_FIELD], "test").store == ST.HOST
