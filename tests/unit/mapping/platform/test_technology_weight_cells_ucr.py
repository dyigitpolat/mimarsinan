"""[U-CR C-3, SD-20] A core type naming a technology whose chip description stores a weight in a block of cells states
its block, or is refused by name at the run door (in the shape of ``conversion_conflict``, where the run resolves its
declarations): a core type that states none would be read at one cell per weight without a word.

A target bundle's geometry counts cells and its ``core_fields`` state the block (NeuRRAM: a signed weight is a
differential pair, 2 cells on 2 adjacent rows, wan2022a Methods PDF p.10; ISAAC-like: 8 two-bit cells on one row). A run
that replaces the bundle's cores (``target_overrides.cores``) and names the technology on a core type
(``write_physics``, ``compute_physics``) without the block read its geometry as one cell per weight: Figure 1's mixed
row priced NeuRRAM's static tiles at 256 x 256 one-cell weights (SD-20: 36,088,676 written cells for as many weights).
The technology's block is now read from its chip description (``target_bundles.technology_blocks``) and a core type
that names the technology states a block (the description's, or its own, such as a bit-sliced pair), with its
``max_axons`` x ``max_neurons`` counting weights.

[targeted pass, review M-1] The technology a core type's weights are priced under is the one it names, else the run's
``platform_physics_profile`` (the target's): a core type that names none is priced under the run's profile, so it too
states the profile's block or is refused by name. W38 r4's arms replace NeuRRAM's cores under ``target: neurram`` with
core types that copy only ``max_axons``, ``max_neurons``, ``count`` and ``has_bias`` (TR/imagenet/driver/configs.py
``arm_cores``): before, the run admitted them at one cell per weight, with no refusal and nothing printed."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.platform.target_bundles import target_bundles
from mimarsinan.mapping.platform.weight_cells import array_extent
from unit.mapping.platform.test_weight_cells_uc4 import _value_document, _verdicts

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "design_point" / "hybrid_neurram_sram_cim.json"
NEURRAM = {"max_axons": 128, "max_neurons": 256, "count": 4, "has_bias": True, "write_physics": "neurram"}
BLOCK = {"cells_per_weight": 2, "weight_cell_rows": 2}
NAMED = "cores[0] names write_physics 'neurram', whose chip description (target 'neurram') stores a weight in 2 cells"
#: W38 r4's arm form (TR/imagenet/driver/configs.py ``arm_cores``): max_axons, max_neurons, count and has_bias only.
ARM = {"max_axons": 64, "max_neurons": 256, "count": 48, "has_bias": False}
RUN_NAMED = ("cores[0] names no write_physics or compute_physics, so the run's platform_physics_profile 'neurram' "
             "stands for them, whose chip description (target 'neurram') stores a weight in 2 cells")


def technology_blocks():
    """The new interface, imported here so that each test fails on its own on a tree without it."""
    from mimarsinan.mapping.platform.target_bundles import technology_blocks as blocks
    return blocks()


def technology_block_errors(cores, blocks, profile=None):
    from mimarsinan.mapping.platform.weight_cells import technology_block_errors as errors
    return errors(cores, blocks) if profile is None else errors(cores, blocks, profile)


class TestTheTechnologyBlocks:
    def test_each_technology_whose_chip_description_states_a_block(self):
        assert technology_blocks() == {"neurram": (("neurram", 2, 2),), "isaac_like": (("isaac_like", 8, 1),)}

    def test_a_technology_whose_description_stores_one_weight_per_cell_states_none(self):
        bundles = target_bundles()
        profiles = {b.values.get("platform_physics_profile") for b in bundles.values()} - {None}
        assert profiles - set(technology_blocks()) >= {"sram_cim_macro", "hermes_64", "generic_estimated_22nm"}


class TestTheRule:
    def test_a_core_type_naming_the_technology_without_its_block_refuses(self):
        errors = technology_block_errors([NEURRAM], technology_blocks())
        assert len(errors) == 1 and errors[0].startswith(NAMED), errors
        assert "states no block" in errors[0] and "cells_per_weight 2" in errors[0]

    @pytest.mark.parametrize("field", ["write_physics", "compute_physics"])
    def test_either_technology_field_names_it(self, field):
        core = {k: v for k, v in NEURRAM.items() if k != "write_physics"}
        core[field] = "neurram"
        errors = technology_block_errors([core], technology_blocks())
        assert len(errors) == 1 and f"names {field} 'neurram'" in errors[0]

    def test_the_stated_block_is_admitted(self):
        assert technology_block_errors([{**NEURRAM, **BLOCK}], technology_blocks()) == []

    @pytest.mark.parametrize("block", [{"cells_per_weight": 4, "weight_cell_rows": 2}, {"cells_per_weight": 1}],
                             ids=["bit_sliced_pair", "one_cell_stated"])
    def test_another_stated_block_is_the_core_type_s_own(self, block):
        """A stated block is the declaration's own (W38's arm W form: an 8-bit weight on two 4-bit differential pairs,
        4 cells on 2 rows); only a core type that states none is refused, since it would be read silently."""
        assert technology_block_errors([{**NEURRAM, **block}], technology_blocks()) == []

    def test_a_technology_with_one_cell_per_weight_and_an_unnamed_one_are_unchanged(self):
        macro = {"max_axons": 128, "max_neurons": 1024, "count": 1, "write_physics": "sram_cim_macro"}
        plain = {"max_axons": 256, "max_neurons": 256, "count": 48}
        assert technology_block_errors([macro, plain], technology_blocks()) == []


class TestTheRunSProfileStandsForAnUnnamedField:
    """[targeted pass, review M-1] A field a core type does not name (``write_physics``, ``compute_physics``) is priced
    under the run's ``platform_physics_profile``: that technology's block is read the same way."""

    def test_an_unnamed_core_type_under_a_profile_with_a_block_refuses(self):
        errors = technology_block_errors([ARM], technology_blocks(), "neurram")
        assert len(errors) == 1 and errors[0].startswith(RUN_NAMED), errors
        assert "states no block" in errors[0] and "cells_per_weight 2" in errors[0]

    def test_a_field_it_does_not_name_reads_the_run_s_profile(self):
        errors = technology_block_errors([{**ARM, "write_physics": "sram_cim_macro"}], technology_blocks(), "neurram")
        assert len(errors) == 1 and errors[0].startswith(
            "cores[0] names no compute_physics, so the run's platform_physics_profile 'neurram' stands for it, whose "
            "chip description (target 'neurram')"), errors

    def test_a_technology_named_and_read_from_the_run_refuses_once(self):
        errors = technology_block_errors([NEURRAM], technology_blocks(), "neurram")
        assert len(errors) == 1 and errors[0].startswith(
            "cores[0] names write_physics 'neurram' and no compute_physics, so the run's platform_physics_profile "
            "'neurram' stands for it, whose chip description (target 'neurram') stores a weight in 2 cells"), errors

    @pytest.mark.parametrize("profile", ["sram_cim_macro", "generic_estimated_22nm", ""])
    def test_a_profile_storing_one_weight_per_cell_or_none_admits_it(self, profile):
        assert technology_block_errors([ARM], technology_blocks(), profile) == []

    @pytest.mark.parametrize("block", [BLOCK, {"cells_per_weight": 4, "weight_cell_rows": 2}],
                             ids=["description_s_block", "bit_sliced_pair"])
    def test_a_stated_block_under_the_profile_is_admitted(self, block):
        assert technology_block_errors([{**ARM, **block}], technology_blocks(), "neurram") == []

    def test_named_technologies_storing_one_weight_per_cell_stand_for_both_fields(self):
        core = {**ARM, "write_physics": "sram_cim_macro", "compute_physics": "sram_cim_macro"}
        assert technology_block_errors([core], technology_blocks(), "neurram") == []


def _replaced(name, cores):
    return {"target": name, "target_overrides": {"cores": {"value": copy.deepcopy(cores), "note": "replaced cores"}}}


class TestATargetSCoresReplaced:
    """[targeted pass, review M-1] ``target_overrides.cores`` replaces the chip description's cores, and the target's
    profile still prices them: a replacement that states no block refuses by name at the run door."""

    @pytest.mark.parametrize("geometry", [(64, 256), (128, 256), (256, 256)])
    def test_w38_r4_s_arm_form_refuses_by_name(self, geometry):
        rows, columns = geometry
        with pytest.raises(ToolchainRefusal, match=re.escape(RUN_NAMED)):
            _resolved(_replaced("neurram", [{**ARM, "max_axons": rows, "max_neurons": columns}]))

    def test_isaac_like_s_cores_replaced_without_the_block_refuse_by_name(self):
        named = ("cores[0] names no write_physics or compute_physics, so the run's platform_physics_profile "
                 "'isaac_like' stands for them, whose chip description (target 'isaac_like') stores a weight in 8 cells")
        with pytest.raises(ToolchainRefusal, match=re.escape(named)):
            _resolved(_replaced("isaac_like", [{"max_axons": 128, "max_neurons": 16, "count": 96}]))

    @pytest.mark.parametrize("name", ["neurram", "isaac_like"])
    def test_the_target_s_own_cores_resolve_unchanged(self, name):
        keys = ("max_axons", "max_neurons", "count", "cells_per_weight", "weight_cell_rows")
        resolved = [{k: c.get(k) for k in keys} for c in _resolved({"target": name})["cores"]]
        assert resolved == [{k: c.get(k) for k in keys} for c in target_bundles()[name].values["cores"]]

    def test_replaced_cores_stating_the_block_are_admitted(self):
        static, = _resolved(_replaced("neurram", [{**ARM, "max_axons": 128, **BLOCK}]))["cores"]
        assert (static["max_axons"], static["cells_per_weight"], static["weight_cell_rows"]) == (128, 2, 2)


class TestTheRunDoor:
    """Where the conversion model's count is held to the core types' (``conversion_conflict``): the run door."""

    def test_a_core_type_naming_the_technology_without_its_block_refuses_at_the_run_door(self, tmp_path):
        _document, _wizard, run = _verdicts(_value_document([NEURRAM]), tmp_path)
        assert isinstance(run, ToolchainRefusal) and NAMED in str(run), repr(run)

    def test_the_stated_block_is_admitted_at_every_door(self, tmp_path):
        document, wizard, run = _verdicts(_value_document([{**NEURRAM, **BLOCK}]), tmp_path)
        assert not [m for m in document if "chip description" in m], document
        assert not [m for m in wizard if "chip description" in m], wizard
        assert run is None, repr(run)


def _resolved(declaration):
    from mimarsinan.config_schema.runtime import run_config
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    return build_platform_constraints_resolved(run_config({}, copy.deepcopy(declaration)))


class TestFigureOnesMixedRow:
    def test_the_fixture_states_neurram_s_block_on_its_static_core_type(self):
        static, operand = _resolved(json.loads(FIXTURE.read_text())["platform_constraints"])["cores"]
        assert (static["max_axons"], static["max_neurons"], static["count"]) == (128, 256, 48)
        assert (static["cells_per_weight"], static["weight_cell_rows"]) == (2, 2)
        assert array_extent(static) == (256, 256)
        assert static["write_physics"] == static["compute_physics"] == "neurram"
        assert "cells_per_weight" not in operand and operand["write_physics"] == "sram_cim_macro"

    def test_the_fixture_s_static_core_type_is_neurram_s_own_core(self):
        bundle, = target_bundles()["neurram"].values["cores"]
        static = json.loads(FIXTURE.read_text())["platform_constraints"]["target_overrides"]["cores"]["value"][0]
        keys = ("max_axons", "max_neurons", "count", "cells_per_weight", "weight_cell_rows")
        assert {k: static[k] for k in keys} == {k: bundle[k] for k in keys}

    def test_the_static_core_type_without_its_block_refuses_by_name(self):
        declaration = json.loads(FIXTURE.read_text())["platform_constraints"]
        static = declaration["target_overrides"]["cores"]["value"][0]
        for key in BLOCK:
            static.pop(key, None)
        static.update(max_axons=256, max_neurons=256)
        with pytest.raises(ToolchainRefusal, match="names write_physics 'neurram'"):
            _resolved(declaration)


class TestTheInstancePrintsTheBlock:
    """Figure 1's document states each core type's block beside its geometry, so a row's 128 x 256 static type reads
    as NeuRRAM's 256 x 256 cells."""

    def test_a_core_type_record_keeps_the_block(self):
        from mimarsinan.deployment_record.design_point.chip_rows import core_type_record
        record = core_type_record({"role": "static", **NEURRAM, **BLOCK, "write_technology": "neurram"})
        assert (record["max_axons"], record["max_neurons"]) == (128, 256)
        assert (record["cells_per_weight"], record["weight_cell_rows"]) == (2, 2)
        plain = core_type_record({"role": "operand", "max_axons": 128, "max_neurons": 1024, "count": 1,
                                  "write_technology": "sram_cim_macro"})
        assert "cells_per_weight" not in plain and "weight_cell_rows" not in plain

    def test_the_instance_schema_admits_the_block(self):
        from mimarsinan.deployment_record.design_point.schema import validate_instance
        from unit.deployment_record.design_point.test_schema import _vit
        document = _vit()
        row = next(r for r in document["rows"] if r["row"] == "rram_static_sram_operand")
        static = next(c for c in row["chip"]["core_types"] if c["role"] == "static")
        static.update(BLOCK)
        assert validate_instance(document) == []
