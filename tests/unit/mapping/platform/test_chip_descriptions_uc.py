"""[U-C C-2 to C-5] Chip descriptions: a bundle carries the write path of its cores (a geometry's cores take the
bundle's declared core fields, so NeuRRAM writes one cell at a time and HERMES-64 one core per write step, each from
its paper), Loihi 2 is registered as a curve-only dense-equivalent geometry with a geometry-only bundle, and the chip's
input width is a property of the chip description: a run may not declare another width except through
``target_overrides`` with a note, which the faithfulness label names and every record prints as a declared
departure (``declared_departures``)."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

import mimarsinan
from mimarsinan.deployment_record.build.builder import with_declared_departures
from mimarsinan.deployment_record.schema import DeploymentRecord
from mimarsinan.mapping.platform.imc_platforms import get_imc_platform
from mimarsinan.mapping.platform.target_bundles import (
    TargetBundleError,
    declared_departures,
    load_bundle,
    target_bundles,
    target_departures,
)
from mimarsinan.mapping.platform.target_expansion import TargetConflictError, expand_target
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

from unit.deployment_record.record_fixtures import make_full_record, make_identity

TARGETS_DIR = Path(mimarsinan.__file__).parent / "mapping" / "platform" / "targets"
#: bundle -> its input width (decision C6) and the page that states it.
INPUT_WIDTH = {"neurram": (4, "wan2022a Extended Data Table 1"), "hermes_64": (8, "Extended Data Table 1"),
               "sram_cim_macro": (4, "jiang2023a"), "isaac_like": (16, "shafiee2016isaac")}


def _file(name):
    return json.loads((TARGETS_DIR / f"{name}.json").read_text())


def _pcfg(name, **platform):
    return build_platform_constraints_resolved(merge_pipeline_config({}, {"target": name, **platform}))


class TestCoreFields:
    def test_neurram_writes_one_cell_at_a_time(self):
        core, = target_bundles()["neurram"].values["cores"]
        assert (core["write_granularity"], core["write_units_in_parallel"]) == ("cell", 1)
        notes = _file("neurram")["notes"]
        assert "individual RRAM cells are selected for read and write" in notes and "PDF p.10" in notes

    def test_hermes_64_writes_one_core_per_write_step(self):
        core, = target_bundles()["hermes_64"].values["cores"]
        assert core["write_granularity"] == "core" and "write_units_in_parallel" not in core
        assert "only 32 SLs are programmed in parallel" in _file("hermes_64")["notes"]

    def test_the_core_fields_extend_the_registered_geometry(self):
        """[U-C4] The registered geometry counts cells: its array is the bundle's cores at their cells per weight
        (NeuRRAM's 256 x 256 cells hold 128 x 256 weights; HERMES-64 stores one weight per cell)."""
        from mimarsinan.mapping.platform.weight_cells import array_extent
        dims = ("max_axons", "max_neurons")
        for name, geometry in (("neurram", "neurram_like"), ("hermes_64", "hermes_64")):
            core, = target_bundles()[name].values["cores"]
            registered, = get_imc_platform(geometry).cores
            assert {k: v for k, v in core.items() if k in registered and k not in dims} == {
                k: v for k, v in registered.items() if k not in dims}
            assert array_extent(core) == tuple(registered[d] for d in dims)

    def test_a_resolved_run_carries_them(self):
        core, = _pcfg("neurram", activation_bits=4)["cores"]
        assert core["write_granularity"] == "cell"

    @pytest.mark.parametrize("fields, message", [
        ({"max_axons": 8}, "set by the geometry"),
        ({"write_granularity": "page"}, "write_granularity"),
        ({"no_such_field": 1}, "no_such_field"),
    ])
    def test_a_core_field_the_geometry_sets_or_the_registry_refuses_is_refused(self, tmp_path, fields, message):
        data = {**_file("hermes_64"), "core_fields": fields}
        path = tmp_path / "hermes_64.json"
        path.write_text(json.dumps(data))
        with pytest.raises(TargetBundleError, match=message):
            load_bundle(path)


class TestLoihi2:
    def test_registered_curve_only_at_128_by_1024_with_8_bit_weights(self):
        platform = get_imc_platform("loihi2_like")
        assert [dict(c) for c in platform.cores] == [
            {"max_axons": 128, "max_neurons": 1024, "count": 128, "has_bias": False}]
        assert platform.weight_bits == 8 and platform.claim_eligibility == "curve-only"
        for quote in ("Up to 128 KB synaptic memory", "Up to 8192 neurons", "Max # Neuron Cores/Chip 128"):
            assert quote in platform.provenance, quote

    def test_a_geometry_only_bundle_refuses_physics_by_name(self):
        """The geometry-only convention (no keys; refused, no physics profile): the brief states no per-event
        energy or step time; the notes carry the dense-equivalent rule and the permissions the brief states."""
        bundle, pcfg = target_bundles()["loihi2_like"], _pcfg("loihi2_like")
        assert bundle.geometry == "loihi2_like" and bundle.claim_class == "curve-only"
        # [U-CHIPKEYS] beside the geometry, the one key the brief states: spiking cores
        assert set(bundle.values) == {"cores", "weight_bits", "core_semantics"}
        assert bundle.values["core_semantics"] == "spiking"
        assert pcfg["platform_physics_resolved"] is None
        assert pcfg["target_faithfulness_class"] == "refused: no physics profile"
        notes = _file("loihi2_like")["notes"]
        for quote in ("Up to 128 KB synaptic memory", "including all synapses connecting to the neurons",
                      "Core fan-out list compression", "0921/SBAI/KC/PDF"):
            assert quote in notes, quote

    @pytest.mark.parametrize("name, cls", [("hermes_64", "headline-eligible"), ("puma_node_like", "curve-only"),
                                           ("loihi2_like", "curve-only")])
    def test_the_chip_card_classes(self, name, cls):
        """Decision C5: HERMES-64 headline-eligible, the PUMA node and Loihi 2 curve-only."""
        assert target_bundles()[name].claim_class == cls


class TestInputWidth:
    @pytest.mark.parametrize("name", sorted(INPUT_WIDTH))
    def test_the_chip_description_fixes_its_input_width(self, name):
        bits, source = INPUT_WIDTH[name]
        assert target_bundles()[name].values["activation_bits"] == bits
        assert source in _file(name)["notes"]

    def test_the_synthetic_point_binds_none(self):
        """The generic estimated point is no chip: its input width is the run's declaration (route (b)'s T-INT)."""
        assert "activation_bits" not in target_bundles()["generic_estimated_22nm"].values

    @pytest.mark.parametrize("name", sorted(INPUT_WIDTH))
    def test_an_undeclared_width_takes_the_chip_s(self, name):
        _dp, pc = expand_target({}, {"target": name}, write_derived=False)
        assert pc["activation_bits"] == INPUT_WIDTH[name][0]

    def test_a_run_may_not_declare_another_width(self):
        with pytest.raises(TargetConflictError, match="activation_bits=8 differs"):
            expand_target({}, {"target": "neurram", "activation_bits": 8}, write_derived=True)

    def test_a_declared_departure_is_admitted_and_named(self):
        override = {"activation_bits": {"value": 8, "note": "8-bit inputs on the 4-bit chip (W38 arm W)"}}
        pcfg = _pcfg("neurram", target_overrides=override)
        assert pcfg["activation_bits"] == 8
        assert pcfg["target_faithfulness_class"].endswith("; activation_bits declared")
        assert target_departures(pcfg) == ({"key": "activation_bits", "bundle": 4, "declared": 8,
                                            "note": "8-bit inputs on the 4-bit chip (W38 arm W)"},)

    def test_route_b_s_t_int_form_is_admitted_unchanged(self):
        """generic_estimated_22nm with its declared 4-bit width and the cores and weight_bits overrides."""
        overrides = {"cores": {"value": [{"max_axons": 256, "max_neurons": 256, "count": 16, "has_bias": True}],
                               "note": "256x256 cores"},
                     "weight_bits": {"value": 4, "note": "4-bit weights"}}
        pcfg = _pcfg("generic_estimated_22nm", activation_bits=4, target_overrides=overrides)
        assert pcfg["activation_bits"] == 4
        assert "activation_bits" not in pcfg["target_faithfulness_class"]
        assert [d["key"] for d in target_departures(pcfg)] == ["cores", "weight_bits"]


class TestDeclaredDepartures:
    def test_a_record_prints_every_override_by_name(self):
        override = {"activation_bits": {"value": 8, "note": "arm W"}}
        pcfg = _pcfg("neurram", target_overrides=override)
        record = replace(make_full_record(), identity=replace(make_identity(), platform=pcfg))
        sealed = with_declared_departures(record)
        data = sealed.to_dict()
        assert data["declared_departures"] == [{"key": "activation_bits", "bundle": 4, "declared": 8,
                                                "note": "arm W"}]
        assert DeploymentRecord.from_dict(data) == sealed

    def test_a_record_without_departures_is_byte_for_byte_unchanged(self):
        record = replace(make_full_record(), identity=replace(make_identity(), platform=_pcfg("neurram")))
        assert with_declared_departures(record).to_dict() == record.to_dict()
        assert "declared_departures" not in record.to_dict()

    def test_a_declaration_that_replaces_a_bundle_key_departs_from_it(self):
        """The ladder's and the design point's declarations: a differing value departs, an equal one or a key the
        bundle does not set departs from nothing; a core field likewise."""
        assert declared_departures("neurram", {"activation_bits": 8, "core_semantics": "mvm"}) == (
            {"key": "activation_bits", "bundle": 4, "declared": 8},)
        assert declared_departures("neurram", {"activation_bits": 4, "inference_batch_K": 4}) == ()
        assert declared_departures("neurram", None, {"write_granularity": "row"}) == (
            {"key": "cores.write_granularity", "bundle": "cell", "declared": "row"},)
        assert declared_departures("odin", {"allow_coalescing": True}) == (
            {"key": "allow_coalescing", "bundle": False, "declared": True},)
        assert declared_departures("generic_estimated_22nm", {"activation_bits": 8}) == ()


SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"


def _load_script(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"{name}_uc_targeted", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ladder_script():
    return _load_script("scaling_ladder")


@pytest.fixture(scope="module")
def design_point_script():
    return _load_script("design_point_instance")


class TestDeparturePrints:
    """[targeted M4, M3(b)] The exploration's ladder rows and the design point's chip blocks print, by name, every key
    and core field their own declarations set to another value than the chip description's: the ladder per row from
    ``--declare`` and ``--core-field``, the design point from its build declarations and the spec's core fields, per
    core type where a row declares several."""

    def test_the_ladder_labels_name_each_departing_declaration(self, ladder_script):
        labels = ladder_script.departure_labels
        assert labels("neurram", {"activation_bits": 8, "core_semantics": "mvm"}, {"write_granularity": "cell"}) == {
            "declared_departures": [{"key": "activation_bits", "bundle": 4, "declared": 8}]}
        assert labels("odin", {"allow_coalescing": True}, None) == {
            "declared_departures": [{"key": "allow_coalescing", "bundle": False, "declared": True}]}
        assert labels("neurram", {"activation_bits": 4}, {"write_granularity": "cell"}) == {}
        assert labels("generic_22nm_on_imc_256x256", {"activation_bits": 8}, None) == {}  # a legacy point: no bundle
        assert labels("declared:16x16x4:value", {"activation_bits": 8}, None) == {}

    def test_every_row_of_a_ladder_run_carries_the_labels(self, ladder_script, monkeypatch):
        """A run whose network stops by name still writes one row per point, each with the labels."""
        from mimarsinan.common.refusal import ToolchainRefusal

        class _Stopped(ToolchainRefusal, ValueError):
            pass

        def stop(*args, **kwargs):
            raise _Stopped("stopped for the test")

        monkeypatch.setattr(ladder_script, "build_network", stop)
        rows = list(ladder_script.run(["resnet20"], ["neurram"], [1.0], [1, 16], None, False,
                                      declared={"activation_bits": 8}, point_budget_s=None))
        assert len(rows) == 2 and all(r["refusal"]["name"] == "_Stopped" for r in rows)
        assert all(r["declared_departures"] == [{"key": "activation_bits", "bundle": 4, "declared": 8}] for r in rows)
        plain = list(ladder_script.run(["resnet20"], ["neurram"], [1.0], [1], None, False, point_budget_s=None))
        assert plain and all("declared_departures" not in r for r in plain)

    def test_the_design_point_chip_block_names_the_build_s_departures(self, design_point_script):
        from mimarsinan.mapping.platform.target_bundles import target_bundles

        block = design_point_script.chip_block
        assert design_point_script.BUILD_DECLARATIONS["activation_bits"] == 8
        rram = block({}, {"target": "neurram", "core_fields": {"write_granularity": "cell"}, "arithmetic_class": "t"},
                     4)
        # [U-CHIPKEYS] the build's lane-free program departs, named, from a chip that declares on-chip classes
        assert rram["faithfulness"]["named"] == [*target_bundles()["neurram"].faithfulness.named,
                                                 "activation_bits declared", "onchip_arithmetic declared"]
        row = block({}, {"target": "neurram", "core_fields": {"write_granularity": "core"}, "arithmetic_class": "t"},
                    4)
        assert row["faithfulness"]["named"][-3:] == ["activation_bits declared", "onchip_arithmetic declared",
                                                     "cores.write_granularity declared"]
        for name in ("hermes_64", "generic_estimated_22nm"):  # an 8-bit chip, and the point that binds no width
            plain = block({}, {"target": name, "arithmetic_class": "t"}, 4)
            assert [n for n in plain["faithfulness"]["named"] if n.endswith(" declared")] == [
                "onchip_arithmetic declared"], name

    def test_a_row_of_several_core_types_names_each_type_s_departure(self, design_point_script):
        operand = {"payload_bits": 8, "write_source": "host"}
        row = {"core_types": [{"role": "static", "target": "neurram"},
                              {"role": "operand", "target": "sram_cim_macro", "operand": operand}],
               "arithmetic_class": "t"}
        block = design_point_script.chip_block({}, row, None)
        assert block["faithfulness"]["named"] == [  # [U-CHIPKEYS] the build's lane-free program departs, named
            "activation_bits declared (static core type, bundle neurram)",
            "onchip_arithmetic declared (static core type, bundle neurram)",
            "activation_bits declared (operand core type, bundle sram_cim_macro)"]
        eight = {"core_types": [{"role": "static", "target": "hermes_64"}], "arithmetic_class": "t"}
        assert design_point_script.chip_block({}, eight, None)["faithfulness"]["named"] == [
            "onchip_arithmetic declared (static core type, bundle hermes_64)"]
