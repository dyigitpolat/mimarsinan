"""The target-bundle registry (plan CH-B0): bundle data, its load-time judges, decision classes, pricing refusal."""

import copy
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import mimarsinan
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.build import declared_paths, effective_class
from mimarsinan.config_schema.registry.decision_classes import FUNCTION_BEARING, class_of_path
from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics
from mimarsinan.mapping.platform.imc_platforms import get_imc_platform, imc_platform_names
from mimarsinan.mapping.platform.target_bundles import (
    PC,
    TargetBundleError,
    absolute_pricing_refusal,
    function_bearing_differences,
    load_bundles,
    target_bundles,
    target_decision_class,
    target_departures,
)
from mimarsinan.mapping.platform.target_expansion import target_assignments
from unit.mapping.platform.hand_descriptions import described
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

PHYSICS_BUNDLES = {
    "odin": "odin", "truenorth": "truenorth", "loihi": "loihi", "isaac_like": "isaac_like",
    "generic_estimated_22nm": "generic_estimated_22nm",
    # [CL2] the three page-sourced profiles (G3's sub-deliverable)
    "hermes_64": "hermes_64", "neurram": "neurram", "sram_cim_macro": "sram_cim_macro",
    # [U-E4 A-1] the co-design study's chips (COMPASS, VR-like and ISAAC-like tiles at the study's constants)
    "study_compass": "study_compass", "study_vr_like": "study_vr_like", "study_isaac_like": "study_isaac_like",
}
#: The geometries of five shipped profiles are bundled once, as physics bundles (plan sections 8.1 and 14 CL2).
PHYSICS_GEOMETRY = {"truenorth": "truenorth_like", "loihi": "loihi_dense_equiv_128x1024", "isaac_like": "isaac_like",
                    "neurram": "neurram_like", "hermes_64": "hermes_64"}
#: Physics bundles whose geometry is declared in bundle data (no registered geometry): name -> geometry.
DECLARED_GEOMETRY = {"odin": "odin_stock_core", "generic_estimated_22nm": "generic_22nm_256x256",
                     "sram_cim_macro": "dimc_macro_128x1024",
                     # [U-E4 A-1]
                     "study_compass": "study_compass_256x256x1", "study_vr_like": "study_vr_like_256x64x4",
                     "study_isaac_like": "study_isaac_like_128x128x2"}
SYNTHETIC_GRIDS = {"imc_128x128", "imc_256x256", "imc_512x512", "imc_mixed_tile"}
TARGETS_DIR = Path(mimarsinan.__file__).parent / "mapping" / "platform" / "targets"
LITERATURE = set(imc_platform_names()) - SYNTHETIC_GRIDS


def _bundle(name="hand", geometry="recom_64x64_256", keys=None, faithfulness=None, **extra):
    """[U-CHIPKEYS] A hand-made chip under the loader rule: value cores, every computation key named unstated."""
    data = {"name": name, "geometry": geometry, "keys": keys if keys is not None else {},
            "faithfulness": faithfulness or {"class": "exact"}}
    data.update(extra)
    return described(data)


def _write(directory, *bundles):
    for data in bundles:
        (directory / f"{data['name']}.json").write_text(json.dumps(data))
    return directory


class TestShippedRegistry:
    def test_eleven_physics_bundles_and_ten_geometry_only_bundles(self):
        """[U-C C-4] Loihi 2 (``loihi2_like``) is the tenth geometry-only bundle. [U-E4 A-1] The co-design study's
        three chips are the ninth to eleventh physics bundles."""
        bundles = target_bundles()
        physics = {name for name, b in bundles.items() if "platform_physics_profile" in b.values}
        assert {name: bundles[name].values["platform_physics_profile"] for name in physics} == PHYSICS_BUNDLES
        geometry_only = set(bundles) - physics
        assert geometry_only == LITERATURE - set(PHYSICS_GEOMETRY.values())
        assert len(geometry_only) == 10 and "loihi2_like" in geometry_only
        for name in geometry_only:
            assert bundles[name].geometry == name
            # [U-CHIPKEYS] beside its geometry, a geometry-only bundle sets only the keys its sources state
            sourced = {path for path, s in bundles[name].statements.items()
                       if "." not in path and (s["kind"] == "stated" or "assumed" in s)}
            assert set(bundles[name].values) - {"cores", "weight_bits"} == sourced - {"weight_bits"}
            assert "core_semantics" in sourced

    def test_every_literature_geometry_is_bundled_once_and_no_synthetic_grid(self):
        geometries = [b.geometry for b in target_bundles().values()]
        assert len(geometries) == len(set(geometries))
        assert LITERATURE <= set(geometries)
        assert not SYNTHETIC_GRIDS & set(geometries)
        for name, geometry in PHYSICS_GEOMETRY.items():
            assert target_bundles()[name].geometry == geometry
        assert set(geometries) - LITERATURE == set(DECLARED_GEOMETRY.values())
        for name, geometry in DECLARED_GEOMETRY.items():
            assert target_bundles()[name].geometry == geometry and geometry not in imc_platform_names()

    def test_a_registered_geometry_is_taken_from_the_registry(self):
        """[U-C C-2] with the bundle's own core fields (its cores' write path) beside the geometry's; [U-C4] the
        geometry counts cells, so the cores hold it over their cells per weight and their array is the geometry."""
        from mimarsinan.mapping.platform.weight_cells import array_extent, weight_capacity
        for bundle in target_bundles().values():
            if bundle.geometry in imc_platform_names():
                platform = get_imc_platform(bundle.geometry)
                fields = json.loads((TARGETS_DIR / f"{bundle.name}.json").read_text()).get("core_fields", {})
                declared = [{**dict(core), **fields} for core in platform.cores]
                assert bundle.values["cores"] == weight_capacity(declared)[0]
                assert [array_extent(c) for c in bundle.values["cores"]] == [
                    (core["max_axons"], core["max_neurons"]) for core in platform.cores]
                assert bundle.values["weight_bits"] == platform.weight_bits
                assert bundle.claim_class == platform.claim_eligibility

    def test_odin_and_the_22nm_point_declare_their_geometry_with_claim_classes(self):
        odin, point = target_bundles()["odin"], target_bundles()["generic_estimated_22nm"]
        assert odin.geometry not in imc_platform_names() and point.geometry not in imc_platform_names()
        # CL-SG (2026-10-20: 256 x 256, per-row sign): the physical core, one static sign per axon row.
        assert odin.values["cores"] == [{"max_axons": 256, "max_neurons": 256, "count": 1, "has_bias": False,
                                         "weight_sign_granularity": "per_axon"}]
        assert odin.values["weight_bits"] == 4 and odin.claim_class == "quarantined"
        # The chip's permissions as the odin-fpga-infra declaration states them (f08c8efa): no inter-core
        # partial-sum transfer, no re-thresholding of a partial sum.
        assert odin.values["allow_coalescing"] is False and odin.values["allow_neuron_splitting"] is False
        assert point.values["cores"] == [{"max_axons": 256, "max_neurons": 256, "count": 1000}]
        assert point.values["weight_bits"] == 8 and point.claim_class == "synthetic"

    def test_every_physics_profile_resolves_and_no_bundle_sets_the_override_block(self):
        for name, profile in PHYSICS_BUNDLES.items():
            assert resolve_platform_physics(profile, {}) is not None, name
        for bundle in target_bundles().values():
            assert "platform_physics_overrides" not in bundle.values

    def test_faithfulness_classes(self):
        labels = {name: b.faithfulness.label() for name, b in target_bundles().items()}
        assert labels["loihi"] == 'approximate: named "dense-equivalent capacity"'  # [U-CHIPKEYS6] quoted
        # [U-CHIPKEYS] ODIN states its RTL's soma law (per-event compare, an 8-bit unsigned saturating membrane, reset to
        # zero): the two departures it named are gone, and it is the one exact description
        # [U-CHIPKEYS2] with the configuration the exact class rests on, which its label states
        assert labels["odin"] == 'exact, given "leakage and SDSP learning off by the deployment"'
        from mimarsinan.mapping.platform.descriptions.computation_keys import departures
        for name in set(labels) - set(PHYSICS_BUNDLES):
            # [U-CHIPKEYS3] a departing statement's departure follows the class's own (DYNAP-SE's weight grid)
            own = departures(target_bundles()[name].statements)
            assert labels[name] == "; ".join(("refused: no physics profile",) + own), name
        assert [name for name, label in labels.items() if label.startswith("exact")] == ["odin"]


class TestLoadJudges:
    def test_a_geometry_bundled_twice_refuses_by_name(self, tmp_path):
        _write(tmp_path, _bundle("first"), _bundle("second"))
        with pytest.raises(TargetBundleError, match=r"recom_64x64_256.*already bundled by 'first'"):
            load_bundles(tmp_path)

    def test_a_declared_geometry_under_a_registered_name_refuses(self, tmp_path):
        geometry = {"name": "neurram_like", "cores": [{"max_axons": 8, "max_neurons": 8}], "weight_bits": 4,
                    "provenance": "hand", "claim_class": "synthetic"}
        _write(tmp_path, _bundle(geometry=geometry))
        with pytest.raises(TargetBundleError, match="second geometry under a registered name"):
            load_bundles(tmp_path)

    def test_a_bundle_declaring_platform_physics_overrides_refuses_by_name(self, tmp_path):
        _write(tmp_path, _bundle(keys={"platform_physics_overrides": {}}))
        with pytest.raises(TargetBundleError, match="platform_physics_overrides"):
            load_bundles(tmp_path)

    @pytest.mark.parametrize("keys,named", [
        ({"core_semantics": "quantum"}, "core_semantics"),
        ({"no_such_key": 1}, "no_such_key"),
        ({"target_claim_class": "synthetic"}, "target_claim_class"),
        ({"target": "odin"}, "target"),
        ({"weight_bits": 4}, "weight_bits is set by the geometry"),
        ({"tile_grid_rows": -1}, "tile_grid_rows"),
        ({"readout_law": {"adc_bits": 1, "range": "calibrated", "mult_bits": 8}}, "readout_law.adc_bits"),
    ], ids=lambda x: str(x)[:30])
    def test_every_key_is_judged_by_the_registry(self, tmp_path, keys, named):
        _write(tmp_path, _bundle(keys=keys))
        with pytest.raises(TargetBundleError, match=named):
            load_bundles(tmp_path)

    @pytest.mark.parametrize("faithfulness", [
        {"class": "exact", "named": ["something"]}, {"class": "approximate"}, {"class": "refused"},
        {"class": "faithful"}, {"class": "exact", "notes": "a typo"},
    ], ids=str)
    def test_a_malformed_faithfulness_class_refuses(self, tmp_path, faithfulness):
        # [U-CHIPKEYS4] 3(f) the chip states its pricing (a profile), so its class names what departs on its own
        _write(tmp_path, _bundle(faithfulness=faithfulness, keys={"platform_physics_profile": "generic_estimated_22nm"}))
        with pytest.raises(TargetBundleError, match="faithfulness"):
            load_bundles(tmp_path)

    def test_closed_fields_and_the_file_name(self, tmp_path):
        _write(tmp_path, _bundle(keyz={}))
        with pytest.raises(TargetBundleError, match="keyz"):
            load_bundles(tmp_path)
        other = tmp_path / "other"
        other.mkdir()
        (other / "file.json").write_text(json.dumps(_bundle("inner")))
        with pytest.raises(TargetBundleError, match="differs from the file name"):
            load_bundles(other)

    def test_an_unregistered_geometry_name_and_an_invalid_declared_geometry_refuse(self, tmp_path):
        _write(tmp_path, _bundle(geometry="no_such_chip"))
        with pytest.raises(TargetBundleError, match="no_such_chip"):
            load_bundles(tmp_path)
        bad = tmp_path / "bad"
        bad.mkdir()
        _write(bad, _bundle(geometry={"name": "x", "cores": [{"max_axons": 0, "max_neurons": 8}],
                                      "weight_bits": 4, "provenance": "hand", "claim_class": "synthetic"}))
        with pytest.raises(TargetBundleError, match="positive"):
            load_bundles(bad)

    def test_load_refusals_are_toolchain_refusals(self):
        assert issubclass(TargetBundleError, ToolchainRefusal) and issubclass(TargetBundleError, ValueError)

    def test_a_valid_hand_bundle_carries_any_registry_key(self, tmp_path):
        """A key a later wave declares rides a bundle with no code change (the value schema judges it)."""
        _write(tmp_path, _bundle(keys={"write_ports": 2, "readout_law": {
            "adc_bits": 6, "range": "calibrated", "mult_bits": 16}}))
        bundle = load_bundles(tmp_path)["hand"]
        # [U-CHIPKEYS4] 3(f) the hand chip states no pricing: its label names the pricing it resolves (none)
        assert bundle.values["write_ports"] == 2 and bundle.faithfulness.label() == "approximate: no physics profile"


class TestDecisionClass:
    """Acceptance 8: the class of ``target: <name>`` and the function-bearing keys that tell bundles apart."""

    @pytest.mark.parametrize("name", sorted(target_bundles()))
    def test_the_target_class_is_the_join_over_the_keys_the_expansion_sets(self, name):
        written = target_assignments({}, {"target": name}, write_derived=False)
        values = {key: value for section in written.values() for key, value in section.items()}
        assert effective_class(declared_paths(values)) == target_decision_class(name)
        assert target_decision_class(name) == FUNCTION_BEARING

    def test_every_bundle_pair_prints_exactly_the_function_bearing_keys_that_differ(self):
        bundles = target_bundles()
        for first, second in itertools.combinations(sorted(bundles), 2):
            a, b = bundles[first], bundles[second]
            listed = set(function_bearing_differences(a, b))
            for path in set(declared_paths(a.values)) | set(declared_paths(b.values)):
                bearing = class_of_path(REGISTRY, path) == FUNCTION_BEARING
                key, _, field = path.partition(".")
                values = [_field(x.values, key, field) for x in (a, b)]
                assert (path in listed) == (bearing and values[0] != values[1]), (first, second, path)

    def test_a_chip_naming_key_is_never_one_class_for_two_chips(self):
        bundles = target_bundles()
        assert "core_semantics" in function_bearing_differences(bundles["odin"], bundles["isaac_like"])
        # [U-CHIPKEYS] TrueNorth and Loihi both add a bias on chip (a leak, a bias current); they differ in the weight
        # grid, the membrane and the threshold compare, and ODIN's bias rides a row
        assert {"weight_bits", "membrane_bits", "thresholding_mode"} <= set(
            function_bearing_differences(bundles["truenorth"], bundles["loihi"]))
        assert "cores.has_bias" in function_bearing_differences(bundles["odin"], bundles["loihi"])


def _field(values, key, field):
    value = values.get(key)
    if not field:
        return value
    if isinstance(value, list):
        return tuple(item.get(field) for item in value)
    return value.get(field) if isinstance(value, dict) else None


class TestGeometryOnlyBundles:
    """Acceptance 5: the geometry resolves, absolute pricing refuses physics by name, the claim class prints."""

    @staticmethod
    def _pcfg(name):
        """The resolved platform of ``target: name``, float weights where the bundle's signed weight grid has no
        nonzero level under weight quantization (key amendment K1 refuses that by name)."""
        bits = target_bundles()[name].values.get("weight_bits")
        dp = {"weight_quantization": False} if bits is not None and bits < REGISTRY["weight_bits"].applied_schema.low else {}
        return build_platform_constraints_resolved(merge_pipeline_config(dp, {"target": name}))

    @pytest.mark.parametrize("name", sorted(set(target_bundles()) - set(PHYSICS_BUNDLES)))
    def test_geometry_resolves_and_physics_refuses_by_name(self, name):
        pcfg = self._pcfg(name)
        bundle = target_bundles()[name]
        cores = copy.deepcopy(bundle.values["cores"])
        assert pcfg["cores"] == cores
        assert pcfg["weight_bits"] == bundle.values["weight_bits"]
        assert pcfg["platform_physics_resolved"] is None
        refusal, reason = absolute_pricing_refusal(pcfg)
        assert refusal == "absolute_pricing"
        assert name in reason and "no physics profile" in reason
        assert pcfg["target_claim_class"] == bundle.claim_class
        # [U-CHIPKEYS3] the class's own departure first, then its statements' (DYNAP-SE's weight grid departs)
        assert pcfg["target_faithfulness_class"] == bundle.faithfulness.label()
        assert bundle.faithfulness.label().startswith("refused: no physics profile")

    @pytest.mark.parametrize("name", sorted(PHYSICS_BUNDLES))
    def test_a_physics_bundle_has_no_pricing_refusal(self, name):
        pcfg = self._pcfg(name)
        assert pcfg["platform_physics_resolved"]["name"] == PHYSICS_BUNDLES[name]
        assert absolute_pricing_refusal(pcfg) is None

    def test_no_target_no_refusal(self):
        assert absolute_pricing_refusal(build_platform_constraints_resolved(merge_pipeline_config({}, {}))) is None


@pytest.mark.parametrize("module", ["mimarsinan.mapping.platform.target_bundles",
                                    "mimarsinan.mapping.platform.target_expansion", "mimarsinan.config_schema.registry"])
def test_imports_first_in_a_fresh_interpreter(module):
    """The registry's package imports the entry points, which import the expansion: no order may cycle, and the
    config entry points never import the absolute pricer (``absolute_pricing_refusal`` returns a plain pair)."""
    env = {**os.environ, "PYTHONPATH": str(Path(mimarsinan.__file__).resolve().parents[1])}
    code = (f"import sys; import {module}; import mimarsinan.mapping.platform.target_bundles as b; "
            "b.target_bundles(); import mimarsinan.mapping.platform.target_expansion; "
            "import mimarsinan.config_schema; "
            "assert 'mimarsinan.deployment_record.cost.absolute' not in sys.modules, 'pricer imported'")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=50)
    assert proc.returncode == 0, proc.stderr[-2000:]


class TestBundlesAreImmutable:
    """A bundle read never hands out the cached bundle's own objects (review round 2): a caller that edits what a
    read returns, a report formatter normalizing cores for instance, changes no later read or expansion."""

    ODIN_POOL = {"value": [{"max_axons": 256, "max_neurons": 256, "count": 64, "has_bias": False,
                            "weight_sign_granularity": "per_axon"}], "note": "a pool"}

    def test_a_departure_s_bundle_value_is_a_copy(self):
        before = copy.deepcopy(target_bundles()["odin"].values)
        platform = {"target": "odin", "target_overrides": {"cores": copy.deepcopy(self.ODIN_POOL)}}
        departure = target_departures(platform)[0]
        departure["bundle"][0]["count"] = 99
        departure["bundle"].append({"max_axons": 1, "max_neurons": 1, "count": 1})
        departure["declared"][0]["count"] = 7
        assert target_bundles()["odin"].values == before
        assert platform["target_overrides"]["cores"] == self.ODIN_POOL  # the declared value is a copy too
        assert target_departures({"target": "odin", "target_overrides": {"cores": self.ODIN_POOL}})[0]["bundle"] == (
            before["cores"])

    def test_each_read_of_a_bundle_s_values_is_a_fresh_copy(self):
        before = copy.deepcopy(target_bundles()["odin"].values)
        values = target_bundles()["odin"].values
        values["cores"][0]["count"] = 99
        values["weight_bits"] = 99
        del values["allow_coalescing"]
        assert target_bundles()["odin"].values == before
        written = target_assignments({}, {"target": "odin"}, write_derived=False)[PC]
        assert written["cores"] == before["cores"] and written["weight_bits"] == 4

    def test_the_shipped_mapping_and_its_bundles_are_read_only(self):
        bundles = target_bundles()
        with pytest.raises(TypeError):
            bundles["odin"] = bundles["loihi"]  # type: ignore[index]
        with pytest.raises(AttributeError):
            bundles["odin"].claim_class = "headline-eligible"  # type: ignore[misc]
        assert sorted(load_bundles(Path(mimarsinan.__file__).parent / "mapping" / "platform" / "targets")) == (
            sorted(bundles))
