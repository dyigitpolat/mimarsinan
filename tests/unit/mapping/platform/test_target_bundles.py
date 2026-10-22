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
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

PHYSICS_BUNDLES = {
    "odin": "odin", "truenorth": "truenorth", "loihi": "loihi", "isaac_like": "isaac_like",
    "generic_estimated_22nm": "generic_estimated_22nm",
}
#: The geometries of three shipped profiles are bundled once, as physics bundles (plan section 8.1).
PHYSICS_GEOMETRY = {"truenorth": "truenorth_like", "loihi": "loihi_dense_equiv_128x1024", "isaac_like": "isaac_like"}
SYNTHETIC_GRIDS = {"imc_128x128", "imc_256x256", "imc_512x512", "imc_mixed_tile"}
LITERATURE = set(imc_platform_names()) - SYNTHETIC_GRIDS


def _bundle(name="hand", geometry="recom_64x64_256", keys=None, faithfulness=None, **extra):
    data = {"name": name, "geometry": geometry, "keys": keys if keys is not None else {},
            "faithfulness": faithfulness or {"class": "exact"}}
    data.update(extra)
    return data


def _write(directory, *bundles):
    for data in bundles:
        (directory / f"{data['name']}.json").write_text(json.dumps(data))
    return directory


class TestShippedRegistry:
    def test_five_physics_bundles_and_nine_geometry_only_bundles(self):
        bundles = target_bundles()
        physics = {name for name, b in bundles.items() if "platform_physics_profile" in b.values}
        assert {name: bundles[name].values["platform_physics_profile"] for name in physics} == PHYSICS_BUNDLES
        geometry_only = set(bundles) - physics
        assert geometry_only == LITERATURE - set(PHYSICS_GEOMETRY.values())
        assert len(geometry_only) == 9
        for name in geometry_only:
            assert bundles[name].geometry == name
            assert set(bundles[name].values) == {"cores", "weight_bits"}

    def test_every_literature_geometry_is_bundled_once_and_no_synthetic_grid(self):
        geometries = [b.geometry for b in target_bundles().values()]
        assert len(geometries) == len(set(geometries))
        assert LITERATURE <= set(geometries)
        assert not SYNTHETIC_GRIDS & set(geometries)
        for name, geometry in PHYSICS_GEOMETRY.items():
            assert target_bundles()[name].geometry == geometry

    def test_a_registered_geometry_is_taken_from_the_registry(self):
        for bundle in target_bundles().values():
            if bundle.geometry in imc_platform_names():
                platform = get_imc_platform(bundle.geometry)
                assert bundle.values["cores"] == [dict(core) for core in platform.cores]
                assert bundle.values["weight_bits"] == platform.weight_bits
                assert bundle.claim_class == platform.claim_eligibility

    def test_odin_and_the_22nm_point_declare_their_geometry_with_claim_classes(self):
        odin, point = target_bundles()["odin"], target_bundles()["generic_estimated_22nm"]
        assert odin.geometry not in imc_platform_names() and point.geometry not in imc_platform_names()
        assert odin.values["cores"] == [{"max_axons": 128, "max_neurons": 256, "count": 1, "has_bias": False}]
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
        assert labels["loihi"] == "approximate: dense-equivalent capacity"
        assert labels["odin"] == "approximate: per-cycle soma law; unbounded membrane for the 8-bit saturating one"
        for name in set(labels) - set(PHYSICS_BUNDLES):
            assert labels[name] == "refused: no physics profile", name
        assert all(not label.startswith("exact") for label in labels.values())


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
        _write(tmp_path, _bundle(faithfulness=faithfulness))
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
        assert bundle.values["write_ports"] == 2 and bundle.faithfulness.label() == "exact"


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
        assert {"weight_bits", "cores.has_bias"} <= set(
            function_bearing_differences(bundles["truenorth"], bundles["loihi"]))


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
        assert pcfg["target_faithfulness_class"] == "refused: no physics profile"

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

    ODIN_POOL = {"value": [{"max_axons": 128, "max_neurons": 256, "count": 64, "has_bias": False}], "note": "a pool"}

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
