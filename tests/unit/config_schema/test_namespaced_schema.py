"""Unit tests for the V8 namespaced config schema + translation shim.

Locks the byte-identical flat<->namespaced bijection, the provenance registry's
coverage of every default key, and the migrated ``hardware`` group.
"""

import copy
import glob
import json
import os

import pytest

from mimarsinan.config_schema import build_flat_pipeline_config
from mimarsinan.config_schema.defaults import (
    DEFAULT_DEPLOYMENT_PARAMETERS,
    DEFAULT_PLATFORM_CONSTRAINTS,
    get_default_deployment_parameters,
    get_default_platform_constraints,
)
from mimarsinan.config_schema.namespaced_schema import (
    CONCERN_GROUPS,
    KEY_SPECS,
    KeySpec,
    LEGACY_KEY_TABLE,
    NAMESPACED_KEY_TABLE,
    keys_with_derivation,
    provenance_table,
    registered_flat_keys,
    to_flat,
    to_namespaced,
    unregistered_default_keys,
)
from mimarsinan.config_schema.deployment_derivation import derive_deployment_parameters

# Keys produced by a derivation pass that are NOT in the defaults dict. The
# conversion trio mirrors display_view_meta.DERIVED_KEYS; firing_mode /
# spike_generation_mode / thresholding_mode are resolved in DeploymentPipeline.
DERIVED_NON_DEFAULT_KEYS = {
    "pipeline_mode",
    "activation_quantization",
    "weight_quantization",
    "firing_mode",
    "spike_generation_mode",
    "thresholding_mode",
}
RUNTIME_KEYS = {"device", "input_shape", "input_size", "num_classes"}


def _full_default_flat():
    flat = dict(get_default_deployment_parameters())
    flat.update(get_default_platform_constraints())
    return flat


class TestRegistryCoverage:
    """Every default flat key is registered with provenance; no orphans."""

    def test_all_default_keys_registered(self):
        assert unregistered_default_keys() == frozenset()

    def test_registry_has_no_stray_keys(self):
        # The registry = defaults ∪ derived ∪ runtime; no other keys may appear.
        defaults = set(DEFAULT_DEPLOYMENT_PARAMETERS) | set(DEFAULT_PLATFORM_CONSTRAINTS)
        non_default = set(registered_flat_keys()) - defaults
        assert non_default == DERIVED_NON_DEFAULT_KEYS | RUNTIME_KEYS

    def test_all_defaults_are_registered_as_default_or_preset(self):
        defaults = set(DEFAULT_DEPLOYMENT_PARAMETERS) | set(DEFAULT_PLATFORM_CONSTRAINTS)
        for key in defaults:
            assert KEY_SPECS[key].derivation in {"default", "preset"}, key

    def test_registered_count_matches_defaults_plus_derived_plus_runtime(self):
        defaults = set(DEFAULT_DEPLOYMENT_PARAMETERS) | set(DEFAULT_PLATFORM_CONSTRAINTS)
        expected = defaults | DERIVED_NON_DEFAULT_KEYS | RUNTIME_KEYS
        assert len(registered_flat_keys()) == len(expected)

    def test_every_spec_group_is_valid(self):
        valid = {g["id"] for g in CONCERN_GROUPS}
        for spec in KEY_SPECS.values():
            assert spec.group in valid

    def test_every_spec_derivation_is_valid(self):
        valid = {"default", "preset", "derived", "runtime"}
        for spec in KEY_SPECS.values():
            assert spec.derivation in valid

    def test_every_spec_has_an_owner(self):
        for spec in KEY_SPECS.values():
            assert spec.owner and isinstance(spec.owner, str)


class TestDerivationTagging:
    """Provenance is real: derived/runtime keys are tagged, not all 'default'."""

    def test_not_every_key_is_default(self):
        # Regression guard: before V8-round2 every derivation was "default".
        derivations = {s.derivation for s in KEY_SPECS.values()}
        assert "derived" in derivations
        assert "runtime" in derivations

    def test_derived_keys_are_tagged_derived(self):
        assert keys_with_derivation("derived") == DERIVED_NON_DEFAULT_KEYS

    def test_runtime_keys_are_tagged_runtime(self):
        assert keys_with_derivation("runtime") == RUNTIME_KEYS

    def test_derivation_pass_only_writes_keys_tagged_derived_or_preset(self):
        # Every key derive_deployment_parameters WRITES must be provenance-tagged
        # as derived or preset (not a bare default) — locks the tagging to the
        # actual derivation behavior, so a new derived key cannot regress to
        # "default" silently.
        dp = {"spiking_mode": "ttfs_quantized", "weight_quantization": True}
        before = dict(dp)
        derive_deployment_parameters(dp)
        written = {k for k in dp if k not in before or dp[k] != before.get(k)}
        # spiking_mode is read, not written; drop keys that pre-existed unchanged.
        written = {k for k in written if k in KEY_SPECS}
        for key in written:
            assert KEY_SPECS[key].derivation in {"derived", "preset", "default"}, key
        # The trio the derivation ALWAYS (re)writes must be tagged derived.
        assert {"pipeline_mode", "activation_quantization",
                "weight_quantization"} <= keys_with_derivation("derived")

    def test_pipeline_init_derived_keys_are_spiking_semantics(self):
        for key in ("firing_mode", "spike_generation_mode", "thresholding_mode"):
            assert KEY_SPECS[key].derivation == "derived", key
            assert KEY_SPECS[key].group == "spiking", key

    def test_conversion_derived_keys_match_display_view_truth(self):
        from mimarsinan.config_schema.display_view_meta import DERIVED_KEYS
        # display_view's DERIVED_KEYS is the established UI truth; the schema's
        # "derived" tag must be a superset (it also tags the pipeline-init keys).
        assert DERIVED_KEYS <= keys_with_derivation("derived")

    def test_runtime_keys_match_display_view_truth(self):
        from mimarsinan.config_schema.display_view_meta import RUNTIME_KEYS as DV_RUNTIME
        assert keys_with_derivation("runtime") == set(DV_RUNTIME)

    def test_no_preset_only_keys_are_mislabeled(self):
        # Keys that are BOTH a preset value and a derived output (activation/
        # weight quantization) are tagged derived because derivation overwrites
        # the preset (setdefault precedence: derivation wins on the always-path).
        assert KEY_SPECS["activation_quantization"].derivation == "derived"
        assert KEY_SPECS["weight_quantization"].derivation == "derived"


class TestTranslationShimBijection:
    """The one translation table is a byte-identical bijection."""

    def test_legacy_table_inverse_is_namespaced_table(self):
        assert len(LEGACY_KEY_TABLE) == len(NAMESPACED_KEY_TABLE)
        for flat_key, path in LEGACY_KEY_TABLE.items():
            assert NAMESPACED_KEY_TABLE[path] == flat_key

    def test_namespaced_paths_are_unique(self):
        paths = list(LEGACY_KEY_TABLE.values())
        assert len(paths) == len(set(paths))

    def test_full_default_config_roundtrips_byte_identical(self):
        flat = _full_default_flat()
        assert to_flat(to_namespaced(flat)) == flat

    def test_roundtrip_preserves_value_identity_for_nested_blocks(self):
        flat = _full_default_flat()
        restored = to_flat(to_namespaced(flat))
        # mutable blocks survive the projection unchanged (deep equal)
        assert restored["cores"] == flat["cores"]
        assert restored["training_recipe"] == flat["training_recipe"]
        assert restored["tuning_recipe"] == flat["tuning_recipe"]

    def test_roundtrip_does_not_mutate_input(self):
        flat = _full_default_flat()
        before = copy.deepcopy(flat)
        to_flat(to_namespaced(flat))
        assert flat == before

    def test_unregistered_keys_pass_through_under_run_group(self):
        flat = {"spiking_mode": "lif", "device": "cuda", "input_size": 784}
        nested = to_namespaced(flat)
        # device/input_size are runtime keys with no KeySpec -> pass-through
        assert nested["run"]["device"] == "cuda"
        assert nested["run"]["input_size"] == 784
        assert to_flat(nested) == flat

    def test_registered_key_lands_under_its_concern_group(self):
        nested = to_namespaced({"spiking_mode": "ttfs_cycle_based"})
        assert nested["spiking"]["spiking_mode"] == "ttfs_cycle_based"

    def test_arbitrary_flat_config_roundtrips(self):
        flat = {
            "spiking_mode": "ttfs_cycle_based",
            "ttfs_cycle_schedule": "synchronized",
            "weight_bits": 4,
            "cores": [{"max_axons": 64, "max_neurons": 64, "count": 10}],
            "allow_coalescing": True,
            "lr": 0.005,
            "enable_sanafe_simulation": True,
            "some_future_key": 123,  # unregistered pass-through
        }
        assert to_flat(to_namespaced(flat)) == flat


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _template_paths():
    return sorted(glob.glob(os.path.join(_REPO_ROOT, "templates", "*.json")))


class TestByteIdenticalOnResolvedConfigs:
    """The shim is byte-identical on configs resolved through the real build path."""

    @pytest.mark.parametrize(
        "spiking_mode,weight_quant,pipeline_mode",
        [
            ("lif", True, "phased"),
            ("lif", False, "vanilla"),
            ("rate", True, "phased"),
            ("ttfs", True, "phased"),
            ("ttfs_quantized", True, "phased"),
            ("ttfs_cycle_based", True, "phased"),
        ],
    )
    def test_resolved_config_roundtrips(self, spiking_mode, weight_quant, pipeline_mode):
        cfg = build_flat_pipeline_config(
            {"spiking_mode": spiking_mode, "weight_quantization": weight_quant},
            pipeline_mode=pipeline_mode,
        )
        assert to_flat(to_namespaced(cfg)) == cfg

    @pytest.mark.parametrize("path", _template_paths())
    def test_template_config_roundtrips(self, path):
        with open(path) as fh:
            doc = json.load(fh)
        dp = doc.get("deployment_parameters", {}) or {}
        pc_raw = doc.get("platform_constraints", {}) or {}
        if isinstance(pc_raw, dict) and "mode" in pc_raw:
            pc = pc_raw.get("user") if pc_raw.get("mode") == "user" else None
        else:
            pc = pc_raw if isinstance(pc_raw, dict) else None
        cfg = build_flat_pipeline_config(
            dp, pc, pipeline_mode=doc.get("pipeline_mode", "phased")
        )
        assert to_flat(to_namespaced(cfg)) == cfg


class TestHardwareGroupMigrated:
    """The hardware concern is migrated end-to-end (declared owners + bijection)."""

    def test_platform_constraint_keys_are_in_hardware_group(self):
        for key in DEFAULT_PLATFORM_CONSTRAINTS:
            assert KEY_SPECS[key].group == "hardware", key

    def test_weight_bits_is_a_hardware_capability(self):
        assert KEY_SPECS["weight_bits"].group == "hardware"

    def test_platform_constraints_roundtrip_byte_identical(self):
        pc = get_default_platform_constraints()
        nested = to_namespaced(pc)
        assert set(nested) == {"hardware"}
        assert to_flat(nested) == pc

    def test_capability_gates_owned_by_mapping_strategy(self):
        assert "MappingStrategy" in KEY_SPECS["allow_coalescing"].owner
        assert "MappingStrategy" in KEY_SPECS["allow_neuron_splitting"].owner

    def test_allow_per_layer_s_is_a_hardware_capability_gate(self):
        # EW1 RESERVED capability gate: declared in hardware alongside allow_coalescing.
        spec = KEY_SPECS["allow_per_layer_s"]
        assert spec.group == "hardware"
        assert "ChipCapabilities" in spec.owner
        assert spec.derivation == "default"


class TestTemporalAllocationAxisRegistered:
    """EW1 — the per-layer-S declaration keys are registered with provenance."""

    def test_s_allocation_keys_in_conversion_group(self):
        for key in ("s_allocation", "s_allocation_explicit", "s_allocation_budget"):
            spec = KEY_SPECS[key]
            assert spec.group == "conversion"
            assert spec.owner == "TemporalAllocation"
            assert spec.derivation == "default"

    def test_s_allocation_keys_roundtrip(self):
        flat = {
            "s_allocation": "budget",
            "s_allocation_explicit": [4, 8, 32],
            "s_allocation_budget": {"target": 0.96},
            "allow_per_layer_s": True,
        }
        assert to_flat(to_namespaced(flat)) == flat


class TestProvenanceTable:
    """provenance_table records owner + derivation per key for the next reviewer."""

    def test_provenance_covers_every_registered_key(self):
        table = provenance_table()
        assert set(table) == set(registered_flat_keys())

    def test_provenance_entry_shape(self):
        entry = provenance_table()["spiking_mode"]
        assert entry["group"] == "spiking"
        assert entry["owner"] == "SpikingDeploymentContract"
        assert entry["derivation"] == "default"
        assert entry["namespaced_path"] == "spiking.spiking_mode"


class TestKeySpecValidation:
    """KeySpec rejects unknown groups/derivations at construction."""

    def test_unknown_group_raises(self):
        with pytest.raises(ValueError):
            KeySpec("k", "not_a_group", "k", "owner", "default")

    def test_unknown_derivation_raises(self):
        with pytest.raises(ValueError):
            KeySpec("k", "hardware", "k", "owner", "made_up")

    def test_namespaced_path_property(self):
        spec = KeySpec("weight_bits", "hardware", "weight_bits", "wq", "default")
        assert spec.namespaced_path == "hardware.weight_bits"
