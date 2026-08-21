"""The early soma-axes fold, its contradictions, and the registry wiring.

The two deployment axes are folded EARLY — before the recipe fold and the
``sim_enables`` derivation — so nothing downstream reads a raw config key. The
fold is total and idempotent over every historical dict shape, and the
bits-driven contradictions are keyed, remediable errors.
"""

import pytest

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.config_schema.defaults import (
    CONFIG_KEYS_SET,
    DEFAULT_DEPLOYMENT_PARAMETERS,
    DEFAULT_PLATFORM_CONSTRAINTS,
)
from mimarsinan.config_schema.deployment_derivation import (
    derive_pipeline_runtime_parameters,
)
from mimarsinan.config_schema.derivation.soma import (
    enforce_soma_axes_contract,
    fold_soma_axes,
    soma_contract_error_rows,
)
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.config_schema.runtime import build_flat_pipeline_config

_SOMA_DEPLOYMENT_KEYS = ("firing_granularity", "membrane_arithmetic")
_SOMA_PLATFORM_KEYS = ("membrane_bits", "weight_sign_granularity")

_BIASLESS_CORES = [{"max_axons": 128, "max_neurons": 256, "count": 4,
                    "has_bias": False}]


def _document(dp=None, pc=None) -> dict:
    return {
        "data_provider_name": "MNIST_DataProvider",
        "experiment_name": "soma",
        "generated_files_path": "./generated",
        "platform_constraints": dict(pc or {}),
        "deployment_parameters": {
            "model_type": "lenet5",
            "model_config": {"variant": "lenet5"},
            **(dp or {}),
        },
    }


class TestRegistryWiring:
    @pytest.mark.parametrize("key", _SOMA_DEPLOYMENT_KEYS + _SOMA_PLATFORM_KEYS)
    def test_the_key_is_registered_and_in_the_live_key_set(self, key):
        assert key in REGISTRY
        assert key in CONFIG_KEYS_SET

    @pytest.mark.parametrize("key", _SOMA_DEPLOYMENT_KEYS)
    def test_the_deployment_axes_carry_no_schema_default(self, key):
        """No DEFAULT_DEPLOYMENT_PARAMETERS entry — the derived default is the
        SSOT, so the golden snapshot diff stays a pure addition."""
        entry = REGISTRY[key]
        assert key not in DEFAULT_DEPLOYMENT_PARAMETERS
        assert not entry.has_default()
        assert entry.derived_default is not None
        assert entry.legal_values is not None
        assert entry.provenance == "derivation rule"
        assert entry.domain == "event"
        assert entry.group == "spiking"
        assert entry.section == "deployment_parameters"

    @pytest.mark.parametrize("key,default", [
        ("membrane_bits", 0), ("weight_sign_granularity", "per_synapse"),
    ])
    def test_the_platform_widths_default_beside_weight_bits(self, key, default):
        assert DEFAULT_PLATFORM_CONSTRAINTS[key] == default
        assert REGISTRY[key].section == "platform_constraints"
        assert REGISTRY[key].default == default


class TestTheFoldIsTotalAndIdempotent:
    @pytest.mark.parametrize("dp", [
        {},
        {"spiking_family": "lif"},
        {"spiking_family": "lif", "spiking_variant": "streamed"},
        {"spiking_family": "lif", "spiking_variant": "synchronized"},
        {"spiking_family": "ttfs", "spiking_variant": "analytical"},
        {"spiking_mode": "lif"},
        {"spiking_mode": "ttfs_cycle_based", "ttfs_cycle_schedule": "synchronized"},
        {"core_semantics": "mvm"},
        {"spiking_family": "banana"},
    ])
    def test_the_fold_resolves_both_axes_and_is_idempotent(self, dp):
        folded = dict(dp)
        fold_soma_axes(folded)
        for key in _SOMA_DEPLOYMENT_KEYS:
            assert folded[key] is not None, key
        once = dict(folded)
        fold_soma_axes(folded)
        assert folded == once

    def test_the_fold_never_overwrites_a_declaration(self):
        dp = {"spiking_family": "lif", "spiking_variant": "streamed",
              "firing_granularity": "per_event"}
        fold_soma_axes(dp)
        assert dp["firing_granularity"] == "per_event"

    def test_the_fold_writes_exactly_the_two_deployment_axes(self):
        dp = {"spiking_family": "lif"}
        before = set(dp)
        fold_soma_axes(dp)
        assert set(dp) - before == set(_SOMA_DEPLOYMENT_KEYS)


class TestBitsDrivenContradictions:
    def test_an_explicit_unbounded_against_a_declared_width_raises(self):
        cfg = {"spiking_family": "lif", "spiking_variant": "streamed",
               "membrane_bits": 8, "membrane_arithmetic": "unbounded"}
        with pytest.raises(ValueError, match="membrane_arithmetic"):
            enforce_soma_axes_contract(cfg)

    def test_saturating_without_a_width_raises(self):
        cfg = {"spiking_family": "lif", "spiking_variant": "streamed",
               "membrane_arithmetic": "saturating_unsigned"}
        with pytest.raises(ValueError, match="membrane_bits"):
            enforce_soma_axes_contract(cfg)

    def test_the_contradiction_is_a_keyed_remediable_row(self):
        rows = soma_contract_error_rows(
            {"spiking_family": "lif", "spiking_variant": "streamed",
             "membrane_arithmetic": "unbounded"},
            {"membrane_bits": 8},
        )
        assert [row["key"] for row in rows] == ["membrane_arithmetic"]
        assert rows[0]["rule_id"] == "soma_law_contract"
        actions = {(r["action"], r["key"]) for r in rows[0]["remedies"]}
        assert ("clear", "membrane_arithmetic") in actions
        assert ("clear", "membrane_bits") in actions

    def test_a_consistent_declaration_is_silent(self):
        enforce_soma_axes_contract({
            "spiking_family": "lif", "spiking_variant": "streamed",
            "membrane_bits": 8, "membrane_arithmetic": "saturating_unsigned",
            "cores": _BIASLESS_CORES,
        })
        assert soma_contract_error_rows({}, {}) == []


class TestPerEventCrossKeyRequirements:
    def _per_event(self, **dp):
        return {"spiking_family": "lif", "spiking_variant": "streamed",
                "firing_granularity": "per_event", "cores": _BIASLESS_CORES, **dp}

    def test_per_event_requires_a_param_encoded_bias(self):
        cfg = self._per_event()
        cfg["cores"] = [{"max_axons": 128, "max_neurons": 256, "count": 4,
                         "has_bias": True}]
        with pytest.raises(ValueError, match="firing_granularity"):
            enforce_soma_axes_contract(cfg)

    def test_a_param_encoded_bias_satisfies_the_requirement(self):
        enforce_soma_axes_contract(self._per_event())

    @pytest.mark.parametrize("init", [1.0, 1.5, -0.25])
    def test_per_event_refuses_a_membrane_init_outside_the_unit_window(self, init):
        with pytest.raises(ValueError, match="lif_membrane_init"):
            enforce_soma_axes_contract(self._per_event(lif_membrane_init=init))

    @pytest.mark.parametrize("init", [0.0, 0.5, 0.99])
    def test_per_event_admits_a_membrane_init_inside_the_window(self, init):
        enforce_soma_axes_contract(self._per_event(lif_membrane_init=init))

    def test_the_contract_is_inert_under_the_default_point(self):
        enforce_soma_axes_contract({"spiking_family": "lif",
                                    "lif_membrane_init": -0.5})


class TestTheContractIsTotalOverAdversarialGrids:
    """The contract judges a RAW, un-normalized draft grid at BOTH seams, so no
    grid shape may raise: a shape it cannot parse as a core grid leaves the row
    silent (the document's own shape validators own that complaint), and a grid
    that genuinely declares a bias lane is a KEYED row, never a crash."""

    # Shapes no reader can parse as a core grid — the soma row stays silent.
    _UNJUDGEABLE_GRIDS = ["nope", 7, ["a", "b"], {"count": 2}, True]
    # Well-shaped grids whose core types default to an on-chip bias lane.
    _LANE_GRIDS = [[{}], [{"count": 2}], [{"max_axons": 128, "max_neurons": 256}]]
    # Well-shaped grids that declare the param-encoded bias per_event needs.
    _BIASLESS_GRIDS = [_BIASLESS_CORES, [{"count": 2, "has_bias": False}]]
    _ALL_GRIDS = _UNJUDGEABLE_GRIDS + _LANE_GRIDS + _BIASLESS_GRIDS + [None, []]

    _DOCUMENT_FAMILIES = {
        "per_event": {"spiking_family": "lif", "spiking_variant": "streamed",
                      "firing_granularity": "per_event"},
        "saturating": {"spiking_family": "lif", "spiking_variant": "streamed",
                       "membrane_arithmetic": "saturating_unsigned"},
        "mvm": {"core_semantics": "mvm", "firing_granularity": "per_event"},
        "legacy_only": {"spiking_mode": "lif"},
        "default_point": {"spiking_family": "lif"},
    }

    def _draft(self, family, cores):
        pc = {} if cores is None else {"cores": cores}
        return _document(self._DOCUMENT_FAMILIES[family], pc)

    def _rows(self, family, cores):
        return [(row["key"], row["rule_id"])
                for row in resolve_draft(self._draft(family, cores)).errors]

    @pytest.mark.parametrize("cores", _ALL_GRIDS)
    @pytest.mark.parametrize("family", sorted(_DOCUMENT_FAMILIES))
    def test_resolve_draft_returns_keyed_rows_and_never_raises(self, family, cores):
        for row in resolve_draft(self._draft(family, cores)).errors:
            assert row["rule_id"] and "message" in row

    @pytest.mark.parametrize("cores", _UNJUDGEABLE_GRIDS)
    @pytest.mark.parametrize("family", sorted(_DOCUMENT_FAMILIES))
    def test_an_unjudgeable_grid_contributes_no_bias_row(self, family, cores):
        """The bias question stays silent on a shape it cannot parse — the row
        set is the one a grid that satisfies the bias rule produces."""
        assert self._rows(family, cores) == self._rows(family, _BIASLESS_CORES)

    @pytest.mark.parametrize("cores", _LANE_GRIDS)
    def test_a_declared_bias_lane_is_a_keyed_row_not_a_crash(self, cores):
        rows = [row for row in resolve_draft(self._draft("per_event", cores)).errors
                if row["rule_id"] == "soma_law_contract"]
        assert [row["key"] for row in rows] == ["firing_granularity"]
        assert any(remedy["action"] == "clear"
                   and remedy["key"] == "firing_granularity"
                   for remedy in rows[0]["remedies"])

    @pytest.mark.parametrize("cores", _BIASLESS_GRIDS)
    def test_a_biasless_grid_needs_no_dimensions_to_satisfy_per_event(self, cores):
        assert not [row for row in self._rows("per_event", cores)
                    if row[1] in ("soma_law_contract", "derivation")]

    @pytest.mark.parametrize("cores", _ALL_GRIDS)
    @pytest.mark.parametrize("family", sorted(_DOCUMENT_FAMILIES))
    def test_the_run_path_never_raises_a_lookup_error(self, family, cores):
        """``derive_pipeline_runtime_parameters`` may refuse the point — with a
        DESIGNED keyed ValueError, never a KeyError/AttributeError from a grid
        key the bias question never needed."""
        dp = {**self._DOCUMENT_FAMILIES[family],
              **({} if cores is None else {"cores": cores})}
        try:
            derive_pipeline_runtime_parameters(dp)
        except ValueError as exc:
            assert any(key in str(exc) for key in
                       ("firing_granularity", "membrane_arithmetic",
                        "membrane_bits", "lif_membrane_init"))

    @pytest.mark.parametrize("cores", _LANE_GRIDS)
    def test_the_run_path_refuses_a_lane_grid_by_name(self, cores):
        dp = {**self._DOCUMENT_FAMILIES["per_event"], "cores": cores}
        with pytest.raises(ValueError, match="firing_granularity"):
            derive_pipeline_runtime_parameters(dp)

    @pytest.mark.parametrize("cores", _BIASLESS_GRIDS + _UNJUDGEABLE_GRIDS)
    def test_the_run_path_admits_what_the_resolve_channel_admits(self, cores):
        dp = {**self._DOCUMENT_FAMILIES["per_event"], "cores": cores}
        derive_pipeline_runtime_parameters(dp)
        assert dp["firing_granularity"] == "per_event"


class TestLegalityLandsThroughTheGenericMachinery:
    @pytest.mark.parametrize("variant", ["synchronized"])
    def test_per_event_under_a_windowed_variant_is_a_keyed_error(self, variant):
        resolution = resolve_draft(_document({
            "spiking_family": "lif", "spiking_variant": variant,
            "firing_granularity": "per_event",
        }))
        rows = [e for e in resolution.errors if e["rule_id"] == "legal_value_set"]
        assert [r["key"] for r in rows] == ["firing_granularity"]
        assert any(r["action"] == "clear" and r["key"] == "firing_granularity"
                   for r in rows[0]["remedies"])

    def test_per_event_under_ttfs_is_a_keyed_error(self):
        resolution = resolve_draft(_document({
            "spiking_family": "ttfs", "spiking_variant": "analytical",
            "firing_granularity": "per_event",
        }))
        rows = [e for e in resolution.errors if e["key"] == "firing_granularity"]
        assert rows and rows[0]["rule_id"] == "legal_value_set"

    def test_saturating_under_ttfs_is_a_keyed_error(self):
        resolution = resolve_draft(_document({
            "spiking_family": "ttfs", "spiking_variant": "analytical",
            "membrane_arithmetic": "saturating_unsigned",
        }))
        rows = [e for e in resolution.errors if e["key"] == "membrane_arithmetic"]
        assert rows and rows[0]["rule_id"] == "legal_value_set"


class TestTheDefaultPointResolvesInert:
    def test_every_resolved_config_carries_the_default_point(self):
        resolved = build_flat_pipeline_config(
            {"spiking_family": "lif"}, {"cores": _BIASLESS_CORES},
            pipeline_mode="phased",
        )
        assert resolved["firing_granularity"] == "per_cycle"
        assert resolved["membrane_arithmetic"] == "unbounded"
        assert resolved["membrane_bits"] == 0
        assert resolved["weight_sign_granularity"] == "per_synapse"
        assert SomaLaw.resolve(resolved).is_default_point

    def test_the_mvm_domain_resolves_the_inert_point_too(self):
        resolved = build_flat_pipeline_config(
            {"core_semantics": "mvm"}, {"cores": _BIASLESS_CORES},
            pipeline_mode="phased",
        )
        assert SomaLaw.resolve(resolved).is_default_point

    def test_a_membrane_width_is_dormant_in_an_mvm_document(self):
        """The width is event-domain: an mvm draft parks it (the wizard keeps
        it for switch-back) instead of resolving a temporal grid it has not."""
        resolution = resolve_draft(_document(
            {"core_semantics": "mvm"}, {"membrane_bits": 8},
        ))
        assert "membrane_bits" in resolution.dormant
        assert "membrane_bits" not in resolution.resolved or (
            resolution.resolved["membrane_bits"] == 0
        )

    def test_a_membrane_width_is_event_domain(self):
        from mimarsinan.config_schema.registry.domain_rules import mvm_document_errors

        errors = mvm_document_errors({
            "deployment_parameters": {"core_semantics": "mvm"},
            "platform_constraints": {"membrane_bits": 8},
        })
        assert any("membrane_bits" in message for message in errors)


class TestSimEnablesDeriveFromTheSamePointAwareQuery:
    """Under a point nothing executes, every backend derives OFF — and an
    explicit ON gets the keyed capability error the recipe fold already
    produces for a capability-off backend."""

    _ENABLES = ("enable_nevresim_simulation", "enable_sanafe_simulation",
                "enable_loihi_simulation")

    def _per_event_dp(self, **extra):
        return {"spiking_family": "lif", "spiking_variant": "streamed",
                "firing_mode": "Novena", "firing_granularity": "per_event",
                **extra}

    def _pc(self):
        return {"cores": _BIASLESS_CORES, "membrane_bits": 8}

    def test_the_default_point_keeps_every_enable_derivation(self):
        resolved = build_flat_pipeline_config(
            {"spiking_family": "lif"}, {"cores": _BIASLESS_CORES},
            pipeline_mode="phased",
        )
        assert resolved["enable_nevresim_simulation"] is True
        assert resolved["enable_sanafe_simulation"] is True
        assert resolved["enable_loihi_simulation"] is True

    def test_a_per_event_point_derives_every_backend_off(self):
        resolved = build_flat_pipeline_config(
            self._per_event_dp(), self._pc(), pipeline_mode="phased",
        )
        for key in self._ENABLES:
            assert resolved[key] is False, key

    @pytest.mark.parametrize("key", _ENABLES)
    def test_an_explicit_enable_against_the_point_is_a_keyed_error(self, key):
        with pytest.raises(ValueError, match=key):
            build_flat_pipeline_config(
                self._per_event_dp(**{key: True}), self._pc(),
                pipeline_mode="phased",
            )
