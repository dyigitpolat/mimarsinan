"""THE legal-value-set law (round-6 item 3+4).

For any key whose legality depends on other config, the derivation exposes the
LEGAL VALUE SET for the current state. |legal| == 1 locks the field; |legal| > 1
allows an override restricted to those options; an explicit illegal value in a
document is a KEYED, remediable error — never an uncaught exception.

The rules live in the SSOTs (``chip_simulation.spiking_semantics``,
``tuning.orchestration.temporal_allocation``) and are wired into the registry;
nothing here (and nothing in JS) may hardcode a per-mode ladder.
"""

import itertools

import pytest

from mimarsinan.chip_simulation.firing_strategy import FiringStrategy, FiringMode
from mimarsinan.chip_simulation.spiking_semantics import (
    ALL_SPIKING_MODES,
    legal_firing_modes,
    legal_spike_generation_modes,
    legal_thresholding_modes,
)
from mimarsinan.config_schema.deployment_derivation import (
    derive_pipeline_runtime_parameters,
)
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.resolve import legal_values_view, resolve_draft
from mimarsinan.config_schema.validation import legality_errors
from mimarsinan.models.spiking.spiking_config import (
    FIRING_MODES,
    SPIKE_MODES,
    THRESHOLDING_MODES,
)

LEGALITY_KEYS = ("firing_mode", "spike_generation_mode", "thresholding_mode")

_MODES = sorted(ALL_SPIKING_MODES)


def _document(**deployment_parameters) -> dict:
    return {
        "data_provider_name": "MNIST_DataProvider",
        "experiment_name": "legality",
        "generated_files_path": "./generated",
        "start_step": None,
        "platform_constraints": {},
        "deployment_parameters": {
            "model_type": "lenet5",
            "model_config": {"variant": "lenet5"},
            **deployment_parameters,
        },
    }


class TestTheLegalitySsotMatchesItsEnforcer:
    """The legal sets are not a second opinion: they must agree exactly with
    the runtime validators that raise. A mutation of either side fails here."""

    @pytest.mark.parametrize("spiking_mode", _MODES)
    @pytest.mark.parametrize("firing_mode", sorted(FIRING_MODES))
    def test_firing_legality_agrees_with_the_firing_strategy_validator(
        self, spiking_mode, firing_mode
    ):
        strategy = FiringStrategy(mode=FiringMode(firing_mode), thresholding_mode="<=")
        legal = firing_mode in legal_firing_modes(spiking_mode)
        if legal:
            strategy.validate_for_spiking_mode(spiking_mode)
        else:
            with pytest.raises(ValueError):
                strategy.validate_for_spiking_mode(spiking_mode)

    @pytest.mark.parametrize("spiking_mode", _MODES)
    def test_thresholding_legality_agrees_with_the_validator(self, spiking_mode):
        assert set(legal_thresholding_modes(spiking_mode)) == set(THRESHOLDING_MODES)
        for mode in THRESHOLDING_MODES:
            FiringStrategy(
                mode=FiringMode(legal_firing_modes(spiking_mode)[0]),
                thresholding_mode=mode,
            ).validate_for_spiking_mode(spiking_mode)

    @pytest.mark.parametrize("spiking_mode", _MODES)
    def test_spike_generation_legality_excludes_the_rejected_encoder(self, spiking_mode):
        """The rate encoder refuses 'TTFS' and the TTFS modes demand it."""
        legal = legal_spike_generation_modes(spiking_mode)
        assert set(legal) <= SPIKE_MODES
        if spiking_mode == "lif":
            assert "TTFS" not in legal
        else:
            assert legal == ("TTFS",)

    def test_legal_sets_are_subsets_of_the_registry_options(self):
        for key in LEGALITY_KEYS:
            options = set(REGISTRY[key].resolved_options() or ())
            for spiking_mode in _MODES:
                legal = REGISTRY[key].legal_values({"spiking_mode": spiking_mode})
                assert legal, (key, spiking_mode)
                assert set(legal) <= options, (key, spiking_mode, legal)

    def test_an_unknown_mode_rules_nothing_out(self):
        """Legality must always be computable; an unknown mode is reported by
        the mode's own error, never by a spuriously empty legal set."""
        for key in LEGALITY_KEYS:
            legal = REGISTRY[key].legal_values({"spiking_mode": "banana"})
            assert set(legal) == set(REGISTRY[key].resolved_options() or ())


class TestTheLegalValueSets:
    def test_ttfs_family_locks_firing_and_spike_generation(self):
        for spiking_mode in ("ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            assert legal_firing_modes(spiking_mode) == ("TTFS",)
            assert legal_spike_generation_modes(spiking_mode) == ("TTFS",)

    def test_lif_leaves_firing_and_spike_generation_overridable(self):
        assert legal_firing_modes("lif") == ("Default", "Novena")
        assert legal_spike_generation_modes("lif") == (
            "Uniform", "Deterministic", "Stochastic",
        )

    def test_thresholding_is_never_locked(self):
        for spiking_mode in _MODES:
            assert len(legal_thresholding_modes(spiking_mode)) == 2

    def test_s_allocation_is_locked_to_the_one_wired_mode(self):
        """Only 'uniform' is wired; the reserved modes are loud-rejected. The
        law therefore LOCKS the field — a knob that can only error is not a knob."""
        legal = REGISTRY["s_allocation"].legal_values({})
        assert legal == ("uniform",)

    def test_the_view_serves_every_legality_bearing_key(self):
        view = legal_values_view({"spiking_mode": "ttfs"})
        assert set(view) == {
            key for key, entry in REGISTRY.items() if entry.legal_values is not None
        }
        assert view["firing_mode"] == ["TTFS"]
        assert view["thresholding_mode"] == ["<", "<="]

    def test_locked_keys_are_exactly_the_singleton_legal_sets(self):
        lif = legal_values_view({"spiking_mode": "lif"})
        ttfs = legal_values_view({"spiking_mode": "ttfs"})
        assert [k for k, v in lif.items() if len(v) == 1] == ["s_allocation"]
        assert sorted(k for k, v in ttfs.items() if len(v) == 1) == [
            "firing_mode", "s_allocation", "spike_generation_mode",
        ]


def _illegal_values(key: str, spiking_mode: str):
    options = REGISTRY[key].resolved_options() or ()
    legal = set(REGISTRY[key].legal_values({"spiking_mode": spiking_mode}))
    return [option for option in options if option not in legal]


class TestTheFullLegalityMatrix:
    """Every spiking_mode x every illegal firing / spike-gen / thresholding
    value yields a KEYED wizard error with a one-click remedy — not an
    exception, not a 500, not a silently corrupted run."""

    @pytest.mark.parametrize(
        "spiking_mode,key",
        list(itertools.product(_MODES, LEGALITY_KEYS)),
    )
    def test_every_illegal_value_is_a_keyed_remediable_error(self, spiking_mode, key):
        illegal = _illegal_values(key, spiking_mode)
        for value in illegal:
            document = _document(spiking_mode=spiking_mode, **{key: value})
            resolution = resolve_draft(document)  # must not raise
            rows = [e for e in resolution.errors if e["rule_id"] == "legal_value_set"]
            assert rows, f"{spiking_mode}/{key}={value}: no keyed legality error"
            assert [r["key"] for r in rows] == [key]
            assert value in rows[0]["message"]
            remedies = rows[0]["remedies"]
            assert remedies, "a legality error must prescribe a one-click remedy"
            assert any(r["action"] == "clear" and r["key"] == key for r in remedies)

    def test_the_reported_bug_is_a_keyed_error_not_a_valueerror(self):
        """spiking_mode='lif' + firing_mode='TTFS' used to raise
        ValueError out of DeploymentPlan.resolve (round-6 item 4)."""
        resolution = resolve_draft(_document(spiking_mode="lif", firing_mode="TTFS"))
        assert not resolution.ok
        row = next(e for e in resolution.errors if e["key"] == "firing_mode")
        assert row["rule_id"] == "legal_value_set"
        assert "Default" in row["message"] and "Novena" in row["message"]

    def test_a_singleton_legal_set_remedy_offers_the_only_legal_value(self):
        resolution = resolve_draft(_document(spiking_mode="ttfs", firing_mode="Novena"))
        row = next(e for e in resolution.errors if e["key"] == "firing_mode")
        assert {"action": "set", "key": "firing_mode", "value": "TTFS"} in [
            {k: r[k] for k in ("action", "key", "value")}
            for r in row["remedies"] if r["action"] == "set"
        ]

    def test_no_hypothetical_values_survive_a_legality_error(self):
        resolution = resolve_draft(_document(spiking_mode="lif", firing_mode="TTFS"))
        assert resolution.resolved == {}
        assert resolution.derived == {}

    @pytest.mark.parametrize("mode", ("explicit", "budget"))
    def test_reserved_s_allocation_modes_are_keyed_to_their_own_key(self, mode):
        resolution = resolve_draft(_document(s_allocation=mode))
        rows = [e for e in resolution.errors if e["key"] == "s_allocation"]
        assert rows and not resolution.ok

    @pytest.mark.parametrize("spiking_mode", _MODES)
    def test_every_legal_combination_still_resolves(self, spiking_mode):
        for firing, spike_gen, thresholding in itertools.product(
            legal_firing_modes(spiking_mode),
            legal_spike_generation_modes(spiking_mode),
            legal_thresholding_modes(spiking_mode),
        ):
            resolution = resolve_draft(_document(
                spiking_mode=spiking_mode, firing_mode=firing,
                spike_generation_mode=spike_gen, thresholding_mode=thresholding,
            ))
            assert resolution.errors == [], (spiking_mode, firing, spike_gen)


class TestTheProgrammaticRaiseSurvives:
    """The wizard never 500s, but the derivation still fails LOUD for a
    programmatic caller that bypasses the resolve error channel."""

    @pytest.mark.parametrize("spiking_mode,key", list(itertools.product(_MODES, LEGALITY_KEYS)))
    def test_derive_raises_on_every_illegal_value(self, spiking_mode, key):
        for value in _illegal_values(key, spiking_mode):
            dp = {"spiking_mode": spiking_mode, key: value}
            with pytest.raises(ValueError, match=key):
                derive_pipeline_runtime_parameters(dp)

    @pytest.mark.parametrize("spiking_mode", _MODES)
    def test_derive_supplies_the_mode_aware_default(self, spiking_mode):
        dp = {"spiking_mode": spiking_mode}
        derive_pipeline_runtime_parameters(dp)
        for key in LEGALITY_KEYS:
            assert dp[key] in REGISTRY[key].legal_values(dp), key
        expected_ttfs = spiking_mode != "lif"
        assert (dp["firing_mode"] == "TTFS") is expected_ttfs
        assert (dp["spike_generation_mode"] == "TTFS") is expected_ttfs
        assert dp["thresholding_mode"] == "<="

    @pytest.mark.parametrize("spiking_mode", _MODES)
    def test_an_explicit_legal_value_wins_over_the_derived_default(self, spiking_mode):
        legal = legal_thresholding_modes(spiking_mode)
        dp = {"spiking_mode": spiking_mode, "thresholding_mode": legal[0]}
        derive_pipeline_runtime_parameters(dp)
        assert dp["thresholding_mode"] == legal[0]


class TestLegalityErrorsAreGeneric:
    def test_legality_errors_reads_the_registry_not_a_ladder(self):
        """Only DECLARED keys are checked: a derived value is legal by
        construction, so an absent key never errors."""
        assert legality_errors({"spiking_mode": "ttfs"}, declared={}) == []
        rows = legality_errors(
            {"spiking_mode": "ttfs", "firing_mode": "Default"},
            declared={"firing_mode": "Default"},
        )
        assert [r["key"] for r in rows] == ["firing_mode"]

    def test_an_unknown_value_outside_the_options_is_also_caught(self):
        rows = legality_errors(
            {"spiking_mode": "lif", "firing_mode": "Banana"},
            declared={"firing_mode": "Banana"},
        )
        assert rows and rows[0]["rule_id"] == "legal_value_set"
