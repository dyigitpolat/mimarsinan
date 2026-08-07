"""The family×variant taxonomy: vocabulary, legacy bridge, resolution, fold."""

import pytest

from mimarsinan.chip_simulation.activation_semantics import (
    ALL_SPIKING_VARIANTS,
    is_streamed_lif,
    ActivationSemantics,
    LIF_VARIANTS,
    LIF_VARIANTS_DESIGNED,
    RETIRED_SPIKING_KEYS,
    SPIKING_FAMILIES,
    TTFS_VARIANTS,
    axes_from_legacy,
    canonical_mode_id,
    derived_spiking_variant,
    effective_legacy_spiking_mode,
    fold_spiking_axes,
    legal_spiking_families,
    legal_spiking_variants,
    require_known_spiking_axes,
    resolve_activation_semantics,
)

LEGAL_POINTS = (
    ("lif", "synchronized", "lif_sync", "lif", "cascaded"),
    ("ttfs", "analytical", "ttfs", "ttfs", "cascaded"),
    ("ttfs", "quantized", "ttfs_quantized", "ttfs_quantized", "cascaded"),
    ("ttfs", "synchronized", "ttfs_sync", "ttfs_cycle_based", "synchronized"),
    ("ttfs", "cascaded", "ttfs_cascaded", "ttfs_cycle_based", "cascaded"),
)


class TestVocabulary:
    def test_families_and_variants(self):
        assert SPIKING_FAMILIES == ("lif", "ttfs")
        assert set(LIF_VARIANTS_DESIGNED) == {"streamed", "synchronized"}
        assert set(TTFS_VARIANTS) == {
            "analytical", "quantized", "synchronized", "cascaded",
        }
        assert set(ALL_SPIKING_VARIANTS) == (
            set(LIF_VARIANTS_DESIGNED) | set(TTFS_VARIANTS)
        )

    def test_legal_sets_follow_the_family(self):
        assert legal_spiking_families({}) == SPIKING_FAMILIES
        assert legal_spiking_variants({"spiking_family": "lif"}) == LIF_VARIANTS
        assert legal_spiking_variants({"spiking_family": "ttfs"}) == TTFS_VARIANTS
        assert legal_spiking_variants({}) == LIF_VARIANTS  # family defaults lif
        # legacy dicts resolve their family through the bridge
        assert legal_spiking_variants({"spiking_mode": "ttfs"}) == TTFS_VARIANTS
        assert derived_spiking_variant({"spiking_mode": "ttfs_quantized"}) == (
            "quantized"
        )

    def test_streamed_is_a_legal_lif_variant(self):
        assert "streamed" in ALL_SPIKING_VARIANTS
        assert LIF_VARIANTS == ("streamed", "synchronized")
        assert require_known_spiking_axes("lif", "streamed") == ("lif", "streamed")
        assert is_streamed_lif(
            {"spiking_family": "lif", "spiking_variant": "streamed"}
        )
        assert is_streamed_lif({"spiking_family": "lif"})  # [P4] default
        assert not is_streamed_lif({"spiking_mode": "lif"})
        assert not is_streamed_lif({"spiking_family": "banana"})

    def test_unknown_family_rules_nothing_out(self):
        assert legal_spiking_variants({"spiking_family": "banana"}) == (
            ALL_SPIKING_VARIANTS
        )

    def test_derived_variant_per_family(self):
        # [P4] streamed is the event-driven default LIF discipline.
        assert derived_spiking_variant({}) == "streamed"
        assert derived_spiking_variant({"spiking_family": "lif"}) == "streamed"
        assert derived_spiking_variant({"spiking_family": "ttfs"}) == "analytical"
        # legacy dicts keep their historical windowed meaning.
        assert derived_spiking_variant({"spiking_mode": "lif"}) == "synchronized"

    def test_retired_keys(self):
        assert set(RETIRED_SPIKING_KEYS) == {
            "spiking_mode", "ttfs_cycle_schedule",
            "lif_execution_discipline", "lif_per_hop_retiming",
        }


class TestModeIdsAndLegacyBridge:
    @pytest.mark.parametrize("family,variant,mode_id,legacy,schedule", LEGAL_POINTS)
    def test_canonical_ids_and_legacy_axes(
        self, family, variant, mode_id, legacy, schedule
    ):
        sem = ActivationSemantics(family, variant)
        assert sem.mode_id == mode_id
        assert sem.legacy_spiking_mode == legacy
        assert sem.legacy_ttfs_cycle_schedule == schedule

    def test_streamed_points_have_ids_and_discipline(self):
        assert ActivationSemantics("lif", "streamed").mode_id == "lif"
        assert ActivationSemantics("lif", "streamed").is_streamed
        assert ActivationSemantics("ttfs", "cascaded").is_streamed
        assert not ActivationSemantics("lif", "synchronized").is_streamed

    def test_discipline_predicates(self):
        assert ActivationSemantics("lif", "synchronized").is_windowed
        assert ActivationSemantics("ttfs", "synchronized").is_windowed
        assert ActivationSemantics("ttfs", "analytical").is_analytical
        assert ActivationSemantics("ttfs", "quantized").is_analytical
        assert not ActivationSemantics("ttfs", "cascaded").is_analytical

    @pytest.mark.parametrize("family,variant,mode_id,legacy,schedule", LEGAL_POINTS)
    def test_reverse_bridge_round_trips(
        self, family, variant, mode_id, legacy, schedule
    ):
        assert axes_from_legacy(legacy, schedule) == (family, variant)

    def test_reverse_bridge_defaults(self):
        # old lif was the WINDOWED semantics; the bridge preserves meaning.
        assert axes_from_legacy("lif") == ("lif", "synchronized")
        # ttfs_cycle_based with no schedule was the cascaded default.
        assert axes_from_legacy("ttfs_cycle_based") == ("ttfs", "cascaded")

    def test_removed_legacy_mode_raises_through_the_bridge(self):
        with pytest.raises(ValueError, match="removed"):
            axes_from_legacy("rate")


class TestResolve:
    def test_axes_only(self):
        sem = resolve_activation_semantics(
            {"spiking_family": "ttfs", "spiking_variant": "quantized"}
        )
        assert (sem.family, sem.variant) == ("ttfs", "quantized")

    def test_empty_config_defaults_to_streamed_lif(self):
        sem = resolve_activation_semantics({})
        assert (sem.family, sem.variant) == ("lif", "streamed")
        assert sem.mode_id == "lif"

    def test_legacy_only_preserves_meaning(self):
        assert canonical_mode_id({"spiking_mode": "lif"}) == "lif_sync"
        assert canonical_mode_id(
            {"spiking_mode": "ttfs_cycle_based", "ttfs_cycle_schedule": "synchronized"}
        ) == "ttfs_sync"

    def test_family_with_legacy_mode_supplies_the_variant(self):
        sem = resolve_activation_semantics(
            {"spiking_family": "ttfs", "spiking_mode": "ttfs_quantized"}
        )
        assert sem.variant == "quantized"

    def test_merged_default_family_never_masquerades_as_a_declaration(self):
        # DEFAULT_DEPLOYMENT_PARAMETERS merges spiking_family='lif' into every
        # legacy dict; with the variant absent the legacy mode WINS outright.
        sem = resolve_activation_semantics(
            {"spiking_family": "lif", "spiking_mode": "ttfs_quantized"}
        )
        assert (sem.family, sem.variant) == ("ttfs", "quantized")

    def test_family_absent_variant_derives(self):
        sem = resolve_activation_semantics({"spiking_family": "ttfs"})
        assert sem.variant == "analytical"

    def test_consistent_folded_config_resolves(self):
        sem = resolve_activation_semantics({
            "spiking_family": "ttfs", "spiking_variant": "cascaded",
            "spiking_mode": "ttfs_cycle_based", "ttfs_cycle_schedule": "cascaded",
        })
        assert sem.mode_id == "ttfs_cascaded"

    def test_mode_contradiction_raises_only_under_authored_variant(self):
        # an explicit variant marks axes-authored intent — legacy must agree.
        with pytest.raises(ValueError, match="contradiction"):
            resolve_activation_semantics({
                "spiking_family": "lif", "spiking_variant": "synchronized",
                "spiking_mode": "ttfs",
            })

    def test_schedule_contradiction_raises(self):
        with pytest.raises(ValueError, match="contradiction"):
            resolve_activation_semantics({
                "spiking_family": "ttfs", "spiking_variant": "synchronized",
                "ttfs_cycle_schedule": "cascaded",
            })

    def test_inert_schedule_never_contradicts_off_cycle(self):
        # merged historical configs carried schedule='cascaded' for every mode.
        sem = resolve_activation_semantics(
            {"spiking_family": "lif", "ttfs_cycle_schedule": "cascaded"}
        )
        assert sem.mode_id == "lif"  # streamed default; schedule stays inert

    def test_streamed_variant_resolves(self):
        sem = resolve_activation_semantics(
            {"spiking_family": "lif", "spiking_variant": "streamed"}
        )
        assert sem.mode_id == "lif" and sem.is_streamed
        assert sem.legacy_spiking_mode == "lif"

    def test_unknown_axis_values_raise(self):
        with pytest.raises(ValueError, match="unknown spiking_family"):
            resolve_activation_semantics({"spiking_family": "banana"})
        with pytest.raises(ValueError, match="not legal for"):
            resolve_activation_semantics(
                {"spiking_family": "lif", "spiking_variant": "cascaded"}
            )


class TestFold:
    def test_fold_writes_all_four_keys(self):
        dp = {"spiking_family": "ttfs", "spiking_variant": "synchronized"}
        fold_spiking_axes(dp)
        assert dp["spiking_mode"] == "ttfs_cycle_based"
        assert dp["ttfs_cycle_schedule"] == "synchronized"
        assert dp["spiking_family"] == "ttfs"
        assert dp["spiking_variant"] == "synchronized"

    def test_fold_is_idempotent(self):
        dp = {"spiking_family": "ttfs", "spiking_variant": "cascaded"}
        fold_spiking_axes(dp)
        snapshot = dict(dp)
        fold_spiking_axes(dp)
        assert dp == snapshot

    def test_fold_lifts_legacy_only_dicts(self):
        dp = {"spiking_mode": "ttfs_quantized"}
        fold_spiking_axes(dp)
        assert dp["spiking_family"] == "ttfs"
        assert dp["spiking_variant"] == "quantized"
        assert dp["spiking_mode"] == "ttfs_quantized"

    def test_fold_defaults_empty_dict(self):
        dp = {}
        fold_spiking_axes(dp)
        assert dp == {
            "spiking_family": "lif", "spiking_variant": "streamed",
            "spiking_mode": "lif", "ttfs_cycle_schedule": "cascaded",
        }


class TestEffectiveLegacyMode:
    def test_prefolded_reads_directly(self):
        assert effective_legacy_spiking_mode({"spiking_mode": "ttfs"}) == "ttfs"

    def test_raw_document_resolves(self):
        assert effective_legacy_spiking_mode({"spiking_family": "ttfs"}) == "ttfs"
        assert effective_legacy_spiking_mode({}) == "lif"

    def test_total_over_invalid_documents(self):
        # an invalid axis rules nothing out — its keyed error is the truth.
        assert effective_legacy_spiking_mode({"spiking_family": "banana"}) == "lif"
        # a contradiction doc honors the raw legacy mode (pre-P0 behavior);
        # the retired-key error owns the verdict.
        assert effective_legacy_spiking_mode(
            {"spiking_family": "lif", "spiking_mode": "ttfs"}
        ) == "ttfs"
