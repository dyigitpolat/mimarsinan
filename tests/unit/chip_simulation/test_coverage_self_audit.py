"""Frontier P1 — the SELF-AUDITING coverage instrument: collapse needs an ARTIFACT.

Locks the keystone the reviewers demanded: the coverage DENOMINATOR is a function of
each axis's SCREENING STATUS so collapse-on-a-hunch is structurally impossible.

* every ``HypervolumeAxis`` carries a ``screening_status`` in
  {SCREENED_COLLAPSED, ENUMERATED_INTERACTING, ASSERTED_UNSCREENED}; a
  SCREENED_COLLAPSED axis REQUIRES a non-empty ``screening_artifact`` — constructing
  one without an artifact RAISES;
* the collapse / denominator logic CONSUMES the status: ONLY SCREENED_COLLAPSED
  collapses to one representative; ENUMERATED_INTERACTING and ASSERTED_UNSCREENED are
  BOTH counted as interacting (enumerated) in the honest denominator — so an axis
  cannot inflate coverage without a linked artifact;
* the report ALWAYS prints the claimed sub-product SIZE next to the fraction, NEVER
  merges VALID + VALID_FLAGGED into one covered total, and carries a per-region
  ATTRIBUTION-FIDELITY field (ATTRIBUTION vs VALUE_DOMAIN_ONLY);
* the CI guard FAILS on each violation (a SCREENED_COLLAPSED axis with no artifact, a
  report headline that merges the two valid tiers, or a flag aged past threshold with
  no owner).
"""

import pytest

from mimarsinan.chip_simulation.coverage_ledger import (
    AXES,
    AttributionFidelity,
    HypervolumeAxis,
    KNOWN_CRACKED_REGIONS,
    ScreeningStatus,
    AxisKind,
    CoverageStatus,
    claimed_subproduct,
    collapse_orthogonal_axes,
    coverage_report,
    honest_claimed_subproduct,
    interacting_axes,
)
from mimarsinan.chip_simulation.coverage_ci import (
    CoverageGuardError,
    assert_axes_screening_sound,
    assert_no_merged_valid_tiers,
    assert_no_aged_unowned_flags,
    audit_coverage_instrument,
)


# --------------------------------------------------------------------------- #
# (1) screening_status — the per-axis classification + the artifact requirement.
# --------------------------------------------------------------------------- #

class TestScreeningStatus:
    def test_every_axis_has_a_screening_status(self):
        for axis in AXES:
            assert isinstance(axis.screening_status, ScreeningStatus)

    def test_collapsed_axis_requires_a_nonempty_artifact(self):
        # Constructing a SCREENED_COLLAPSED axis with NO artifact must RAISE — a
        # hunch can never collapse an axis; the artifact is the structural gate.
        with pytest.raises(ValueError):
            HypervolumeAxis(
                name="bogus",
                kind=AxisKind.ORTHOGONAL,
                values=("a", "b"),
                screening_status=ScreeningStatus.SCREENED_COLLAPSED,
                representative="a",
                screening_artifact="",
            )

    def test_collapsed_axis_with_artifact_is_allowed(self):
        axis = HypervolumeAxis(
            name="ok",
            kind=AxisKind.ORTHOGONAL,
            values=("a", "b"),
            screening_status=ScreeningStatus.SCREENED_COLLAPSED,
            representative="a",
            screening_artifact="docs/research/PROGRAM_CHECKPOINT.md offload==subsume ~1e-6",
        )
        assert axis.collapsed is True
        assert axis.screening_artifact

    def test_unscreened_and_interacting_axes_need_no_artifact(self):
        # An ENUMERATED_INTERACTING or ASSERTED_UNSCREENED axis is counted interacting,
        # so it carries no collapse claim and needs no artifact to construct.
        for status in (
            ScreeningStatus.ENUMERATED_INTERACTING,
            ScreeningStatus.ASSERTED_UNSCREENED,
        ):
            axis = HypervolumeAxis(
                name="x",
                kind=AxisKind.INTERACTING,
                values=("a", "b"),
                screening_status=status,
            )
            assert axis.collapsed is False

    def test_collapsed_property_is_derived_from_status(self):
        # ``collapsed`` is no longer an independent flag — it IS
        # ``screening_status is SCREENED_COLLAPSED``.
        for axis in AXES:
            assert axis.collapsed == (
                axis.screening_status is ScreeningStatus.SCREENED_COLLAPSED
            )


# --------------------------------------------------------------------------- #
# (3) the mandated reclassification of the real axes.
# --------------------------------------------------------------------------- #

class TestAxisReclassification:
    def _status(self, name):
        return HypervolumeAxis.get(name).screening_status

    def test_encoding_placement_is_screened_collapsed_with_artifact(self):
        axis = HypervolumeAxis.get("encoding_placement")
        assert axis.screening_status is ScreeningStatus.SCREENED_COLLAPSED
        assert axis.collapsed is True
        # The artifact is the offload==subsume ~1e-6 cheap-screen result.
        assert axis.screening_artifact
        assert "1e-6" in axis.screening_artifact or "subsume" in axis.screening_artifact.lower()

    @pytest.mark.parametrize("name", ["firing", "sync", "quantization", "S", "depth"])
    def test_firing_family_is_enumerated_interacting(self, name):
        assert self._status(name) is ScreeningStatus.ENUMERATED_INTERACTING

    @pytest.mark.parametrize("name", ["dataset", "vehicle"])
    def test_dataset_and_vehicle_are_enumerated_interacting_with_artifact(self, name):
        # dataset & vehicle PROVE interaction (the dual-axis depth×dataset law +
        # architecture-dependent onset), so they are ENUMERATED_INTERACTING and carry
        # the proving artifact.
        axis = HypervolumeAxis.get(name)
        assert axis.screening_status is ScreeningStatus.ENUMERATED_INTERACTING
        assert axis.screening_artifact

    @pytest.mark.parametrize("name", ["pruning", "regime"])
    def test_unscreened_axes_are_asserted_unscreened(self, name):
        # No screen yet → ASSERTED_UNSCREENED → counted interacting until P3. pruning &
        # regime are SEMANTIC knobs (they change the trained result), so they CANNOT
        # collapse on a fidelity artifact — they stay ASSERTED_UNSCREENED.
        assert self._status(name) is ScreeningStatus.ASSERTED_UNSCREENED

    @pytest.mark.parametrize("name", ["backend", "mapping_strategy"])
    def test_faithfulness_axes_are_screened_collapsed_with_artifact(self, name):
        # backend & mapping_strategy are FAITHFULNESS axes — different simulators /
        # packings of the SAME deployment contract — so they collapse on a measured
        # PARITY/FIDELITY artifact (faithful sims & equivalent packings agree on the
        # deployed value). The collapse is FIDELITY-ONLY: the artifact must scope it.
        axis = HypervolumeAxis.get(name)
        assert axis.screening_status is ScreeningStatus.SCREENED_COLLAPSED
        assert axis.collapsed is True
        assert axis.representative
        assert axis.screening_artifact
        scope = axis.screening_artifact.lower()
        assert "fidelity-only" in scope
        assert "cost" in scope  # cost/utilization explicitly NOT collapsed

    def test_backend_representative_is_a_known_backend(self):
        axis = HypervolumeAxis.get("backend")
        assert axis.representative in axis.values

    def test_mapping_strategy_representative_is_a_known_strategy(self):
        axis = HypervolumeAxis.get("mapping_strategy")
        assert axis.representative in axis.values


# --------------------------------------------------------------------------- #
# (2) the denominator CONSUMES the status — collapse only on SCREENED_COLLAPSED.
# --------------------------------------------------------------------------- #

class TestDenominatorConsumesStatus:
    def test_only_screened_collapsed_axes_drop_from_the_active_product(self):
        active = {a.name for a in collapse_orthogonal_axes(AXES)}
        # The SCREENED_COLLAPSED axes (encoding_placement + the two faithfulness axes
        # backend & mapping_strategy) are the ones dropped from the active product.
        for name in ("encoding_placement", "backend", "mapping_strategy"):
            assert name not in active
        # The ASSERTED_UNSCREENED SEMANTIC axes are counted interacting → they SURVIVE
        # as coordinates (they cannot collapse without a real GPU equivalence screen).
        for name in ("pruning", "regime"):
            assert name in active

    def test_interacting_axes_are_every_non_collapsed_axis(self):
        names = {a.name for a in interacting_axes()}
        collapsed = {
            a.name
            for a in AXES
            if a.screening_status is ScreeningStatus.SCREENED_COLLAPSED
        }
        all_names = {a.name for a in AXES}
        # interacting == everything that is NOT screened-collapsed.
        assert names == (all_names - collapsed)
        # encoding_placement (precedent) + the two faithfulness axes now collapse.
        assert collapsed == {"encoding_placement", "backend", "mapping_strategy"}

    def test_asserted_unscreened_axis_enlarges_the_honest_denominator(self):
        # The HEADLINE invariant: an ASSERTED_UNSCREENED SEMANTIC axis (e.g. pruning,
        # 2 values) is ENUMERATED in the honest denominator — it multiplies the claim
        # size rather than collapsing to one default. A bigger denominator = LOWER
        # honest coverage.
        pinned = honest_claimed_subproduct(vehicle=["deep_cnn"], dataset=["mnist"])
        # Pin pruning down to one value → strictly fewer cells (the unscreened semantic
        # axis was being enumerated over its whole domain).
        one_pruning = honest_claimed_subproduct(
            vehicle=["deep_cnn"], dataset=["mnist"], pruning=["dense"]
        )
        assert len(pinned) > len(one_pruning)
        n_pruning = len(HypervolumeAxis.get("pruning").values)
        assert len(pinned) == len(one_pruning) * n_pruning

    def test_collapsed_faithfulness_axis_does_not_enlarge_the_denominator(self):
        # Pinning a SCREENED_COLLAPSED faithfulness axis (backend / mapping_strategy) to
        # ALL its values does NOT change the honest denominator — it collapsed to one
        # representative, exactly like encoding_placement.
        base = honest_claimed_subproduct(vehicle=["deep_cnn"], dataset=["mnist"])
        all_backends = honest_claimed_subproduct(
            vehicle=["deep_cnn"], dataset=["mnist"],
            backend=["nevresim", "sanafe", "hcm", "lava"],
        )
        all_strategies = honest_claimed_subproduct(
            vehicle=["deep_cnn"], dataset=["mnist"],
            mapping_strategy=["packed", "identity", "neuron_split", "coalesced"],
        )
        assert len(base) == len(all_backends) == len(all_strategies)

    def test_screened_collapsed_axis_does_not_enlarge_the_denominator(self):
        # Pinning the collapsed encoding_placement to BOTH values does not change the
        # honest denominator — a screened axis collapses to one representative.
        base = honest_claimed_subproduct(vehicle=["deep_cnn"], dataset=["mnist"])
        both = honest_claimed_subproduct(
            vehicle=["deep_cnn"],
            dataset=["mnist"],
            encoding_placement=["subsume", "offload"],
        )
        assert len(base) == len(both)

    def test_honest_denominator_is_at_least_the_pinned_default_denominator(self):
        # The honest (enumerate-unscreened) denominator is never SMALLER than the
        # legacy single-default claim — collapse-on-a-hunch only ever shrank it.
        honest = honest_claimed_subproduct(vehicle=["deep_cnn"], dataset=["mnist"])
        legacy = claimed_subproduct(vehicle=["deep_cnn"], dataset=["mnist"])
        assert len(honest) >= len(legacy)


# --------------------------------------------------------------------------- #
# (4) coverage_report discipline — size, never-merge, attribution-fidelity.
# --------------------------------------------------------------------------- #

def _row(model, dataset, schedule, tier, **extra):
    row = {
        "model": model,
        "dataset": dataset,
        "schedule": schedule,
        "spiking_mode": "ttfs_cycle_based",
        "backend": "sanafe",
        "deployment_validity": tier,
    }
    row.update(extra)
    return row


class TestReportDiscipline:
    def _ledger(self):
        return [
            _row("deep_cnn", "mnist", "cascaded", "VALID_on_chip_majority"),
            _row("deep_cnn", "fmnist", "cascaded", "VALID_FLAGGED_placement"),
            _row("deep_mlp", "mnist", "cascaded", "INVALID_host_majority"),
        ]

    def test_report_carries_the_claimed_subproduct_size(self):
        claim = honest_claimed_subproduct(vehicle=["deep_cnn"], dataset=["mnist"])
        report = coverage_report(self._ledger(), claimed_subproduct=claim)
        # The size is exposed as a first-class field, not just derivable.
        assert report.claimed_cell_count == len(claim)
        assert report.claimed_subproduct_size == len(claim)

    def test_report_never_merges_valid_and_valid_flagged(self):
        report = coverage_report(self._ledger())
        # The two valid tiers are ALWAYS separate counts — there is no merged
        # "covered" headline that adds them.
        assert report.tier_counts[CoverageStatus.VALID] == 1
        assert report.tier_counts[CoverageStatus.VALID_FLAGGED] == 1
        d = report.to_dict()
        # No headline key fuses the two valid tiers into one number.
        assert "valid_total" not in d
        assert "covered_valid_total" not in d
        # to_dict keeps them separate.
        assert d["tier_counts"]["VALID"] == 1
        assert d["tier_counts"]["VALID_FLAGGED"] == 1

    def test_report_marks_known_cracked_regions_value_domain_only(self):
        # GAP-1 (coalescing+output-tiling per-neuron attribution at VGG scale; kept
        # VALUE_DOMAIN_ONLY after the Wave-2 C3 reconciliation — see
        # TestGap1ReconciliationAfterC3) and the residual Tier-1 merge are KNOWN-CRACKED
        # → their attribution fidelity is VALUE_DOMAIN_ONLY, not full ATTRIBUTION.
        report = coverage_report(self._ledger())
        fidelity = report.attribution_fidelity
        assert isinstance(fidelity, dict)
        cracked = {
            region
            for region, fid in fidelity.items()
            if fid is AttributionFidelity.VALUE_DOMAIN_ONLY
        }
        assert cracked  # at least the GAP-1 region is flagged
        joined = " ".join(cracked).lower()
        assert "gap-1" in joined or "coalesc" in joined or "neuron_split" in joined

    def test_attribution_fidelity_survives_serialization(self):
        report = coverage_report(self._ledger())
        d = report.to_dict()
        assert "attribution_fidelity" in d
        assert any(v == AttributionFidelity.VALUE_DOMAIN_ONLY.value for v in d["attribution_fidelity"].values())


# --------------------------------------------------------------------------- #
# (6b) GAP-1 reconciliation after Wave-2 C3 — the instrument must be HONEST about
# what C3 fixed (the harness reassembler keying) vs what remains (the PRODUCTION
# NF↔SCM gate is identity-mapping-only, so the coalescing+output-tiling fragment
# attribution path is NOT exercised in deployment). C3 did NOT close GAP-1 in
# production, so GAP-1 stays VALUE_DOMAIN_ONLY — but its text must be sharpened so
# it is not stale-misleading.
# --------------------------------------------------------------------------- #


class TestGap1ReconciliationAfterC3:
    def _gap1_region(self) -> str:
        gap1 = [r for r in KNOWN_CRACKED_REGIONS if "gap-1" in r.lower()]
        assert gap1, (
            "GAP-1 must remain a KNOWN-CRACKED region — Wave-2 C3 fixed only the "
            "fidelity-harness reassembler keying; the PRODUCTION NF↔SCM gate still "
            "asserts identity-mapping-only, so coalescing+output-tiling per-neuron "
            "attribution is NOT exercised in deployment"
        )
        return gap1[0]

    def test_gap1_is_not_removed_after_c3(self):
        # SHARPEN branch evidence: production nf_scm_parity._group_record_by_perceptron
        # asserts len(core_placements)==1 (identity mapping) and split_group_id is None
        # (no neuron-split fragments). It builds its OWN identity mapping
        # (build_identity_mapping_for_pipeline) and never runs on the deployed
        # coalesced/tiled mapping. So the fragment reassembly is exercised only by the
        # test-only fidelity harness (integration._split_reassembly). GAP-1 STAYS.
        assert self._gap1_region() in KNOWN_CRACKED_REGIONS

    def test_both_known_cracked_regions_remain(self):
        # The two VALUE_DOMAIN_ONLY regions are unchanged: GAP-1 + the residual Tier-1
        # merge. C3 must NOT have silently dropped either.
        joined = " ".join(KNOWN_CRACKED_REGIONS).lower()
        assert "gap-1" in joined
        assert "residual tier-1 merge" in joined
        assert len(KNOWN_CRACKED_REGIONS) == 2

    def test_gap1_remains_value_domain_only_in_the_report(self):
        report = coverage_report([])
        fidelity = report.attribution_fidelity
        gap1 = self._gap1_region()
        assert fidelity.get(gap1) is AttributionFidelity.VALUE_DOMAIN_ONLY

    def test_gap1_text_credits_what_c3_fixed_and_names_what_remains(self):
        # Honesty: the GAP-1 text must (a) credit C3's reassembler keying fix so the
        # instrument is not stale, AND (b) name the residual — the production gate is
        # identity-mapping-only, so the fragment attribution path is not exercised in
        # deployment. Without (b) the region would over-claim "fixed".
        text = self._gap1_region().lower()
        # (a) credits the C3 fix (the reassembler / joint keying / harness).
        assert "c3" in text or "reassembl" in text or "harness" in text
        # (b) names what remains uncovered in production (identity-mapping-only gate).
        assert "identity" in text or "production" in text or "deploy" in text


# --------------------------------------------------------------------------- #
# (7) flag owner + aging — minimal flag metadata + the aging check.
# --------------------------------------------------------------------------- #

class TestFlagAging:
    def _flagged_row(self, **extra):
        row = _row("vit_b", "mnist", "cascaded", "VALID_FLAGGED_unsupported_op")
        row.update(extra)
        return row

    def test_report_surfaces_flag_metadata_with_owner_and_age(self):
        # A flagged row that carries a flag owner + timestamp is surfaced with both in
        # the report's flag-metadata table.
        row = self._flagged_row(flag_owner="dyigit", flag_ts="2026-01-01")
        report = coverage_report([row], now_ts="2026-01-31")
        meta = report.flag_metadata
        assert meta  # non-empty
        entry = meta[0]
        assert entry.owner == "dyigit"
        assert entry.age_days == pytest.approx(30, abs=1)

    def test_unowned_flag_has_no_owner_and_is_aged(self):
        row = self._flagged_row(flag_ts="2026-01-01")  # no owner
        report = coverage_report([row], now_ts="2026-06-01")
        entry = report.flag_metadata[0]
        assert entry.owner is None
        assert entry.age_days > 90


# --------------------------------------------------------------------------- #
# (6) CI guard — fails on each violation.
# --------------------------------------------------------------------------- #

class TestCiGuard:
    def test_real_axes_pass_the_screening_soundness_check(self):
        # The shipped AXES are all sound — every collapse has an artifact.
        assert_axes_screening_sound(AXES)  # does not raise

    def test_guard_fails_on_collapsed_axis_without_artifact(self):
        # We cannot construct one (the dataclass raises), so the guard is tested
        # against a status-tampered axis via object.__setattr__ on a frozen instance.
        good = HypervolumeAxis(
            name="x",
            kind=AxisKind.ORTHOGONAL,
            values=("a", "b"),
            screening_status=ScreeningStatus.ENUMERATED_INTERACTING,
        )
        object.__setattr__(good, "screening_status", ScreeningStatus.SCREENED_COLLAPSED)
        object.__setattr__(good, "screening_artifact", "")
        with pytest.raises(CoverageGuardError):
            assert_axes_screening_sound([good])

    def test_guard_fails_when_a_headline_merges_valid_tiers(self):
        merged = {"covered_valid_total": 5, "tier_counts": {"VALID": 3, "VALID_FLAGGED": 2}}
        with pytest.raises(CoverageGuardError):
            assert_no_merged_valid_tiers(merged)

    def test_guard_passes_a_report_dict_that_keeps_tiers_separate(self):
        report = coverage_report(
            [_row("deep_cnn", "mnist", "cascaded", "VALID_on_chip_majority")]
        )
        assert_no_merged_valid_tiers(report.to_dict())  # does not raise

    def test_guard_fails_on_aged_flag_without_owner(self):
        row = _row("vit_b", "mnist", "cascaded", "VALID_FLAGGED_unsupported_op",
                   flag_ts="2026-01-01")  # no owner, very old
        report = coverage_report([row], now_ts="2026-06-01")
        with pytest.raises(CoverageGuardError):
            assert_no_aged_unowned_flags(report, max_age_days=90)

    def test_guard_passes_aged_flag_with_owner(self):
        row = _row("vit_b", "mnist", "cascaded", "VALID_FLAGGED_unsupported_op",
                   flag_owner="dyigit", flag_ts="2026-01-01")
        report = coverage_report([row], now_ts="2026-06-01")
        assert_no_aged_unowned_flags(report, max_age_days=90)  # owned → ok

    def test_guard_passes_fresh_unowned_flag(self):
        row = _row("vit_b", "mnist", "cascaded", "VALID_FLAGGED_unsupported_op",
                   flag_ts="2026-05-25")  # no owner but fresh
        report = coverage_report([row], now_ts="2026-06-01")
        assert_no_aged_unowned_flags(report, max_age_days=90)  # fresh → ok

    def test_audit_runs_all_guards_over_real_axes_and_a_clean_report(self):
        report = coverage_report(
            [_row("deep_cnn", "mnist", "cascaded", "VALID_on_chip_majority")]
        )
        audit_coverage_instrument(AXES, report)  # does not raise


# --------------------------------------------------------------------------- #
# The HEADLINE end-to-end: the honest denominator slashes a hunch-inflated
# coverage fraction on a deep_cnn-shaped ledger.
# --------------------------------------------------------------------------- #

class TestHonestDeepCnnFraction:
    def _deep_cnn_ledger(self):
        # deep_cnn arch_dataset rows: ttfs_cycle_based, S=4, sanafe/packed (the screened
        # defaults), mnist/fmnist/kmnist tested cascaded + synchronized; svhn/cifar10
        # not tested.
        def cnn(dataset):
            r = _row("deep_cnn", dataset, None, "VALID_on_chip_majority", S=4)
            r["cascaded_deployed_mean"] = 0.98
            r["synchronized_deployed_mean"] = 0.99
            return r
        return [cnn("mnist"), cnn("fmnist"), cnn("kmnist")]

    def test_honest_denominator_is_strictly_larger_and_fraction_lower(self):
        ledger = self._deep_cnn_ledger()
        pins = dict(
            firing=["ttfs_cycle_based"],
            vehicle=["deep_cnn"],
            dataset=["mnist", "fmnist", "kmnist", "svhn", "cifar10"],
            sync=["cascaded", "synchronized"],
        )
        legacy = coverage_report(ledger, claimed_subproduct=claimed_subproduct(**pins))
        honest = coverage_report(
            ledger, claimed_subproduct=honest_claimed_subproduct(**pins)
        )
        # The honest denominator ENUMERATES the non-collapsed unpinned axes
        # (quantization, pruning, regime — backend & mapping_strategy now COLLAPSE on a
        # fidelity artifact), so it is strictly larger and the honest fraction is
        # strictly LOWER — collapse-on-a-hunch can no longer inflate it.
        assert honest.claimed_subproduct_size > legacy.claimed_subproduct_size
        assert honest.coverage_fraction < legacy.coverage_fraction
        # The enumerated-axis blow-up factor over the legacy single-default claim: the
        # unpinned NON-COLLAPSED axes the legacy claim collapsed to one default —
        # quantization(4)×pruning(2)×regime(2) = 16. backend & mapping_strategy are now
        # SCREENED_COLLAPSED, so they no longer multiply the denominator.
        blow_up = 1
        for name in ("quantization", "pruning", "regime"):
            blow_up *= len(HypervolumeAxis.get(name).values)
        assert blow_up == 16
        assert honest.claimed_subproduct_size == legacy.claimed_subproduct_size * blow_up

    def test_honest_report_passes_the_full_self_audit(self):
        ledger = self._deep_cnn_ledger()
        honest = coverage_report(
            ledger,
            claimed_subproduct=honest_claimed_subproduct(
                firing=["ttfs_cycle_based"], vehicle=["deep_cnn"], dataset=["mnist"]
            ),
        )
        audit_coverage_instrument(AXES, honest)  # the honest report is self-audit-clean
