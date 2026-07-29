"""[IMC][W5] The 12 literature-sourced IMC platforms, pinned + provenance-ratcheted.

Registration source of truth: the structured-elimination paper's extraction-card
pipeline, `papers/structured_elimination_aaai/research_artifacts/13_chip_geometries.json`
(curated table: the sibling `13_chip_geometries.md`). This test pins the transcribed
geometry/weight_bits/eligibility for each of the 12 named platforms and enforces the
provenance ratchet: nothing carrying PLACEHOLDER_PROVENANCE may reach a run through
the named registry.
"""

import pytest

from mimarsinan.mapping.platform.imc_platforms import (
    CLAIM_ELIGIBILITY_CLASSES,
    PLACEHOLDER_PROVENANCE,
    IMCPlatform,
    get_imc_platform,
    imc_platform_names,
    register_imc_platform,
)

# name -> (max_axons, max_neurons, count, weight_bits), single homogeneous core
# type per platform. Values transcribed EXACTLY from 13_chip_geometries.json —
# never round or invent a number here; if a platform's geometry changes, the
# JSON extraction card changes first and this pin follows it.
EXPECTED_GEOMETRY = {
    "truenorth_like": (256, 256, 4096, 1),
    "neurram_like": (256, 256, 48, 4),
    "hermes_core_like": (256, 256, 1, 8),
    "isaac_like": (128, 128, 16128, 16),
    "meng_subarray_144x32": (144, 32, 4096, 4),
    "xformer_pe_like": (128, 128, 1728, 8),
    "admm_crossbar_128x64": (128, 64, 2048, 8),
    "recom_64x64_256": (64, 64, 256, 8),
    "prime_mat_256x256": (256, 256, 128, 4),
    "ibm_analog_ai_512x512x34": (512, 512, 34, 8),
    "loihi_dense_equiv_128x1024": (128, 1024, 128, 8),
    "dynap_se_64x256x4": (64, 256, 4, 2),
}

# Claim-eligibility class per platform, from `13_chip_geometries.md` §"S2 /
# allocation-claim eligibility" (charter gate G5, red-team VR#4/MS#2).
EXPECTED_ELIGIBILITY = {
    "truenorth_like": "headline-eligible",
    "neurram_like": "headline-eligible",
    "recom_64x64_256": "headline-eligible",
    "ibm_analog_ai_512x512x34": "headline-eligible",
    "isaac_like": "curve-only",
    "xformer_pe_like": "curve-only",
    "prime_mat_256x256": "curve-only",
    "loihi_dense_equiv_128x1024": "curve-only",
    "dynap_se_64x256x4": "curve-only",
    "meng_subarray_144x32": "occupancy-only",
    "admm_crossbar_128x64": "occupancy-only",
    "hermes_core_like": "quarantined",
}


class TestTwelveLiteraturePlatformsArePinned:
    @pytest.mark.parametrize("name", sorted(EXPECTED_GEOMETRY))
    def test_geometry_and_weight_bits_match_the_extraction_card(self, name):
        platform = get_imc_platform(name)
        assert len(platform.cores) == 1, (
            f"{name}: expected a single homogeneous core type")
        core = platform.cores[0]
        actual = (
            int(core["max_axons"]), int(core["max_neurons"]),
            int(core["count"]), int(platform.weight_bits),
        )
        assert actual == EXPECTED_GEOMETRY[name]

    @pytest.mark.parametrize("name", sorted(EXPECTED_GEOMETRY))
    def test_provenance_is_non_empty_and_non_placeholder(self, name):
        provenance = get_imc_platform(name).provenance
        assert provenance.strip()
        assert provenance != PLACEHOLDER_PROVENANCE

    def test_all_twelve_names_are_covered(self):
        assert set(EXPECTED_GEOMETRY) == set(EXPECTED_ELIGIBILITY)
        assert len(EXPECTED_GEOMETRY) == 12
        assert set(EXPECTED_GEOMETRY) <= set(imc_platform_names())


class TestClaimEligibilityMetadata:
    @pytest.mark.parametrize("name", sorted(EXPECTED_ELIGIBILITY))
    def test_eligibility_class_matches_the_curated_table(self, name):
        assert get_imc_platform(name).claim_eligibility == EXPECTED_ELIGIBILITY[name]

    def test_eligibility_is_always_one_of_the_declared_classes(self):
        for name in imc_platform_names():
            assert get_imc_platform(name).claim_eligibility in CLAIM_ELIGIBILITY_CLASSES

    def test_curve_only_pi_condition_platforms_are_never_headline_eligible(self):
        # loihi/dynap geometry is PI-conditioned modeling (dense-equivalent /
        # CAM abstraction over a non-crossbar substrate) — this must never
        # silently become a headline claim.
        for name in ("loihi_dense_equiv_128x1024", "dynap_se_64x256x4"):
            assert get_imc_platform(name).claim_eligibility != "headline-eligible"

    def test_hermes_is_quarantined_not_headline_eligible(self):
        # Single-core (demo used 2): degenerate population, excluded from
        # S-metric / multi-core allocation aggregates by charter gate G5.
        assert get_imc_platform("hermes_core_like").claim_eligibility == "quarantined"


class TestProvenanceRatchet:
    """Nothing carrying PLACEHOLDER_PROVENANCE may reach a run via the registry."""

    def test_every_registered_platform_is_non_placeholder(self):
        for name in imc_platform_names():
            assert get_imc_platform(name).provenance != PLACEHOLDER_PROVENANCE, (
                f"{name} still carries PLACEHOLDER_PROVENANCE")

    def test_preexisting_synthetic_platforms_declare_themselves_synthetic(self):
        # These are capability-exercise fixtures, not chips — they must say so
        # explicitly rather than silently inheriting the placeholder string.
        for name in ("imc_128x128", "imc_256x256", "imc_512x512", "imc_mixed_tile"):
            platform = get_imc_platform(name)
            assert platform.claim_eligibility == "synthetic"
            assert "not a real chip" in platform.provenance
            assert platform.provenance != PLACEHOLDER_PROVENANCE

    def test_registry_retrieval_refuses_placeholder_provenance(self):
        # Ad-hoc programmatic IMCPlatform() construction (as other tests in
        # this suite do, e.g. provenance="placeholder") stays completely
        # unaffected — only registration + retrieval through the named
        # registry is ratcheted.
        bad = IMCPlatform(
            name="_ratchet_test_placeholder_platform",
            cores=({"max_axons": 8, "max_neurons": 8, "count": 1},),
            weight_bits=4,
            provenance=PLACEHOLDER_PROVENANCE,
        )
        register_imc_platform(bad)
        try:
            with pytest.raises(ValueError, match="PLACEHOLDER_PROVENANCE"):
                get_imc_platform("_ratchet_test_placeholder_platform")
        finally:
            from mimarsinan.mapping.platform import imc_platforms as _mod
            _mod._REGISTRY.pop("_ratchet_test_placeholder_platform", None)

    def test_validate_for_run_accepts_a_real_provenance_string(self):
        good = IMCPlatform(
            name="_ratchet_test_good_platform",
            cores=({"max_axons": 8, "max_neurons": 8, "count": 1},),
            weight_bits=4,
            provenance="author2020fake: \"a made-up quote for this test\" (p.1)",
        )
        assert good.validate_for_run() is good

    def test_ad_hoc_platform_construction_is_unaffected_by_the_ratchet(self):
        # .validate() (used directly by other tests in this suite) never
        # checks provenance — only .validate_for_run() (used by the registry
        # retrieval path) does.
        placeholder = IMCPlatform(
            name="demo", cores=({"max_axons": 8, "max_neurons": 8, "count": 1},),
            weight_bits=4, provenance="placeholder",
        )
        assert placeholder.validate() is placeholder
