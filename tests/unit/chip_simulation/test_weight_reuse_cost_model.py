"""GAP-R P2 — the DEFENSIBLE per-phase weight-reuse cost model + uncertainty band.

The round-1 `weight_reuse_mj` term took opaque `mj_per_reprogram` / `mj_per_sync`
coefficients that default to 0 ⇒ the scheduled deployment mode had an EMPTY cost
column. This P2 model makes the cost DEFENSIBLE: it decomposes each coefficient into
NAMED physical units with a CITED range, and carries a low/nominal/high UNCERTAINTY
BAND so a cost output is a RANGE, never a false-precision point. Locks:

* the per-phase model reproduces a HAND-computed reprogram/reuse cost for a small
  schedule (reprogram = params·bytes_per_param·E_dma + E_sync; reuse = act_bytes·E_dma
  + E_sync);
* the band BRACKETS the nominal (low <= nominal <= high) on every output;
* the VGG16@224 scheduled schedule yields a stated-model nominal + band + the
  reprogram-vs-reuse split (how much cheaper weight-reuse makes it);
* the zero-coefficient path is exactly 0.0 (byte-identical to the default-off term);
* the new model agrees with the existing `weight_reuse_mj` when the opaque coefficient
  is the decomposed `bytes_per_param · e_dma_per_byte_mj`.
"""

import pytest

from mimarsinan.chip_simulation.cost_extraction import CostRecord, weight_reuse_mj
from mimarsinan.chip_simulation.certification import CertificationCell
from mimarsinan.chip_simulation.weight_reuse_cost_model import (
    DEFAULT_COEFFICIENT_BAND,
    CoefficientBand,
    CostBand,
    DmaCostCoefficients,
    PhaseCostBreakdown,
    phase_cost_band,
    phase_cost_model,
    vgg16_224_scheduled_cost,
)


# --------------------------------------------------------------------------- #
# DmaCostCoefficients — the NAMED physical decomposition (defensible units).
# --------------------------------------------------------------------------- #

class TestDmaCostCoefficients:
    def test_zero_coefficients_is_byte_identical(self):
        # The default-off guarantee: zero coefficients ⇒ every phase costs 0.0,
        # exactly the legacy `weight_reuse_mj` default behavior.
        coeffs = DmaCostCoefficients(
            e_dma_per_byte_mj=0.0, bytes_per_param=0.0, e_sync_barrier_mj=0.0
        )
        breakdown = phase_cost_model(
            reprogram_passes=16,
            reuse_passes=142,
            params_reloaded=66_600_000,
            activation_bytes_moved=2_000_000,
            coeffs=coeffs,
        )
        assert breakdown.total_mj == 0.0
        assert breakdown.reprogram_mj == 0.0
        assert breakdown.reuse_mj == 0.0

    def test_mj_per_reprogram_is_decomposed(self):
        # The decomposition is the whole point: the opaque `mj_per_reprogram` of the
        # legacy term = bytes_per_param · e_dma_per_byte_mj (DMA the weight bytes).
        coeffs = DmaCostCoefficients(
            e_dma_per_byte_mj=2e-9, bytes_per_param=1.0, e_sync_barrier_mj=0.0
        )
        assert coeffs.mj_per_reprogrammed_param == pytest.approx(2e-9)
        coeffs2 = DmaCostCoefficients(
            e_dma_per_byte_mj=2e-9, bytes_per_param=4.0, e_sync_barrier_mj=0.0
        )
        assert coeffs2.mj_per_reprogrammed_param == pytest.approx(8e-9)

    def test_mj_per_activation_byte_is_e_dma(self):
        coeffs = DmaCostCoefficients(
            e_dma_per_byte_mj=2e-9, bytes_per_param=1.0, e_sync_barrier_mj=0.0
        )
        assert coeffs.mj_per_activation_byte == pytest.approx(2e-9)


# --------------------------------------------------------------------------- #
# phase_cost_model — the per-phase model reproduces a HAND computation.
# --------------------------------------------------------------------------- #

class TestPhaseCostModelHandComputation:
    def _coeffs(self):
        # Simple round numbers so the hand computation is unambiguous.
        return DmaCostCoefficients(
            e_dma_per_byte_mj=1e-6,   # 1 nJ/byte (= 1e-6 mJ/byte)
            bytes_per_param=1.0,      # 1 byte/param (quantized weights)
            e_sync_barrier_mj=1e-3,   # 1 uJ/barrier
        )

    def test_reprogram_phase_hand_computation(self):
        # reprogram_mj = params_reloaded · bytes_per_param · E_dma + N_reprogram · E_sync
        # = 1000 · 1 · 1e-6  +  2 · 1e-3 = 1e-3 + 2e-3 = 3e-3
        breakdown = phase_cost_model(
            reprogram_passes=2,
            reuse_passes=0,
            params_reloaded=1000,
            activation_bytes_moved=0,
            coeffs=self._coeffs(),
        )
        assert breakdown.reprogram_dma_mj == pytest.approx(1e-3)
        assert breakdown.reprogram_sync_mj == pytest.approx(2e-3)
        assert breakdown.reprogram_mj == pytest.approx(3e-3)

    def test_reuse_phase_hand_computation(self):
        # reuse_mj = activation_bytes_moved · E_dma + M_reuse · E_sync
        # = 500 · 1e-6  +  3 · 1e-3 = 5e-4 + 3e-3 = 3.5e-3
        breakdown = phase_cost_model(
            reprogram_passes=0,
            reuse_passes=3,
            params_reloaded=0,
            activation_bytes_moved=500,
            coeffs=self._coeffs(),
        )
        assert breakdown.reuse_dma_mj == pytest.approx(5e-4)
        assert breakdown.reuse_sync_mj == pytest.approx(3e-3)
        assert breakdown.reuse_mj == pytest.approx(3.5e-3)

    def test_total_is_reprogram_plus_reuse(self):
        breakdown = phase_cost_model(
            reprogram_passes=2,
            reuse_passes=3,
            params_reloaded=1000,
            activation_bytes_moved=500,
            coeffs=self._coeffs(),
        )
        # reprogram: 1000·1e-6 + 2·1e-3 = 3e-3 ; reuse: 500·1e-6 + 3·1e-3 = 3.5e-3
        assert breakdown.total_mj == pytest.approx(3e-3 + 3.5e-3)

    def test_reprogram_vs_reuse_split_reported(self):
        breakdown = phase_cost_model(
            reprogram_passes=2,
            reuse_passes=3,
            params_reloaded=1000,
            activation_bytes_moved=500,
            coeffs=self._coeffs(),
        )
        assert breakdown.reprogram_fraction == pytest.approx(
            3e-3 / (3e-3 + 3.5e-3)
        )
        assert breakdown.reuse_fraction == pytest.approx(
            3.5e-3 / (3e-3 + 3.5e-3)
        )

    def test_split_zero_total_is_well_defined(self):
        # Zero coefficients ⇒ zero total ⇒ a 0.0 split (never a divide-by-zero).
        breakdown = phase_cost_model(
            reprogram_passes=2,
            reuse_passes=3,
            params_reloaded=1000,
            activation_bytes_moved=500,
            coeffs=DmaCostCoefficients(0.0, 0.0, 0.0),
        )
        assert breakdown.total_mj == 0.0
        assert breakdown.reprogram_fraction == 0.0
        assert breakdown.reuse_fraction == 0.0


# --------------------------------------------------------------------------- #
# Cross-check against the legacy `weight_reuse_mj` term — the DMA part must agree.
# --------------------------------------------------------------------------- #

class TestAgreesWithLegacyTerm:
    def test_dma_part_matches_weight_reuse_mj(self):
        # With NO sync-barrier charge, the model's DMA total must equal the legacy
        # `weight_reuse_mj` whose `mj_per_reprogram` is the decomposed product.
        coeffs = DmaCostCoefficients(
            e_dma_per_byte_mj=1.6e-7, bytes_per_param=1.0, e_sync_barrier_mj=0.0
        )
        breakdown = phase_cost_model(
            reprogram_passes=16,
            reuse_passes=142,
            params_reloaded=66_600_000,
            activation_bytes_moved=3_000_000,
            coeffs=coeffs,
        )
        legacy = weight_reuse_mj(
            params_reloaded=66_600_000,
            activation_bytes_moved=3_000_000,
            mj_per_reprogram=coeffs.mj_per_reprogrammed_param,
            mj_per_sync=coeffs.mj_per_activation_byte,
        )
        assert breakdown.total_mj == pytest.approx(legacy)

    def test_record_reuse_mj_feeds_the_decomposed_coefficient(self):
        # A CostRecord's `reuse_mj` consumes the SAME decomposed coefficient — so the
        # record-level term and the model agree when sync-barrier charge is 0.
        cell = CertificationCell("lif", None, "sanafe")
        record = CostRecord(
            cell_key=cell.cell_key, mode="lif", backend="sanafe",
            acc_deploy=0.97, mj_per_sample=0.04, spikes=0, latency_steps=0,
            cores=0, s_global=4, depth=1,
            reprogram_passes=16, reuse_passes=142,
            params_reloaded=66_600_000, activation_bytes_moved=3_000_000,
        )
        coeffs = DmaCostCoefficients(
            e_dma_per_byte_mj=1.6e-7, bytes_per_param=1.0, e_sync_barrier_mj=0.0
        )
        model_total = phase_cost_model(
            reprogram_passes=record.reprogram_passes,
            reuse_passes=record.reuse_passes,
            params_reloaded=record.params_reloaded,
            activation_bytes_moved=record.activation_bytes_moved,
            coeffs=coeffs,
        ).total_mj
        record_total = record.reuse_mj(
            mj_per_reprogram=coeffs.mj_per_reprogrammed_param,
            mj_per_sync=coeffs.mj_per_activation_byte,
        )
        assert model_total == pytest.approx(record_total)


# --------------------------------------------------------------------------- #
# The uncertainty band — low <= nominal <= high (a RANGE, not a point).
# --------------------------------------------------------------------------- #

class TestCoefficientBand:
    def test_default_band_is_ordered_per_unit(self):
        band = DEFAULT_COEFFICIENT_BAND
        # Each NAMED coefficient is bracketed low <= nominal <= high.
        assert band.low.e_dma_per_byte_mj <= band.nominal.e_dma_per_byte_mj
        assert band.nominal.e_dma_per_byte_mj <= band.high.e_dma_per_byte_mj
        assert band.low.e_sync_barrier_mj <= band.nominal.e_sync_barrier_mj
        assert band.nominal.e_sync_barrier_mj <= band.high.e_sync_barrier_mj

    def test_default_band_coefficients_are_positive(self):
        # A defensible band is strictly positive (an empty/zero column is the thing
        # this fixes); the byte-identical zero stays available as an explicit override.
        band = DEFAULT_COEFFICIENT_BAND
        assert band.nominal.e_dma_per_byte_mj > 0.0
        assert band.nominal.bytes_per_param > 0.0
        assert band.nominal.e_sync_barrier_mj > 0.0

    def test_default_band_dma_per_byte_in_cited_range(self):
        # The cited DMA-energy-per-byte literature band: HBM2 ~31 pJ/byte (low) to
        # Horowitz-2014 45nm off-chip DRAM worst case ~320 pJ/byte (high), nominal at
        # the DDR3 / Horowitz lower-end ~160 pJ/byte. (1 pJ = 1e-9 mJ.)
        band = DEFAULT_COEFFICIENT_BAND
        assert band.low.e_dma_per_byte_mj == pytest.approx(31e-9, rel=0.2)
        assert band.nominal.e_dma_per_byte_mj == pytest.approx(160e-9, rel=0.2)
        assert band.high.e_dma_per_byte_mj == pytest.approx(320e-9, rel=0.2)


class TestPhaseCostBand:
    def test_band_brackets_nominal(self):
        band = phase_cost_band(
            reprogram_passes=16,
            reuse_passes=142,
            params_reloaded=66_600_000,
            activation_bytes_moved=3_000_000,
            coefficient_band=DEFAULT_COEFFICIENT_BAND,
        )
        assert band.low_mj <= band.nominal_mj <= band.high_mj

    def test_band_low_uses_low_coefficients(self):
        # The band's low endpoint IS the model evaluated at the low coefficient set.
        band = phase_cost_band(
            reprogram_passes=16,
            reuse_passes=142,
            params_reloaded=66_600_000,
            activation_bytes_moved=3_000_000,
            coefficient_band=DEFAULT_COEFFICIENT_BAND,
        )
        expected_low = phase_cost_model(
            reprogram_passes=16,
            reuse_passes=142,
            params_reloaded=66_600_000,
            activation_bytes_moved=3_000_000,
            coeffs=DEFAULT_COEFFICIENT_BAND.low,
        ).total_mj
        assert band.low_mj == pytest.approx(expected_low)

    def test_band_carries_nominal_breakdown(self):
        band = phase_cost_band(
            reprogram_passes=16,
            reuse_passes=142,
            params_reloaded=66_600_000,
            activation_bytes_moved=3_000_000,
            coefficient_band=DEFAULT_COEFFICIENT_BAND,
        )
        assert isinstance(band.nominal_breakdown, PhaseCostBreakdown)
        assert band.nominal_mj == pytest.approx(band.nominal_breakdown.total_mj)

    def test_band_zero_when_all_coefficients_zero(self):
        zero = DmaCostCoefficients(0.0, 0.0, 0.0)
        zero_band = CoefficientBand(low=zero, nominal=zero, high=zero)
        band = phase_cost_band(
            reprogram_passes=16,
            reuse_passes=142,
            params_reloaded=66_600_000,
            activation_bytes_moved=3_000_000,
            coefficient_band=zero_band,
        )
        assert band.low_mj == 0.0
        assert band.nominal_mj == 0.0
        assert band.high_mj == 0.0


# --------------------------------------------------------------------------- #
# VGG16@224 SCHEDULED cost — the headline measurement-with-assumptions.
# --------------------------------------------------------------------------- #

class TestVgg16_224ScheduledCost:
    def test_schedule_shape(self):
        result = vgg16_224_scheduled_cost()
        # N ≈ 16 reprogram (13 conv + 3 FC) + M ≈ 142 reuse phases.
        assert result.reprogram_passes == 16
        assert result.reuse_passes == 142
        assert result.params_reloaded == pytest.approx(66.6e6, rel=0.05)

    def test_yields_band_bracketing_nominal(self):
        result = vgg16_224_scheduled_cost()
        band = result.cost_band
        assert band.low_mj <= band.nominal_mj <= band.high_mj
        # A defensible band is a genuine RANGE — high strictly above low.
        assert band.high_mj > band.low_mj

    def test_nominal_is_dominated_by_weight_reload(self):
        # The stated-model claim: the reprogram (weight-reload) term dominates — that
        # is WHY weight-reuse (M reuse phases that reload nothing) is the lever.
        result = vgg16_224_scheduled_cost()
        breakdown = result.cost_band.nominal_breakdown
        assert breakdown.reprogram_mj > breakdown.reuse_mj
        assert breakdown.reprogram_fraction > 0.5

    def test_reports_naive_reprogram_every_pass_baseline(self):
        # The split's VALUE: if every one of the 158 passes reprogrammed its full
        # resident bank (the pre-reuse model), params reloaded would balloon by the
        # reuse multiplier. The result reports the naive baseline so the savings are
        # quantified, and weight-reuse must be strictly cheaper.
        result = vgg16_224_scheduled_cost()
        assert result.naive_all_reprogram_mj_nominal > result.cost_band.nominal_mj

    def test_savings_factor_is_large(self):
        # Weight-reuse collapses ~137,775 position-reprograms to 16 ⇒ a large
        # reprogram-reload saving. The reported savings factor must exceed ~10x.
        result = vgg16_224_scheduled_cost()
        assert result.reprogram_savings_factor > 10.0

    def test_stated_model_string_names_assumptions(self):
        # The number is only useful WITH its stated model — the description must name
        # the per-phase decomposition and the cited coefficient sources.
        result = vgg16_224_scheduled_cost()
        text = result.stated_model.lower()
        assert "dma" in text
        assert "sync" in text
        assert "horowitz" in text or "hbm" in text or "ddr" in text
