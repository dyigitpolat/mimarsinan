"""Defensible per-phase weight-reuse cost model with a low/nominal/high uncertainty band."""

from __future__ import annotations

from dataclasses import dataclass


__all__ = [
    "DmaCostCoefficients",
    "CoefficientBand",
    "PhaseCostBreakdown",
    "CostBand",
    "Vgg16ScheduledCost",
    "DEFAULT_COEFFICIENT_BAND",
    "phase_cost_model",
    "phase_cost_band",
    "vgg16_224_scheduled_cost",
]


_PJ_PER_BYTE_TO_MJ = 1e-9  # 1 pJ = 1e-9 mJ.


@dataclass(frozen=True)
class DmaCostCoefficients:
    """The NAMED physical coefficients of the per-phase cost model (all default 0.0 ⇒ byte-identical)."""

    e_dma_per_byte_mj: float = 0.0
    bytes_per_param: float = 0.0
    e_sync_barrier_mj: float = 0.0

    @property
    def mj_per_reprogrammed_param(self) -> float:
        """The decomposed ``mj_per_reprogram``: ``bytes_per_param · e_dma_per_byte_mj``."""
        return self.bytes_per_param * self.e_dma_per_byte_mj

    @property
    def mj_per_activation_byte(self) -> float:
        """The decomposed ``mj_per_sync`` (per-byte part): ``e_dma_per_byte_mj``."""
        return self.e_dma_per_byte_mj


@dataclass(frozen=True)
class CoefficientBand:
    """The low / nominal / high coefficient sets bracketing the uncertainty band (``low <= nominal <= high``)."""

    low: DmaCostCoefficients
    nominal: DmaCostCoefficients
    high: DmaCostCoefficients


@dataclass(frozen=True)
class PhaseCostBreakdown:
    """The per-phase cost decomposition at one coefficient set (each named contribution, auditable)."""

    reprogram_dma_mj: float
    reprogram_sync_mj: float
    reuse_dma_mj: float
    reuse_sync_mj: float

    @property
    def reprogram_mj(self) -> float:
        """The N reprogram phases' total: weight DMA + sync barriers."""
        return self.reprogram_dma_mj + self.reprogram_sync_mj

    @property
    def reuse_mj(self) -> float:
        """The M reuse phases' total: activation DMA + sync barriers."""
        return self.reuse_dma_mj + self.reuse_sync_mj

    @property
    def total_mj(self) -> float:
        """The whole-schedule energy = reprogram + reuse phases."""
        return self.reprogram_mj + self.reuse_mj

    @property
    def reprogram_fraction(self) -> float:
        """Fraction of total energy spent reprogramming (0.0 when total is 0)."""
        if self.total_mj == 0.0:
            return 0.0
        return self.reprogram_mj / self.total_mj

    @property
    def reuse_fraction(self) -> float:
        """Fraction of total energy spent in reuse phases (0.0 when total is 0)."""
        if self.total_mj == 0.0:
            return 0.0
        return self.reuse_mj / self.total_mj


@dataclass(frozen=True)
class CostBand:
    """A cost as a RANGE (low / nominal / high mJ) + the nominal's auditable breakdown."""

    low_mj: float
    nominal_mj: float
    high_mj: float
    nominal_breakdown: PhaseCostBreakdown


DEFAULT_COEFFICIENT_BAND = CoefficientBand(
    low=DmaCostCoefficients(
        e_dma_per_byte_mj=31e-9,   # HBM2 ~3.9 pJ/bit
        bytes_per_param=0.5,       # 4-bit weights
        e_sync_barrier_mj=1e-4,    # ~0.1 uJ/barrier
    ),
    nominal=DmaCostCoefficients(
        e_dma_per_byte_mj=160e-9,  # DDR3 ~20 pJ/bit / Horowitz 45nm lower end
        bytes_per_param=1.0,       # 8-bit (quantized) weights
        e_sync_barrier_mj=1e-3,    # ~1 uJ/barrier
    ),
    high=DmaCostCoefficients(
        e_dma_per_byte_mj=320e-9,  # Horowitz 2014 45nm off-chip worst case
        bytes_per_param=2.0,       # 16-bit weights
        e_sync_barrier_mj=1e-2,    # ~10 uJ/barrier
    ),
)


def phase_cost_model(
    *,
    reprogram_passes: int,
    reuse_passes: int,
    params_reloaded: int,
    activation_bytes_moved: int,
    coeffs: DmaCostCoefficients,
) -> PhaseCostBreakdown:
    """The defensible per-phase cost at one coefficient set (0.0 at zero coefficients)."""
    reprogram_dma_mj = float(params_reloaded) * coeffs.mj_per_reprogrammed_param
    reprogram_sync_mj = float(reprogram_passes) * coeffs.e_sync_barrier_mj
    reuse_dma_mj = float(activation_bytes_moved) * coeffs.mj_per_activation_byte
    reuse_sync_mj = float(reuse_passes) * coeffs.e_sync_barrier_mj
    return PhaseCostBreakdown(
        reprogram_dma_mj=reprogram_dma_mj,
        reprogram_sync_mj=reprogram_sync_mj,
        reuse_dma_mj=reuse_dma_mj,
        reuse_sync_mj=reuse_sync_mj,
    )


def phase_cost_band(
    *,
    reprogram_passes: int,
    reuse_passes: int,
    params_reloaded: int,
    activation_bytes_moved: int,
    coefficient_band: CoefficientBand = DEFAULT_COEFFICIENT_BAND,
) -> CostBand:
    """Evaluate the per-phase model across the band ⇒ a low/nominal/high cost RANGE (the model is monotone)."""
    def total(coeffs: DmaCostCoefficients) -> PhaseCostBreakdown:
        return phase_cost_model(
            reprogram_passes=reprogram_passes,
            reuse_passes=reuse_passes,
            params_reloaded=params_reloaded,
            activation_bytes_moved=activation_bytes_moved,
            coeffs=coeffs,
        )

    nominal_breakdown = total(coefficient_band.nominal)
    return CostBand(
        low_mj=total(coefficient_band.low).total_mj,
        nominal_mj=nominal_breakdown.total_mj,
        high_mj=total(coefficient_band.high).total_mj,
        nominal_breakdown=nominal_breakdown,
    )


_VGG16_REPROGRAM_PASSES = 16
_VGG16_REUSE_PASSES = 142
_VGG16_PARAMS_RELOADED = 66_600_000

_VGG16_SPATIAL_POSITIONS = 137_788

_VGG16_ACTIVATION_BYTES_MOVED = 16_000_000


@dataclass(frozen=True)
class Vgg16ScheduledCost:
    """VGG16@224 scheduled cost as a measurement-with-assumptions: schedule shape, cost band, naive baseline, stated model."""

    reprogram_passes: int
    reuse_passes: int
    params_reloaded: int
    activation_bytes_moved: int
    cost_band: CostBand
    naive_all_reprogram_mj_nominal: float
    reprogram_savings_factor: float
    stated_model: str


def vgg16_224_scheduled_cost(
    coefficient_band: CoefficientBand = DEFAULT_COEFFICIENT_BAND,
) -> Vgg16ScheduledCost:
    """The VGG16@224 scheduled-deployment cost as a low/nominal/high RANGE + naive baseline + savings factor."""
    cost_band = phase_cost_band(
        reprogram_passes=_VGG16_REPROGRAM_PASSES,
        reuse_passes=_VGG16_REUSE_PASSES,
        params_reloaded=_VGG16_PARAMS_RELOADED,
        activation_bytes_moved=_VGG16_ACTIVATION_BYTES_MOVED,
        coefficient_band=coefficient_band,
    )

    position_multiplier = _VGG16_SPATIAL_POSITIONS / _VGG16_REPROGRAM_PASSES
    naive_params_reloaded = int(_VGG16_PARAMS_RELOADED * position_multiplier)
    naive_breakdown = phase_cost_model(
        reprogram_passes=_VGG16_SPATIAL_POSITIONS,
        reuse_passes=0,
        params_reloaded=naive_params_reloaded,
        activation_bytes_moved=0,
        coeffs=coefficient_band.nominal,
    )
    naive_mj = naive_breakdown.total_mj
    savings_factor = (
        naive_mj / cost_band.nominal_mj if cost_band.nominal_mj > 0.0 else 0.0
    )

    stated_model = (
        "Per-phase weight-reuse cost model (GAP-R P2): "
        "reprogram_phase = params_reloaded·bytes_per_param·E_dma_per_byte + N·E_sync_barrier; "
        "reuse_phase = activation_bytes_moved·E_dma_per_byte + M·E_sync_barrier. "
        "Coefficient band (low/nominal/high): E_dma_per_byte = 31/160/320 pJ/byte "
        "(HBM2 / DDR3 / Horowitz-2014 45nm off-chip DRAM); bytes_per_param = 0.5/1.0/2.0 "
        "(4/8/16-bit weights); E_sync_barrier = 0.1/1/10 uJ/barrier (Loihi-style NoC "
        "flush). Weight DMA dominates; the sync-barrier term is sub-dominant. "
        "VGG16@224 schedule: N=16 reprogram + M=142 reuse phases, params_reloaded=66.6M."
    )

    return Vgg16ScheduledCost(
        reprogram_passes=_VGG16_REPROGRAM_PASSES,
        reuse_passes=_VGG16_REUSE_PASSES,
        params_reloaded=_VGG16_PARAMS_RELOADED,
        activation_bytes_moved=_VGG16_ACTIVATION_BYTES_MOVED,
        cost_band=cost_band,
        naive_all_reprogram_mj_nominal=naive_mj,
        reprogram_savings_factor=savings_factor,
        stated_model=stated_model,
    )
