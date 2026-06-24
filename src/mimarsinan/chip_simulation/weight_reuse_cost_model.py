"""GAP-R P2 — the DEFENSIBLE per-phase weight-reuse cost model + uncertainty band.

The round-1 weight-reuse term (`cost_extraction.weight_reuse_mj`) charged opaque
`mj_per_reprogram` / `mj_per_sync` coefficients that DEFAULT TO 0 ⇒ the scheduled
deployment mode has an EMPTY cost column. An empty column is honest-but-useless; an
indefensible non-zero coefficient is WORSE (authoritative-looking wrong numbers
propagate into the E5 Pareto and every cross-cell comparison). This module makes the
cost DEFENSIBLE rather than precise: it decomposes the opaque coefficients into NAMED
physical units, each with a CITED literature range, and carries a low/nominal/high
UNCERTAINTY BAND so a cost output is a RANGE, not a false-precision point.

The per-phase model (named assumptions)
---------------------------------------
A scheduled deployment = N reprogram phases + M reuse phases (see
`mapping/weight_reuse.py` and `docs/research/WEIGHT_REUSE_SCHEDULING_DESIGN.md`). The
ONLY energy a reprogram phase costs over a reuse phase is the WEIGHT DMA onto the
cores; both pay a per-phase sync-barrier flush:

    reprogram_phase_mj = params_reloaded · bytes_per_param · E_dma_per_byte
                       + N_reprogram     · E_sync_barrier
    reuse_phase_mj     = activation_bytes_moved · E_dma_per_byte
                       + M_reuse              · E_sync_barrier

This is the SAME term as `weight_reuse_mj` with its opaque `mj_per_reprogram` made
explicit (`= bytes_per_param · E_dma_per_byte`) PLUS a per-phase sync-barrier charge
the legacy term folded into nothing. At zero coefficients the whole model is 0.0 ⇒
byte-identical to the default-off term.

The coefficients (defensible sources / ranges — NOT fabricated precision)
-------------------------------------------------------------------------
* **E_dma_per_byte** — energy to move one weight/activation byte from off-core memory
  onto a core. Grounded in the off-chip DRAM access literature, expressed as a RANGE,
  not a point:
    - LOW   ~31 pJ/byte — HBM2 (~3.9 pJ/bit); the most efficient stacked DRAM.
    - NOM   ~160 pJ/byte — DDR3 (~20 pJ/bit); also the lower end of Horowitz 2014's
      45 nm off-chip 64-bit DRAM access (1300 pJ / 8 B ≈ 162 pJ/byte).
    - HIGH  ~320 pJ/byte — Horowitz 2014's 45 nm worst-case off-chip 64-bit DRAM
      access (2600 pJ / 8 B ≈ 325 pJ/byte).
  Sources: M. Horowitz, "Computing's energy problem (and what we can do about it),"
  ISSCC 2014 (the canonical 45 nm energy table); standard DDR3 / HBM2 per-bit access
  energy figures. 1 pJ = 1e-9 mJ.
* **bytes_per_param** — bytes DMA'd per resident weight. Nominal 1.0 (the chip stores
  quantized 8-bit weights); the band spans 0.5 (4-bit) to 2.0 (16-bit) to bracket
  weight precision, but the nominal model is 1 byte/param.
* **E_sync_barrier** — energy of one inter-phase activation-gather barrier (a NoC
  flush + time-step-advance broadcast across the resident cores, the Loihi barrier
  mechanism). Bounded by ~(active cores) × (per-hop packet energy, 1–100 pJ); modeled
  as a flat per-barrier energy with a wide band (~0.1–10 µJ, nominal ~1 µJ). It is the
  SUB-DOMINANT term — the headline cost is weight DMA — so its wide relative
  uncertainty does not swing the verdict; the band carries it honestly regardless.

This module runs nothing and changes no sim behavior. It is consumed opt-in: a caller
that wants a cost number passes a `CoefficientBand` (or the cited `DEFAULT_COEFFICIENT_
BAND`); the default-off `weight_reuse_mj` path is untouched and stays byte-identical.
"""

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
    """The NAMED physical coefficients of the per-phase cost model.

    Each is a defensible physical quantity, NOT a fitted free parameter:
    ``e_dma_per_byte_mj`` (off-core DMA energy per byte), ``bytes_per_param`` (weight
    bytes DMA'd per resident param), ``e_sync_barrier_mj`` (per-barrier NoC-flush
    energy). All default 0.0 ⇒ the model is 0.0 ⇒ byte-identical to the default-off
    ``weight_reuse_mj`` term.
    """

    e_dma_per_byte_mj: float = 0.0
    bytes_per_param: float = 0.0
    e_sync_barrier_mj: float = 0.0

    @property
    def mj_per_reprogrammed_param(self) -> float:
        """The decomposed ``mj_per_reprogram``: ``bytes_per_param · e_dma_per_byte_mj``.

        This is exactly the opaque coefficient the legacy ``weight_reuse_mj`` took,
        made explicit as "DMA the weight bytes of one reloaded param".
        """
        return self.bytes_per_param * self.e_dma_per_byte_mj

    @property
    def mj_per_activation_byte(self) -> float:
        """The decomposed ``mj_per_sync`` (per-byte part): ``e_dma_per_byte_mj``.

        A reuse phase moves activation BYTES at the sync barrier (the same DMA energy
        per byte); the per-barrier fixed flush is the separate ``e_sync_barrier_mj``.
        """
        return self.e_dma_per_byte_mj


@dataclass(frozen=True)
class CoefficientBand:
    """The low / nominal / high coefficient sets — the uncertainty band's endpoints.

    The band brackets the defensible coefficient ranges so a cost output is a RANGE.
    Invariant (checked by the default band's tests): each NAMED coefficient is ordered
    ``low <= nominal <= high``.
    """

    low: DmaCostCoefficients
    nominal: DmaCostCoefficients
    high: DmaCostCoefficients


@dataclass(frozen=True)
class PhaseCostBreakdown:
    """The per-phase cost decomposition at one coefficient set (a single point).

    Carries each named contribution so the number is auditable: the reprogram phase's
    weight-DMA + sync parts, the reuse phase's activation-DMA + sync parts, the total,
    and the reprogram-vs-reuse split (the headline "how much cheaper weight-reuse is").
    """

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
    """A cost as a RANGE (low / nominal / high mJ) + the nominal's auditable breakdown.

    The whole point of P2: a cost number is reported WITH its uncertainty band, never
    as a false-precision point. ``low_mj <= nominal_mj <= high_mj`` by construction
    (the model is monotone in every non-negative coefficient).
    """

    low_mj: float
    nominal_mj: float
    high_mj: float
    nominal_breakdown: PhaseCostBreakdown


# The cited default coefficient band (see module docstring for sources / ranges).
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
    """The defensible per-phase cost (one coefficient set) — see module docstring.

    ``reprogram_mj = params_reloaded · bytes_per_param · E_dma + N · E_sync``;
    ``reuse_mj = activation_bytes_moved · E_dma + M · E_sync``. At zero coefficients
    the whole breakdown is 0.0 (byte-identical to the default-off ``weight_reuse_mj``).
    """
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
    """Evaluate the per-phase model across the band ⇒ a low/nominal/high cost RANGE.

    The model is monotone non-decreasing in every (non-negative) coefficient, so the
    low/nominal/high coefficient sets map straight to the low/nominal/high cost
    endpoints. The nominal's full :class:`PhaseCostBreakdown` is carried for audit.
    """
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


# --------------------------------------------------------------------------- #
# VGG16@224 scheduled cost — the headline measurement-with-assumptions.
# --------------------------------------------------------------------------- #

# VGG16: 13 conv + 3 FC weight-distinct banks ⇒ N = 16 reprogram phases. Over the
# full 224²-spatial feature-map positions the positions are batched into core-budget
# passes ⇒ M ≈ 142 reuse phases (the design's headline N≈16 / M≈142 split; see
# docs/research/WEIGHT_REUSE_SCHEDULING_DESIGN.md). params_reloaded ≈ 66.6M is the Σ
# over the 16 distinct banks of the resident weight count (each bank counted ONCE, not
# once per reused spatial position).
_VGG16_REPROGRAM_PASSES = 16
_VGG16_REUSE_PASSES = 142
_VGG16_PARAMS_RELOADED = 66_600_000

# The irreducible spatial-unroll position count for the VGG16@224 conv stack (E4 doc:
# ~137,800 softcores / output positions). In the PRE-reuse model every one of these
# positions reprograms its full resident bank; weight-reuse collapses them to the 16
# distinct banks. The naive baseline reloads the resident weights once PER reuse pass
# instead of once per bank, so its params-reloaded balloons by the position multiplier.
_VGG16_SPATIAL_POSITIONS = 137_788

# Activation bytes gathered at the sync barriers (Σ over total_passes-1 barriers of the
# gathered slice size). VGG16@224's largest feature maps are ~3.2 MB (224²·64·1 B); the
# per-barrier gathered slices sum to ~the whole activation working set. A defensible
# order-of-magnitude figure for the schedule (sub-dominant to the weight DMA): ~16 MB.
_VGG16_ACTIVATION_BYTES_MOVED = 16_000_000


@dataclass(frozen=True)
class Vgg16ScheduledCost:
    """VGG16@224 scheduled cost AS A MEASUREMENT-WITH-ASSUMPTIONS.

    Carries the schedule shape (N reprogram + M reuse, params reloaded), the cost as a
    RANGE (``cost_band``: low/nominal/high mJ + nominal breakdown), the naive
    "reprogram-every-pass" baseline (so the weight-reuse saving is quantified), and the
    ``stated_model`` string that NAMES the per-phase decomposition and cited sources —
    because the number is only useful WITH its model + band.
    """

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
    """The VGG16@224 scheduled-deployment cost: nominal + band + reprogram/reuse split.

    A MEASUREMENT-WITH-ASSUMPTIONS, not an optimization: applies the defensible
    per-phase model to the VGG16@224 schedule (N≈16 reprogram + M≈142 reuse,
    params_reloaded≈66.6M) and reports the cost as a low/nominal/high RANGE, the naive
    reprogram-every-pass baseline, and the savings factor weight-reuse buys.
    """
    cost_band = phase_cost_band(
        reprogram_passes=_VGG16_REPROGRAM_PASSES,
        reuse_passes=_VGG16_REUSE_PASSES,
        params_reloaded=_VGG16_PARAMS_RELOADED,
        activation_bytes_moved=_VGG16_ACTIVATION_BYTES_MOVED,
        coefficient_band=coefficient_band,
    )

    # The naive baseline: every spatial position reprograms its full resident bank, so
    # the weight DMA scales by (positions / distinct banks) — i.e. the resident banks
    # are reloaded once per reuse pass instead of once. We charge the SAME nominal DMA
    # per byte over the ballooned params (positions/banks × the per-bank reload).
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
