# GAP-R P2 — A Defensible Per-Phase Weight-Reuse Cost Model (with uncertainty band)

## The problem this fixes

Round-1 (`WEIGHT_REUSE_SCHEDULING_DESIGN.md`) landed the reprogram-vs-reuse phase
classifier and the cost fields (`reprogram_passes` / `reuse_passes` / `params_reloaded`
/ `activation_bytes_moved`) plus the `weight_reuse_mj(mj_per_reprogram, mj_per_sync)`
term — but the chip coefficients **default to 0**, so the scheduled deployment mode has
an **empty cost column**. An empty column is honest-but-useless. An *indefensible*
non-zero coefficient is **worse**: an authoritative-looking wrong number propagates into
the E5 Pareto and every cross-cell comparison.

The reviewers' bar: a cost number is only useful **with** a stated, defensible per-phase
model **and** an uncertainty band. The goal here is **honest measurement** (the coverage
frame), not optimization.

## The per-phase model (named assumptions)

A scheduled deployment = **N reprogram phases** + **M reuse phases**. The *only* energy a
reprogram phase costs over a reuse phase is the **weight DMA** onto the cores; both pay a
per-phase **sync-barrier flush**:

```
reprogram_phase_mj = params_reloaded · bytes_per_param · E_dma_per_byte   (weight DMA)
                   + N_reprogram     · E_sync_barrier                     (barriers)
reuse_phase_mj     = activation_bytes_moved · E_dma_per_byte              (activation DMA)
                   + M_reuse              · E_sync_barrier                (barriers)
```

This is the *same* term as `weight_reuse_mj`, with its opaque `mj_per_reprogram` made
explicit (`= bytes_per_param · E_dma_per_byte`, "DMA the weight bytes of one reloaded
param") **plus** a per-phase sync-barrier charge the legacy term folded into nothing. At
zero coefficients the whole model is `0.0` ⇒ byte-identical to the default-off term.

## The coefficients — defensible sources/ranges (NOT fabricated precision)

Each coefficient is a physical quantity with a **cited literature range**, carried as a
**low / nominal / high** band so a cost output is a RANGE, not a false-precision point.

| coefficient | low | nominal | high | source / range |
|---|---|---|---|---|
| `E_dma_per_byte` | 31 pJ/byte | 160 pJ/byte | 320 pJ/byte | HBM2 (~3.9 pJ/bit) → DDR3 (~20 pJ/bit) / Horowitz-2014 45 nm off-chip 64-bit DRAM lower end (1300 pJ / 8 B ≈ 162 pJ/byte) → Horowitz-2014 45 nm worst case (2600 pJ / 8 B ≈ 325 pJ/byte) |
| `bytes_per_param` | 0.5 | 1.0 | 2.0 | 4-bit / 8-bit (quantized) / 16-bit weights |
| `E_sync_barrier` | 0.1 µJ | 1 µJ | 10 µJ | one inter-phase activation-gather barrier — a Loihi-style NoC flush + time-step-advance broadcast across resident cores, bounded by ~(active cores) × (per-hop packet energy, 1–100 pJ); modeled flat with a wide band |

Primary source: **M. Horowitz, "Computing's energy problem (and what we can do about
it)," ISSCC 2014** — the canonical 45 nm energy table (off-chip DRAM access 1300–2600 pJ
for a 64-bit word). DDR3 (~20.3 pJ/bit) and HBM2 (~3.9 pJ/bit) per-bit access figures
bracket the modern-DRAM end. The barrier-energy band is grounded in the Loihi
many-core barrier-synchronization mechanism (Davies et al., 2018) and per-synaptic-op /
per-hop NoC packet energies (1–100 pJ).

**Weight DMA dominates**; the sync-barrier term is sub-dominant, so its (deliberately
wide) relative uncertainty does not swing the verdict — but the band carries it honestly.

## VGG16@224 scheduled cost — measurement-with-assumptions

Schedule shape: **N = 16** reprogram (13 conv + 3 FC weight-distinct banks) + **M = 142**
reuse phases; `params_reloaded ≈ 66.6M` (Σ over the 16 distinct banks, each counted
**once**, not once per reused spatial position). Activation bytes moved at the barriers
≈ 16 MB (order-of-magnitude; sub-dominant).

Applying the model across the default band:

| quantity | value |
|---|---|
| **nominal cost** | **13.4 mJ / inference** |
| **band (low — high)** | **1.5 — 49.3 mJ / inference** |
| reprogram (weight DMA + sync), nominal | 10.7 mJ (**79.8%** of total) |
| reuse (activation DMA + sync), nominal | 2.7 mJ (20.2%) |

**Reprogram-vs-reuse split — the weight-reuse saving.** If every one of the ~137,788
spatial positions reprogrammed its full resident bank (the *pre-reuse* model), the weight
DMA balloons by the position multiplier (positions / distinct banks): naive
reprogram-every-pass ≈ **92,000 mJ** nominal. Weight-reuse collapsing those positions to
**16** distinct-bank reloads buys a **~6900× reprogram-reload saving**. That collapse is
*why* the scheduled VGG16@224 cost is dominated by — but bounded to — the 16-bank weight
DMA rather than the 137 K-position naive reload.

## Default-off preserved

The existing `weight_reuse_mj` default-0 path is **byte-identical** — untouched, and the
61 existing `test_cost_extraction` / `test_weight_reuse` cases pass unchanged. The
defensible model is **opt-in**: a caller passes a `CoefficientBand` (or the cited
`DEFAULT_COEFFICIENT_BAND`) to `phase_cost_band(...)`; nothing reads it until a consumer
chooses to. The model decomposes the SAME coefficients the round-1 term took, so the two
agree exactly when the opaque `mj_per_reprogram` is fed the decomposed
`bytes_per_param · E_dma_per_byte` (locked by a cross-check test).

## Code

| concern | file |
|---|---|
| defensible model + band + VGG16@224 | `src/mimarsinan/chip_simulation/weight_reuse_cost_model.py` |
| round-1 opaque term (unchanged) | `src/mimarsinan/chip_simulation/cost_extraction.py::weight_reuse_mj` |
| phase classifier (N / M / params) | `src/mimarsinan/mapping/weight_reuse.py` |
| tests | `tests/unit/chip_simulation/test_weight_reuse_cost_model.py` |

## Honest limitations / next round

- `activation_bytes_moved` for VGG16@224 is an order-of-magnitude figure, not a measured
  `SegmentIOSlice` walk (deferred to the R2 replica-fill build per the round-1 design);
  it is sub-dominant to the weight DMA, so the nominal verdict is robust to it.
- `params_reloaded ≈ 66.6M` is the design's stated Σ-over-16-banks figure; the per-bank
  resident weight counts are not re-derived here.
- The coefficient band is a *literature* band, not a measurement on a specific target
  chip — which is exactly the point: the number is reported as a defensible RANGE, and a
  target-chip datasheet would narrow it.
