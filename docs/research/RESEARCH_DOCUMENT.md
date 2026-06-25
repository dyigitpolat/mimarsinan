# mimarsinan — a generic, auditable ANN→SNN deployment toolchain

*The research document (deliverable F5). System · research process · quantitative insights · genuine
future gaps. Companion to `ROADMAP.md` (plan), `HYPERVOLUME.md` (axis SSOT), `PROGRAM_CHECKPOINT_v2.md`
(state). Every coverage/cost number here is **measured by the instruments**, re-runnable from the
cited command — none is asserted. Sections marked ⟦pending Wave-3⟧ are filled from the E5/D3 landings.*

---

## 1. What the system is

mimarsinan is a **composable experimentation environment for deploying ANNs as SNNs on a
multi-core neuromorphic chip**. Its deliverable is not a single accuracy number; it is **honest,
measured, auditable coverage of a deployment-configuration hypervolume**, plus tools that turn any
covered configuration into comparable per-cell outputs (cost, accuracy, fidelity). Energy / accuracy
/ speed are **per-cell outputs**, never the success metric.

The organizing abstraction is the **deployment hypervolume**: a typed product of axes

```
firing × sync × encoding_placement × quantization × pruning × backend
      × mapping_strategy × S × depth × vehicle × dataset × regime
```

A *cell* is one fully-specified deployment; a *campaign* fills cells; the *coverage instrument*
measures what fraction of the honestly-counted hypervolume has actually been run, and at what
validity tier. The scientific claim the toolchain exists to support is **transferable genericity**:
the tuning/conversion recipe is generic across models *on the operations the chip can map*, and the
un-mappable operations (attention, LayerNorm) are a flagged research frontier, not a silent scope cut.

## 2. Architecture — the pipeline and its instruments

**Pipeline.** `convert` (torch ANN → IR) → `map` (IR → soft-cores → hard-cores, packing /
neuron-splitting / coalescing / scheduling) → `deploy` (per-backend runner: nevresim, SANA-FE, HCM
analytical reference, Lava) → `certify` (parity gates + cost extraction). One `run.py` drives it.

**Instruments built to make genericity *measurable* (the toolchain's real contribution):**

| Instrument | File(s) | What it measures / guarantees |
|---|---|---|
| Tiered validity gate-v2 | `mapping/verification/onchip_fraction.py`, `onchip_majority.py` | on-chip param+MAC fraction → INVALID (<20%) / VALID_FLAGGED (20–50%) / VALID (≥50%); static, pre-run |
| Capacity + scheduling diagnostic | `mapping/verification/capacity/estimate.py` | cores needed; peak-phase vs sum; `scheduled` verdict + phase_count; early `CapacityExceededError` |
| Coverage ledger + **P1 self-audit** | `chip_simulation/coverage_ledger.py`, `coverage_ci.py` | genericity as a MEASURED fraction; the **denominator is a function of each axis's screening status** (collapse-on-a-hunch is structurally impossible — a collapse RAISES without a linked artifact); flag aging; per-region attribution fidelity; CI-enforced |
| Defensible cost model + band | `chip_simulation/weight_reuse_cost_model.py`, `cost_extraction.py` | weight-reuse mJ as a `(lo, mid, hi)` band from cited DRAM/HBM coefficients; default 0.0 byte-identical |
| Cross-simulator parity screen | `chip_simulation/cross_sim_parity.py` | 3-state AGREE(max_abs_diff) / DISAGREE(gap) / INAPPLICABLE(capability); `justifies_collapse` honesty gate |
| Pareto decision layer | `chip_simulation/pareto.py` | ⟦pending Wave-3 E5⟧ cascaded-vs-synchronized verdict + `propose_recipe` |
| Campaign loop | `scripts/campaign/{scheduler,director}.py`, `scripts/gpu/campaign_runner.py` | scheduler FILLS · runner DRAINS · director GROWS+FLAGS · research-round CONSOLIDATES; validity+capacity pre-checks reject infeasible/invalid configs at enqueue |
| Self-defense guards | `scripts/campaign/guards.py` | base-check (stale-base trap), stash-intact, `fcntl` singleton |

## 3. The research process (how every increment was made)

The program ran as a **disciplined incremental loop**, each increment doing exactly one of: make the
measurement trustworthy · raise honest coverage · instrument a per-cell output · add a hypervolume
region. The method per increment:

1. **Decompose** into INDEPENDENT units with disjoint file ownership.
2. **Allocate each to an isolated parallel dynamic Workflow** (Build→Verify, or
   Research→Design→Prototype→Verify), agents in pre-created git worktrees off current `main`.
3. Every unit is **tests-first · adversarially verified** (a default-refute skeptic re-derives the
   claim from the diff, re-runs the tests, mutation-checks, attacks the integrity claim) ·
   **default-off byte-identical** (a new capability ships gated off; the default path provably
   unchanged) · returns a **patch for review** (never auto-merges framework code).
4. **Land only tested/byte-identical work**; keep preliminary research isolated on a branch.
5. **After each land, re-price the honest measurement** — the headline number is re-run, not edited.

This is itself a finding: **adversarial verification caught real things** (a stale-base worktree, two
guard bugs found red→green, and it forced a coverage-collapse claim to prove it was measurement-backed
and fidelity-scoped, not a hunch).

## 4. Quantitative insights (measured)

### 4.1 The coverage methodology — the denominator is a function of screening status
The central honesty mechanism: an axis collapses to one representative **only with a linked screening
artifact**; otherwise it is enumerated (counted interacting). A bigger denominator = a lower, honest
coverage fraction. This makes "collapse-on-a-hunch" — the original sin the reviewers caught (a legacy
0.60 that silently collapsed four unscreened axes) — structurally impossible.

### 4.2 Faithfulness axes vs semantic knobs (the rule for *which* axes may collapse)
A load-bearing distinction this program introduced:
- **Faithfulness axes** — `backend`, `mapping_strategy`, `encoding_placement` — are different
  *simulators / packings / placements of the SAME deployment contract*. They collapse on a **measured
  parity/fidelity artifact** (a disagreement would be a BUG, not an interaction), **scoped
  fidelity-only**: capability (which backend×mode runs) and cost/utilization stay un-collapsed
  frontiers.
- **Semantic knobs** — `pruning`, `regime`, `quantization` (and `S`/`depth`/`vehicle`/`dataset`) —
  *change the trained result*. They CANNOT collapse on a fidelity argument; they need a real
  cross-product equivalence screen (GPU) or a capability build. They stay enumerated.

This rule was applied to **raise honest deep_cnn coverage from 0.23% (6/2560) to 3.75% (6/160)** by
collapsing `backend` and `mapping_strategy` on measured artifacts (a live cross-sim screen at
`max_abs_diff = 0.0`; the bit-exact torch↔sim fidelity lock) — the keystone working in the legitimate
direction, with the covered-cell count (97) and tier split (VALID 43 / FLAGGED 38 / INVALID 16)
invariant under the pure-denominator collapse. Remaining denominator factors (pruning×2 · regime×2 ·
quantization×4 = 16×) are honestly un-collapsed pending real screens. *(Re-run:
`coverage_report.py --vehicle deep_cnn --dataset … --sync cascaded synchronized --firing ttfs_cycle_based`.)*

### 4.3 The validity frontier — on-chip majority
A deployment is only valid if ≥50% of params+MACs run on-chip (gate-v2; 20% floor = VALID_FLAGGED).
This retired host-majority configurations as INVALID (e.g. deep_mlp w64 at 19.7–36.4% on-chip) and
classified ViT-B as VALID_FLAGGED (0.33/0.33 — the MAC metric does not rescue attention). deep_cnn is
the valid trainable-deep vehicle; SqueezeNet (added this campaign) is VALID at frac 1.0 (offload),
942-of-1000 cores, single-phase.

### 4.4 The death-cascade science (firing × sync, depth × dataset)
Genuine cascaded single-spike TTFS suffers a depth-driven **death cascade** — a correctable
firing-gain deficit, architecture- and dataset-dependent in onset (dataset dominant). Synchronized
execution is the **lossless, depth-stable** default (measured deep_cnn d10/S4 mnist: synchronized
0.9903 vs cascaded 0.9297). `firing × sync` and `depth × dataset` are PROVEN-interacting axes
(enumerated, never collapsed).

### 4.5 The scale frontier and weight-reuse scheduling
ImageNet-scale conv does **not** map simultaneously (no on-chip weight sharing): VGG16@224 ≈ **138K
irreducible soft-cores**. The **Scheduled path** makes it *feasible-via-scheduling* by
time-multiplexing across ~158 reprogram phases; **time-domain weight-reuse** (load weights once,
stream data) re-factors this into ~**16 reprogram + 142 reuse** phases. The defensible cost model
prices VGG@224 ≈ **13.4 mJ** (band 1.5–49.3, ~80% weight-DMA) — a *model-estimate with an
uncertainty band*, the honest form for a number with no measured ground truth yet. ⟦Wave-3 D3
confirms the phase decomposition + bit-exactness on a real built graph.⟧

### 4.6 Attribution fidelity — value-domain vs per-neuron
The instrument distinguishes value-domain bit-exactness (deployed accuracy) from per-neuron
attribution. **GAP-1** (per-neuron attribution under coalescing + output-tiling-under-compaction-
reorder) was fixed this campaign via joint `(perceptron_output_slice, ir_id)` keying — bit-exact,
value-domain byte-identical. The residual Tier-1 merge remains the sole value-domain-only region.

### 4.7 Decision science — cascaded vs synchronized ⟦pending Wave-3 E5⟧
*(Filled from the E5 Pareto landing: the cascaded-vs-synchronized Pareto verdict under the
cost-band, the retire-or-regime recommendation, and `propose_recipe(budget)` behavior.)*

## 5. Genuine future gaps (open research / capability, not deferral of understood work)

These are *genuine* frontiers — each is either un-instrumented or needs a capability we do not yet
have. Things we already understand or can fix in scope were NOT deferred here.

1. **Per-sample energy instrumentation.** Cost is currently a defensible model-estimate-with-band;
   measured per-(cell, schedule) energy/latency is not yet in the ledger. The largest single
   instrumentation gap for a true cost-accuracy Pareto.
2. **Pruning / regime equivalence screens.** Semantic knobs cannot collapse on fidelity; collapsing
   them honestly needs a real GPU cross-product equivalence screen — which needs the pruning
   deployment capability (D4) and the pretrained bridge (B3).
3. **On-chip attention / LayerNorm (the transformer frontier).** ViT-B is VALID_FLAGGED because
   attention+LN have no on-chip SNN mapping yet; this is *the* headline capability gap (the cheap
   path is foreclosed — it is real research).
4. **Pretrained near-SOTA + ImageNet headline (GPU-weeks).** Reachable via the Scheduled path at a
   costed phase budget, gated on the pretrained bridge + timm/torchvision import.
5. **Residual Tier-1** (on-chip param-free merge) and the residual-merge attribution.
6. **Published-baseline head-to-heads** (RMP/QCFS/percentile-norm) on the covered/valid cells.

## 6. How to reproduce / audit

- Honest coverage: `python scripts/campaign/coverage_report.py [--vehicle … --dataset … --sync …]`
- CI integrity (no collapse-without-artifact, no merged tiers, no aged-unowned flags):
  `python scripts/campaign/coverage_report.py --ci`
- Validity of a model: `mapping.verification.onchip_fraction.classify_validity(...)`
- Capacity/scheduling: `mapping.verification.capacity.estimate.estimate_cores_needed(...)`
- Cross-sim parity screen: `chip_simulation.cross_sim_parity` + `assert_cross_sim_screen_sound`
- The screening artifacts: `docs/research/findings/`.
