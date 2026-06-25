# mimarsinan — a generic, auditable ANN→SNN deployment toolchain

*The research document (deliverable F5). System · research process · quantitative insights · genuine
future gaps. Companion to `ROADMAP.md` (plan), `HYPERVOLUME.md` (axis SSOT), `PROGRAM_CHECKPOINT_v2.md`
(state). Every coverage/cost number here is **measured by the instruments**, re-runnable from the
cited command — none is asserted.*

---

## 0. Executive summary

mimarsinan is a composable environment for deploying ANNs as SNNs on a multi-core neuromorphic chip,
whose deliverable is **honest, measured, auditable coverage of a deployment hypervolume** rather than a
single accuracy number. This document reports the toolchain and the quantitative findings it produced.

**Headline results (all measured, re-runnable):**

| Result | Value | Where |
|---|---|---|
| Honest deep_cnn coverage (self-auditing denominator) | **3.75%** (6/160), raised from 0.23% by two artifact-backed faithfulness-axis collapses | §4.1–4.2 |
| The integrity rule for axis collapse | **faithfulness axes** (backend/mapping/placement) collapse on a *measured fidelity* artifact; **semantic knobs** (pruning/regime/quant) cannot | §4.2 |
| Cascaded vs synchronized (decision science) | **REGIME_DEPENDENT** — synchronized is the accuracy default (+6.06/+7.19/+11.34pp); cascaded is retained for the hard-latency budget (~2.7–2.9× lower) | §4.7 |
| Death cascade | deep_cnn = sharp **d6 onset → bounded ~4–7pp plateau** through d12 (distinct from deep_mlp's monotone widening); synchronized lossless; no working config-level rescue lever | §4.4 |
| Scale frontier | ImageNet conv = 138K soft-cores → Scheduled path → ~16 reprogram + 142 reuse; **scheduled build confirmed bit-exact** end-to-end | §4.5 |
| Validity is **architecture-dependent** (3 flag causes) | ViT research-gap-flagged; ResNet-18 structural-host-flagged (offload **REFUTED** by test — `supported_host` residual shortcuts, 0.42 param/0.999 MAC); ResNet-50 **VALID** (Bottleneck param-majority 0.666, scheduled-feasible) | §4.3 |
| Per-neuron attribution (GAP-1) | fixed bit-exact under coalescing+output-tiling; one residual VALUE_DOMAIN_ONLY region remains | §4.6 |

**The honesty is self-defending** (CI-enforced: no collapse-without-artifact, no merged tiers, no
aged-unowned flags). The remaining DoD is a **GPU-weeks commitment** (publication rigor + ImageNet) and
one research build (on-chip attention) — §5.

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
| Pareto decision layer | `chip_simulation/pareto.py` | cascaded-vs-synchronized verdict (REGIME_DEPENDENT, conditional on a cost band) + `propose_recipe(budget)` |
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
942-of-1000 cores, single-phase. The pretrained bridge (B3) then revealed that **VALID_FLAGGED splits
into two structurally distinct classes**: (a) a **research-gap** flag — unsupported host ops, e.g.
ViT-B (attention+LN), the genuine frontier; and (b) a **structural host-placement** flag — e.g. a stock
torchvision **ResNet-18**, measured **VALID_FLAGGED at param 0.423 / MAC 0.999 with `research_gap_ops=[]`**
(no unsupported op at all). The tempting fix was *tested and REFUTED* (Wave 6): `offload` does NOT lift
ResNet-18 to VALID (param only 0.4223→0.4232) — the host param-majority is **11 residual-boundary
`supported_host` shortcut/downsample Sequentials**, not offloadable encoders, so the deep_mlp
`offload` analogy does not transfer. The deeper resolution is **architecture-dependence**: **ResNet-50**
is measured **VALID** (param 0.666 — its Bottleneck trunk holds the param-majority on-chip natively,
scheduled-feasible at peak 208 / 16–17 phases) with *no* placement trick. So VALID_FLAGGED resolves into
**three** causes the instrument names per cell — *unsupported-op* (research frontier; ViT),
*offloadable-encoder* (placement-fixable; deep_mlp d8), and *structural-host-residual* (NOT offload-fixable,
BasicBlock-vs-Bottleneck architecture-dependent; ResNet-18). Testing the hypothesis rather than asserting
it is what kept this section honest.

### 4.4 The death-cascade science (firing × sync, depth × dataset)
Genuine cascaded single-spike TTFS suffers a depth-driven **death cascade** — a correctable
firing-gain deficit, architecture- and dataset-dependent in onset (dataset dominant). Synchronized
execution is the **lossless, depth-stable** default (measured deep_cnn d10/S4 mnist: synchronized
0.9903 vs cascaded 0.9297). For deep_cnn the cascaded→synchronized gap shows a **sharp d6
onset then a BOUNDED ~4–7pp plateau through d12** (MNIST, n200+n1000; the load-bearing d12 rung is
pending seed-firming) — **distinct from deep_mlp's monotone widening** (an earlier lower-n two-point
read suggested monotone growth to ~9.5pp; the fuller pooled ladder corrects it to a bounded plateau).
Synchronized is lossless at every depth, and there is **no working config-level firing-gain rescue
lever** at the convnet onset (theta_cotrain is rc=1-broken; the convnet staircase-STE and
conversion_policy are both net-negative), so synchronized is the unconditional accuracy default. `firing × sync` and `depth × dataset` are PROVEN-interacting
axes (enumerated, never collapsed).

### 4.5 The scale frontier and weight-reuse scheduling
ImageNet-scale conv does **not** map simultaneously (no on-chip weight sharing): VGG16@224 ≈ **138K
irreducible soft-cores**. The **Scheduled path** makes it *feasible-via-scheduling* by
time-multiplexing across ~158 reprogram phases; **time-domain weight-reuse** (load weights once,
stream data) re-factors this into ~**16 reprogram + 142 reuse** phases. The defensible cost model
prices VGG@224 ≈ **13.4 mJ** (band 1.5–49.3, ~80% weight-DMA) — a *model-estimate with an
uncertainty band*, the honest form for a number with no measured ground truth yet. **D3 confirmed the
mechanism end-to-end (measured):** a small model overflowing a 6-core budget genuinely triggers the
Scheduled path (`phase_count=3>1`, stages `[6,6,2]` vs single-pool `[12,2]` — a real build-level
split, not an estimate echo), realizes 3 reprogram + 33 reuse passes over 3 weight banks, and is
**bit-exact** (`max|Δ|=0.0` scheduled-vs-reference and scheduled-vs-torch, non-degenerate control).
The VGG@224 16-reprogram/142-reuse figure is thus *confirmed-by-mechanism*; the full ImageNet build
itself remains a GPU-weeks run, not yet done.

### 4.6 Attribution fidelity — value-domain vs per-neuron
The instrument distinguishes value-domain bit-exactness (deployed accuracy) from per-neuron
attribution. **GAP-1** (per-neuron attribution under coalescing + output-tiling-under-compaction-
reorder) was fixed this campaign via joint `(perceptron_output_slice, ir_id)` keying — bit-exact,
value-domain byte-identical. The residual Tier-1 merge remains the sole value-domain-only region.

### 4.7 Decision science — cascaded vs synchronized (E5, measured)
The Pareto decision layer (`pareto.py`) reads per-schedule accuracy from the ledger and prices a
defensible cost band, then emits the verdict. **Measured accuracy gap (synchronized − cascaded):**
mnist **+6.06pp** (d10/S4), kmnist **+7.19pp** (d8/S4), fmnist **+11.34pp** (d8/S4) — synchronized
tracks its ANN reference within noise; the gap is the cascaded firing-gain deficit (§4.4), not a
synchronized loss. **Cost** is a *model-estimate-with-band, not measured energy*: latency is a tight
derivation from the documented execution model (synchronized `sim_time = S×groups`; cascaded pipelined
`S+groups` → cascaded ~2.7–2.9× lower latency), and energy is a `cores×active_steps` proxy present
only when cores are known; absolute per-sample spike energy is **UNINSTRUMENTED** (flagged, not
invented — the single biggest cost gap). **Verdict: `REGIME_DEPENDENT` on every measured dataset —
cascaded is NOT retired.** Synchronized is the accuracy-front schedule everywhere, but cascaded's
lower pipelined latency keeps it on the (latency, accuracy) Pareto front, so it is the hard-latency-
budget code; the verdict is explicitly **conditional on the cost band** (if measured per-sample energy
turned out higher for cascaded it could flip to `RETIRE_CASCADED`). `propose_recipe(budget)` picks
synchronized at an accuracy budget and cascaded at a hard-latency budget. This is independently
corroborated by the campaign's research-round: the convnet staircase-STE *regresses* the deep_cnn d6
onset (−5pp FMNIST, −4.33pp KMNIST) and does not compose with conversion_policy → there is **no
working config-level firing-gain rescue lever** at the convnet onset, so synchronized stays the
unconditional accuracy default while cascaded is retained purely for its latency regime.

## 5. Genuine future gaps (open research / capability, not deferral of understood work)

These are *genuine* frontiers — each is either un-instrumented or needs a capability we do not yet
have. Things we already understand or can fix in scope were NOT deferred here.

1. **Per-sample energy instrumentation (now wired at the source; data fills forward).** Cost
   extraction is now invoked at the end of the deployment path (`sanafe_simulation_step` emits a
   measured `cost_record.json`, exception-isolated + result-byte-identical), and E5 prefers the measured
   cost over the proxy when a record is resolvable. The remaining gap is purely temporal: the *existing*
   ledger rows predate emission, so the cost-accuracy Pareto becomes fully *measured* (not proxy) only as
   the campaign re-runs cells with emission on — and per-schedule run-dir→cost resolution is still
   best-effort.
2. **Pruning / regime equivalence screens.** Semantic knobs cannot collapse on fidelity; collapsing
   them honestly needs a real GPU cross-product equivalence screen. The **pretrained bridge now exists**
   (B3 — a stock torchvision model is mappable + validity-classified), so the from-scratch↔pretrained
   *regime* screen is now *runnable*; the remaining piece is the GPU run itself (the bridge enables it,
   it is not a substitute). Pruning still needs the D4 pruning-deployment capability.
3. **On-chip attention / LayerNorm (the transformer frontier).** ViT-B is VALID_FLAGGED because
   attention+LN have no on-chip SNN mapping yet; this is *the* headline capability gap (the cheap
   path is foreclosed — it is real research).
4. **Pretrained near-SOTA + ImageNet headline (GPU-weeks).** The pretrained bridge **now lands** (B3:
   ResNet-18 imports, maps, and validity-classifies — VALID_FLAGGED for a *placement* reason, no
   unsupported op); the remaining piece is the GPU-weeks run via the Scheduled path at a costed phase
   budget. ViT/ImageNet at full breadth additionally needs D5 (on-chip attention/LN).
5. **Residual Tier-1** (on-chip param-free merge) — *characterized as intrinsically `1/T`-bounded*: an
   in-segment on-chip merge cannot be bit-exact to the Tier-0 host-add reference (the in-segment IF head
   re-quantizes the merged spike train, differing by exactly 1 spike = `1/T` by construction; matching it
   needs a host round-trip = Tier 0). It is feasible as a `1/T`-characterized deployment, not a
   host-add-identical one; a closeable shared-HCM-fill alignment (Component A) separately tightens
   NF==HCM to `atol=0`. See `findings/residual_tier1_intrinsic_limit.md`.
6. **Published-baseline head-to-heads** (RMP/QCFS/percentile-norm). The **percentile-norm method now
   lands** (D7 — a selectable, default-off activation-scale policy, numerically verified); the
   remaining piece is the GPU head-to-head comparison on the covered/valid cells.

## 6. How to reproduce / audit

- Honest coverage: `python scripts/campaign/coverage_report.py [--vehicle … --dataset … --sync …]`
- CI integrity (no collapse-without-artifact, no merged tiers, no aged-unowned flags):
  `python scripts/campaign/coverage_report.py --ci`
- Validity of a model: `mapping.verification.onchip_fraction.classify_validity(...)`
- Capacity/scheduling: `mapping.verification.capacity.estimate.estimate_cores_needed(...)`
- Cross-sim parity screen: `chip_simulation.cross_sim_parity` + `assert_cross_sim_screen_sound`
- The screening artifacts: `docs/research/findings/`.
