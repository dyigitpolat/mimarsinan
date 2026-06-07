# NF↔SCM behavioral-contract unification — situation report & engineering record

**Status:** implemented and verified (R1–R8 + the stale-bias root-cause fix)
**Commit:** `ef4bdf9` — "lock parity contract across backends and spike modes" (89 files, +3476/−2512)
**Design doc:** `docs/behavior_contract_unification.md` (the incident analysis and round plan this work executed)
**Date:** 2026-06-07
**Test status:** full `tests/unit` green (2544 passed; the only failures are the 6 pre-existing
on the parent commit — 3 architecture meta-tests + 3 compilagent, verified in a clean worktree)

---

## 1. Situation report

### 1.1 What was broken

The 2026-06-06 incident: a synchronized `ttfs_cycle_based` deployment **passed every
gate** while carrying a silent 3.8 pp torch-side ↔ sim-side accuracy split
(NF 0.904 → SCM 0.866). The 15 % `degradation_tolerance` absorbed it. Root causes,
in increasing depth:

1. `TTFSCycleAdaptationTuner` was schedule-blind — it installed the **cascaded**
   spike forward unconditionally, so synchronized runs fine-tuned, committed, and
   reported through dynamics the chip never executes (~2.7 pp of the split).
2. The behavioral SSOT (`NeuralBehaviorConfig`) spanned only the sim side and missed
   five semantic axes; 13 files independently re-derived what the schedule implies.
3. Accuracy-tolerance gates are categorically too coarse to detect semantic drift:
   they cannot distinguish "the mapping costs 1 pp" from "the NF is computing a
   different function".

During the verification sweeps, three further latent defects surfaced and were fixed:

4. The synchronized q(x) wire snap was wired to `is_encoding_layer` — **inert under
   `encoding_layer_placement=offload`** (nothing is flagged) and the wrong seam under
   subsume (it wrapped the host encoder's input instead of the first on-chip core's).
5. The per-perceptron parity comparison assumed conv core emission order equals the
   torch flatten order (it does not — values were bit-exact up to a permutation).
6. **`TTFSActivation` held `perceptron.layer.bias` by reference, and normalization
   fusion replaces `perceptron.layer` wholesale** — orphaning the reference. The
   cascaded NF then subtracted a stale bias every cycle, silently inflating
   torch-side metrics ~5 pp over the deployed truth for offload runs.

### 1.2 What is now true (verified on real runs)

| Run (mmixcore MNIST) | Before | After |
|---|---|---|
| synchronized + offload (regression config) | fine-tune 0.8609; NF 0.904 vs SCM 0.866 (silent −3.8 pp) | fine-tune **0.9559**; NF 0.9685 → SCM 0.968 → HCM 0.968 → SANA-FE 0.968 — flat; per-neuron gate **0/122880 mismatches** with segment-entry q(x) |
| cascaded + subsume | (untested per-neuron) | NF↔identity-executor argmax agreement **1.0000** on 512 MNIST samples |
| cascaded + offload | agreement 0.8535; NF acc 0.8848 *inflated* vs deployed 0.8398 | with live bias refs: agreement **1.0000**, NF acc ≡ deployed acc = 0.8398 (the old NF number was an artifact) |
| lif / continuous ttfs | — | end-to-end green with the gate enforced |

The honest consequence: offload-cascaded torch-side metrics drop to the true deployed
value on re-runs. The old higher numbers measured a model that never deploys.

---

## 2. The gate ladder (the new verification architecture)

Each rung isolates exactly one concern; a regression now names its layer instead of
hiding inside an end-to-end accuracy budget.

| Rung | Executor | Mapping | Catches | Gate |
|---|---|---|---|---|
| 1 | schedule-correct torch NF (driver / analytical staircase) | model graph | training/deploy objective drift | training metric + tuner commit |
| 2 | `SpikingHybridCoreFlow` over **identity mapping** (`run_scm_identity_metric`) | 1:1 `NeuralCore`→`HardCore`, no pool/pad/reindex | IR semantics: weights, shifts, banks, segment partition, wire effects | SCM accuracy + **NF↔SCM parity gate** (§4) |
| 3 | same executor | packed mapping (`run_hcm_mapping_metric`) | packing: placement, padding, reindex, coalescing, splitting, scheduling | HCM accuracy |
| 4 | SANA-FE / nevresim / Lava | packed | per-backend execution | bit-exact record parity (atol 1e-12) |

Rung 2 is new: previously SCM and HCM both measured the *same packed object*
(`run_hcm_mapping_metric` twice), so a packing bug and an IR bug were
indistinguishable, and rung 1↔2 compared *different algorithms*. SCM no longer
caches the packed `hybrid_mapping`; HCM builds it on demand
(`load_hybrid_mapping_for_step`).

---

## 3. Architecture surface changes

### 3.1 `chip_simulation/deployment_contract.py` — the cross-side SSOT (new)

`SpikingDeploymentContract` composes the existing `NeuralBehaviorConfig` (identity
axes: spiking/firing/thresholding/spike-generation modes, seed) and adds the four
axes that were loose kwargs: `simulation_steps`, `ttfs_cycle_schedule`,
`encoding_layer_placement`, `bias_mode`.

- **Single-reader invariant:** `from_pipeline_config` is the *only* place these
  config keys are read — enforced by a grep-guard test
  (`test_deployment_contract.py::TestSingleReaderInvariant`) with an explicit
  allowlist (`config_schema/`, `deployment_specs.py` step selection).
- **Derived getters** replace per-consumer re-derivation (the design doc's D4):
  `is_synchronized()`, `is_cascaded()`, `quantize_stage_input_to_grid()`,
  `training_forward_kind()` ∈ {`segment_spike`, `analytical_staircase`,
  `lif_cycle`, `rate`}, `wire()`. A parametrized truth-table test locks every
  (mode × schedule) answer.
- **Reserved seam:** every derived getter accepts `core=None` (ignored) so per-core
  heterogeneity can land without re-plumbing.
- Consumers rewired: `simulation_factory` (`build_deployment_contract`),
  `ttfs_executor` (`contract=` supersedes loose kwargs), `SanafeRunner`
  (`contract=`), the SANA-FE step, the tuner.

### 3.2 `models/spiking/wire_semantics.py` — kernel pairs (new)

Each TTFS wire op is defined **once** with torch+numpy twins in identical operation
order (bit-equal in float64; cross-twin tests sweep exact grid points k·θ/S ±1 ULP
for S ∈ {1,2,3,4,8,16}):

| Op | Definition | Twins |
|---|---|---|
| `ttfs_quantized_staircase` | `(S − clamp(ceil(S·(1−V/θ)), 0, S−1))/S`, fire mask `k<S` | torch + `_np` |
| `ttfs_spike_time` | `round(S·(1−clamp(x,0,1)))` (round-half-to-even both sides) | torch + `_np` |
| `ttfs_grid_quantize` | encode→decode round trip `(S − spike_time)/S`; `k≥S` → 0 | torch + `_np` |
| `floor_staircase` | `floor(x·levels)/levels` (generic act-quant; non-integer levels) | torch + `_np` |

`WireSemantics(S, compare_mode)` bundles them per deployment (`contract.wire()`);
`compare_mode="<"` implements nevresim `StrictCompare` tie semantics
(`floor(S(1−r))+1` — exact grid ties fire one cycle later). C++ `<` parity is not
yet harness-covered (documented).

Delegations: `ttfs_kernels.py` (names kept), `ttfs_encoding.py` numpy halves,
`spike_modes.to_ttfs_latched_spikes`. The NF activations (`TTFSActivation`
analytical branch, `TTFSCycleActivation`) switched from the floor `StaircaseFunction`
to a new `TTFSStaircaseFunction` STE wrapping the deployment ceil kernel.
`StaircaseFunction` itself stays floor: `QuantizeDecorator` calls it with
**non-integer** levels (`levels/c`), where the floor/ceil identity
`floor(y) = S − ceil(S − y)` (valid only for integer S) breaks.

### 3.3 Identity mapping (rung 2) — `mapping/packing`

- `HardCoreMapping.map_identity(softcore_mapping)`: one exactly-sized `HardCore`
  per soft core; shares `_finalize_sources` (source remap + off-padding, a no-op
  for identity) with the packing `map()`.
- `build_identity_hybrid_mapping(*, ir_graph)` in `hybrid_build_pool.py`
  (exported via `hybrid_hardcore_mapping.py`); `_flush_neural_segment` gained
  `identity: bool`.
- Equivalence is *proven*, not assumed: `test_identity_hybrid_mapping.py` asserts
  contract-runner stage outputs identical (atol 0) between identity and packed
  mappings on FC and conv/mean toy graphs — i.e. **packing is value-preserving**,
  which is exactly what licenses splitting rungs 2 and 3.
- `models/spiking/hybrid/identity_flow.py::build_identity_spiking_flow`: a
  pipeline-free drop-in for the retired `SpikingUnifiedCoreFlow` (same positional
  signature; computes IR latencies only when missing). **Caveat:** hybrid TTFS
  output is count-scaled (×T); the retired unified flow returned normalized values.

### 3.4 `models/spiking/unified/` — deleted (D3/D7)

`SpikingUnifiedCoreFlow` (an independent torch re-implementation of the executor —
the second source of truth the design doc condemned) is gone. ~15 test files
migrated to the identity-mapped hybrid flow with assertions preserved verbatim;
4 obsolete diagnostic/migration scripts deleted; remaining mentions in
`mappers/base.py`, `ir/types.py`, `ir/legacy_convert.py` were docstring/message text
only.

### 3.5 `pipelining/core/nf_scm_parity.py` — the NF↔SCM gate (new)

Hooked into `SoftCoreMappingStep` **before** the model→cpu move (see trap §6.4).

| Mode | Gate | Default budget | Healthy measurement |
|---|---|---|---|
| synchronized ttfs_cycle | per-neuron, normalized [0,1] domain | `nf_scm_parity_max_mismatch_fraction` = 0.02 | **0/122880** (bit-exact) |
| continuous ttfs | per-neuron | 0.25 (uncalibrated; passes) | small |
| cascaded ttfs_cycle | decision-level: argmax agreement vs the genuine identity executor (`assert_cascaded_nf_scm_agreement_or_raise`) | `nf_scm_parity_min_agreement` = 0.98 over `nf_scm_parity_samples_cascaded` = 64 | **1.0000** |
| ttfs_quantized | **excluded by design** | — | (≈46 % step flips at a 0.2 pp accuracy gap — see §5.4) |
| lif / rate | not applicable (LIF has its own exact per-neuron story) | — | — |

Mechanics: per-perceptron NF activations captured via forward hooks, normalized by
`activation_scale`, compared against identity-contract per-core records grouped by
`perceptron_index` (psum partials excluded) — **order-insensitive per
(perceptron, sample) row** (`compare_normalized_records`, sorted multisets; §5.5
explains why this is sound). Fails fast with an actionable message on a stale
instance forward (legacy synchronized caches). `MIMARSINAN_NF_SCM_PARITY_DEBUG=1`
prints per-perceptron mismatch stats.

### 3.6 Tuner (`ttfs_cycle_adaptation_tuner.py`) — schedule-aware (R1) + entries (R7b)

- `training_forward_kind()` selects the NF: cascaded keeps `_SegmentSpikeForward`
  (genuine spike walk) pinned at rate 1.0 through commit/recovery/downstream;
  synchronized installs **no instance forward** (the class forward through the
  ramped `TTFSActivation` blend *is* the analytical staircase) and ramps the blend
  naturally — the incident's "natural adaptation 0.0000" anomaly disappeared with
  the correct objective and is now a regression assertion.
- Synchronized trains through the wire's stage-input grid snap via
  `TTFSInputGridQuantizer` STE placed on **segment-entry perceptrons**
  (`torch_mapping/encoding_layers.py::segment_entry_perceptrons` — the first
  on-chip core per neural segment), not `is_encoding_layer`.
- Optional `ttfs_finetune_kd_against_rung2` (default off): the KD teacher becomes
  `_Rung2TeacherFlow` — the frozen pre-step snapshot IR-mapped and evaluated
  through the identity contract flow (÷T, per-output node scales).

### 3.7 Cascade driver observability (`spiking/segment_policy_ttfs.py`)

`TtfsSegmentPolicy.node_value_recorder` + `TTFSSegmentForward.forward_with_node_values(x)`
→ `(output, {mapper_node: decoded value})`. This is simultaneously the NF capture
for the cascaded gate and the per-layer bisect instrument that localized both R8
findings.

### 3.8 The stale-bias fix (`refresh_perceptron_bias_references`)

`models/nn/activations/ttfs_spiking.py::refresh_perceptron_bias_references(perceptron)`
re-points every `TTFSActivation` under a perceptron at the live `layer.bias`.
Called at both **layer-replacing seams**: `transformations/normalization_fusion.py
::fuse_into_perceptron` and the SCM step's `bring_back_bias` loop. Locked by
`test_fusion_refreshes_ttfs_bias.py` plus a driver-level regression
(replace layer with shifted bias → refresh → driver ≡ executor).

### 3.9 Tolerance plumbing

`Pipeline.step_tolerances` (per-step retention override; `_step_tolerance(name)`),
fed by the opt-in `scm_degradation_tolerance` (e.g. 0.02) — safe to tighten now
that rungs 1↔2 share semantics. New config keys registered in
`config_schema/defaults.py`: `scm_degradation_tolerance`, `nf_scm_parity_samples`,
`nf_scm_parity_samples_cascaded`, `nf_scm_parity_atol`,
`nf_scm_parity_max_mismatch_fraction`, `nf_scm_parity_min_agreement`,
`ttfs_finetune_kd_against_rung2`.

---

## 4. Rigorous relationships (the load-bearing identities)

### 4.1 Staircase forms

For integer S and any real y: `floor(y) = S − ceil(S − y)`. Hence on the clamped
unit domain the legacy floor staircase `floor(r·S)/S` and the deployment ceil
kernel `(S − ceil(S(1−r)))/S` are **mathematically identical** — but their float
evaluations (`S·r` vs `S·(1−r)`) can round to opposite sides of an integer, so the
two *forms* may disagree under ULP noise at grid boundaries. The kernel pair fixes
the form; the cross-twin tests fix the bits. For **non-integer** S the identity
fails entirely — which is why `QuantizeDecorator` (levels = `target_tq/c`) keeps
the floor form.

### 4.2 q(x) and tie semantics

`q(x) = (S − round(S(1−clamp(x))))/S` with `k = S → 0`. Two non-obvious facts:

- `round(S(1−x)) ≠ S − round(S·x)` at half-step ties for **odd** S
  (round-half-to-even parity argument); the torch twin therefore mirrors the numpy
  op order exactly rather than using the "equivalent" complementary form.
- The decode must be computed as `(S − k)/S`, **not** `1 − k/S`: for non-dyadic S
  the latter differs by 1 ULP (e.g. S=3: `(3−2)/3 = 0.3333…33` vs
  `1 − 2/3 = 0.3333…37`).

### 4.3 Cascade dynamics (single-spike, ramp reconstruction)

Per consumer neuron with inputs spiking once at times `t_j`:

```
membrane(t) = Σ_j w_j · (t − t_j)₊  +  b · (t + 1),    fire once at min t with membrane ≥ θ
```

- The chip executor computes this with `W_eff` (integers after WQ) against float θ;
  the driver computes the θ-normalized arrangement (`w/θ` vs 1.0). With WQ'd
  integer weights the chip membranes are **exact integers**, so no tie window
  exists, and the two arrangements provably cannot flip (float error ~1e-15 vs a
  minimum threshold distance of `|m − θ|/θ ≥ ~0.03`). This is what ultimately
  falsified the tie-flip hypothesis (§5.6).
- The greedy (cascaded) schedule fires on the running membrane, possibly **before
  late inputs arrive** — so the genuine cascade legitimately differs from the
  analytical staircase of the total `V = Σ w_j v_j`. They are different reference
  functions, not a bug (§5.3).
- Bias equivalence: on-chip per-cycle bias and a param-encoded always-on axon
  firing at the window start both give cumulative `b·(t_local+1)`.

### 4.4 Stale-bias error term

`TTFSActivation` recovers the weighted input as `weighted_raw = pre − b_held`. If a
step replaces the layer so that `pre = conv(s_t) + b_new` while `b_held = b_old`:

```
weighted_raw = conv(s_t) + (b_new − b_old)        ← constant ≠ 0 every cycle
ramp(t)     += (b_new − b_old) each cycle  ⇒  membrane error grows ~ t²·(b_new − b_old)/2
```

A constant bias error becomes a *quadratically growing* membrane error — which is
why the divergence was so visible on bias-dominated (empty-patch) entry cores and
directional rather than noise-like.

### 4.5 Window/decoding contracts

Driver node window `[depth, depth+T)` with depth = perceptron-hops from segment
entry ≡ executor core window `[lat, lat+T)` with IR latency; both decode
`value = (window_end − fire)/T` via latch accumulation. The flow's final TTFS
output is count-scaled: `flow(x) = value_domain × T`.

### 4.6 Mode × NF-convention matrix (why each gate is what it is)

| Mode | NF convention | Deployed kernel | Per-neuron equality? |
|---|---|---|---|
| synchronized ttfs_cycle | ceil staircase (`TTFSStaircaseFunction`) + entry q(x) | same kernel | **yes — bit-exact** (gate: per-neuron, 0.02) |
| continuous ttfs | clamp/relu | same | yes in exact arithmetic (gate: per-neuron, 0.25 uncalibrated) |
| cascaded ttfs_cycle | genuine spike walk (driver) | genuine greedy executor | **yes — exactly**, given live bias refs (gate: decision-level 0.98; per-logit atol is meaningless through host compute ops) |
| ttfs_quantized | floor staircase at tq **+ half-step bias compensation** (`apply_ttfs_quantization_bias_compensation`) | ceil kernel | **no, by design** — agreement only within one step/layer, compounding (excluded; protected by accuracy + rung 3/4) |

---

## 5. A-ha moments (chronological, each = wrong assumption → evidence → correction)

### 5.1 "Schedule-blind tuner" was a *class* of bug, not an instance
The tuner hardcoded the cascade forward because cascaded was designed first and
synchronized silently reused it. The deep lesson: a predicate SSOT
(`is_synchronized_ttfs` consulted in 13 files) is not a *behavior* SSOT — nothing
forces the 14th consumer to branch at all. Hence `training_forward_kind()` on the
contract: consumers ask for the *answer*, not the ingredients.

### 5.2 The committed model itself can carry the bug (pickled forwards)
Legacy synchronized caches hold `model.__dict__["forward"] = _SegmentSpikeForward`
— resuming them runs wrong dynamics no matter how correct the new code is. The
parity gate fails fast on this with an actionable message. (And the same mechanism
later led to the stale-bias discovery: state baked at fine-tune time can desync
from state mutated by later steps.)

### 5.3 The contract record is **not** the genuine cascade (the analytical-reference trap)
`run_ttfs_hybrid_contract` / `TtfsAnalyticalExecutor` computes the **analytical
staircase composition** per core, for `ttfs_cycle_based` included — it is the
SANA-FE *contract* reference, not the deployed greedy dynamics. The greedy cascade
legitimately fires early relative to it (early-arriving positive ramp crosses θ
before late negative inputs land). Half a day of bisecting compared the driver
against the wrong reference before this clicked. **Rule: the deployed cascade
truth is the hybrid flow's cascade executor.**

### 5.4 Per-neuron equality is a property of the *NF convention*, not a universal goal
ttfs_quantized's NF deliberately trains floor-at-tq + half-step bias compensation —
its existing lock is `max_diff < 1.5/tq` per layer, i.e. one-step agreement, and
step flips compound through depth: a perfectly healthy run measured ~46 % per-neuron
flips at a 0.2 pp accuracy gap. Forcing a per-neuron gate there asserts a false
invariant. The gate matrix (§4.6) follows the conventions, not a slogan.

### 5.5 Conv core order ≠ torch flatten order (and why sorted comparison is sound)
p0's per-neuron "mismatch" was a **pure permutation** (sorted rows bit-identical) —
while its *consumers* were elementwise-exact. That consumer-exactness is also the
soundness argument for multiset comparison: cross-wired neurons within a layer
would corrupt the next layer's values (consumers read producers by index), and the
final outputs are held by accuracy + bit-exact rung 3/4. Positional correctness is
enforced transitively.

### 5.6 Integer membranes kill the tie-flip hypothesis
The seductive explanation for the cascaded offload residual — "12 % of values sit
on S-grid ties; float arrangements flip them" — died on arithmetic: WQ'd chip
membranes are exact integers, θ = 8.2406…, so the minimum relative distance to
threshold is ~3 %, dwarfing 1e-15 float error. When the algebra says flips are
impossible, the observed flips must come from *different inputs* — which pointed
the drill at the data the two engines consume.

### 5.7 Per-core spike counts are traffic, not values
`output_spike_count` for single-spike modes is 1 per fired neuron — reading it as
a value made every fired neuron "0.25" and produced a phantom one-step-systematic
signature that perfectly mimicked an encode-rounding seam. Instrument legend
matters as much as the instrument.

### 5.8 The empty patch was the microscope
The decisive observation: an all-zero MNIST corner patch where the chip fired
neuron 0 at cycle 1 from **bias alone** (`5·(t+1) ≥ 8.2406`), while the assumed
model-bias dynamics said never. Bias-only dynamics have no confounders — they
exposed in one comparison that the driver's held bias `[-0.0112, 0.133, …]` was a
different tensor from the live `layer.bias` `[0.6068, 0, 0.7281, …]`.
`ttfs._bias is p0.layer.bias → False` ended a multi-hypothesis investigation in
one line.

### 5.9 The pattern of immunity was the fingerprint of the cause
Why offload-cascaded only? The patch-embed conv is the **only** perceptron with
BatchNorm → the only layer fusion replaces. Subsume: that conv is a host op
(bias-agnostic analytical compute). Synchronized: the analytical forward never
reads `_bias`. FC biases: mutated in place (references survive). One mechanism,
four placements, four different symptoms — a consistency check worth running on
any future "mode-specific" mystery.

### 5.10 A "parity fix" can lower your metrics — and that's the point
With live bias references the offload-cascaded NF accuracy fell from 0.8848 to
0.8398 — *equal to the deployment*. The old number measured a model that never
ships. Parity work doesn't make models better; it makes the numbers mean something.

---

## 6. Traps & guardrails (for future engineers)

1. **Never re-read schedule/semantic config keys.** The grep-guard test will trip;
   extend `SpikingDeploymentContract` getters instead.
2. **Never compare the genuine cascade against `run_ttfs_hybrid_contract` records**
   (analytical reference, §5.3). Use the identity-mapped hybrid flow.
3. **`output_spike_count` in single-spike modes is traffic.** Reconstruct values
   from fire cycles `(lat + T − fire)/T` or use the node-value recorder.
4. **`model.to("cpu")` does not move mapper-graph compute modules** (they are not
   registered children). Anything that runs the model (the parity gate does) must
   run before the SCM step's device shuffle — the gate is ordered accordingly.
5. **Any step that replaces `perceptron.layer` must call
   `refresh_perceptron_bias_references(perceptron)`.** Current seams: normalization
   fusion, `bring_back_bias`. (Consider this the contract of `TTFSActivation._bias`.)
6. **`is_encoding_layer` ≠ "segment entry."** Under offload nothing is flagged; the
   wire seam for q(x) is `segment_entry_perceptrons` (first on-chip core per
   segment).
7. **Identity flow needs IR latencies.** Intra-segment evaluation order depends on
   them; `build_identity_spiking_flow` computes them only when absent (preset
   latencies are respected).
8. **Hybrid TTFS flow output is ×T** (count-scaled). The retired unified flow was
   normalized — migrated tests divide by T.
9. **Resume-from-cache caveat:** legacy synchronized caches carry the stale cascade
   instance forward; the gate rejects them with a message naming the fix.

---

## 7. Verified end-to-end evidence

- **Synchronized regression config** (`regression_…_045154`): fine-tune 0.8609 →
  **0.9559** (+9.5 pp from training the right objective); NF 0.9685 → SCM 0.968 →
  HCM 0.968 → SANA-FE 0.968. Per-neuron gate: 0.92 % on the committed pre-R7b
  model; **0/122880** with segment-entry q(x) (the form fresh runs take).
- **Cascaded subsume** (`…_065922`): trajectory flat 0.9131 → 0.914 → … → SANA-FE
  0.914; NF↔executor agreement 1.0000/512.
- **Cascaded offload** (`…_065924`): stale-bias agreement 0.8535 (NF inflated to
  0.8848 vs deployed 0.8398); with the fix, agreement 1.0000 and NF ≡ deployed.
- **Identity ≡ packed**: contract outputs bit-identical (atol 0) on FC and
  conv/mean graphs — packing is value-preserving (rung 2/3 separation is sound).
- **Driver ≡ executor** (cascaded): bit-exact (atol 1e-9) on single/multi-segment,
  subsume/offload, non-unit scales, and post-layer-replacement-with-refresh.

---

## 8. Follow-up surface (known, deliberately not done here)

1. **ttfs_quantized convention unification** — give it the R1/R3 treatment (ceil
   kernel NF, retire the half-step bias compensation) so it can join the per-neuron
   gate. Touches training behavior; needs its own test-first round.
2. **Continuous-ttfs budget calibration** — currently 0.25 (passes; unmeasured
   margin). Collect the printed per-run fractions, then tighten.
3. **`compare_mode="<"` C++ parity** — the Python strict-tie kernels exist and are
   unit-locked; no harness exercises a `<` TTFS deployment against nevresim/SANA-FE.
4. **R6 in anger** — `ttfs_finetune_kd_against_rung2` is wired and tested but not
   yet evaluated on a real run; relevant if anyone chases the last wire-residual
   via training.
5. **Offload-cascaded model quality** — with honest metrics, offload trails subsume
   (~0.84 vs 0.914 on this config). That is now a visible modeling/training
   question (entry-aware fine-tuning, KD), not a parity artifact.
6. **Pre-existing failures** (unrelated, on parent commit too): 3 architecture
   meta-tests (module budget / package flatness / undefined-names baseline) and
   3 compilagent optimizer tests.
