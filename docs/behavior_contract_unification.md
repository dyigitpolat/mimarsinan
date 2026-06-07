# Behavioral-contract unification: NF ↔ SCM parity and the deployment-semantics SSOT

**Status:** implemented (R1–R5 landed 2026-06-07, plus R6 = config-gated rung-2 KD
`ttfs_finetune_kd_against_rung2` and the opt-in `scm_degradation_tolerance`;
see `pipelining/core/nf_scm_parity.py`, `chip_simulation/deployment_contract.py`,
`models/spiking/wire_semantics.py`, `mapping/packing` identity build).
Verified on the fresh synchronized regression run (20260607_045154): fine-tune
0.9559 (was 0.8609), NF 0.9685 → SCM 0.968 → HCM 0.968 → SANA-FE 0.968 with the
per-neuron gate enforced — the incident's 3.8 pp split is gone. The per-neuron
gate covers synchronized ttfs_cycle and continuous ttfs; **cascaded** gets a
decision-level gate (argmax agreement vs the genuine identity executor,
default ≥0.98; healthy agreement is exactly 1.0 — driver ≡ executor, locked by
`test_ttfs_segment_node_recorder`; the 2026-06-07 0.85 readings were stale
`TTFSActivation` bias references after normalization fusion replaced
`perceptron.layer` — fixed via `refresh_perceptron_bias_references` at the
layer-replacing seams). **ttfs_quantized is
excluded by design** — its NF trains the floor-staircase + half-step-bias
convention, which matches the chip ceil kernel only within one step per layer,
so per-neuron equality is not its invariant (~46 % step-flip fraction measured
on a healthy mmixcore run with a 0.2 pp accuracy gap). NOTE: the contract
runner's ttfs_cycle record is the ANALYTICAL staircase reference (rung-4
SANA-FE contract fields), not the deployed greedy dynamics.
**Date:** 2026-06-06
**Incident run:** `generated/regression_phased_deployment_run_20260606_231505`
(`ttfs_cycle_based`, `synchronized`, `offload`, S=4, weight-quantized, MNIST mixer)
**Companion docs:** `segment_boundary_parity_report.md` (§9 synchronized off-grid contract),
the SCM identity-gate proposal (§5.3 below).

---

## 1. The incident

The run *passed* every gate, yet its metric trajectory shows a silent 3.8 pp
torch-side ↔ sim-side split:

| Step | Metric | Side |
|---|---|---|
| TTFS Cycle Fine-Tuning | 0.8609 | torch (NF) |
| Weight Quantization | 0.9040 | torch (NF) |
| Normalization Fusion | 0.9039 | torch (NF) |
| **Soft Core Mapping** | **0.8660** | **sim (mapped)** |
| Hard Core Mapping | 0.8660 | sim |
| SANA-FE Simulation | 0.8660 | sim (parity gate ✅) |

All sim backends agree with each other bit-for-bit (the SANA-FE TTFS parity gate
passes at contract atol=1e-12). The discrepancy is exclusively **NF (torch
forward) vs the mapped semantics** — exactly the boundary the behavioral SSOT
was supposed to close. The 15 % `degradation_tolerance` absorbed it, so nothing
tripped.

### 1.1 Experimental decomposition (cached model, full MNIST test set)

| NF variant | Accuracy | Δ vs SCM 0.866 |
|---|---|---|
| As cached — `_SegmentSpikeForward` installed (cascade dynamics) | 0.9043 | **+3.8 pp** |
| Instance forward stripped → analytical staircase (synchronized semantics) | 0.8772 | +1.1 pp |
| Analytical staircase + wire `q(x)` (S=4 input grid snap) | 0.8797 | +1.4 pp |

- **≈2.7 pp — wrong NF algorithm for the schedule** (cascade vs synchronized).
  Dominant cause; see §2.1.
- **≈1.1 pp — residual mapping-level wire effects** the torch graph never sees:
  per-core decomposition of convs into partial-sum cores whose intermediate
  values are themselves S-grid TTFS timings (`layout_ir_mapping_fc_psum`),
  segment-boundary grid snap, and float32-torch vs float64-numpy tie flips on a
  4-level staircase (S=4 puts a large share of pre-activations exactly on grid
  boundaries; cf. the 12 %-exact-ties measurement in
  `segment_boundary_parity_report.md` §9).
- Input quantization `q(x)` is **not** a contributor here (0.9043 → 0.9043 on
  the cascade forward; slightly *helps* the analytical one).

---

## 2. Root causes

### 2.1 Immediate: `TTFSCycleAdaptationTuner` is schedule-blind

`src/mimarsinan/tuning/tuners/ttfs_cycle_adaptation_tuner.py` installs
`_SegmentSpikeForward` (→ `TTFSSegmentForward` → `SegmentForwardDriver` +
`TtfsSegmentPolicy`, the **cascaded** single-spike ramp walk) unconditionally —
the file never reads `ttfs_cycle_schedule`. Its module docstring says "The
cascaded `ttfs_cycle_based` deployment…": the tuner was *designed* for cascaded
and the synchronized schedule silently reuses it.

Consequences under `synchronized`:

1. Fine-tuning **trains through the wrong dynamics** (cascade ramp instead of
   the staircase composition the chip executes), so the optimizer optimizes an
   objective the deployment never runs.
2. `_after_run` keeps the cascade forward installed through commit, recovery,
   and every downstream torch metric (WQ, NormFusion) — by design for cascaded
   (see the fine-tune↔deploy parity fix, parity report §8), but for
   synchronized it propagates the wrong semantics through the whole torch side.
3. Secondary anomaly in the same run:
   `TTFSCycleAdaptationTuner: natural adaptation reached only 0.0000; _after_run
   will force to 1.0` — the KD blend never progressed naturally; worth a
   separate look when making the tuner schedule-aware (the cascade objective on
   a synchronized-calibrated model may simply not adapt).

The correct synchronized NF is the analytical staircase composition (the
ReLU↔TTFS equivalence that `unified/ttfs_step.py` cites) **plus the wire
contract** (S-grid input snap, boundary semantics) — i.e. exactly what the
shared contract runner already computes.

### 2.2 Architectural: the behavioral SSOT only spans the sim side

`NeuralBehaviorConfig` (`chip_simulation/behavior_config.py`) was meant to be
the SSOT for activation semantics. Two structural gaps:

**(a) Its field set is a subset of the deployment semantics.**

| Semantic axis | In `NeuralBehaviorConfig`? | Where it actually lives |
|---|---|---|
| spiking_mode, firing_mode, thresholding_mode, spike_generation_mode | ✅ | — |
| `ttfs_cycle_schedule` | ❌ | separate kwarg threaded ad hoc (e.g. `SanafeRunner(..., behavior=…, ttfs_cycle_schedule=…)`) |
| `simulation_steps` (S/T) | ❌ | separate kwarg everywhere |
| `encoding_layer_placement` | ❌ | converter/mapping flags |
| `bias_mode` | ❌ | `resolve_bias_mode` at each consumer |
| wire grid-quantization (q(x)) | ❌ | `quantize_input_to_ttfs_grid` flag, computed per caller |

**(b) The torch side consumes none of it.** Consumers of
`NeuralBehaviorConfig`: `sanafe/runner/core.py`, `lava_loihi/runner.py`,
`loihi_simulation_step.py`, `sanafe_simulation_step.py`,
`simulation_factory.py` — sim side only. The torch side re-derives the same
semantics from loose config keys per class (`TTFSActivation.__init__` takes
T/thresholding/firing/encoding/bias_mode as kwargs; tuners read
`pipeline.config` directly; flows read `is_cascaded_ttfs(...)` themselves).

**(c) Schedule interpretation is decentralized.** `is_cascaded_ttfs` /
`is_synchronized_ttfs` are consulted in **13 files**; each consumer
independently decides what the schedule *implies* (flows pick forwards; runners
pick somas and sim lengths; `deployment_specs.py` picks steps; the executor
picks input quantization) — and the tuner simply forgot to. A predicate SSOT
(`spiking_semantics.py`) is not a *behavior* SSOT: nothing forces a new
schedule-dependent decision to be made consistently everywhere.

---

## 3. Who decides what, today (evidence map)

| Decision | torch side | sim side |
|---|---|---|
| Which NF algorithm per schedule | `hybrid/flow.py:170`, `unified/flow.py:185`, `rate_forward.py:67` (each re-branches); tuner: **no branch (bug)** | `neural_stage.py:97-102` (soma + T_eff + preset selection) |
| TTFS staircase kernel | `StaircaseFunction` (floor, float32, `activations/autograd.py`); `ttfs_quantized_activation` (ceil-form, torch, `models/spiking/ttfs_kernels.py`) | `ttfs_quantized_activation_np` (numpy), nevresim `TTFSQuantizedCompute<S,Compare>` (C++), SANA-FE `mimarsinan_ttfs_*_soma.cpp` (C++) |
| Value→spike-time encode | `spike_modes.to_spikes(spike_mode="TTFS")` (torch); `TTFSActivation` encoding branch (membrane sim) | `ttfs_encoding.py` (numpy SSOT); `_spike_encoding.py` (Lava/SANA-FE batch); nevresim `TTFSSpikeGenerator` (C++) |
| Wire grid snap `q(x)` | hybrid flow only (wired 2026-06-06) | contract runner + SANA-FE (gated on `is_synchronized_ttfs`) |
| Thresholding compare | `TTFSActivation` validates mode but kernels don't consult it | nevresim/SANA-FE config-driven `Compare` policies |
| Bias delivery | `TTFSActivation.bias_mode` (no forward branch, doc-only) | per-backend handling (param_encoded vs on_chip), parity-locked |
| Per-cycle fire/reset policy | `cycle_policy.py` (LIF/cascade-TTFS, explicitly "mirrors nevresim's fire_policy") | nevresim `FirePolicy`, SANA-FE somas |

`cycle_policy.py` is the proof-of-concept that mirroring works — it just needs
to be *one* abstraction rather than a hand-maintained mirror.

---

## 4. Duplication catalog (boy-scout findings along the way)

| # | Duplicated mechanism | Sites | Proposed abstraction |
|---|---|---|---|
| D1 | TTFS quantized staircase | `StaircaseFunction` (floor/torch), `ttfs_quantized_activation` (torch), `ttfs_quantized_activation_np` (numpy), nevresim C++, 3 SANA-FE somas | **Kernel pair module**: one `ttfs_kernels` source defining each op once with torch+numpy twins, cross-tested element-wise (`assert torch == numpy` over a sweep incl. exact grid points); C++ stays separate but is parity-locked against the pair by the existing harnesses. `StaircaseFunction` becomes a thin STE wrapper over the shared kernel — today its floor form and the ceil form agree only by mathematical accident and disagree under ULP noise at grid boundaries. |
| D2 | Value→spike-time encode (`round(S(1−clamp(x)))`) | `ttfs_encoding.py`, `spike_modes.to_spikes`, `_spike_encoding.py`, `TTFSActivation` encoding branch, nevresim generator | Same kernel-pair treatment: `ttfs_spike_time` + `ttfs_input_grid_quantize` get torch twins in the same module; all four Python sites import them. (`q(x)` already had to be hand-mirrored into the hybrid flow — the torch twin removes that class of drift.) |
| D3 | NF executors | `SpikingHybridCoreFlow` (+ per-stage contract delegation), `SpikingUnifiedCoreFlow` (independent torch reimpl.), `SegmentForwardDriver` policies | Retire the unified flow (§5.3); evaluation NF becomes the contract executor over an identity mapping; only the *training* forward stays torch-native (gradients). |
| D4 | Schedule→behavior branching | 13 files consult `is_cascaded_ttfs`/`is_synchronized_ttfs` and locally decide forwards/somas/sim-lengths/quantization | `SpikingDeploymentContract` (§5.1) answers *derived* questions (`training_forward()`, `wire.quantize_input`, `soma_name()`, `sim_length(groups)`), so consumers stop re-deriving implications. Predicates stay for low-level code. |
| D5 | Config plumbing to activations | `TTFSActivation(T, thresholding, firing, encoding, bias_mode)` kwargs duplicated at every construction site (tuner `_make_target_activation`, adaptation manager, tests) | Construct from the contract: `TTFSActivation.from_contract(contract, perceptron)`; one place maps contract→ctor args. |
| D6 | Stage input assembly + shift + snap | numpy (`hybrid_execution.py` + executor) and torch (`hybrid/ttfs_step.py` converts to numpy per stage — already unified de facto) | Keep numpy as the execution SSOT; document that torch-side stage execution must delegate (it already does); forbid a second torch reimplementation (that's what the unified flow was). |
| D7 | SCM/HCM gates measure the same object twice | `SoftCoreMappingStep` metric packs+runs; `HardCoreMappingStep` re-runs the identical packed mapping | Identity-mapped SCM gate (§5.3) makes the two gates measure different layers. |

---

## 5. Refactoring design

### 5.1 `SpikingDeploymentContract` — one object, both sides

Extend (and eventually supersede) `NeuralBehaviorConfig` with the full
deployment semantics, constructed **once** from pipeline config:

```python
@dataclass(frozen=True)
class SpikingDeploymentContract:
    # identity (existing NeuralBehaviorConfig fields)
    spiking_mode: str
    firing_mode: str
    thresholding_mode: str
    spike_generation_mode: str
    spike_encoding_seed: int | None
    # the axes that were missing
    simulation_steps: int                  # S / T
    ttfs_cycle_schedule: str               # normalized via ttfs_cycle_schedule()
    encoding_layer_placement: str          # subsume | offload
    bias_mode: str                         # on_chip | param_encoded

    @classmethod
    def from_pipeline_config(cls, cfg) -> "SpikingDeploymentContract": ...

    # ---- derived behavior (the D4 killer) ----
    def is_synchronized(self) -> bool: ...
    def quantize_stage_input_to_grid(self) -> bool:
        # the q(x) wire rule, decided HERE, not per caller
        return self.is_synchronized()
    def training_forward_kind(self) -> str:
        # 'segment_spike' (cascaded) | 'analytical_staircase' (synchronized)
        ...
    def wire(self) -> WireSemantics: ...   # encode/decode/grid ops (kernel pairs, D1/D2)
    # existing per-backend helpers migrate verbatim:
    # nevresim_*_policy(), sanafe_reset_mode(), lava_zero_reset(), ...
```

Consumers (the unification):

| Consumer | Today | After |
|---|---|---|
| `TTFSCycleAdaptationTuner` | reads 4 config keys, hardcodes cascade forward | `contract.training_forward_kind()` selects `_SegmentSpikeForward` **or** analytical-staircase KD (with `wire.quantize_input` on encoding entries, trained through STE) |
| NF flows (hybrid) | re-branch on `is_cascaded_ttfs`, pass `quantize_input_to_ttfs_grid` manually | take `contract`, ask it |
| Contract runner / SANA-FE / nevresim / Lava | `behavior` + loose kwargs (`ttfs_cycle_schedule=`, `simulation_length=`) | take `contract` (or read the extra fields from it) |
| `record_ttfs_hcm_reference`, parity gates | thread schedule by hand | take `contract` |
| `deployment_specs.py` | predicate calls | unchanged (step selection is fine on predicates) |

Single non-negotiable invariant: **`from_pipeline_config` is the only place
that reads these config keys.** Everything downstream takes the contract.
(Reserved seam: per-core/per-layer overrides — the contract stays global for
now, but derived getters take an optional `core`/`node` argument signature so
heterogeneity can land without another plumbing round.)

### 5.2 Kernel pairs (`WireSemantics`)

One module owning each wire op with torch+numpy twin implementations and a
cross-twin equality test (including exact-grid sweeps, because S=4 puts mass on
the boundaries):

- `spike_time(x)`, `grid_quantize(x)` (D2 — numpy halves already exist in `ttfs_encoding.py`)
- `quantized_staircase(V, θ)` (D1 — replaces `StaircaseFunction`'s body and both
  `ttfs_quantized_activation` variants; STE wrapper stays in `autograd.py`)
- boundary encode/decode re-exports from `spiking/segment_boundary.py` (already SSOT)

C++ (nevresim, SANA-FE somas) cannot share code; they stay parity-locked
against the pair via the existing integration harnesses — the pair gives them
*one* Python reference instead of four.

### 5.3 The gate ladder (integrates the SCM identity-gate proposal)

With one contract and one executor, the pipeline's verification becomes a
ladder where each rung isolates one concern:

| Rung | Executor | Mapping | Catches |
|---|---|---|---|
| 1. Training NF metric | schedule-correct torch forward (5.1) | model graph | training/deploy objective drift |
| 2. **SCM gate** | hybrid flow (contract runner) | **identity-mapped soft cores** (new `build_identity_hybrid_mapping`, 1:1 `NeuralCore`→`HardCore`, no pool/pad/reindex) | IR semantics: weights, shifts, banks, segment partition, wire effects (psum decomposition!) |
| 3. **HCM gate** | same | packed mapping (existing `run_hcm_mapping_metric` at `hard_core_mapping_step.py:94`) | packing: placement, padding, reindex, coalescing, splitting, scheduling |
| 4. Backend parity | SANA-FE/nevresim/Lava vs contract record | packed | per-backend execution |
| — | `SpikingUnifiedCoreFlow` | — | **deleted**; its ~15 mapping tests migrate to rung 2's identity mapping |

Why this closes the incident class: rung 1↔2 currently compare *different
algorithms* (3.8 pp slipped under a 15 % accuracy tolerance). After 5.1 they
share semantics, so the rung-1↔2 residual is exactly the *mapping-level wire
effects* (the honest ≈1.1 pp) — and rung 2 becomes the **true NF for
evaluation purposes**. Additionally:

- **New parity check (not just accuracy):** for analytic schedules
  (synchronized, ttfs, ttfs_quantized), add a per-neuron NF↔SCM parity record on
  N samples (mirror `compare_ttfs_contract_records`) between rung 1's analytical
  forward and rung 2's stage records. Cascaded already has its per-neuron story
  (unified driver == HCM). Accuracy tolerance alone is demonstrably too coarse.
- The remaining honest gap (rung 1 vs 2: psum/grid/dtype wire effects) is then a
  *modeling* decision: either accepted and reported per-run, or closed by
  training through rung-2 semantics (KD on identity-mapped outputs) — config
  knob, out of scope here.

### 5.4 Migration plan (test-first rounds)

1. **R1 — fix the bug at its seam (small, urgent).** Schedule-aware
   `TTFSCycleAdaptationTuner`: under synchronized, do not install
   `_SegmentSpikeForward`; train/commit through the analytical staircase (+
   `q(x)` STE on encoding entries). Tests: tuner installs the right forward per
   schedule; committed metric matches a contract-runner evaluation within a
   tight bound on a toy model. Also investigate the `natural adaptation 0.0000`
   warning here. *(Expected effect on the incident run: NF ≈0.877 → gap vs SCM
   ≈1.1 pp, honestly attributed.)*
2. **R2 — `SpikingDeploymentContract`.** Introduce the dataclass + factory;
   port `NeuralBehaviorConfig` consumers (compat alias retained); convert the
   `ttfs_cycle_schedule`/`simulation_length`/`quantize_input_to_ttfs_grid`
   kwarg threads to contract reads. Tests: factory SSOT (config→contract),
   derived-getter truth table per (mode × schedule).
3. **R3 — kernel pairs.** `WireSemantics` module; replace `StaircaseFunction`
   body, both staircase variants, `to_spikes(TTFS)`; cross-twin sweep tests +
   existing parity suites green.
4. **R4 — SCM identity gate + unified retirement.** `build_identity_hybrid_mapping`
   (+ tests vs packed equivalence on toy graphs), switch `SoftCoreMappingStep`
   metric, keep `HardCoreMappingStep` rung-3 gate, migrate unified-flow tests,
   delete `models/spiking/unified/` and the `build_spiking_flow_for_metric` alias.
5. **R5 — NF↔SCM per-neuron parity gate** for analytic schedules (N-sample
   record comparison, loud failure with `format_first_*_diff`-style messages).

Each round leaves the tree green and independently shippable; R1 alone
resolves the incident's dominant cause.

---

## 6. Open questions

1. **R1 scope:** for synchronized, should fine-tuning *train through* `q(x)`
   (STE) and the staircase only, or also through a differentiable surrogate of
   the psum-decomposition wire effects (i.e., KD against rung-2 outputs)? The
   former is cheap and removes 2.7 pp of self-deception; the latter chases the
   last ≈1.1 pp at real complexity cost.
2. **Contract granularity:** global per-deployment now; per-core overrides
   (mixed-schedule or mixed-mode chips) reserved via getter signatures — is that
   sufficient for the roadmap?
3. **Gate tolerances:** once rungs share semantics, the accuracy
   `degradation_tolerance` between rung 1 and 2 can tighten substantially
   (e.g. 2 pp) plus the per-neuron parity check; what budget is right per rung?
4. **`thresholding_mode` in Python kernels:** the C++ sides consult Compare
   policies; the Python staircase kernels don't branch on `<` vs `<=`. Verify
   `<` deployments are parity-covered, else fold the compare mode into the
   kernel pair (it belongs in `WireSemantics` anyway).

---

## Appendix A — incident reproduction

```bash
# resume only the SANA-FE step against the cached run (see memory: pipeline-resume-from-cached-run)
# experiments used:
#   /tmp/nf_scm_gap_experiment.py   (input-quantization eval: no effect)
#   /tmp/nf_forward_semantics.py    (forward-semantics decomposition: 0.9043 / 0.8772 / 0.8797)
#   /tmp/nf_scm_bisect.py           (45-capture cascade-walk signature; per-stage records)
```

Key code locations cited above:

- `tuning/tuners/ttfs_cycle_adaptation_tuner.py:72-108` — unconditional cascade forward
- `chip_simulation/behavior_config.py` — sim-only SSOT, missing axes
- `chip_simulation/spiking_semantics.py` — predicates (13 consumer files)
- `models/spiking/cycle_policy.py` — existing torch↔nevresim mirror precedent
- `chip_simulation/ttfs/ttfs_executor.py` — contract runner (+ `quantize_input_to_ttfs_grid`)
- `pipelining/core/simulation_factory.py:219` / `pipeline_steps/mapping/hard_core_mapping_step.py:94` — duplicated gate

---

## 7. Follow-up (2026-06): unified gradual ramp + the LIF r=0 leak

**Status:** implemented. One value-domain blend ramp now backs both
`LIFAdaptationTuner` and `TTFSCycleAdaptationTuner`; the genuine cross-layer
dynamics are installed only at finalize. See
`tuning/orchestration/kd_blend_adaptation_tuner.py`
(`_InstalledForward`, `CascadeForwardInstall`, `_ramp_forward`/
`_finalize_forward` hooks), `tuning/tuners/lif_adaptation_tuner.py`,
`tuning/tuners/ttfs_cycle_adaptation_tuner.py`.

### 7.1 Two defects, one root cause

The cascaded `TTFSCycleAdaptationTuner` **pinned rate to 1.0** and installed
`_SegmentSpikeForward` for the whole ramp — a one-shot jump from ReLU
semantics to the full single-spike cascade, recovered only by KD (the
`natural adaptation reached only 0.0000` symptom; §2.1's dominant cause). The
gradual, non-destructive `SmartSmoothAdaptation` ramp that LIF relies on was
bypassed entirely.

Investigating LIF as the golden example surfaced a **latent LIF leak**: with
`cycle_accurate_lif_forward` (default true), the ramp installed
`_CycleAccurateForward`, applying the per-perceptron `BlendActivation`
*per spike frame* inside `run_cycle_accurate`. At rate 0 the output is
`mean_t ReLU(W s_t + b)` — a convex-Jensen-biased, layer-compounding
rate-coded value, **not** the continuous teacher. The KD teacher (a pre-blend
continuous snapshot) and the rate-0 student already disagreed; the
"non-destructive ramp" was leaky.

Both reduce to the same root cause: **blending a cross-layer cascade/cycle
forward per frame is not the per-perceptron value blend** the ramp needs.

### 7.2 The fix

`BlendActivation(v) = (1−r)·ReLU(v) + r·OnChipAct(v)` in the **value domain**
(the plain class forward, no per-frame/cascade wrapper) already is the
per-perceptron, monotone, bit-exact-endpoint blend (`r=0` == continuous
teacher, `r=1` == pointwise on-chip composition — LIF rate / TTFS staircase).
It is what synchronized TTFS and non-cycle-accurate LIF already ramped
through. So:

- The ramp runs in the value domain (`_ramp_forward()` → `None`).
- The genuine cross-layer dynamics are installed **only at finalize**
  (`_finalize_forward()`): the single-spike cascade for cascaded TTFS, the
  chip-aligned segment forward for cycle-accurate LIF. It stays bit-identical
  at the committed `r=1`, preserving every existing parity gate (which all run
  against the finalized forward).

The legacy LIF per-frame ramp is retained behind `legacy_lif_blend_ramp`
(default **off** — the value-domain ramp is the default) for fallback; the
deployed finalize forward is identical for both ramp choices.

### 7.3 Phase B (gradual genuine-cascade *entry*) — viable, deferred

A second phase was scoped: gradually fade in the genuine cascade *output*
during the ramp (rather than the one-shot finalize entry), so even the cascade
entry is non-destructive.

It is **feasible** — the cascade is already a per-core, node-by-node cycle walk
that propagates exact spike trains (interior cores read the predecessor's exact
1-cycle-delayed spike; cross-segment boundaries already decode→re-encode). A
per-core blend that keeps the exact intra-segment trains flowing and blends
only the per-core decoded readout / boundary values stays bit-identical at
`r=1`. (The one thing that is *not* losslessly expressible as decode→re-encode
at every core is the **intra-segment sub-window spike phase** — re-encoding a
decoded value as a fresh single spike loses the partial-ramp arrival timing —
so a blend must leave those interior trains untouched.)

It is **not yet implemented**: Phase A (value-domain ramp + genuine forward at
finalize + KD + `_stabilize_at_full_rate`) already mirrors the golden LIF
pattern, which itself enters its cross-layer forward in one shot at finalize.
Revisit if a measured finalize-entry regression appears on a deep cascaded
graph.

---

## 8. Genuine-gradual ramp (2026-06): train through the deployed dynamics

**Status: REMOVED 2026-06-08** (`genuine_gradual_cascade_ramp` switch and
`BlendedGenuineForward` deleted in "tuner unification phase 2"). It is
architecturally clean but empirically regresses accuracy — the r=1 hand-off is
catastrophic because the ε-weighted continuous term is a margin oracle (see §9.5
and `docs/fine_tuning_research_directions.md` §2.6). The section below is the
historical design record; the idea lives on as research direction D2/D3 in the
research doc. The shipped ramp is value-domain only.

### 8.1 The two incidents

A subsume-vs-offload comparison (`ttfs_cycle_based`, cascaded, S=T=4, mmixcore;
runs `…174755` vs `…174757`) exposed two problems with the §7 value-domain ramp:

1. **Offload ≪ subsume.** Final SANA-FE 0.966 (subsume) vs 0.924 (offload). The
   IR differs: subsume runs `patch_embed` as a **host ComputeOp at full
   precision**; offload folds it into the **first on-chip core**, so the raw
   pixels are TTFS-encoded at **S=4 (4 timing levels)**. The §7 ramp trains the
   pointwise staircase **proxy**, whose first layer sees full-precision pixels —
   it never experiences offload's dominant loss term, which only appears in the
   genuine cascade installed at finalize (a 2.6 pp finalize cliff).
2. **The gradual phase did almost no work.** Phase timing: the ramp was
   ~6–12 s of a ~149 s step; the rest was recovery + a `2×` stabilization pass
   (`2 * max_training_steps` ≈ 3562 steps). The genuine deployed dynamics were
   trained only in "stabilization" — the proxy ramp was cheap-but-wrong, the
   real training was relabelled as polish.

### 8.2 Why the blend must be at the whole-model output

The continuous teacher (`F_continuous`, real-domain activations) and the genuine
cascade (`F_genuine`, normalized-rate boundaries + spike trains) are **different
computational graphs with non-corresponding intermediate representations**:

- **Scale.** `decode_segment_output = counts / T` (normalized rate) is the
  parity-locked HCM inter-stage value for LIF; TTFS carries `decode · scale`
  (real). A per-core/per-boundary value at the two endpoints differs by a
  per-layer scale, so blending them at an interior point is ill-defined.
- **Sub-cycle phase.** Keeping r=1 bit-exact requires the genuine spike trains
  to flow **untouched** within a segment (a core fires partway through its
  window; the next core integrates that partial ramp). Any per-core
  decode→blend→re-encode destroys that phase, so r=1 would not be bit-exact.

Hence the **only** representation where the two graphs are directly, bit-exactly
comparable is the **final model output**. The blend is therefore at the output:

```
out(x) = (1 - r) * F_continuous(x) + r * F_genuine(x)
```

`F_continuous` = the class forward with every `BlendActivation` rate pinned to 0
(ReLU teacher); `F_genuine` = the deployed cascade forward (rates pinned to 1).
r=0 == continuous teacher bit-exact (non-destructive); r=1 == deployed cascade
bit-exact; every r>0 trains through the deployed dynamics — **including offload's
S=4 input encode**. Unified for LIF (`_ChipAlignedNFForward`) and TTFS
(`_SegmentSpikeForward`). This supersedes §7.3's per-core "Phase B": per-core is
not merely risky but ill-posed for a continuous↔genuine blend.

### 8.3 Consequences (instrumented) and the empirical verdict

The genuine-gradual ramp behaves exactly as designed:

- **Finalize cliff ≈ 0** (instrumented `_finalize_cliff`): the ramp's r=1 forward
  IS the deployed forward, so the `_finalize_forward` swap is a no-op.
- **Gradual phase carries the training** (`_phase_seconds`): 152–181 s of genuine
  training (31 fine-grained cycles) vs the proxy's ~6–12 s; `_stabilization_multiplier`
  0.5 (rollback guard retained).

**But it regresses final accuracy**, so it is **default off** (opt-in). Measured
on mmixcore (S=T=4, seed 0), final SCM/SANA-FE:

| ramp | subsume | offload |
|---|---|---|
| value-domain proxy (default) | **0.966** | 0.924 |
| genuine-gradual | 0.948 | 0.926 |
| value-domain proxy + input-encode STE | 0.948 | 0.932 |

The premise that the proxy is "self-deception" is empirically false: the smooth
staircase proxy is a **better-conditioned optimizer**, and its optimum transfers
*favorably* to the genuine cascade (proxy-trained subsume: genuine SCM 0.966,
*higher* than its torch metric). Directly descending the single-spike cascade
(noisy surrogate gradients through T cycles) lands in a worse basin. The old
"proxy ramp → genuine stabilization" was already curriculum learning
(pretrain-on-proxy, polish-genuine-from-good-init).

### 8.4 Open problems (not closed by this work)

- **Offload TTFS-cycle < 96%** (0.924–0.932 across all ramp strategies). The
  subsume↔offload difference is solely the on-chip patch_embed conv cascade
  (raw-pixel S=4 encode + **conv psum decomposition** with S=4 intermediate
  timings) vs host full-precision. The cap is in the **deployment path**, not the
  tuner; the conv-psum S=4 encode is the prime suspect. A surprising lead: the
  input-encode STE *hurts* subsume (0.966→0.948), suggesting its grid-snap model
  may not match the actual deployment boundary encode (a possible bug to chase).
- **NF↔SCM residual** (~0.8 pp; mapping wire effects: psum/grid/dtype) — the
  designed lever is `ttfs_finetune_kd_against_rung2`, currently synchronized-only.

---

## 9. Cascaded+offload incident closure (2026-06-08): effective bias + stale mapping

Run `mnist_mmixcore_ttfs_cycle_60_offload…234317` showed two "impossible" drops:
Normalization Fusion (analytical) lost 2.7 pp (0.9468 → 0.9199) and the final
sim landed at 0.916. Root causes, both structural:

### 9.1 Raw-vs-effective bias in the cascade walk (the fusion cliff)

`TTFSActivation` (non-encoding, cycle mode) receives the **post-norm**
pre-activation `norm(W s_t + b)` per cycle but subtracted the **raw**
`layer.bias`; under a non-identity normalization (mmixcore's `patch_bn` on
`patch_embed_full`) the residual `fused_b − b` poured into `ramp_current` and
was double-integrated. Only **offload** exposes it: subsume runs patch_embed as
the value-domain encoding entry (bias-agnostic); offload runs it as an interior
cascade core. The fine-tuner then optimized the wrong dynamics, and fusion —
which *corrects* the dynamics (norm folded, bias refreshed) — destroyed the
tuned operating point. Empirically: installing the effective bias pre-fusion
makes pre≡post **bit-exact** (argmax agreement 1.0).

**Fix (SSOT chain):** `models.nn.layers.norm_affine_params` (frozen-stats
affine `(u, β, mean)`, differentiable) → `perceptron.effective_preactivation_bias()`
(the additive constant the chip charges) → consumed by BOTH
`fuse_into_perceptron` (fused bias) and `TtfsSegmentPolicy.prepare()` (installs
it on every `TTFSActivation` fresh per drive; `finalize()` restores the raw
`layer.bias` reference for pickling). Drive-time recompute structurally closes
the stale-bias-reference class (`refresh_perceptron_bias_references` is now a
stored-contract nicety, not a correctness dependency).
`test_fusion_invariant_cascade_forward.py` pins fusion invariance per-node
(offload, subsume, biasless+norm'd) + the restore contract + norm-param grads.

### 9.2 Stale packed mapping across resumes (the HCM "seam" that wasn't)

After the bias fix, a resumed rerun showed SCM 0.954 but HCM 0.916 — exactly
the old number, because `load_hybrid_mapping_for_step` reused the flat-key
`hybrid_mapping` cache entry built from the *previous* run's ir_graph.
**Fix:** every fresh `IRGraph` carries a `build_token` (uuid);
`build_hybrid_mapping_for_pipeline` stamps it as `source_ir_build_token`; the
consumer rebuilds on mismatch (legacy tokenless pairs still pair up).
`test_hybrid_mapping_staleness.py`. With the rebuild: SCM = HCM = nevresim =
SANA-FE = **0.954** (NF↔SCM per-neuron agreement 1.0000).

### 9.3 The ramp was never gradual (SmartSmoothAdaptation drift)

The KD blend steps ran: one-shot `_adaptation(1.0)` first (commit if within
~3 pp!), else a `step=0.5` ladder (2 cliffs, each licensed to lose
`rollback_tolerance`), then 2× stabilization carried the real training — the
"gradual" phase was 6–12 s of a 150+ s step. **Fix:**
`SmartSmoothAdaptation(initial_step, growth)`; KD blend tuners set
`_skip_one_shot=True`, `_initial_ramp_step=0.125`, `_ramp_step_growth=1.0`
(uniform 8-rung ladder, rollback still halves, clamped to budget `min_step`).
`test_kd_blend_gradual_ramp.py` pins the contract at step level.

### 9.4 Status / remaining

- §8.3's "genuine-gradual regresses" benchmark is **confounded by §9.1** (the
  cascade it trained through had the wrong bias dynamics for offload). Worth
  re-measuring `genuine_gradual_cascade_ramp` after the fix.
- Remaining gaps to the acceptance gates (cascaded+offload): FT 0.9432 (gate
  0.96), sim 0.954 (gate 0.97); the measured `finalize_cliff` of the
  value-domain proxy is ~0.23 pre-recovery — the proxy↔genuine swap is now the
  dominant loss mechanism.
- **Cliff decomposition** (post-stabilization model, 1000 test samples):
  genuine cascade 0.938, staircase proxy 0.902, cascade with an ideal-value
  entry (input encode disabled) 0.939. §8.4's "conv-psum S=4 input encode is
  the prime suspect" is REFUTED post-§9.1: the input encode contributes ~0.1 pp;
  the gap is the cascade's partial-ramp sub-cycle nonlinearity vs the pointwise
  staircase — a different transfer function, which only training through the
  genuine forward can close.

### 9.5 Measured ramp verdicts (cascaded+offload ttfs_cycle, resumed from the
same Activation Analysis cache; mmixcore S=T=4, seed 0)

| ramp (post-§9.1/§9.2 fixes) | FT step | final sim (SCM=HCM=nevresim=SANA-FE) |
|---|---|---|
| historical (one-shot + 0.5 cliffs) | 0.9432 | **0.954** |
| gradual value ladder (0.125 × 8) | 0.9436 | 0.942 |
| genuine-gradual (`BlendedGenuineForward`) | 0.9124 | 0.932 |

- **Genuine-gradual is rejected with a sharper mechanism than §8.3:** the ramp
  holds ≥0.96 to r=0.9999 and r=1.0 is *catastrophic* (14 attempts). The
  ε-weighted continuous output resolves the argmax for samples where the
  genuine cascade's logit gaps are tiny — the high-r blend is a tie/margin
  oracle, structurally unable to hand off at r=1. Default stays off.
- The value-ladder ≈ cliff ramp on TTFS-cycle (the loss is the proxy↔genuine
  function gap, not ramp granularity; A↔B spread is run noise).
- **LIF is the opposite story:** the gradual ladder lifted LIF offload from
  LIF Adaptation 0.953 / sim 0.966 to **0.9649 / 0.976** (gate ≥0.97 passes) —
  LIF's chip-aligned finalize has no proxy↔genuine gap, so ramp granularity
  is the binding constraint there.
- Error structure of the deployed cascade (2000 samples): no argmax ties, no
  1-quantum near-misses — a pure optimization gap on the cascade's
  partial-ramp transfer function. Next lever: training budget through the
  genuine dynamics (`tuning_budget_scale`).

### 9.6 Post-finalize polish: stabilization rounds with LR restarts

Run E (2× `tuning_budget_scale`) reached **sim 0.97** (SCM=HCM=nevresim=
SANA-FE) — the cascaded+offload gate. Its WQ step then lifted the genuine
model 0.9473 → 0.9585 (full test) *purely because WQ re-finds its LR* — the
FT stabilization had plateaued at a constant LR. Mechanism added:
`_stabilize_at_full_rate` supports `_max_stabilization_rounds` (default 1;
KD blend tuners = 3): extra rounds restart from a freshly found LR while the
previous round still gains > `accuracy_se()/2`, all inside the existing
rollback guard. This is deployed-dynamics polish (the proxy↔genuine gap is
not reachable by ramping), distinct from the gradual-ramp contract which is
preserved.

### 9.7 Deployment-aware activation-scale calibration: REJECTED (negative result)

The cascade's back-loaded partial-ramp integration suggested a theta seam: a
greedy per-perceptron `activation_scale` multiplier search through the genuine
forward gained **+2.1pp** — but only when calibrated ON TEST (selection bias).
The honest protocol (calibrate on the validation split, evaluate on test)
transfers **+0.08pp**: val 0.972→0.9748, test 0.9345→0.9353. Wired into the
pipeline it was noise-neutral-to-negative (post-finalize seam −1.6pp because
training undid it; post-stabilization seam within run noise). The feature was
reverted.

**What the experiment actually exposed:** the genuine-cascade model carries a
~3.7pp val→test generalization gap (val 0.972 vs test 0.9345 on the same
weights) — the residual cascaded deficit is OVERFITTING of the
deployed-dynamics training phase, not a deployment-semantics seam. Levers for
a future round: stronger KD weighting / label smoothing / training noise in
the FT phase, weight averaging (SWA) across stabilization, or best-state
selection on a larger validation sample.



### 9.8 What landed vs what remains (cascaded ttfs_cycle)

**Landed (all well-tested):** effective-bias SSOT (§9.1, NF now lossless under
the cascade), build-token stale-mapping guard (§9.2), genuinely-gradual ramp
(§9.3), multi-round stabilization with LR-refind on the deployed forward
(stabilization's first round was training the genuine cascade at the *proxy's*
cached LR). Rejected with evidence: genuine-gradual ramp (§9.5 margin oracle),
theta calibration (§9.7 selection bias), pretraining best-checkpoint (regressed
pretraining ~0.9pp on the annealing schedule — reverted).

**Controlled result (same 0.9824 pretrain, resumed):** offload cascaded sim
0.916 → **0.954**; the NF-destruction the user reported (0.9468→0.9199) is
**eliminated** (NF lossless). Subsume cascaded FT 0.9457 → 0.9616.

**Remaining gap to ≥0.97 (cascaded only):** the proxy↔genuine single-spike
cascade is a different, worse-conditioned transfer function at S=4 (4 timing
levels); the value-domain proxy the ramp optimizes does not transfer past
~0.95–0.96, and the genuine-cascade model carries a ~3.7pp val→test
generalization gap (overfit of the deployed-dynamics stabilization). These are
algorithmic, not seams. Future levers (untried): dithered thresholds /
training noise through the genuine cascade, SWA over stabilization rounds,
larger S. Synchronized ttfs_cycle (regression config) already clears 0.97 —
its deployed dynamics are analytically differentiable, so it has no proxy gap.
