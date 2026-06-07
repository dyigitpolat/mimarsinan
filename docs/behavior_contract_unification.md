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
