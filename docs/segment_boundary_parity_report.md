# Segment-Boundary Parity — Change Report & Cross-Mode Genericity Assessment

**Scope:** the work landed in commits `01f03d1 … 751b3fc` (on top of `e51a1ab`).
**Goal of the work:** make torch-side (NF, "normalization fusion") spike behaviour
match the simulators (HCM / SANA-FE / Lava / nevresim) at neural-segment boundaries,
for the **LIF** spiking mode, starting from the `mlp_mixer_core` (mmixcore) template.

This report (1) inventories the changes and the architectural edit surface, (2) assesses
whether each change is **generic** across the other deployment modes — `rate` and the
**TTFS** family (`ttfs`, `ttfs_quantized`, `ttfs_cycle_based`) — and (3) gives a
well-engineered action plan for the gaps, including the Case-B subsumed-boundary clamp.

---

## 1. Executive summary

- The work is **correct and validated for LIF**, and most of it is **mode-agnostic mapping
  infrastructure** (provenance, subsume/offload) or **LIF-local** changes that bring LIF up
  to parity with what the TTFS path **already did**.
- The TTFS family is **not broken** by these changes: it has **parallel, independent, and
  already-segment-aware** implementations (`TTFSActivation`, `TTFSSegmentForward`,
  `TtfsAnalyticalExecutor`). The "SSOT" is therefore a **single source of truth for the
  LIF/rate boundary**, not yet a single source across *all* modes.
- **One genuine cross-mode gap:** the **negative-value shift** is gated LIF-only, yet TTFS
  has the *same* underlying problem (a negative-producing ComputeOp boundary is clamped and
  lost). This is the priority action item.
- **Two smaller items:** (a) the **Case-B** subsumed-encoding clamp is unshifted in *both*
  LIF and TTFS (rare topology, fails loud where unsupported); (b) **subsume/offload is
  implemented mode-agnostically but only tested for LIF**.

---

## 2. Change inventory & architectural edit surface

Five commits, ~1.3k LOC across 41 files. Grouped by concern:

### 2.1 Root-cause fix — signed integrate-and-fire LIF (`01f03d1`)
- **`models/nn/activations/lif.py`**, **`spiking/spike_trains.py`** — removed `F.relu` from the
  LIF membrane charge (`x/scale`, no relu) in `_forward_single_step`, `_spikes_and_scale`,
  `lif_spike_train`. The membrane may now go negative and recover, matching the chip / HCM
  (`memb += W@s + b`). This is what made per-neuron NF↔HCM parity *exact* (the relu-IF
  diverged 524/600 on mixed-sign inputs; signed-IF: 0/300).

### 2.2 SSOT segment-boundary module (`01f03d1`)
- **`spiking/segment_boundary.py`** (new; renamed from `segment_encoding.py`) — owns boundary
  **encode** (`encode_segment_input`, `encode_compute_boundary`) and **decode**
  (`decode_segment_output` = `counts/T`, `decode_segment_output_torch`), plus the declarative
  `SegmentBoundary` descriptor and `BoundaryConfig`.
- **`chip_simulation/hybrid_run/hybrid_semantics.py`**, **`sanafe/runner/*`** — the decode is
  re-exported / consumed here so HCM/SANA-FE/Lava/nevresim route through the SSOT. Legacy
  `segment_encoding` shim, `lif_inter_stage_from_spike_counts`, `SegmentEncodingConfig`, and
  the `build_/emit_` aliases were **removed** (no back-compat).
- **`spiking/__init__.py`** — public surface updated to the SSOT names.

### 2.3 Per-neuron provenance (`01f03d1`)
- **`mapping/packing/softcore/{soft_core,hard_core_mapping}.py`**, **`mapping/ir/legacy_convert.py`**
  — `merge_softcore_into` stamps `perceptron_index`, `perceptron_output_slice`, `psum_role`
  on each placement; `placement.ir_node_id == IR NeuralCore.id`. Enables per-neuron NF↔HCM
  reconstruction.

### 2.4 Segment-aware NF forward (`472d207`)
- **`spiking/chip_aligned_nf.py`** — `chip_aligned_segment_forward` (replaces the deleted
  `chip_aligned_nf_forward`). Walks the mapper exec graph keeping a per-cycle spike `train`
  (intra-segment perceptron cascade) and a `rate` (`count/T`); **each ComputeOp runs once on
  the decoded rate** (not per-cycle on spikes). Fixes the old forward, which ran mid-graph
  ComputeOps per-cycle on spikes — correct only for *linear* ops (which is why mmixcore
  worked; a `LayerNorm` boundary diverged NF-vs-HCM by 0.5).
- **`tuning/tuners/lif_adaptation_tuner.py`** — the live NF probe now installs the segment-aware forward.

### 2.5 Subsume vs offload switch (`06ad0b7`)
- **`torch_mapping/encoding_layers.py`** — `mark_encoding_layers(repr, *, placement)`:
  `"offload"` clears `is_encoding_layer` so the encoding perceptron maps **on-chip** (NeuralCore)
  instead of a host ComputeOp; the flow's existing uniform/TTFS segment-input encode handles
  its raw input.
- **`torch_mapping/converter.py`**, **`pipeline_steps/config/torch_mapping_step.py`**,
  **`config_schema/{defaults,display_view_meta}.py`**, **`gui/static/{wizard.html,js/wizard.js}`**
  — `encoding_layer_placement` config plumbed end-to-end incl. GUI.

### 2.6 Negative-value shift (`751b3fc`, superseding the initial `26af464/88c3a80/b806d2d`)
- **`mapping/support/neg_shift_bias.py`** — `apply_negative_value_shifts(model, calib_x, T)`
  (pre-mapping): calibrate per-ComputeOp output minima via the NF forward, walk structural
  nodes (executing them on the shift to align it) to the consuming perceptron, bake its bias
  **once** (`B' = B − W·s`, both NF + HCM inherit), tag `ComputeOpMapper._negative_shift`.
  `transfer_negative_shifts_to_ir` (mapper→IR) + `propagate_negative_shifts_to_hybrid`
  (IR→hybrid `node_output_shifts`) carry it to HCM.
- **`spiking/chip_aligned_nf.py`** — NF adds the shift to `node_rate[ComputeOp]`.
- **`models/spiking/hybrid/{rate_forward,lif_step}.py`**, **`mapping/packing/hybrid_types.py`** —
  HCM applies `node_output_shifts` in `_apply_input_shifts`.
- **`pipelining/.../soft_core_mapping_step.py`**, **`pipelining/core/simulation_factory.py`** —
  pipeline wiring (opt-in `negative_value_shift`).
- **`config_schema/*`, `gui/*`** — config + GUI toggle.

### 2.7 Tests (all commits)
`test_lif_step_vs_activation_parity` (dynamics linchpin), `test_segment_boundary[_encode]`,
`test_nf_hcm_per_node_spike_parity_mmixcore` (exact per-neuron), `test_nf_hcm_multisegment_parity`
(LayerNorm decode/re-encode + negative-shift consistency), `test_encoding_offload`,
`test_negative_shift_flow`, `test_neg_shift_bias`.

---

## 3. Cross-mode genericity assessment

Verdict legend: **GENERIC** (mode-agnostic) · **LIF-local / TTFS-parallel-OK** (LIF change, TTFS
already has an independent correct path) · **GAP** (LIF-only, TTFS has the same problem unaddressed).

| # | Change | LIF | rate | TTFS family | Verdict |
|---|--------|-----|------|-------------|---------|
| 1 | Signed-IF (relu removal) | fixed | n/a (analytical) | TTFS uses `TTFSActivation`, **already signed** (relu only on the encoding *charge*, not the membrane) | **LIF-local / TTFS-parallel-OK** |
| 2 | SSOT `segment_boundary` (encode/decode) | uses it | uses it | TTFS uses `ttfs_segment_forward.py` + `ttfs_executor.py`; decode contract (`count/T`, `store_neural_segment_output`) **shared via `hybrid_semantics`**, but the **encode side is separate** | **LIF-local / TTFS-parallel-OK; SSOT not yet unified** |
| 3 | Segment-aware NF forward | new | n/a | `TTFSSegmentForward` was **already segment-aware** (decodes boundary values, runs ComputeOps on values, never per-cycle on spikes) | **LIF-local / TTFS-parallel-OK** (Round 3 brought LIF *up to* TTFS's design) |
| 4 | **Negative-value shift** | done | not wired | **same problem, no mechanism** — TTFS clamps negative ComputeOp outputs at the fire-time encode (`relu(x)/scale .clamp(0,1)`), unrecoverable | **GAP** |
| 5 | Subsume vs offload | done+tested | works | mode-agnostic; offload uses `spike_generation_mode="TTFS"` — **plausibly works, untested** | **GENERIC (TTFS untested)** |
| 6 | Per-neuron provenance | used | inert | mapping-only metadata, mode-independent | **GENERIC** |
| 7 | Case-B subsumed-encoding clamp | unshifted | unshifted | TTFS bakes the clamp into the encoding node; **also unshifted** | **GAP (both modes, rare)** |

### Why TTFS is largely "parallel-OK", not broken
- **`TTFSActivation`** (`models/nn/activations/ttfs_spiking.py`) integrates ramps into a signed
  membrane; the only `relu` is on the *value→spike-time charge* (TTFS encodes a non-negative
  rate by design). So the LIF relu-bug never applied to TTFS.
- **`TTFSSegmentForward`** (`models/spiking/training/ttfs_segment_forward.py`) is explicitly
  segment-aware: *"any region node consumed by a value node (or the output) is decoded … the
  encoding layer consumes the decoded values of upstream segments, never their per-cycle
  spikes."* This is exactly the decode→compute→re-encode contract Round 3 added to LIF.
- The **decode/store contract** (`decode_segment_output` = `count/T`,
  `store_neural_segment_output`, `is_ttfs_spiking_mode`) is already shared in
  `hybrid_semantics.py` — TTFS uses ramp/activation inter-stage there.

**Consequence:** the "single source of truth" today is genuinely single for **LIF + rate +
all sims' decode**, but the **encode side and the cycle-accurate NF reference are two parallel
implementations** (`chip_aligned_segment_forward` for LIF, `TTFSSegmentForward` for TTFS).
That is acceptable and correct, but it is *not* the fully unified boundary model the
`SegmentBoundary` descriptor was scaffolded for.

---

## 4. Identified gaps & action plan

### GAP-1 (Priority 1) — Negative-value shift is LIF-only; TTFS has the same loss
**Problem.** `SoftCoreMappingStep._apply_negative_value_shift_compensation` early-returns unless
`spiking_mode == "lif"` (`soft_core_mapping_step.py:210`). For TTFS, a negative-producing
ComputeOp boundary (LayerNorm/GELU/signed Linear → segment) is clamped to `[0,1]` at the
fire-time encode (`ttfs_spiking.py` charge `relu(x)/scale .clamp`; `ttfs_encoding.py`
`np.clip(rate,0,1)`) → the negative information **fires immediately or never, unrecoverably**.

**Why it's not a gate-flip.** The shift *value* is mode-agnostic (`s = max(0,−min F)`), and the
linear bias compensation `B' = B − W·s` is valid for TTFS's linear membrane integration. But the
**wiring is mode-specific**:
- **Calibration** must run via `TTFSSegmentForward` (TTFS produces the boundary values), not
  `chip_aligned_segment_forward`. → factor the calibration recorder behind a small interface so
  either forward can drive it.
- **Encode-side application** must add the shift in the TTFS encode path (the TTFS analogue of
  NF's `node_rate[ComputeOp] += s` and HCM's `_apply_input_shifts`). Confirm the TTFS analytical
  executor / `node_output_shifts` consumption exists for TTFS or add it.
- **Bias bake** reuses `apply_negative_shift_bias` (perceptron-level, mode-agnostic) — verify the
  TTFS consumer's effective-weight/threshold convention matches `PerceptronTransformer`.

**Action.**
1. Introduce a calibration abstraction `calibrate_compute_op_minima(model, x, T, *, forward_fn)`
   in `neg_shift_bias.py`; have LIF pass `chip_aligned_segment_forward` and TTFS pass
   `TTFSSegmentForward`-driven calibration.
2. Generalise `apply_negative_value_shifts` to call it; keep the perceptron bake + IR/hybrid
   propagation unchanged.
3. Add TTFS encode-side shift application (mirror `node_rate[ComputeOp] += s`) and verify HCM
   TTFS reads `node_output_shifts` (the TTFS executor path `ttfs_step.py` / `ttfs_executor.py`).
4. Relax the gate to the TTFS family; **add `test_negative_shift_ttfs_nf_hcm` (LayerNorm
   2-segment, TTFS)** mirroring `test_nf_hcm_multisegment_parity`.
5. **Interim safety (ship first):** in `_apply_negative_value_shift_compensation`, if
   `negative_value_shift` is requested for a non-LIF mode, **raise a clear `NotImplementedError`
   / warn** rather than silently no-op (today it silently returns).

### GAP-2 (Priority 2) — Unify the boundary SSOT across modes (or document the split)
**Problem.** `chip_aligned_segment_forward` (LIF) and `TTFSSegmentForward` (TTFS) independently
re-implement the same decode→compute→re-encode walk; `SegmentBoundary` is consumed by neither.

**Action (choose one).**
- **(2a) Unify** — extract the exec-graph walk + node-classification (`spike-producing` vs
  `value-producing`, encoding-layer subsume) into a shared driver parameterised by a
  per-mode "neuron step" (LIF signed-IF cascade vs TTFS ramp-fire) and a per-mode encoder
  (uniform vs TTFS). Drive it from the `SegmentBoundary` descriptor. Highest cohesion; larger.
- **(2b) Document** — if the parallel paths are intentional, add an `ARCHITECTURE.md` note that
  the SSOT covers LIF/rate encode + all-mode decode, and TTFS encode lives in
  `ttfs_segment_forward.py`. Lower effort; keeps the split explicit.

Recommend **2b now, 2a when a third boundary-bearing mode appears**.

### GAP-3 (Priority 2) — Subsume/offload untested for TTFS
**Problem.** `mark_encoding_layers(placement="offload")` is mode-agnostic, but
`test_encoding_offload` only covers LIF.

**Action.** Add a TTFS offload test: assert the encoding perceptron maps on-chip and that
TTFS NF == TTFS HCM under offload (mirror `test_offload_runs_through_hcm_and_matches_subsume`
with `spiking_mode="ttfs"`, `spike_generation_mode="TTFS"`).

### GAP-4 (Priority 3) — TTFS multi-segment / non-linear-ComputeOp parity is untested
**Problem.** `TTFSSegmentForward` is designed segment-aware, but no test exercises a non-linear
mid-graph ComputeOp (LayerNorm) through TTFS NF↔HCM (the analogue of
`test_nf_hcm_multisegment_parity`, which only exists for LIF).

**Action.** Add `test_ttfs_nf_hcm_multisegment_parity` (LayerNorm 2-segment, TTFS) to lock the
TTFS boundary contract the way LIF's is locked.

---

## 5. Case B — the subsumed-encoding-boundary clamp (applies to LIF *and* TTFS)

**What it is.** When a *bare, unbounded* `Linear`/`Conv` `ComputeOp` feeds an **encoding
perceptron that is subsumed host-side**, the encoding perceptron's input is assembled and
**clamped to `[0,1]`** *without* the negative-shift:
- **LIF:** `segment_boundary._gather_op_input_train` — `raw_input[:, idx].clamp(0,1)` and
  `rate[:, idx].clamp(0,1)` (the subsumed-boundary input path), **distinct** from the segment-input
  path `_apply_input_shifts` and the NF `node_rate[ComputeOp] += s` where the shift *is* applied.
- **TTFS:** the equivalent clamp is baked into the encoding node
  (`TTFSActivation(encoding=True)`: `relu(x)/scale .clamp(0,1)`).

**Why it's currently acceptable.**
- The **common** encoding inputs are already safe: raw network input is in `[0,1]`, and a
  *bounded* upstream op (the normal case) doesn't go negative.
- The negative-shift's structural walk **fails loud** on the genuinely unsupported topology
  (a ComputeOp feeding *another* ComputeOp → `NotImplementedError` in `_bake_consumer_perceptrons`),
  so nothing is silently wrong.
- A *bare unbounded Linear/Conv directly before an encoding perceptron* (Case B) is an uncommon
  shape (the Linear/Conv is normally fused into the perceptron itself).

**Risk if left as-is.** If such a model is deployed, the subsumed encoder's negative input is
clamped and lost — silently (no error), because this path is neither the ComputeOp→segment path
(shift-wired) nor the fail-loud ComputeOp→ComputeOp path.

**Action (Priority 2).**
1. **Make it loud, not silent.** In `_gather_op_input_train` (and the TTFS encoding node),
   detect a subsumed encoder whose upstream is an *unbounded* ComputeOp producing negatives
   (reuse `_wraps_unbounded_raw_linear_or_conv` + the calibrated minima) and **raise/warn** that
   negative-shift does not yet cover the subsumed-encoder input boundary.
2. **Then wire it (optional).** Extend `apply_negative_value_shifts` to treat the
   *subsumed-encoder input* as a shiftable boundary: calibrate the encoder's input min, shift it
   in `_gather_op_input_train` (LIF) / the encoding node (TTFS), and bake the **encoding
   perceptron's** bias (it *is* a perceptron, so `apply_negative_shift_bias` applies directly).
   This closes Case B symmetrically with the ComputeOp→segment case.
3. **Test.** `input → bare Linear (signed) → encoding perceptron → …`, assert NF == HCM with
   shift and recovery, for LIF and TTFS.

---

## 6. Prioritised action summary

| Priority | Item | Effort | Files |
|---|---|---|---|
| **P1** | Negative-shift for TTFS (calibration abstraction + encode-side + gate + test) | M–L | `neg_shift_bias.py`, `soft_core_mapping_step.py:210`, `ttfs_step.py`/`ttfs_executor.py`, new test |
| **P1 (ship first)** | Fail loud when `negative_value_shift` requested for non-LIF (no silent no-op) | S | `soft_core_mapping_step.py:210` |
| **P2** | Case-B: make the subsumed-encoder negative clamp loud; then wire the shift | S→M | `segment_boundary._gather_op_input_train`, `ttfs_spiking.py`, `neg_shift_bias.py` |
| **P2** | TTFS offload test | S | new `test_encoding_offload` TTFS case |
| **P2** | Document the LIF/TTFS boundary split (or unify) in `ARCHITECTURE.md` | S | `spiking/ARCHITECTURE.md` |
| **P3** | TTFS multi-segment non-linear-ComputeOp parity test | S | new test |

**Bottom line.** The delivered work is correct and complete for **LIF** and **rate**, and is
**safe** for TTFS (parallel correct paths; nothing regressed). The one substantive cross-mode
debt is the **negative-value shift**, which is LIF-only by an explicit gate while TTFS has the
same loss — the recommended first step is to make that gate **fail loud** for non-LIF, then
generalise the (already mode-agnostic) shift/bake behind a per-mode calibration + encode hook.

---

## 7. Round status addendum (2026-06-04) — gaps closed

All actions from §4–§6 landed (TDD, full suite green vs. pre-existing baseline):

| Item | Status | Where |
|---|---|---|
| **GAP-2 (2a) — unified SSOT driver** | **Done.** One mode-parameterized walk: `spiking/segment_forward.py` (`SegmentForwardDriver`) + `spiking/segment_policies.py` (`LifSegmentPolicy` / `TtfsSegmentPolicy` / `AnalyticalSegmentPolicy`) + shared classification in `spiking/segment_partition.py` (every host ComputeOp is a value boundary, matching HCM). `chip_aligned_segment_forward` and `TTFSSegmentForward` are thin wrappers; the dead `SegmentBoundary` descriptor was deleted. Locked by atol-0 characterization tests + the existing exact-parity suite. | `tests/unit/spiking/test_segment_forward_driver.py` |
| **GAP-1 — negative shift for the TTFS family** | **Done.** `apply_negative_value_shifts(..., forward_fn)` + `calibration_forward_for_mode` (LIF chip-aligned / TTFS segment driver / analytical driver; `NotImplementedError` otherwise — the silent LIF-only gate is gone). Analytical HCM applies `node_output_shifts` via `apply_input_shifts_numpy` in `run_ttfs_contract_neural_stage`; cascaded `ttfs_cycle_based` inherits `_apply_input_shifts` (locked by test). Note: the analytical contract path consumes boundaries **linearly**, so there the shift is an exact identity whose purpose is the spike-train backends' `clip(rate,0,1)` encode domain. | `test_negative_shift_ttfs_nf_hcm.py`, `test_negative_shift_cascaded_ttfs.py`, `test_negative_shift_gate.py` |
| **Case B — subsumed-encoder clamp** | **Done (LIF).** `_gather_op_input_train` lifts producer rates by `node_output_shifts` before its clamps and **warns once per op** on unshifted negatives; the encoder bake needed no relaxation (at calibration time the encoder is still a `PerceptronMapper`). The NF LIF policy now mirrors HCM's two encoder outputs (host rate value + cycle-emitted train for plain LIF; Conv wrappers stay uniform), making Case-B parity exact. Leading-dim-split producers (`{name}_col{i}`) get per-column shift rows in the IR transfer. New fail-loud: a baked subsumed encoder feeding a ComputeOp. TTFS subsume needs no Case-B fix (host encoder consumes values linearly). | `test_case_b_subsumed_encoder_shift.py` |
| **GAP-3 — TTFS offload** | **Done.** Analytical TTFS HCM: offload == subsume exactly. | `test_encoding_offload.py::test_offload_runs_through_hcm_and_matches_subsume_ttfs` |
| **GAP-4 — TTFS multisegment parity** | **Done, including the fix it exposed.** Cascaded NF fed *constant ideal values* to non-encoding segment entries while HCM TTFS-encodes the boundary; `TtfsSegmentPolicy` now feeds those entries single-spike TTFS trains (the rising edge of HCM's latched encode — ramp reconstruction equals the greedy core's latched integration). Cascade NF == HCM across a LayerNorm boundary **exactly**, with and without the shift. Analytical (`ttfs_quantized`) multisegment parity is covered by the GAP-1 tests. | `test_ttfs_nf_hcm_multisegment_parity.py` |

Residual notes:
- Continuous `ttfs` NF (staircase surrogate) vs the continuous executor remains a
  known non-goal for exact parity (the pipeline already warns); shift calibration
  works for it regardless.
- SANA-FE / nevresim / Lava do not yet read `node_output_shifts` in their own
  stage loops — backend-side shift parity is the natural next round.
- The boundary single-spike encode at non-encoding cascade entries is a hard
  (non-differentiable) encode; gradient flows through encoding entries only.

---

## 8. `ttfs_cycle_based` fine-tune ↔ deploy parity (follow-up, same day)

A `mnist_mmixcore_ttfs_cycle_60` run failed the SCM gate (HCM 0.594 vs reported 0.943).
Forensics on the saved artifacts:

- **NF↔HCM was already exact** with the unified driver (`SegmentForwardDriver` == HCM
  bit-for-bit); the old `TTFSSegmentForward` had diverged (max|Δ|=0.365), so the driver
  *improved* it. The gate failure was **not** a segment-boundary regression.
- **Root cause:** `TTFSActivation`'s interior-cascade forward (`cycle_accurate=True`,
  single-spike ramp) does **not** equal its analytical staircase forward — node-level
  max|Δ|≈0.94 through one layer, compounding to a 0.90→0.62 gap across the mixer. For
  **LIF** these two match per-neuron (the parity guarantee), so LIF trains on one and
  deploys the other safely; TTFS-cycle could not.
- **Tuner asymmetry vs LIF:** `LIFAdaptationTuner._after_run` re-installs the genuine
  cycle-accurate forward (`_ChipAlignedNFForward`) and keeps it through the committed
  metric + recovery + every downstream step. `TTFSCycleAdaptationTuner._finalize`
  **stripped** the genuine `_SegmentSpikeForward` before the committed metric, so
  fine-tuning recovered/committed the analytical staircase (0.90) while only HCM ran the
  genuine cascade (0.62) — the gap surfaced at SCM.

**Fix:** `TTFSCycleAdaptationTuner` now mirrors LIF — `_after_run` re-installs
`_SegmentSpikeForward` after subsuming the decorators and **keeps it installed** through
`_ensure_pipeline_threshold` (recovery) and all downstream steps, so fine-tuning
validates/recovers/commits the exact deployed single-spike cascade. A scaled
`ttfs_cycle_based` run is now metric-consistent end-to-end (Fine-Tuning == SCM ==
Simulation) and no longer trips the gate. Locked by
`test_ttfs_cycle_adaptation_step.py::TestFinalState::test_genuine_spike_forward_persists_after_step`
(+ the stale `TTFSCycleActivation` target expectation corrected to the genuine
`TTFSActivation` node). The cascade's accuracy ceiling for a given model/T is now
honestly reported at fine-tuning instead of deferred to SCM — reaching high accuracy is
a modeling concern (T/architecture/KD), no longer a hidden parity bug.

## 9. Synchronized + offload + off-grid input (2026-06-06) — SANA-FE parity gate

A `mnist_mmixcore_ttfs_cycle_30_offload_sync` deployment failed the SANA-FE parity gate
at stage 0 (`ref=0.5 actual=0.75`, exactly one activation level). Two independent root
causes, both fixed:

- **Bank weight-slice transpose (all TTFS modes).** `segment_ttfs_arrays_from_mapping`
  sliced `register_weight_bank`'s `(axons, neurons)` storage with the ranges swapped —
  a no-op for square cores, but it zeroed every weight beyond `used_neurons` on
  non-square bank-backed cores (exactly the offloaded conv). Fixed
  (`bank_mat[axon_range, neuron_range].T`); locked by
  `test_segment_arrays_bank.py`. Also fixed two pre-existing
  `test_hardcore_ttfs_equivalence` failures on non-square cores.
- **Reference must see the TTFS-grid-quantized stage input (synchronized only).** The
  synchronized cycle soma reconstructs V from single-spike *timings*, so every neural-
  stage input axon is quantized to the 1/S grid on the wire
  (`q(x) = (S − round(S(1−clamp(x))))/S`, 0 when the spike falls off-grid). The
  analytical HCM reference fed the continuous raw value to the offloaded encoding conv
  and saw information the hardware cannot carry. Fix: `ttfs_input_grid_quantize`
  (encode→decode round-trip SSOT in `ttfs_encoding.py`) applied to the **assembled,
  post-`node_output_shifts` stage input** in the shared contract runner
  (`run_ttfs_contract_neural_stage(quantize_input_to_ttfs_grid=…)`), gated on
  `is_synchronized_ttfs`. The SANA-FE runner reuses the contract's `seg_input` for its
  spike encode (idempotent round-trip), so reference and wire agree by construction.
  The seam matters: quantizing the *raw* input before shifts leaves ±1-level diffs
  (the earlier forensics' "residual ~6"); quantizing after shifts — where the hardware
  encodes — resolves the gate completely (contract atol=1e-12 passes on the real model).
- **Cascaded needs none of this**: both its reference and its soma reconstruct the same
  ramp from the same quantized spike timing (`_boundary_single_spike_train` /
  `ramp_current`), so the continuous-vs-quantized asymmetry is synchronized-only.

**Known structural sensitivity (documented, not a defect).** On the real S=30 weight-
quantized model, 12% of fire events sit *exactly* on an integer `ceil(S(1−V/θ))`
boundary (min non-zero distance 3.2e-4 — bimodal). Parity holds because the soma's
arrival-order float64 sum and the reference's BLAS dot are currently bit-identical; an
FP-noise source on either side (different BLAS reduction order, GPU compute — see the
`CUBLAS_WORKSPACE_CONFIG` note in `run.py`) would flip ties by ±1/S and trip the gate
loudly (1/S ≫ tolerances). Debug aid: `MIMARSINAN_TTFS_CYCLE_TRACE=<path>` makes the
synchronized soma dump per-event CSV (input decode, dropped-late deliveries, hexfloat
fire-step resolution). Locked by `test_sanafe_ttfs_cycle_offgrid_parity.py` and
`test_ttfs_contract_input_quantize.py` / `test_ttfs_input_quantize.py`.
