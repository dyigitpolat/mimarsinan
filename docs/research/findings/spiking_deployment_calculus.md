# The spiking deployment calculus: one typed model of every conversion, and the defect taxonomy that closes it

**Scope.** This memo is the unification layer above the five deployment
theories this codebase has produced — the LIF hop commutation theorem
(`lif_deployment_exactness.md`), the synchronized-TTFS composition theorem
(`sync_deployment_exactness.md`), the conversion boundary algebra
(`conversion_boundary_algebra.md`), the two exact-QAT programs
(`lif_exact_qat_program.md`, `ttfs_exact_qat_program.md`), and the
measurement-numerics rules (`numerical_boundary_consistency.md`). It defines
the objects those memos are instances of, states the single exactness
condition every deployment must satisfy, gives the **defect taxonomy** and the
**decision procedure** that classifies any measured train↔deploy deviation,
compiles the **contract registry** (every precondition, code-anchored, with
its enforcement status), and pre-registers the falsifiable predictions for the
open ViT-family gaps. Nothing here replaces the instance memos; every claim
cites them. The program that consumes this memo is
`~/.claude/plans/spiking_deployment_calculus_program.md`.

**Method mandate (operating law).** Theory precedes measurement: an assay may
run only against a prediction registered here (or in a successor section)
with an expected value and tolerance. The empirical ladder is strict:
per-edge isolated assay → MNIST-scale tier-0 replication → tier-1 full run.
A fix may be built only for a defect confirmed in isolation, and it lands at
the mode-generic SSOT with contract tests over all deployable modes.

---

## 1. Objects

### 1.1 Typed representations (what an edge carries)

Every edge of the deployment graph carries exactly one representation type:

- **V** — host value: float, signed, arbitrary scale. The domain host
  parameters were trained on (LN γ/β, attention projections).
- **R(κ, T)** — rate code: the buffer stores `counts/T ∈ [0,1]` on the grid
  `1/T`; the value is `rate · κ`. The analytic projection of S.
- **S(κ, T, w)** — timed spike train: the cycle-domain realization of R
  inside window `w = [L, L+T)`; carries timing, hence back-loading and
  window physics that R cannot express (`lif_deployment_exactness.md` §1-2).
- **C(κ, S, g)** — TTFS time code: one spike at
  `k = round(S·(1 − clamp(v/κ,0,1)))` (`models/spiking/wire_semantics.py:132-143`),
  latency group `g`; the synchronized schedule serializes groups
  (`sim_time = S × groups`).

### 1.2 Gauges (the per-edge scale/offset state)

Per edge, the gauge triple **(κ_fold, κ_buf, σ)**
(`conversion_boundary_algebra.md` §1):

- **κ_fold** — the consumer-fold currency: the scale the consumer's weight
  fold multiplies back in (`W~ = per_input_scales · W / θ_out`,
  `transformations/perceptron/perceptron_transformer.py:106-113`). ONE
  propagation law, two implementations under a permanent nodewise-equality
  contract: IR-side `compute_node_output_scales`
  (`mapping/support/activation_scales.py:66-88`) ↔ NF-side
  `read_boundary_out_scales` (`spiking/scale_aware_boundaries.py:20-40`).
- **κ_buf** — the buffer gauge: what the runtime buffer stores
  (neural = θ_out, host value-op = 1); the entry divisor is the DERIVED view
  `κ_fold/κ_buf` (`spiking/segment_boundary.py:51-88`).
- **σ** — the signed-seam lift, value-domain. Exactly two application
  sites exist (producer forward lift + consumer baked bias); a third is a
  design error (boundary-algebra P3b refutation: an explicit σ·κ conversion
  double-counts — the gauge conversion lives in the fold, once).

### 1.3 Kernels as twin pairs

Every node is a **twin pair** (value twin ⟦K⟧, temporal twin K^τ) plus a
**commutation contract** C(K) = (preconditions P, defect bound δ when P
holds). Canonical instances:

| kernel | value twin | temporal twin | contract |
|---|---|---|---|
| LIF hop | `lif_count_staircase` (`wire_semantics.py:97-121`): `θ·clamp(F(T·z/θ),0,T)/T`, `F=floor` for `<=`, `ceil−1` for `<` | `lif_fire_and_reset` cycle loop (`models/nn/lif_kernels.py`) | Theorems 0-2 (`lif_deployment_exactness.md` §2); exact off integer ties (\|Δcount\| ≤ 1 on the tie set) |
| ttfsq / sync hop | `ttfs_quantized_staircase` / strict variant (`wire_semantics.py:17-83`), comparator half-step in the ladder (E3) | single-spike fire at the ladder crossing | sync Theorem 1 + P1–P6 (`sync_deployment_exactness.md` §1-2) |
| armed host op | plain forward + `output_value_offset` (`mapping/mappers/compute_op_mapper.py:144`) | `ScaleNormalizingWrapper.forward`: module(**kwargs) → select `output_index` → +offset → /scale (`mapping/support/compute_modules.py:67`) | wire twin ≡ value twin on kwargs/tuple ops (locked, `test_scale_normalizing_wrapper.py`) |
| entry | `ChipInputQuantizer`: `κ·grid_T(clamp(v/κ))` (`models/nn/activations/autograd.py:197-204`) | wire encode of the same round | I1: trained entry == deployed round-trip (T5 lock) |

### 1.4 Transcodes (typed morphisms between representations)

- **encode** `E(v) = grid_T(clamp(v/κ_fold + σ/κ_fold, 0, 1))` — SSOT
  `normalize_boundary_value` (`spiking/compute_boundary.py:18-35`) +
  `uniform_spike_train`; the walk mirror is `LifSegmentPolicy.train_of`
  (`spiking/segment_policies.py:95-110`).
- **decode** `D(r) = r · κ_buf` (gather-lift).
- **entry divisor** `κ_fold/κ_buf` (derived, never a third table).
- **σ-bake** `B' = B − W_eff·(σ/κ)` — entries via
  `apply_negative_shift_bias`; host bias carriers via
  `_bake_shift_into_host_bias` (Linear, packed MHA `in_proj_bias` ONCE —
  `tuning/orchestration/signed_seam_install.py:155-174`); the bake law
  `f_baked(v+σ) == f(v)` is a test (`TestHostBiasCarrierBake`).
- **retime** — per-hop uniform re-encode of the window count,
  count-preserving (`round((c/T)·T) = c`), STE backward
  (`spiking/segment_policies.py:145-153`); restores the constant-drive
  precondition per hop. Its deployed twin is per-hop neural-segment mapping
  (`lif_per_hop_retiming`, both sides consult
  `chip_simulation/spiking_semantics.py::lif_per_hop_retiming_enabled`).
- **apply_ttfs** — the TTFS boundary transcode ≡ gather-lift with
  κ_buf ≡ κ_fold (`spiking/segment_policy_ttfs.py:70-95`): TTFS is the
  all-wire-gauge instance of the same algebra.

## 2. The deployment functor and the one exactness condition

For a mode m and placement p, deployment is a functor **F_{m,p}** from the
trained value composition to the temporal composition, inserting transcodes
at every representation change. The pipeline's job (the I1 half) is to make
*training itself* run the image of F: the entry rounds, σ, currencies, and
staircases are installed BEFORE the QAT so the trained function IS the
deployed composition.

**Exactness condition.** `F_{m,p}` preserves the denotation iff
(i) every kernel twin pair commutes on its reachable input set,
(ii) every inserted transcode satisfies its contract, and
(iii) every entry satisfies I1 (train ≡ deploy round).
Then the deployed accuracy equals the trained read *identically* — the
parity-certified-deployment corollary: accuracy is read once in the cheapest
representation whose certificates are green (the SCM identity read,
`pipelining/pipeline_steps/verification/simulation_step.py:18-20`);
simulators are decision-parity probes, never the census.

**The QAT square (A2).** The exact-QAT arms assert one more commutation: the
square (train staircase endpoint → deploy) ≡ (deploy → temporal read). For
LIF it holds under P-L1…P-L6 **plus the retime pairing**, and the pairing is
load-bearing in both directions (staircase-QAT without retime −2.5 pp;
retime without staircase-QAT +0.04 pp — `lif_exact_qat_program.md` §4.5-4.6).
For the sync family, the recipe SSOT states the deploy-side reduction
outright: *synchronized IS ttfs_quantized at deploy — sync-deploy =
ttfsq-deploy + the synchronized-window backend*
(`tuning/orchestration/conversion_policy.py:196`). Consequence: sync-side
seam fixes belong to the shared ttfs transcode, never to a sync special case.

**The measurement functor (new, this session).** Every in-pipeline
"deployed" read is itself an application of F on a clone: the full-transform
probe = `set_blend_rate(1.0)` → finalize rebuild → the finalize forward
(`tuning/orchestration/kd_blend_adaptation_tuner.py:227-259`,
`mbh_ledger.py:106-124`), with probe ≡ deploy by construction
(`_finalize_forward_for` shared by probe and finalize). A defect in this
composition is a **Type-M** defect: the network is fine, the read is wrong.
The t2_04 LIF entry 0.0115 is the open candidate (PR1).

## 3. The defect taxonomy and the decision procedure

Every measured train↔deploy deviation lands in exactly one class:

- **Type-B — convention bias.** A deterministic per-edge map mismatch (wrong
  currency, dropped/doubled σ, wrong clamp domain, window miss, dead relay).
  Composes linearly across seams (and multiplicatively through the following
  nonlinearity) → chance at depth K. Signature: the temporal family agrees
  with itself bit-for-bit and disagrees with the analytic family; the
  agreement-triage law reads **exactly 0.0**. LAW: Type-B is fixed exactly at
  the SSOT — never adapted around, never absorbed into training.
- **Type-G — grid noise.** \|ε\| ≤ κ/(2T) per seam, zero-mean (mid-tread),
  composes ~√K·κ/(2T). A budget, not a bug (percent-level at T=32, K=12).
- **Type-C — capacity/resolution.** The signal band vs the encodable grid:
  signed-band amputation, saturation, level starvation, half-step first
  moments, `eff_levels = S·R_out/κ` decay. Deterministic but
  **trainable-through when installed pre-training** (the σ-in-the-op law:
  ViT AQ 0.06 → 0.596). FIX: currency design (quantile κ — never max; σ in
  the op; per-channel θ; S allocation) + QAT. Post-training σ stays banned.
- **Type-T — training gap.** The trained function is outside the deployable
  family (QAT absent, wrong basin, surrogate mismatch). FIX: the exact-QAT
  arms and recipes; never a deploy-side patch.
- **Type-M — measurement defect.** The read violates the rules of evidence
  (§5) or the measurement functor mis-composes. FIX: the measurement seam;
  numbers read under a broken functor are void, not "low".

**Decision procedure** for a deviation Δ at a site:
1. Validate the read (§5 checklist). Invalid → **M**; re-read before any
   other conclusion.
2. Repeat the read; run the agreement triage: exactly-0.0 agreement between
   representations → deterministic convention (**B**); ~1/num_classes →
   decoupled garbage (also **B**, different site); 0.9x-ish → noise family.
3. \|δ_e\| within the composed grid envelope → **G**; stop.
4. Band/coverage analysis on captured tensors (clamp fraction, negative
   mass, eff_levels, saturation): a coverage term → **C**; distinguish
   installed-pre-training (adapted, budget) from post-hoc (defect).
5. Else compare the trained function family against the deployable family →
   **T**.
A finding that fits no class is a hole in the calculus — extend the memo
before touching code (that is the falsifiability clause working).

## 4. Composition laws (how per-edge defects become network numbers)

With J_h the float Jacobian past hop h:
`Δz_L ≈ Σ_h J_h · diag(θ_h) · ε_h` — first-order drift, then saturation
(`sync_deployment_exactness.md` §3.2). The per-hop local error of a
staircased activation decomposes as dead-zone / granular / overload with
per-channel first moment μ_c computable from one calibration pass (sync §3.1
eq. 3-4). Type-B terms enter with the same sign at every seam (linear); the
half-step fold is the first-moment correction that nearly completes
(residual ≤ 0.07 grid steps); Type-G enters zero-mean (√K). This is the law
the composition validator (PR8) checks quantitatively: ledger-predicted loss
must match the measured genuine read within ~2·SE once all B/M certificates
are green — a mismatch means a missing edge in the model, and the number is
not shippable until explained.

## 5. Rules of evidence (Type-M checklist; any read that skips these is void)

- **fp32 metric grade**: every gate/parity read under `metric_grade_eval`
  (`tuning/orchestration/mbh_ledger.py:22-26`); fp16 autocast reads are RC2
  (`numerical_boundary_consistency.md`).
- **Seeded + isolated**: RNG fork (CUDA included) and validation-cursor
  restore around every measurement (`mbh_ledger._measurement_guard`,
  `mbh_ledger.py:39-51`); the pipeline seeds RNGs (RC1).
- **Tie conventions**: f32/f64 grid ties (sync P5, LIF P-L3) are structural,
  measured (parity gate), never assumed; strict-`<` on the WQ integer
  lattice is V9 (advisory + `DeadRelayError` mapping guard).
- **Probe ≡ deploy**: the measurement functor must run the SAME emitted
  composition as deployment (shared `_finalize_forward_for`; SNW emission
  unwrap for IR, wrapper for torch twins). A probe that rebuilds its own
  variant is not evidence.
- **SE discipline**: n sized to the decision (SE = √(p(1−p)/n)); entry/final
  reported numbers at census n with 2·SE bands.

## 6. The contract registry (code-anchored; status as of 2026-07-19)

Legend: **ENF** enforced (guard/assert/test) · **CFG** config-invariant
(pinned by recipe SSOT) · **BUD** measured budget · **OPEN** known hole.

### 6.1 R/S-edge (LIF hop) contracts

| id | contract | anchor(s) | status |
|---|---|---|---|
| H1 | window coverage: gap ≤ 1 per live edge (latency invariant + prev-cycle buffer reads) | `mapping/latency/chip.py:145-179`, `models/spiking/hybrid/lif_step.py:117-159`; depth-balancing relays `lif_depth_balancing_relays` | ENF (V6 when violated; post-ChipLatency gap≤1 assert = P-L4) |
| H2 | no terminal overshoot / chase completion | Theorem 2(2,3), Theorem 3 | BUD (V4 ≤ 7·10⁻⁴ rate/hop; guard refuted) |
| H3 | subtractive reset (charge conservation, Theorem 0) | `models/nn/lif_kernels.py`; `firing_strategy` guard | ENF (Novena excluded from the exact family; V7 = expectation-repair only) |
| H4 | uniform encoder trains (deterministic 1/S error, not CLT 1/√S) | `chip_simulation/recording/spike_modes.py:32-41` | CFG (V8) |
| H5 | half-step folded exactly once, owned by the QAT under the exact arm | `conversion_policy.py` `lif_half_step_bias`; `mapping/support/bias_compensation.py:73-102`; P-L6 skip+assert at `soft_core_mapping_step.py:428-446` | ENF |
| H6 | back-loading priced or removed: staircase-QAT ⟺ per-hop retime PAIRED | `spiking_semantics.lif_per_hop_retiming_enabled` (both twins consult); `LifSegmentPolicy(retime)` `segment_policies.py:145-153` | ENF at config derivation (the pairing law; V3) |
| H7 | counts-decode readout on BOTH sides (train loss consumes the deployed decode) | P-L2 (R8); `segment_boundary.py:173-178`; logits = integers in [0,T] | ENF |
| H8 | comparator convention carried in the kernel pair, ties on the lattice measured | `WireSemantics.compare_mode` (`wire_semantics.py:162-205`); V9 advisory + `DeadRelayError` | ENF/BUD |

### 6.2 C-edge (ttfsq / sync) contracts

| id | contract | anchor(s) | status |
|---|---|---|---|
| P1 | scalar θ per hop == wire/decode normalizer (per-channel θ only via the theta-aware seams) | sync memo §2/§4.3 (three scalar-collapse seams) | ENF for scalar; per-channel via `ttfs_theta_cotrain` seams |
| P2 | QAT entry-quantizer set == deployed snap seams | `segment_entry_perceptrons` vs `ttfs_executor.py:193-194` | ENF (snaps idempotent on-grid) |
| P3 | half-step folded once; mapping-time +0.5/Tq skipped for exact-QAT; **coverage: fold skips `is_encoding_layer` — wrong under `subsume`** | `bias_compensation.py:80-83`; all-or-none marker `adaptation_manager.py:58-78` | **OPEN under subsume** (parity-invisible, floor-biased entry — Type-C) |
| P4 | WQ bias lattice represents the folded half-step | `normalization_aware_perceptron_quantization.py:67-83` | BUD (±10-70% worst hops; train/deploy bit-consistent) |
| P5 | dtype tie stability f32↔f64 | parity gate (the monitor) | BUD (0 flips measured; no structural guarantee — keep measured) |
| P6 | readout stays float (host ComputeOp) | falls out of `linear_mixin.py:48-59`; **not a contract anywhere** | **OPEN** (a vehicle whose last layer has an activation silently eats −10 pp at S=4 — needs a fail-loud) |
| P7 | apply_ttfs ≡ gather-lift at κ_buf ≡ κ_fold; **non-homogeneous re-encoded host ops under TTFS wires are NOT armed today** | `segment_policy_ttfs.py:70-95`; arming mode-gated to rate/LIF (`requires_ttfs_firing` gate in `value_domain.py`) | **OPEN** (boundary-algebra §10b cell 5: torch↔deployed 0.0000 on the torch-mixer; the F2 program target) |

### 6.3 Seam contracts (any mode)

| id | contract | anchor(s) | status |
|---|---|---|---|
| B1 | I1 entry identity: trained entry == deployed round (same κ, σ, grid, rounding) | `ChipInputQuantizer` install at the AQ seam (`lif_exact_qat.py:110-174`); T5 lock | ENF |
| B2 | ONE κ_fold propagation, nodewise-equal across IR/NF tables | `activation_scales.py:66-88` ↔ `scale_aware_boundaries.py:20-40`; contract test | ENF |
| B3 | κ_buf is a derived view (divisor = κ_fold/κ_buf); no third table | `segment_boundary.py:51-88` | ENF |
| B4 | arming law: non-homogeneous AND re-encoded host op ⇒ owns_domain (SNW), even at uniform scalar scales | `mark_wire_value_ops` + wrap policy (`mapping/support/value_domain.py`, `per_source_scales.py`) | ENF for rate/LIF; **OPEN for TTFS (=P7)** |
| B5 | σ-scope law: σ lives on (producer → neural-entry) edges only; trained-entry and never-encoded boundaries skip; post-training σ banned | `negative_boundary.apply_negative_boundary_policy` + `trained_entry_boundary` | ENF |
| B6 | σ-in-the-op: armed-only stamping; quantile σ and κ; shift-response consumer bakes; `cat` fail-loud | `signed_seam_install.py:30,44,94,155,177` | ENF (locks in `test_lif_exact_qat.py`, `TestHostBiasCarrierBake`) |
| B7 | κ = QUANTILE, never max (capacity vs resolution) | `_signed_seam_quantiles`; §10f full-width-cover refutation (0.0146) | ENF |
| B8 | σ rank law: per-channel for rank ≤ 2, SCALAR for rank ≥ 3 (mixed-axis not bias-compensable) | recorder mirror `segment_forward.py:172-177`; rank-aware `negative_shift` | ENF |
| B9 | homogeneity allowlist: `wire_transparent` membership proven by f(αx)=α·f(x) property test | T6 | ENF |
| B10 | SNW transparency to the wrapped calling convention (kwargs; tuple select BEFORE offset/scale; no double-index) | `compute_modules.py:67,95-102`; `forward_scale_normalized` delegate (`compute_op_mapper.py:163`) | ENF |

### 6.4 QAT-square contracts (exact arms)

P-L1 BN freeze · P-L2 counts readout · P-L3 dtype ties monitored · P-L4 join
balance (gap ≤ 1) · P-L5 Default-reset only + cycle-accurate forward · P-L6
fold ownership (`lif_exact_qat_program.md` §4.5; predicate + fail-loud in
`tuning/orchestration/lif_exact_qat.py:18-51`). Sync arms:
`sync_exact_qat`, `sync_entry_half_step`, `sync_hop_staged_install`,
two-scale WQ projection for the whole ttfs_cycle family
(`conversion_policy.py:112-133`). Status: ENF (fail-loud predicates).

## 7. Reduction of the existing memos (theory SSOT discipline)

| memo | is the … |
|---|---|
| `lif_deployment_exactness.md` | R/S-edge hop contract: Theorems 0-3, ledger V1-V9, correction series |
| `sync_deployment_exactness.md` | C-edge composition contract: Theorem 1, P1-P6, first-moment law, starvation §4, WQ-lattice §5 |
| `conversion_boundary_algebra.md` | seam contracts B1-B10: κ gauges, σ laws, arming, σ-in-the-op (§10f-g), agreement triage (§10e) |
| `lif_exact_qat_program.md` / `ttfs_exact_qat_program.md` | the commuting QAT square per mode (A2; P-L*, sync arms) |
| `numerical_boundary_consistency.md` | the §5 rules of evidence (RC1-RC3, tie inventory) |

## 8. The census, classified, and the pre-registered predictions

State (2026-07-19): t2_04 LIF ViT parked at the σ-armed AQ cache (staircase
0.5956 entry / 0.6046 post-recovery; analytic 0.7742); its LIF full-transform
read is **0.0115** where the A2 square predicts ≈ 0.60 (mixer precedent
t0_30 held the square within ~4 pts: AQ 0.9559 → LIF 0.9143 → deployed
0.923). t2_05 sync ViT died pre-conversion (0-byte cache at Pruning
Adaptation). t1_02/t1_03/t2_02 (analytical-TTFS ViT) unmeasured. Tier-0 has
no attention-topology vehicle (the isolation-ladder inversion).

| census item | class (predicted) | resolving assay |
|---|---|---|
| LIF full-transform 0.0115 vs 0.60 | **M or B** in exactly one of {retime, manager-rebuild, cycle-trains} | PR1/PR2 (R0-R5 matrix + auditor ledger) |
| sync `apply_ttfs` on mixed-axis host seams (§10b, mixer 0.0000) | **B** (P7/B4 hole) | PR3 |
| pre-softmax score seam units | **B if red** (expected green via SNW) | PR4 |
| residual-stream eff_levels decay 26→1 (§10i) | **C** (static recal REFUTED §10j — stands) | PR5, after B/M green |
| sync P3 subsume coverage; P6 readout | **C** + missing fail-loud | PR6 (t2_05 is offload + host head ⇒ expected inert; verify, don't assume) |
| t2_05 0-byte cache | infrastructure (F3 guard) | truncated-file test |

**Predictions (each with expected value; L1 requires these before the runs):**
- **PR1**: the faithful full-transform repro on the AQ cache reads
  0.011 ± 0.02 at n=512, and at least one single-axis toggle
  (retime / rebuild / cycle-trains) recovers ≥ 0.30.
- **PR2**: if the retime transcode is the killer, the per-edge ledger shows a
  deterministic bias at ARMED σ seams specifically (σ dropped or doubled at
  the re-encode), depth-monotone.
- **PR3**: the sync ViT (and the retired t0_32 mixer repro) violates P7/B4:
  torch↔deployed agreement ≈ 0.0 at the first parity gate until the TTFS
  transcode is brought under the arming + rank laws.
- **PR4**: the SNW-wrapped MHA computes the score seam in absolute units —
  certificate green; red = the ViT-specific Type-B.
- **PR5**: no static currency lifts the genuine LIF read above ~0.30 on the
  un-fixed composition (already measured, §10j); after the B/M fixes the
  Type-C residual is what QAT/adaptation must close.
- **PR6**: t2_05 under offload + host float head is inert to the P3/P6
  holes — the auditor must confirm both certificates, not assume them.
- **PR7 (replication keystone)**: a tiny-ViT carrying {patch-embed conv,
  LN→entry seams, packed MHA + score seam, residual adds, offload}
  reproduces every confirmed Type-B/M defect at MNIST scale — the defect
  classes are topological, not scale-dependent. Failure to reproduce
  falsifies the classification and forces a calculus revision.
- **PR8 (quantitative closure)**: with all B/M certificates green, the
  composed first-moment prediction from the ledger matches the measured
  genuine read within ~2·SE on every vehicle; a mismatch is a missing edge
  in this memo.

## 9. The instrument this memo specifies (Phase I contract)

**The seam-certificate auditor** (`spiking/seam_audit.py`, program §3): given
any cached model + config, emit the per-edge certificate ledger — edge,
representation types, gauges (κ_fold/κ_buf/σ), precondition checklist
(§6 rows evaluable statically), measured per-edge defect δ_e (both twins run
in isolation on captured real activations via the driver recorders,
`segment_forward.py:111-142`), and the §3 classification. The auditor is
**untrusted until it passes defect injection**: synthetic fixtures (the
`test_wire_currency_contract._signed_seam_model` family) with planted
defects — dropped σ, corrupted κ, induced saturation — which it must localize
to the exact edge with the exact class, and a clean fixture on which it must
report no Type-B. Its composed-loss output is the PR8 check. It is the L2
ladder's first rung made executable: no tier-1 run may be used to discover
what a ledger row could have said.

## 10. F1 executed (2026-07-19): PR1 RESOLVED — the finalize-rebuild decorator defect, fixed and locked

Instruments: the R-matrix harness (`scripts/_probes/lif_finalize_twin_bisect.py`
— builds the real `PipelineSession` on the cached σ-armed t2_04 AQ run and
reads every arm through the tuner's own `_mbh_full_transform_forward`, so
probe ≡ deploy by construction) at n=512 (SE 0.022); the seam auditor +
gauge-introspection probes on the same cache.

### 10.1 The measured matrix

| arm | retime | cycle-trains | manager-rebuild | read |
|---|---|---|---|---|
| R0 blended entry (no transform) | — | — | — | 0.8086 |
| R1 pipeline-faithful | T | T | T | **0.0156** |
| R2 | F | T | T | 0.0156 |
| R3 | T | F | T | 0.0156 |
| R4 | T | T | F (fresh LIF @ own θ) | **0.2910** |
| R5 probe replica | F | F | F | 0.2910 |

PR1 confirmed in full (repro 0.0156 ∈ 0.011±0.02; one single-axis toggle
recovers ≥0.30). **PR2's retime hypothesis is REFUTED**: retime and
cycle-trains are both bit-inert on this read (R1=R2=R3; R4=R5). The killer
axis is the **manager rebuild** alone.

### 10.2 The mechanism (Type-M in the taxonomy — the READ ran the wrong function)

`AdaptationManager.update_activation` subsumes clamp/quant/shift under
`lif_active`/`ttfs_active`, but the **activation-replacement decorator**
(`activation_adaptation_rate` → `RateAdjustedDecorator(
ActivationReplacementDecorator(LeakyGradReLU()))`) carried **no subsumption
gate**. On torch-converted vehicles the Activation Adaptation step binds a
buffer-backed rate carrier that persists in the cached manager at **alpha 1.0
even when the float field reads 0.0** (`_rate_is_active`: "a bound buffer
counts as active even at alpha 0.0"). Every finalize rebuild therefore
re-attached the decorator, which at alpha 1.0 substitutes
`LeakyGradReLU(input)` for the installed LIF forward — the deployed twin ran
ReLU per cycle instead of spikes at all 12 layers. Two aggravators made it
invisible: `TransformedActivation.decorators` is a plain Python list (absent
from the module repr — the rebuilt tree LOOKED clean), and native tier-0
vehicles never bind the AA buffer (float 0.0 → decorator absent), which is
exactly why tier-0 lif/sync cells stayed strict-exact while every
torch-converted conversion (ViT, and predictably squeezenet/resnet/deepcnn32
at tiers 1–2) collapsed at the same seam.

### 10.3 The fix (one gate at the SSOT, tests-first)

`adaptation_manager.py::update_activation`: the replacement decorator is now
gated on `not runtime_subsumed` — it vanishes exactly when the conversion
family owns the node, and is untouched before that (the AA phase itself is
byte-identical; `runtime_subsumed`, not `subsumes_decorators`, so lif-mode AA
still trains through it). Locks:
`tests/unit/tuning/test_adaptation_manager_rebuild.py` (lif_active and
ttfs_active rebuilds transparent over the blend; pre-install AA behavior
preserved — the buffer-persisted-alpha state replicated exactly). **Measured
verification: post-fix faithful R1 = 0.2910, bit-identical to R4/R5.** Gate
8268 green, typecheck 0.

### 10.4 The named next lever: the 0.29 → 0.60 residual is a currency-system split

Gauge introspection on the same cache measured **three coexisting currency
systems** at the seams: (a) the trained entry quantizers
(`input_activation_scale ≡ ChipInputQuantizer.activation_scale`,
self-consistent); (b) the weight-fold `per_input_scales` — **diverging from
(a) at entries 1–3** (1.0 vs 2.64, 0.813 vs 1.523, 0.966 vs 1.137), agreeing
deep; (c) the NF walk's re-encode table (`read_boundary_out_scales`
θ-pass-through), a third value at most entries (e.g. 1.037 vs 1.48 at entry
4) — while the armed SNW chain gauges (the σ-install's lifted quantile-κ,
`output_scale ≈ boundary_traffic_scale`) form their own self-consistent
system that the pass-through table never learned. This is a B2-class
candidate one level up: the σ-in-the-op re-propagation updated the
mapping-side gauges but `read_boundary_out_scales` has no term for
ComputeOp-level lifted currencies, and `LifSegmentPolicy.train_of`
re-encodes with the table. Auditor v1 caveat recorded: its chain-interior
`host_twin` certificates normalized wire inputs by the pass-through table, so
those B flags must be re-derived against the node gauges before a verdict;
the entry/boundary certificates stand (out-of-band mass 0.03 → 0.61 growing
with depth — the §10i eff_levels decay measured at the seam level, Type-C).

## 11. Re-consolidation (2026-07-19): the lossless-deployment decomposition

### 11.1 The gap ladder, named

Measured on t2_04 (one cache, one harness): float/finetuned ≈ 0.86 →
analytic trained-clamp 0.774 (census) / 0.809 (n=512) → staircase exact-QAT
endpoint **0.5956 / 0.6046** → genuine full-transform **0.2910** (post-§10
fix) → deployed (unmeasured). Three gaps of three different KINDS:

- **G-A (0.2910 vs 0.6046) — twin exactness.** By the A2 square this must be
  ≈ 0 (mixer slack ~4 pts) with ZERO training. Type-B/M territory only.
- **G-B (0.6046 vs 0.774) — staircase capacity.** The trained-through cost of
  the (σ, κ, θ, S) design at S=32. Type-C plus a Type-T (undertrained-QAT)
  component. Minimized by design + training, floor predicted analytically.
- **G-C (0.774 vs 0.86) — trained-model quality.** The AA/clamp adaptation
  cost. The T-axis; OUT of deployment-exactness scope; tracked separately.

**Definitive lossless deployment ≡ G-A driven to zero exactly (coherence ⇒
A2), G-B driven to its analytically predicted floor and the floor itself
driven small (§11.4), G-C reported honestly on its own axis.**

### 11.2 The Currency Coherence Theorem (G-A's closure)

Per re-encoded edge e, FOUR gauge readers exist in the code today:
**κ_Q** (the trained entry quantizer's scale), **κ_W** (the weight-fold
`per_input_scales`), **κ_T** (the walk re-encode table,
`read_boundary_out_scales`), **κ_S** (the producer's buffer gauge: armed SNW
`output_scale`, else neural θ).

**Theorem (coherence).** The deployed seam composition is value-preserving
in-band iff κ_Q = κ_W = κ_T = κ_S ( = κ*_e). Each pairwise violation is a
distinct, computable Type-B:
- κ_T ≠ κ_S — every twin-side re-encode multiplies values by κ_T/κ_S
  (measured ≈ 0.70× on early blocks);
- κ_S ≠ κ_Q — the deployed clamp band differs from the trained band
  (one-sided amputation/saturation the QAT never saw);
- κ_W ≠ κ_Q — chip-side charge mis-scale, INVISIBLE to the torch twin,
  caught only at SCM parity (the dangerous silent one; measured live at
  entries 1–3: 1.0 vs 2.64, 0.813 vs 1.523, 0.966 vs 1.137).

**Corollary (the one-writer law).** Coherence is maintainable iff exactly ONE
propagation writes every κ field (all others derive). §10f–g's σ-in-the-op
install created node-level currencies (lifted quantile covers) with a
PARTIAL writer set: the mapping-side re-propagation learned them; the
NF-side pass-through table and some entry stamps did not — B2 one level up.
The definitive fix is structural, not a patch: armed-node currencies join
the single κ propagation; a **coherence certificate** (all four gauges per
edge) becomes an install-time fail-loud, a permanent contract test, and an
auditor row (replacing the v1 chain-interior `host_twin` normalization).

**PR9**: with coherence restored, the faithful genuine read rises from
0.2910 to within the mixer's A2 slack of the staircase endpoint
(**≥ 0.55** vs 0.6046) with zero training. If it does not, the calculus is
missing an edge — fail loud, extend §1 before touching more code.

### 11.3 The contaminated-refutation audit (epistemics the taxonomy imposes)

A Type-B/M defect in a measurement composition voids the conclusions
measured through it. Re-opened by §10:

- **§10j ("no static currency recovers; adaptation is real") — UNPROVEN
  again.** Its sweeps ran through the κ_T/κ_S-split walk, and sweeping the
  perceptron θ moved κ_T and the LIF scale TOGETHER while κ_S (host chains)
  and κ_Q stayed — a confounded experiment. **PR10**: the post-coherence
  sweep landscape differs from §10j's (the 0.30 plateau moves). The
  "adaptation is the only path" conclusion is suspended until PR10 runs.
- **§10i (θ "mis-referenced to the stream" ⇒ the 0.011)** — the ratio
  MEASUREMENTS stand as facts; the causal reading is superseded (§10 fixed
  the 0.011 without touching any scale). `eff_levels` remains the Type-C
  instrument.

**Law (added to the rules of evidence):** every refutation records its
measurement composition; when a later Type-B/M fix touches that composition,
the refutation automatically re-enters the open set. Negative results are
only as durable as the functor they were read through.

### 11.4 The staircase-capacity program (G-B's closure)

Decomposition of 0.6046 vs 0.774 under the calculus, each term with its
lever and its instrument:

- **(a) Stale covers.** σ/κ are calibrated ONCE at install and frozen; the
  QAT then moves the activation distributions; the drift is measured as
  per-seam out-of-band mass (0.03 → 0.61 with depth on this cache). Closure:
  the **cover-tracking iteration** — alternate [recalibrate (σ, κ) from the
  CURRENT distributions → the I1 install updates by construction → short
  recovery]. Each recalibration is pre-training with respect to the next
  phase, so the σ-scope law is respected: what is banned is a post-training
  σ the training never trains through; an iterated install trains through
  every new cover. Monitor: max-seam oob, monotone decreasing to ε.
  **PR11**: one iteration + endpoint recovery lifts the staircase endpoint
  ≥ +3 pp (band 0.60 → 0.63–0.70).
- **(b) Level starvation.** `eff_levels ≥ L_min` becomes an explicit
  per-hop constraint; per-channel θ (the existing `per_channel_theta` seams,
  armed for lif + synchronized) reallocates grid within a hop — the sync
  memo §4 starvation theory applied to LIF. **PR12**: per-channel θ raises
  min-channel eff_levels and cuts deep-seam oob — checkable ANALYTICALLY by
  the auditor before any genuine run is spent.
- **(c) QAT depth.** The σ-armed AQ ladder entered at 0.5956 and finished at
  0.6046 (fast_lr_scale 0.25, budget 0.5): G-B is partly Type-T
  (undertrained). The V0 economics (SE-sized evals) buys the budget back.
- **(d) The irreducible floor, predicted — never assumed.** The PR8
  composition validator (per-hop first-moment fold over the ledger, sync
  memo §3) yields an ANALYTIC estimate of the staircase cost at the current
  (σ, κ, θ, S). "Definitive lossless" at fixed S means: measured G-B ≡
  predicted floor within 2·SE, AND the floor itself made small by (a)+(b).
  If the predicted floor at S=32 remains large after both, the theory
  prescribes resolution (S, per-channel θ) — a design decision with a
  price, no longer a mystery.

### 11.5 The enumerability law (the §10 lesson, made structural)

Two invisibility mechanisms enabled the §10 defect: behavior-bearing state
OUTSIDE the module tree (`TransformedActivation.decorators` is a plain list —
repr-invisible) and dual-representation state (buffer vs float, silently
diverging). **Law L6**: every forward-influencing state must be enumerable
through one introspection surface. Enforcement now: the auditor grows a
per-perceptron **forward-influencer census** row (decorator stack types +
carrier alphas, blend rates, quantizer scale refs) so this class is caught
by inspection, not only by behavior; a repr/ratchet follows if it recurs.

### 11.6 The prediction registry after re-consolidation

Open: **PR4** (score-seam certificate) · **PR6** (sync P3/P6 inertness on
t2_05) · **PR7** (tiny-ViT replication keystone — now including the §10
decorator class and the coherence certificate at MNIST scale) · **PR8**
(composition validator match) · **PR9** (coherence ⇒ ≥0.55 no-training) ·
**PR10** (§10j re-run moves) · **PR11** (cover iteration ≥ +3 pp) · **PR12**
(per-channel-θ analytic gains). Resolved: PR1 (rebuild axis, fixed §10.3);
PR2 refuted for retime; PR3/PR5 superseded in part by §11.3 (re-derive
post-coherence). The program of record sequences these as C → D → R → F2 →
V (`~/.claude/plans/spiking_deployment_calculus_program.md`).
