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

## 12. Phase C executed (2026-07-19): coherence LANDED — PR9 exceeded; the genuine read reaches the analytic band

### 12.1 C1 — the writer-census verdict

A fresh `compute_per_source_scales` + `propagate_boundary_input_scales` on an
in-memory copy of the cached AQ model reproduces every stamp bit-for-bit:
the split is IN THE WALKS, not stale ordering. Three formulas coexisted:
the pure table (θ pass-through, no lifts), the mutating boundary walk
(`_traffic_lift` accumulation — the source of in_act 1.523), and the
per-source system (its own lift — the source of s_out 0.813). Second
finding: the ENTIRE block-0 chain (patch-embed → LN → MHA → LN, exec sites
1–9) is UNARMED — `is_wire_value_op` marked, but the unity-gauge arming
condition skipped it — so its raw signed values hit the walk's
clamp-before-scale re-encode, and the armed-only σ law left that seam
σ-free (the measured oob 0.498 at entry 1).

### 12.2 C2 — the one-writer fix (landed, gate 8274 green, typecheck 0)

1. `ComputeOpMapper.propagate_boundary_scale`: an ARMED op's boundary
   out-scale IS its buffer gauge (`output_scale`); unarmed ops keep the
   lifted pass-through.
2. `read_boundary_out_scales` now DELEGATES every non-perceptron node to the
   polymorphic walk — one implementation for both tables (inheriting the
   traffic lifts and the residual-merge max rule, closing two latent splits
   beyond the measured one).
3. `LifSegmentPolicy.train_of` dispatches on representation: ABSOLUTE
   (raw, unarmed-chain) producers transcode via the SSOT divide-first
   `normalize_boundary_value`; wire producers clamp; mixed wire/absolute
   fan-in at a plain host op fails loud (`_absolute_value_nodes`).
4. `verify_boundary_currency_coherence` — the install-seam fail-loud
   certificate, called at the σ-install end.
5. Locks: `test_currency_coherence.py`; and the injection suite tracked the
   theorem — under coherence an output-gauge corruption CANNOT split the
   twins (the table follows it): it now surfaces as a currency defect at
   the consumer stamps; the twin-divergence injection moved to the decode
   gauge (`per_source_scales`).

### 12.3 C3 — the PR9 verdict: EXCEEDED

Faithful full-transform (retime=T, cycle-trains=T, rebuild=T), same cache,
zero training:

| read | n=512 | census n=2496 |
|---|---|---|
| R0 analytic blended | 0.8086 | **0.7742** (≡ the historical post_acc — harness census-validated) |
| R1 faithful genuine | **0.7832** | **0.7397** |

The journey on one artifact: **0.0115 (pre-§10) → 0.2910 (post-§10) →
0.7397 (post-§12)**. Residual G-A = **3.45 pp** (Δ ≈ 3.9·SE — real and
small), at the mixer's A2-slack scale (t0_30: 4.2 pp), decomposable into
the known terms: the σ-free entry-1 seam (oob 0.498, Type-C by the
armed-only law), the entries-2/3 band clips (s_out < κ_Q), hop terms
(V3/V4/V5 family), and ties.

### 12.4 The G-B reinterpretation (§11.3 strikes again)

The "staircase endpoint 0.6046" was itself a contaminated anchor: the AQ
full-transform read re-quantizes already-grid values (entry rounds + the
decorator staircase snap under strict `<`) — the exact double-quantize the
`lif_active` subsumption exists to avoid (`lif_exact_qat_program` §6.1(2)).
The deployable LIF composition does not implement that double snap, so
0.6046 UNDER-reported the deployable function by ~13 pp. The honest ladder
is now **analytic 0.7742 → genuine twin 0.7397**; G-B as originally framed
(17 pp) largely dissolves into (i) the AQ read convention (Type-M on the
train-side twin — an F4 item: align the AQ full-transform read with the
subsumed composition) and (ii) the 3.45 pp physical residual above.

### 12.5 Between here and "deployed ≈ analytic"

(1) the chip-side κ_W half: entries 1–3 `per_input_scales` (1.0 / 0.813 /
0.966) vs κ_Q — invisible to the torch twin; needs the SCM emission-table
coherence + a cross-sim decision-parity spot; (2) the D-phase capacity
items re-anchored to the 3.45 pp residual (entry-1 σ/arming under the
unity-gauge condition; entries-2/3 band re-pin; PR8 floor prediction);
(3) WQ → SCM → parity (Phase V). PR9 CLOSED-EXCEEDED; PR10-12 unchanged.

## 13. The anchored-conversion principle (2026-07-19): the FULL gap is the gate

**Correction of scope (the program's gate metric).** The deployment gate is
**pretrained → deployed**, not analytic → deployed. At S=32 (5-bit
activations) with 8-bit weights and per-neuron θ, the deployable family
F_deploy provably reaches CIFAR-100-ViT accuracy within ~1 pp of float
(the standard QAT regime); therefore the measured 0.8678 → 0.7397 (−12.8 pp)
is a **conversion/tuning-dynamics deficit**, not capacity — and §11.1's G-C
("trained-model quality, out of scope") is RETIRED as a scoping error. One
gate: origin float → deployed, lossless within a declared ε.

### 13.1 The measured decay ledger (t2_04, per-step pipeline metrics)

| step | metric | Δ | classification |
|---|---|---|---|
| Weight Preloading / Torch Mapping | 0.852 | — | structural, lossless |
| Pruning Adaptation (+recovery) | **0.8678** | +1.6 | **the origin anchor** (post-structural float — exactly what `ReferenceTeacherSnapshotStep` freezes, G8) |
| Scale Migration / Activation Analysis | 0.8678 | 0 | scales only, lossless |
| Activation Adaptation (GELU → chip-ReLU) | 0.8062 | **−6.2** | family swap, undertrained (teacher = own entry ≈ origin, so NOT drift: budget/recipe) |
| Clamp Adaptation | 0.8062 | 0 | carried |
| Activation Shifting | 0.836 | **+3.0** | recovery credit — training pulls back even toward a drifted anchor |
| Activation Quantization (σ-armed exact-QAT) | 0.7711 | **−6.5** | family swap trained at **plain CE** — the measured worst arm; the origin-KD lever exists (`lif_exact_qat_kd` + reference teacher) but is default-off and single-step |
| genuine twin (§12) | 0.7397 | −3.1 | the D-phase deploy-side residual |

The entire ladder loss lives in the two **function-family swaps**; every step
after AA distills to a **drifted teacher** (each tuner snapshots its own
entry model — `kd_blend_adaptation_tuner.py:262`,
`activation_adaptation_tuner.py:54`), and the accuracy compact anchors each
floor to the PREVIOUS step (`step_floor(previous_metric, tol)`), licensing
multiplicative decay: with per-step tolerance 0.1 over ~6 conversion steps
the pipeline contractually permits 0.9⁶ ≈ 0.53×. There is no global
restoring force; the target adjuster even relaxes missed targets. The
dynamics are a drift process by construction.

### 13.2 The principle (generic, mode/model/workload-agnostic)

Conversion is constrained optimization: min L(f) s.t. f ∈ F_deploy. The
ladder implements it as greedy sequential family restriction — fine — but
lossless conversion additionally requires, at every step k:

- **L-A (one anchor).** The KD teacher is the ORIGIN function (the
  post-structural float model), not the previous step's endpoint. The
  mechanism already exists as `ReferenceTeacherSnapshotStep` (post-prune
  float, cached as `reference_teacher_model`) — generalize it from
  one-consumer-default-off to the pipeline-wide teacher SSOT: every
  function-changing tuner consults the reference entry when present
  (self-snapshot remains the fallback). One seam (`tuning/teacher.py`), all
  modes.
- **L-B (one budget).** The accuracy compact anchors every floor/target to
  the ORIGIN metric with a cumulative loss budget ε(k) (Σ ε_k = ε_total),
  replacing prev×(1−tol) — no multiplicative license, no relaxation drift;
  the retention envelope re-anchors to origin so cumulative drift is
  MONOTONE bookkeeping, not a random walk of per-rung tolerances. One seam
  (`accuracy_budget.step_floor` + `retention_envelope`).
- **L-C (budget follows distance).** Function-family swaps (AA, AQ, the
  mode conversion) are the expensive projections; per-step training budget
  scales with the swap's measured entry drop, funded by the V0 eval
  economics. The +3.0 pp Shift credit is the existence proof that the
  optimizer recovers when given steps — it currently aims at the wrong
  anchor.
- **L-D (capacity is certified, never assumed).** The PR8/oob analytic
  instruments certify per step that F_deploy at the current (σ, κ, θ, S)
  contains a near-origin function; only a failed certificate may re-classify
  a deficit as capacity (and then prescribes σ/θ/S design, priced).

Elegance claim: no new subsystems. Three existing seams re-anchored
(teacher SSOT, compact SSOT, envelope) + one existing step un-gated and
generalized. Mode-generic because all of them live ABOVE the mode layer.

### 13.3 Predictions (pre-registered)

- **PR13 (CONFIRMED by the ledger above)**: the tuning-ladder loss
  concentrates in the family swaps (−6.2, −6.5), with recovery credits
  proving optimizer headroom; registered as the baseline for every E-phase
  A/B.
- **PR14**: pipeline-wide origin-teacher KD + origin-anchored compact,
  SAME budgets, lifts the AQ-endpoint analytic from 0.7711 to ≥ 0.80 on the
  t2_04 resume (AA entry), tier-0-neutral.
- **PR14b**: with L-C budget scaling on the two swap steps (funded by
  SE-sized evals), ≥ 0.84.
- **PR15**: the capacity certificate at S=32 on the ViT predicts a total
  deploy-side floor ≤ 2 pp — i.e., F_deploy contains a ≥ 0.85 function; a
  failed PR15 re-routes the program to σ/θ/S design with the deficit priced.
- **PR16**: under L-B the ladder's measured cumulative drift never exceeds
  the declared ε at any step (monotone envelope), on tier-0 and the ViT.

**The definitive-lossless DoD is restated on the full gap**: origin float
(0.8678 here) → deployed within ε_total (declared per vehicle; the ViT
target: ≤ 2 pp analytic + the certified deploy floor), with the §12
coherence certificates green and cross-sim parity ≥ 0.98. The program of
record sequences this as Phase E (anchor) → D (deploy residual) → R → F2 →
V.

### 13.4 E4 executed (2026-07-19): PR14 PASSES — and the twin residual is trajectory-dependent

E2 (origin-teacher SSOT) + E3 (origin-anchored compact) landed default-off
(locks in `test_origin_anchor.py`; gate 8286). The A/B: t2_04 resumed at
Reference Teacher Snapshot in an isolated dir with `origin_teacher_kd`,
`origin_anchored_compact`, `activation_adaptation_kd`, `lif_exact_qat_kd`,
`eval_subsample_target=1152` armed — same training budgets as baseline.

**The whole conversion chain (snapshot → AA → Clamp → Shift → AQ) completed
in ONE 10.4-minute window** — the reaper economics dissolved as a side
effect (SE-sized evals + rungs that converge instead of thrash).

| read | baseline | anchored A/B |
|---|---|---|
| AA endpoint (analytic) | 0.8062 | 0.8096 (swap cost −5.8: anchor-INSENSITIVE, as §13.1 predicted — its lever is L-C budget) |
| AQ exact-QAT ladder | 0.5956 → 0.6046 (plain CE, inert) | **0.694 → 0.798 → 0.802 → 0.807 → 0.808 (KD-to-origin, strongly convergent)** |
| AQ endpoint analytic (census) | 0.7711 | **0.8140** — PR14 (≥0.80) PASSES, +4.3 pp at equal budgets |
| genuine twin (census) | 0.7397 (residual 3.45 pp) | **0.7151 (residual 9.9 pp)** |

**The new finding (honest, and the next theory target): the analytic↔genuine
twin residual CO-VARIES with the training trajectory.** On the anchored
model the per-seam evidence exonerates the §12 suspects: out-of-band mass
COLLAPSED (deep seams 0.61 → 0.011–0.047; only the unarmed σ-free entry-1
seam remains at 0.500), per-seam bias ≈ 0, and retime/cycle-trains remain
bit-inert (R2 = R3 exactly). The QAT's acceptance gauge (the staircase
D-hat) is not the deployed composition, so genuine drift is unmonitored and
unconstrained during training; the leading mechanism candidate — pre-
registered, not assumed — is **tie-mass concentration**: staircase-KD pulls
pre-activations onto grid flats where the strict-`<` comparator drops a
level per hop (the P5/V9 hazard, structurally common on grid values).
**PR17**: a tie-mass gauge (mass within ε of the θ·k/T grid, per hop) reads
significantly higher on the anchored model than the baseline, and its
per-hop profile predicts the genuine drop via the composition law. **PR18
(the law candidate)**: adding the cheap analytic gauges (tie-mass +
temporal-A6 + oob) as rung-acceptance CONSTRAINTS — or a small-n genuine
spot-read at commits — bounds the twin residual during QAT without paying
the census; with it, the anchored ladder retains its analytic gain AND a
genuine read ≥ the baseline's. Scoreboard: origin 0.8678 → analytic 0.8140
→ genuine 0.7151; the program's next block is PR17 → PR18 (gate design,
theory-first) + PR14b (AA budget scaling) + the entry-1 arming hole.

## 14. The trajectory-dependent twin residual: the model, three refutations, and the surviving mechanism class (2026-07-19)

### 14.1 The model

Write the genuine composition as g = s + η through the network Jacobian:
s the value/staircase composition, η the per-hop deploy-side terms. The
accuracy cost of η factors as (how often and how large η fires) × (whether
J·η crosses decision margins). Candidate mechanisms, each with a
discriminating instrument: (M1) **tie-mass/STE-parking** — staircase-STE
updates halt at boundary crossings, parking pre-activations where the
deployed comparator coin-flips; (M2) **margin sharpening** — anchored KD
sharpens margins so unchanged grid noise costs more; (M3) **same-sign
per-hop bias** — a deterministic sub-grid drift that the residual stream
INTEGRATES across blocks (the §3 linear-composition branch); (M4) a
within-hop transcode convention split (Type-B proper).

### 14.2 The instrument verdicts (`twin_residual_decomposition.py`, both AQ models, identical batches, n=256)

| read | baseline | anchored |
|---|---|---|
| value forward | 0.8164 | 0.8594 |
| genuine (retimed cascade) | 0.8086 | 0.7383 |
| value + full grid-noise injection (±θ/2T at every activation) | 0.8086 | **0.8633** |
| per-hop d_abs/θ (genuine vs value) | ≤ 0.011 | ≤ 0.021 |
| per-hop d_mean/θ | mixed-sign, ~±0.003 | **same-sign +0.003 … +0.021 at EVERY hop** |

- **M2 REFUTED**: the anchored model is completely robust to iid grid noise
  of the full deployed amplitude — margins are not the carrier.
- **M1 instrument VOID** (honest): the tie-distance column mis-handled the
  dominant negative mass (`frac` of negatives); tie-mass remains unmeasured
  and is now subsumed by the sharper finding below.
- **The surviving signature is M3/M4**: per-hop deltas are sub-grid (below
  one θ/32 step) yet END-TO-END the anchored genuine loses ~12 pp — and the
  per-hop means are same-sign positive on the anchored model while mixed on
  the baseline. A deterministic same-sign sub-grid drift, integrated by the
  residual stream over 12 blocks, is exactly the calculus §3 deterministic-
  bias branch: invisible to iid-noise robustness, tiny per hop, linearly
  compounding. The anchored trajectory did not create NEW physics — it
  aligned the signs of an existing sub-grid term.

### 14.3 The post-hoc lever REFUTED (PR19')

The house first-moment machinery (`match_lif_activation_distributions`,
DFQ per-neuron mean matching to the ORIGIN teacher, keep-best over the
genuine probe) does NOT recover it: probe_best ≡ probe_entry (best_iter 0,
patience-stopped), mean-gap 0.0668 unchanged, genuine 0.7246 → 0.7246
(n=512). Per-neuron output-mean shifts are the wrong coordinate system for
the damage. Consequence: **prevention-during-training outranks post-hoc
correction** — the PR18 gate is promoted from guardrail to primary lever.

### 14.4 The surviving discriminators and the next-block laws

- **PR20 (token structure)**: the classifier reads ONLY the CLS token;
  element-pooled statistics dilute precisely the damage that matters.
  Instrument: the per-hop delta ledger resolved per token (CLS vs patch
  mean) — prediction: the anchored drift concentrates on / is amplified at
  the CLS path.
- **PR21 (within-hop convention bisect)**: replay ONE hop on captured real
  inputs through each transcode stage separately (entry round → LIF count
  vs trained staircase → retime round) — localizes the same-sign term to a
  single stage; a nonzero systematic stage bias is Type-B and gets an exact
  convention fix at the kernel SSOT (`WireSemantics` / the staircase
  decorator pairing).
- **PR18 (the genuine-gauged gate — primary)**: under the exact-QAT arm the
  AQ tuner's full-transform gauge becomes the DEPLOYED (chip-aligned
  genuine) read on the clone — the measurement functor applied to the gate
  itself, via the existing `_finalize_forward_for` hook, at SE-sized n
  (~40 s per rung, affordable since the chain runs in one window).
  Prediction: re-run E4 with the genuine-gauged gate → analytic gains
  retained AND genuine ≥ the baseline's 0.7397 (the gate refuses
  sign-aligning trajectories).
- **The auditor grows the signed-ledger certificate**: per-hop d_mean/θ
  with a Type-B threshold from the §3 law (|mean| · depth vs margin scale)
  — the trajectory-dependence class becomes inspectable forever.

Sequencing: PR20+PR21 (instruments, minutes) → convention fix if PR21 says
Type-B → PR18 gate (one seam) → E4' re-run (one window) → PR14b (AA budget)
→ entry-1 arming → E5/PR15 → D/R/F2/V unchanged.

### 14.5 G1 executed (2026-07-19): two more refutations close the ring

- **PR21/V1 — the seam round is INERT**: inserting the κ-grid seam round
  into the value twin at every entry changes nothing (V0 0.8594 = V1 0.8594
  exactly). The drift is not a seam-round convention.
- **PR20 — REFUTED**: the signed deltas are token-UNIFORM (hop 6: CLS
  +0.0126 vs patch +0.0124; hop 11: CLS +0.0126 < patch +0.0213). No CLS
  concentration.
- **The ring is closed.** With retime, cycle-trains, iid noise, seam rounds,
  token structure, coverage, and per-neuron means ALL exonerated, the
  surviving stage is the only one never toggled: **hop-internal per-cycle
  integration physics** — the V4-family rectified transient (one-sided
  positive by construction) balanced against strict-`<` tie deficits
  (negative); the baseline's mixed per-hop signs are the two terms in
  near-cancellation, and the anchored trajectory tipped the balance
  same-sign. This is temporal physics (Type-C-temporal), not a convention
  bug: there is no exact fix, only the calculus's standard answer —
  **install the true composition in the training loop's gauge and let the
  QAT train through it** like every other deployed term. PR18 is therefore
  not merely primary but UNIQUE among cheap levers, and it is landed:
  `deployed_lif_gauge_forward` (`lif_exact_qat.py`) — fresh LIF at each θ +
  the retimed chip-aligned walk on the gate's clone — consumed by the AQ
  tuner's `_mbh_full_transform_forward` under the exact arm (locks in
  `test_lif_exact_qat.py::TestDeployedLifGauge`; gate 8288, typecheck 0).
  E4' (G3) verdict follows below.

### 14.6 G3 executed (2026-07-19): the gate works, gating is INSUFFICIENT — train-through is mandatory

**Economics forensics first (an M2 hole found and fixed).** The E4' run
thrashed: the genuine gauge at `eval_n_batches=32` (~5 min/read) cannot fit
a 952 s window — and the explicit `eval_subsample_target` could not go
below 32 batches because `TuningBudget` floored it at
`min(validation_steps, total)`. Fixed tests-first (the explicit workload
clamp is now authoritative; default path byte-identical). Corollary: E4's
"SE-sized" reads had silently been census-grade all along.

**E4' completed** (13.2-min window; the F3 atomic-write guarantee is what
makes a mid-kill cache trustworthy): AQ endpoint 0.8069; census verdict on
the artifact:

| model | analytic (census) | genuine (census) |
|---|---|---|
| baseline (plain CE) | 0.7742 | 0.7397 |
| E4 anchored, value-gauged | 0.8140 | 0.7151 |
| **E4' anchored, genuine-gauged** | **0.8377 (best yet)** | **0.6374** |

**PR18-as-sufficient is REFUTED — with the mechanism visible.** The gate did
exactly its designed job: the ladder's genuine D-hat climbed monotonically
from the 0.3214 entry to 0.6374 and `finalize_on_best_deployed` kept the
best state. But an accept/reject ratchet can only SELECT among states the
training trajectory visits — and the objective (KD-to-origin on the value
composition) never sees the temporal term, so the trajectory lives where
the twin gap is wide (analytic even improved, +2.4 pp over E4). The twin
residual is OBJECTIVE-controlled, not acceptance-controlled: post-hoc
correction failed (§14.3), gating failed (here) — **the only remaining
lever class is training THROUGH the genuine composition**, exactly the
pre-registered escalation.

**PR22 (launched)**: the LIF Adaptation step's below-floor recovery trains
through the installed chip-aligned genuine forward against the
origin-anchored floor (0.781) — never yet run on a post-fix artifact. A
BOUNDED slope experiment (budget 0.15, eval 256, one window) is running on
the E4' artifact: prediction — the genuine-through recovery slope is
positive and ≥ +2 pp within the bounded budget; extrapolated, the full-
budget step closes toward the analytic band. A ~zero slope would mean the
temporal term resists the surrogate gradient — then the next candidates are
deployment-noise/dither QAT and the analytic V4 first-moment fold, in that
order.

### 14.7 PR22 CONFIRMED (2026-07-19): train-through works, fast — and re-aligns the twins

The pipeline-step form of the experiment could not fit a window (session +
cache-load + LR-probe overhead precedes any training; no intra-step
resume), so PR22 ran probe-grade (`pr22_train_through_slope.py`: load AQ
model + origin teacher only, fresh-LIF install, KD-to-origin loss computed
ON `chip_aligned_segment_forward(..., retime=True)`, surrogate gradients
through the per-cycle IF and the retime STE). Three arms on the E4'
artifact (entry genuine 0.6387 ≡ the census 0.6374):

| arm | result |
|---|---|
| lr 3e-4 flat, 180 steps | CRATERED to chance in 60 steps — analytic control cratered identically ⇒ ordinary optimization damage (hot flat LR on a pretrained ViT), not a temporal effect |
| lr 2e-5 flat, 180 steps (~5 GPU-min) | genuine 0.6387 → **0.7188** (+8.0 pp); **twin gap collapsed along the trajectory** (genuine ≈ analytic at every checkpoint) |
| lr 2e-5 cosine + keep-best, 400 steps (~11 GPU-min) | genuine → **0.7676 census — the program's best genuine read** (prior best 0.7397); analytic 0.7402: **the twins crossed** — the deployed composition is now the model's best-read function |

Verdict: the temporal term is **surrogate-trainable with a steep slope**
(+12.9 pp in 11 minutes on a naive flat recipe — no LLRD, no warmup, 0.13
epochs), and training through the composition doesn't merely lift the
genuine read, it re-aligns the twins — the term is absorbed into the
weights exactly as the calculus predicts for any trained-through deployed
term. The lever-class elimination is complete and constructive: post-hoc
correction ✗, acceptance gating ✗ (but kept as the gauge/guardrail — it
also raised analytic), **objective-through-composition ✓**.

**Productionization note (the L-family completion): the conversion endpoint
must TRAIN through the deployed composition, not merely be measured by it**
— concretely, the genuine-through recovery becomes the exact-QAT endpoint
stage (the LIF step's recovery with a genuine-appropriate LR: 2e-5-scale,
NOT the pipeline 3e-3 which is the measured crater regime), window-chained
or probe-grade with keep-best. Remaining full-gap ledger after PR22:
origin 0.8678 → best genuine 0.7676 (−10.0), attributed: the AA swap
(−5.8, PR14b budget lever untouched), the analytic↔genuine joint headroom
(the probe traded analytic 0.8377 → 0.7402 under a naive recipe; an
origin-anchored longer run should hold both), entry-1 σ, and WQ/parity
still ahead.

## 15. The deployed-risk principle — the endgame consolidation (2026-07-19)

### 15.1 What the totality of evidence proves

One sentence per phase, all measured on one vehicle: exactness certificates
(coherence + twin + measurement-functor integrity) took the genuine read
0.0115 → 0.7397 with zero training; the anchor laws (one origin teacher,
one origin-ε compact) turned the AQ QAT from inert (0.5956→0.6046) to
convergent (0.694→0.808, analytic to 0.8377); the trajectory phenomenon
showed the twin residual is OBJECTIVE-controlled (acceptance gating selects
but cannot create; post-hoc correction has the wrong coordinates); and
PR22 showed the terminal cure — training THROUGH the deployed composition
lifts genuine +12.9 pp in 11 GPU-minutes and RE-ALIGNS the twins until the
deployed composition is the model's best-read function. Every pathology was
the same mistake in a different costume: **optimizing or measuring a
surrogate of the deployed risk where the surrogate's validity was never
certified**. Every win was a partial restoration of the converse rule.

### 15.2 The principle, stated as the program's final law

Define the deployed risk **R_dep(W) = E[ℓ(g_W(x), y)]** with g_W the image
of the deployment functor (the genuine composition), and the surrogate risk
R_s over the value twin s_W. The **surrogate-validity criterion**: training
and gating may use s_W only where (i) every Type-B/M certificate is green
and (ii) the composition-law bound |R_dep − R_s| ≤ δ (computable from the
signed ledger: Σ_h |d̄_h|·L_h + √K·κ/2T) is tight. Where temporal terms
leave δ loose, gradient descent on R_s is UNCONTROLLED for R_dep — measured
twice (E4: −2.5 pp genuine while analytic rose; E4': −7.6 pp while
analytic rose further). Hence:

**The two-phase conversion theorem.** Lossless conversion decomposes as
(I) **Phase-Value** — family projection and anchored surrogate training
wherever the validity criterion holds: the existing ladder, cheap because
value forwards are S× faster; then (II) **Phase-Deploy** — terminal
minimization of R_dep itself: origin-KD through g_W with surrogate
gradients, keep-best on SE-priced genuine reads. **Alignment corollary**:
at Phase-Deploy convergence the twins re-align (measured: genuine 0.7676 >
analytic 0.7402), so "holding analytic" is not a goal — analytic is
thereafter a free gauge of the same function; the shipped metric is R_dep
and only R_dep. Statistical grounding throughout: optimize the risk you
are evaluated on; make accept/reject decisions at n sized to their
tolerance (SE = √(p(1−p)/n)); pay census only at anchors; keep-best is the
ratcheted estimator of the trajectory minimum.

### 15.3 The cost model — lightning-fast is compatible, quantified

Measured constants (86M ViT, S=32, retimed, bs16): genuine train step
1.6 s ⇒ one epoch ≈ 83 min; the temporal term absorbed at ~0.13 epoch ⇒
**Phase-Deploy ≈ 10–30 min**. Phase-Value chain: 10–15 min (measured, one
window). Instruments: minutes. **PR26 target: end-to-end conversion
wall-clock ≤ 90 min on this vehicle**, with per-N-step checkpointing in
Phase-Deploy decoupling the reaper permanently (the pipeline-step form
failed to fit a window for overhead reasons — the stage must own its
resume).

### 15.4 The generic terminal stage (one SSOT, all modes)

**DeployedRiskFinetune** = the recovery engine pointed at g_W: origin
teacher (the L-A SSOT) + the per-mode deployed forward (the existing
family: `deployed_lif_gauge_forward`, the LIF `_ChipAlignedNFForward`, the
TTFS genuine policies — ONE stage, mode-parameterized by the finalize/gauge
seam that already exists) + keep-best on SE-priced genuine reads + the
**genuine-LR regime lever** (measured: 2e-5-scale; the pipeline recovery
default 3e-3 is the measured crater regime — the lever ships as a derived
default, e.g. deployed-finetune LR = tuning LR × 1e-2-scale, A/B'd once
then frozen) + the house recipe (warmup/cosine/LLRD from `tuning_recipe`)
+ intra-stage checkpointing. Applies unchanged to sync/ttfsq (their genuine
forwards plug into the same seam) and to any vehicle — genericity by
construction, elegance by pointing existing machinery at the right risk.

### 15.5 The victory predictions

- **PR23**: Phase-Deploy from the 0.8377 artifact with the proper recipe
  (warmup + cosine + LLRD + origin-KD, 0.3–0.5 epoch, keep-best) →
  genuine ≥ **0.82** census.
- **PR24**: + PR14b (budget-scaled AA under the origin anchor, the −5.8 pp
  swap) and one iteration → genuine ≥ **0.84**.
- **PR25 (the victory condition)**: pretrained → DEPLOYED within
  **ε_total = 2 pp** including WQ, the SCM identity read, and cross-sim
  decision parity ≥ 0.98 — with the WQ endpoint governed by the same
  principle (train through the quantized deployed forward; the two-scale
  machinery exists).
- **PR26**: end-to-end conversion wall-clock ≤ 90 min.
- Standing: PR15 (the analytic capacity floor certifies nothing is
  physically blocked), PR8 (the composed prediction must match the
  measured read — the theory's own audit), PR7 (tiny-ViT tier-0 locks the
  whole class at MNIST scale).

### 15.6 Risks, honestly

A persistent analytic/genuine trade at budget is IRRELEVANT to the DoD
(the alignment corollary makes genuine the only shipped risk) but would be
watched via PR15 for capacity misattribution. WQ may re-open a twin gap —
the principle prescribes the same cure at that seam. Recipe sensitivity is
bounded by the one-time LR A/B and the keep-best ratchet. Entry-1's
σ-free seam (oob 0.50) remains a known Type-C item with a designed fix
(arm marked wire-value ops feeding re-encoded seams regardless of gauge
unity). Nothing in the remaining ledger lacks a mechanism.

### 15.7 PR23 executed (2026-07-19): MISS at 0.7688 — the plateau diagnosed, H4 promoted to blocking

The full 1200-step Phase-Deploy ran in ONE 39.7-minute window (the reaper
never fired; checkpointing stands as insurance): keep-best 0.668 → 0.793
@256, **census 0.7688 — a hair above PR22's 400-step 0.7676: the curve
SATURATED** (3× budget bought +0.1 pp). Diagnostics on the trained state:

- **The alignment corollary held**: analytic 0.7734 / genuine 0.7930 @512
  — the twins converged (genuine above); the plateau is a JOINT basin
  ceiling under this recipe, not a twin gap.
- **The θ-frozen hypothesis is REFUTED by inspection**: the exact-QAT had
  already promoted activation_scale to trainable Parameters — Phase-Deploy
  co-trained θ throughout.
- **The dominant wound is singular and structural**: entry-1 oob = 0.498
  (every other seam 0.008–0.061). Half of block-0's LN band is amputated
  at the STEM; KD cannot restore information the seam deletes; every
  downstream layer computes on the rectified stem. This is the §12.1
  unity-gauge arming hole, now measured as the binding constraint.

**H4 design (from the C1/§12 mechanics)**: blanket-arming unity-gauge
marked ops would REGRESS the seam currency (unity-combined output_scale=1
vs the 2.64 band, breaking the divide-first pass-through the dispatch
correctly applies today). The correct completion: **the σ-installer
pre-arms marked-but-unarmed ops at their PASS-THROUGH currency** (wrap
slots = unit per-source scales, output_scale = the boundary-table κ),
then stamps σ/κ and re-propagates as usual — arming exists to transport
σ; the `lif_aq_negative_boundary` knob already gates it; the §12.2
coherence certificate verifies the result. Then Phase-Value re-runs
(one window) and Phase-Deploy re-runs from the healed artifact (PR23').

### 15.8 H4 executed (2026-07-19): the stem healed — entry-1 oob 0.498 → 0.036

Two locks landed tests-first: the pre-arm (`_prearm_marked_value_ops`,
currency-inertness proven by a bit-identical walk before/after) and — after
the fail-loud correctly caught the ViT stem's `cat` (the CLS concatenation,
genuinely non-compensable) — the **σ-scope feasibility skip**
(`_bake_walk_feasible`: a dry-run of the consumer walk; infeasible chains
are left trained-clamp with a report, never crash the install). The healed
Phase-Value chain then completed AQ in one 5.3-minute window with exactly
ONE skip (`conv_proj`, the pre-cat patch-embed — collateral, not the seam)
and endpoint 0.8205. The audit on the healed artifact: **entry-1 oob
0.498 → 0.036; all twelve seams now 0.006–0.046 with biases ≈ 0 — the
certificate board is green across the graph for the first time.** PR23'
(Phase-Deploy from the healed artifact) is the running verdict.

### 15.9 PR23' MISS (0.7536) — two more refutations; the stall diagnosed as optimization fidelity

PR23' from the healed artifact: entry 0.5625 → best 0.7773 @256, census
**0.7536** — at or below the unhealed run. Two refutations follow:
(1) **the stem-oob-as-binding-constraint hypothesis is refuted at the
outcome level** — healing 0.498 → 0.036 moved the Phase-Deploy ceiling not
at all (the heal remains correct and stays: the certificates are green and
the capacity is real, but it was not the constraint); (2) **the strict-`<`
tie hypothesis is refuted by direct A/B** — the same trained state reads
0.7540 under BOTH comparators (V9's float-threshold claim confirmed on the
ViT). The ~0.75–0.77 ceiling now reproduces across two structurally
different artifacts, two budgets (400/1200), and both comparators.

**The overlooked tell (see §15.10 for the correction): the KD loss NEVER descended** (~2.03 → ~1.9 across
1200 steps, both runs) while accuracy did all its climbing in the first
~120 steps. The optimizer is STALLED, not converged: the surrogate
gradient through 12 per-cycle IF layers at bs16 is too weak/noisy past the
easy re-alignment. The front-runner is therefore **optimization fidelity**
(the quality of the gradient estimator through the deployed composition),
with the discriminator arm: effective batch 64 (gradient accumulation),
pure-KD α=1 at T=8 (denser, softer signal) — prediction: the loss
DESCENDS and genuine exceeds 0.79. The escalation ladder if confirmed is
already in-house: the **frontier machinery** (`segment_hop_frontier`
k-hybrid, prefix/hop-staged ramps) — progressive genuine-depth training
that keeps gradients strong at every stage — is the designed curriculum
for exactly this class. The PR15 composed floor (still unrun) arbitrates
what remains after fidelity is fixed.

### 15.10 The great elimination closes: the temporal tax model (2026-07-19)

The fidelity arm (accum 4, pure-KD T=8, lr 4e-5, 300 steps on the healed
artifact) fixed the stall — the loss descends and the slope tripled
(+21.5 pp in 300 steps) — **but the ceiling did not move: census 0.7676
again.** (Two honesty notes: keep-best@256 overestimates census by ~2 pp
via selection bias — future gates read keep-best at larger n; and the
"descending loss" was partly a scale artifact — T=8 softening makes the
target easier, so low KL does not imply a closer argmax. The twins also
DECOUPLED under pure-KD (analytic 0.6172): the §15.2 alignment corollary
is recipe-dependent, holding for CE+KD mixtures, not pure soft matching.)

**The S-sweep acquits temporal resolution**: the best trained state,
re-gridded end-to-end (LIF T + entry quantizers + walk) reads T=32 0.7504 /
T=64 0.7604 / T=128 0.7488 — FLAT. The per-hop S-scaled terms (grid noise,
back-loading, transient counts) are NOT the cap.

**The elimination board is now effectively complete** — seams, stem
coverage, comparator ties, retime, cycle-trains, iid noise, margins, token
structure, per-neuron means, optimization stall, and S-resolution are ALL
measured-refuted as the binding constraint. What survives is a
QUANTITATIVE regularity across every artifact and recipe of the program:

    genuine_census ≈ analytic_artifact − τ,   τ ≈ 4–7 pp on this class
    (ViT: 0.8205→0.7536/0.7676, 0.8377→0.7688; mixer precedent t0_30:
     0.9559→0.9143, τ = 4.2 pp)

**The temporal-tax model**: the genuine composition levies an
S-independent, training-resistant, per-design tax τ on the artifact's
analytic accuracy. The endgame therefore has exactly two axes, both
pre-registered and both UNPULLED: (1) **the artifact axis — PR14b**: the
AA swap (−5.8 pp, plain-budget) is the last macro lever; a near-lossless
AA (literature-standard with KD + epochs, now cheap at one-window chains)
raises the taxed base directly; (2) **the tax axis — PR15/PR8 at last**:
the analytic decomposition of τ from the per-hop ledger (first moments ×
Jacobians + the interaction terms the S-sweep says are S-independent),
which either localizes τ to a fixable term (per-channel θ / frontier
curriculum / distmatch-on-aligned) or certifies it as the design floor —
converting the final distance into the priced (θ, S, topology) decision
the calculus reserves for a certified floor. DoD arithmetic: PR25 (≤2 pp)
requires analytic ≈ origin AND τ ≤ 2 — both axes must close; either alone
cannot.

### 15.11 The tax has a mechanism: rectified-transient overfire — and a
### zero-training +21pp lever (2026-07-19)

**PR15a (k-cut prefix-hybrid).** New instrument: pin perceptrons 0..k's
activation outputs to their genuine-walk decoded records (the walk's
`node_value_recorder` seam) and run the suffix analytically; acc(k) vs k
decomposes the tax per hop in accuracy units, with a built-in endpoint
self-check (cut=all must equal the genuine read; W_a validated at 0.4pp,
W_g at 2pp ~ SE). The mapped ViT has 12 LIF hops (θ 0.30→6.09 with depth,
head 2.30). The tax is BIMODAL, not uniform: hops 1–2 (−11pp) and hops
7–9 + head (−10pp, −7pp) carry it; middle hops are free. Both states
agree on the sites. Not a uniform design floor ⇒ regime-local mechanism.

**PR27 (membrane-init pairing).** The kernel pair is floor-centered
(V0=0, soft reset; exact-QAT staircase = ceil−1/floor — A2 commutes on
the floor pair). SHAQ-transfer hypothesis (V0=θ/2 round pair, 35pt→0.7pt
streaming-gap collapse there): isolated single-hop assay CONFIRMS
exactness (uniform 0.0128→0.0000, real spike-train arrivals
0.0282→0.0000) — but the MODEL-level A/B REFUTES it violently (W_a
genuine 0.5469→0.0801): mimarsinan's LIF integrates SIGNED charge, and
emitted spikes are a rectified path-functional (later negative charge
cannot cancel a fire). Pre-charging the membrane raises the rectifier's
operating point ⇒ near-zero/negative-z neurons overfire on positive
transients. SHAQ's win lives in all-positive-charge streams; it does not
transfer to signed ones. LAW: the temporal kernel's quantization center
is charge-sign-regime-dependent (positive-only ⇒ +θ/2 round; signed ⇒ 0
or a small NEGATIVE guard).

**The rectified-transient theorem (the tax mechanism).** For per-cycle
consumer charge z_t = z̄ + ε_t (Σε_t = 0), the count bias
E[N(path) − N(ε=0)] ≥ 0, increasing in transient variance and in V0 —
matching every elimination: S-independent (comb correlation is
T-invariant), iid-injection-immune (deterministic path rectification,
not noise), depth-bimodal (stream statistics), training-resistant
(weights can only absorb the mean, not the timing). And the SOURCE is
ours: `to_uniform_spikes` anchors EVERY active channel's comb at cycle 0
(`floor(cycle % spacing) == 0`) — a layer-wide synchronized burst at
each window start, maximizing cross-channel arrival variance.

**PR27-3/PR28 (both levers, zero-training, W_a artifact):**
- Guard band alone (V0 negative): monotone dose-response 0.5469 →
  0.6289 (−0.10) → 0.6895 (−0.25) → 0.7207 (−0.50) → 0.7324 (−0.75);
  V0=+0.10 hurts (0.4434) — the full sign story.
- Phase dither alone (per-channel comb rotation mod T, counts preserved
  EXACTLY — decode-invariant, analytic twin untouched): 0.5469 →
  **0.7559 (+20.9pp)**.
- Composed: dither + guard(−0.25) = **0.7617** (best); guard's optimum
  shrinks under dither (variance removed at source) — as predicted.
- W_g (genuine-trained at locked physics): flat/slightly down under
  dither — the trained state SPECIALIZED to locked combs; its 0.77
  ceiling was capacity spent absorbing transient noise. Retraining on
  the fixed composition is the unlock.

**Program consequence:** the conversion order inverts — fix the
composition's physics FIRST (dither + guard as SSOT config knobs across
all five backends: encode SSOT + FiringStrategy V0; HCM/nevresim/SANA-FE
must consume the SAME phase function or parity breaks), THEN Phase-Deploy
trains from a 0.76 entry with the noise gone instead of a 0.55 entry
spending its budget on absorption. Composed with the artifact axis
(PR14b) this is the arithmetic path to PR25 (≤2pp).

**Census-grade correction (n=2500):** the winning arm (dither +
guard −0.25) on W_a reads **genuine 0.7376 / analytic 0.8244** (the
n=512 matrix reads flatter by ~2.4pp — the known small-n optimism; all
future gates read census). Zero-training on the fixed composition ≈ the
fully-trained locked-composition state (0.7676): the remaining artifact
composition gap is 8.7pp (was ~27pp), now with training headroom that no
longer pays the absorption tax.

### 15.12 PR29 lands: the physics knobs are SSOT, tier-0-replicated;
### training on the fixed composition is FLAT ⇒ the artifact axis is next
### (2026-07-19)

**Landed (tests-first, gate 8310, typecheck 0):** `spike_phase_dither`
(count-exact per-channel comb rotation at the uniform-encode SSOT —
`uniform_phase_offsets` golden-ratio policy in spike_modes; threaded
through spike_trains → LifSegmentPolicy/chip_aligned walk → BoundaryConfig
→ segment_boundary/compute_boundary → the contract → the HCM flow) and
`lif_membrane_init` (the window-start guard: LIFActivation installs it as
the IFNode reset value so every reset path restores it; the HCM rate loop
pre-charges `memb += V0·θ` via `precharge_lif_states`, LIF-gated at the
flow ctor). Registry-registered, default-off, byte-identical when off.

**Tier-0 replication GREEN (t0_05, both knobs armed):** the full backend
ladder — NF → SCM → HCM → nevresim → Loihi → SANA-FE — deploys LOSSLESS
at 0.9809 (= the pretrained anchor, Δ=+0.0000 at every simulator read)
with Loihi spike parity 1.0. Cross-backend coherence holds because every
side consumes the SAME encode policy and the same guard. (One CUDA
illegal-access on the saturated shared GPU disappeared on rerun —
infrastructure, not the knobs.)

**Phase-Deploy on the fixed composition is FLAT.** From the healed
artifact with dither+guard(−0.25): entry 0.7734@256 / census 0.7376.
(a) DeployedRiskFinetune, warmup + lr 2e-5, 600 steps: entry = best =
final = 0.7734 — zero trainable improvement; (b) the hot recipe (4e-5,
no warmup) actively damages it (0.7617 → 0.6133@60). Reading: the fixed
composition already sits at THIS artifact's joint local capacity — the
+22pp that Phase-Deploy used to deliver was absorption of the transient
noise the knobs now remove at source. The two-phase theorem update: with
composition physics fixed, Phase-Deploy's role shrinks to a small
polish; the remaining 8.7pp (0.7376 vs analytic 0.8244) belongs to the
ARTIFACT (Phase-Value quality + QAT gauged on the true composition).

**Next block = AB4 (in flight): the Phase-Value chain re-run with the
physics knobs armed** — the exact-QAT AQ acceptance gauge
(`deployed_lif_gauge_forward`) now measures candidates on the FIXED
genuine composition, so the ladder optimizes what actually ships;
prediction (pre-registered): AQ endpoint genuine census ≥ 0.78 and the
twin residual ≤ 3pp without any Phase-Deploy. Then PR14b (AA budget) on
top. Refinement noted, not blocking: the value-twin staircase center is
not shifted with the guard (a ≤θ·0.25/T per-hop offset in the A2 square);
if AB4's residual stalls at ~1-2pp, shift the LIFCountStaircase center
with V0 (PR30 candidate).

### 15.13 AB4 verdict + the residual's new address: hops 0–2, structural
### (2026-07-19)

**AB4 (Phase-Value with physics knobs armed) MISSES its pre-registration**:
AQ endpoint analytic 0.8359 (endpoint recovery not even engaged) but
census genuine 0.6668 / analytic 0.8124 — WORSE genuine than AB3's
zero-training 0.7376. Two causes read out: (1) this run's AA landed
0.8678→0.7712 (−9.7 vs AB3's −5.8) — the AA step's RUN-TO-RUN VARIANCE
is now the largest artifact-axis term (PR14b's case strengthens); (2)
gauge-gating the QAT ladder on the fixed-genuine read does NOT transfer
the win into the artifact — acceptance can only SELECT among candidates
the VALUE-staircase objective proposes, and that objective still trains
the unshifted (V0=0) staircase (§14.6's selection-vs-objective law,
recurring one level up). Conversion-order law: physics knobs help AT
DEPLOYMENT on a strong artifact; arming them mid-Phase-Value without
moving the value twin's center buys nothing and costs ladder stability.

**The fixed-composition k-cut curve (AB3 artifact + knobs, n=512):
analytic 0.8398 / genuine 0.7617; the residual CONCENTRATES at hops 0–2**
(−1.6/−5.7/−2.2 = −9.4pp; every later hop flat within noise; endpoint
self-check 0.7559 vs 0.7617 ✓). The locked curve's late-hop and head
terms are CURED by dither+guard. Two follow-up sweeps, both FLAT:
per-hop deeper guards on hops 0–2 (V0 −0.5/−0.75/−1.0: all ±0.2pp —
overfire exhausted there) and the T-sweep re-run on the FIXED
composition (T=32/64/128 → 0.7617/0.7480/0.7520 — resolution acquitted
on clean physics as well; the slight T>32 dip is consistent with knobs
tuned at T=32).

**Standing model:** the remaining ~8pp is a STRUCTURAL early-hop term —
prime suspects, in order: the entry-seam composition (ChipInputQuantizer
grid × walk re-encode double-rounding at the stem, §12's family), true
within-window causality (back-loading the assay showed no V0 can fix),
and the quarter-grid value-twin center mismatch (PR30). Next
instruments/levers: (i) the §14 twin-delta ledger (d_mean/d_abs per hop
at IDENTICAL inputs) re-run on the fixed composition to split hop-internal
vs upstream-seam at hops 0–2; (ii) PR14b budget-scaled AA (artifact
axis, −5.8..−9.7 measured spread); (iii) PR30 center-paired exact-QAT
(staircase shifted with V0, then Phase-Value again).

**Scoreboard after PR29:** origin 0.8678 → best analytic artifact 0.8244
(AB3, census) → **deployed genuine census 0.7376 with ZERO training**
(knobs at deploy; tier-0-replicated lossless across all six backends).
Gap to origin −13.0pp = artifact −4.3 (AA-dominated) + composition −8.7
(hops 0–2 structural). Phase-Deploy is retired as a main lever (flat on
fixed physics); the endgame budget shifts to Phase-Value quality + the
entry-seam term.

### 15.14 The early-hop term dissected: magnitude + structure at a
### sensitive coordinate; attribution ≠ magnitude (2026-07-19)

Parallel-cycle verdicts on the fixed composition (AB3 + knobs):

**PR31 (identical-input hop twin-delta):** hop-internal defect decays
steeply with depth — d_abs/θ = 0.0664 (hop 0, θ=0.30) → 0.0235 → 0.0130
→ 0.0076 → 0.0043 → 0.0001 (hop 9). Hop 0 also carries a systematic
+0.0225·θ overfire remnant.

**Hop-0 subsume REFUTED:** running hop 0 in value domain (the walk's
`is_encoding_layer` path ≡ `encoding_layer_placement=subsume`) changes
nothing (census 0.7344 ≈ 0.7376 baseline). LESSON: matched-input delta
MAGNITUDE is not accuracy ATTRIBUTION — LayerNorm absorbs hop-0's error
entirely. (The k-cut's −1.6 at cut 0 was the value-pin, not the count.)

**The hop-1 discriminator (hop-0 pinned genuine, hop-1 output variants
through the analytic suffix, n=512):** clean 0.8262 / +real-d 0.7676
(reproduces the k-cut drop) / +same-RMS iid 0.7969 / +per-channel-mean-
of-d 0.8203. DECOMPOSITION: the −5.9pp at hop 1 = ~3pp noise MAGNITUDE
at a genuinely sensitive coordinate (iid at that RMS is NOT free on this
artifact — §14's noise-immunity result does not transfer) + ~3pp
higher-order STRUCTURE (token/spatially-correlated deviations), with
per-channel bias INNOCENT (−0.6) — the DFQ/bias-correction fix class is
retired for this term.

**Standing lever candidates for the ~8pp composition term, in order:**
(1) per-hop temporal allocation — give early hops more T within the
budget (`s_allocation` budget objective is a REGISTERED design axis;
hops 3+ are free, hops 0–2 carry everything — the allocation is
maximally lopsided today); (2) the artifact axis (AB5's AA endpoint leg,
in flight); (3) hop-targeted train-through of hops 0–2 only (weak — full
train-through was flat, but the targeted form is untested).

**PR14b LANDED (gate 8312):** ActivationAdaptationTuner gains the funded
endpoint-recovery leg (`aa_endpoint_recovery_steps`, default 0), closing
the one unfunded endpoint among conversion tuners; the endpoint-family
registry entries move to `entries_endpoint.py` (both prior tables at the
300-LOC cap; `config_schema/registry` joins the sibling allowlist with
the structural reason stated). AB5 (AA @1200-step endpoint leg, physics
knobs OFF in-chain per the AB4 lesson) is running.

### 15.15 AB6 on peta: the AA-endpoint lever fires and is REFUTED — the
### artifact deficit is not a training-budget deficit (2026-07-20)

**Infra:** AB6 ran the whole Phase-Value chain (snapshot→AA→Clamp→Shift→
AQ) in ONE uninterrupted 36-min process on peta's A100 (no reaper) —
vs sura's 9-window reaper thrash. peta is now the chain vehicle
(cu126 wheels; `torch.backends.cudnn.enabled=False` for the one ViT conv;
`MIMARSINAN_DISABLE_FFCV=1`; recorded in cluster-topology memory).

**The AA anchor fix WORKED (the leg fired) — and the lever is REFUTED.**
With the target anchored to origin (0.8678, envelope-capped to 0.852) the
AA endpoint leg engaged for the first time: entry 0.8125, budget 1200,
steps_used 1200, engaged=True — **exit 0.8125, reached=False,
rolled_back=False.** 1200 value-domain KD-to-origin steps with a genuine
+4pp of headroom yielded EXACTLY ZERO (keep-best never once beat entry).
Deployed census (knobs at deploy, n=2500): **genuine 0.7368 / analytic
0.8144 — statistically identical to AB3's 0.7376 / 0.8244** (the leg also
did not hurt: non-destructive as designed).

**The finding:** the AA swap deficit (GELU→ReLU, −6..−8pp) is NOT a
training-budget deficit. This is the SECOND artifact-axis training lever
to go flat (Phase-Deploy §15.12 was the first) — both refuted with real
headroom and full budgets. The artifact sits at its joint capacity for
value-domain KD-to-origin; more optimization does not move it. LAW
(artifact-training saturation): once the anchored ladder finalizes, the
value composition is at a KD-to-origin local optimum; neither a terminal
Phase-Deploy nor a funded endpoint leg recovers the swap loss — the swap
LOSES information no post-swap objective can reconstruct.

**Consequence — the artifact lever moves to the SWAP ITSELF, not its
recovery.** The GELU→ReLU morph is the −6..−8pp origin→analytic term;
recovering it post-hoc is refuted, so the lever is a SOFTER/RICHER swap:
(a) a mappable parametric activation nearer GELU held through conversion
(PReLU/│x│-gated families the mapper already supports), or (b) a
two-target morph (GELU→SiLU-approx→ReLU) spreading the cliff. Both are
design changes to Activation Adaptation's target, testable analytic-first
(no deploy needed to read the swap loss).

**On PR14b's status (correcting an earlier mis-statement here):** PR14b is
a SINGLE mode-generic mechanism — `_post_stabilization_hook` reads one
config key `aa_endpoint_recovery_steps` (default 0, off for EVERY
workload) and, when armed, runs the same `run_endpoint_recovery` every
other conversion endpoint uses; its only branch is on the generic
`origin_anchored_compact` lever, never on a model. It is set in ZERO
recipes/templates — the only config that ever armed it is the AB6 ViT
experiment. There is no tier-0-vs-ViT code path, and no evidence it
helps ANY workload (the sole measured arming, on ViT, recovered nothing;
a native-ReLU tier-0 cell does no GELU→ReLU swap, so it has nothing to
recover either). Its justification is the endpoint-funding SYMMETRY — AA
was the one conversion tuner with no fundable endpoint while WQ/AQ/LIF-
adaptation all have one — i.e. an SSOT-consistency completion, kept
default-off, NOT a workload-specific default. The empirical verdict
stands: on the ViT the AA swap loss is non-recoverable by this leg.

**Scoreboard unchanged (both training levers spent):** origin 0.8678 →
analytic ~0.82 → deployed genuine ~0.737 zero-training. The two live
levers are now both STRUCTURAL: the swap (artifact, −4..−8) and hops 0–2
(composition, −8, of which ~3 magnitude reachable via s_allocation +
~3 token-structure). No training lever remains.

### 15.16 Two isolated assays narrow the endgame: s_allocation refuted,
### swap-target shape refuted, artifact gap is budget-or-floor (2026-07-20)

**PR33 (hop-grid RMS vs T, isolated):** the LIF value-twin vs float-clamp
RMS is EXACTLY 1/T at every hop (×7.9 for T×8: hop0 0.0197→0.0025,
hop1 0.0091→0.0011, …). The analytic grid shrinks with T as designed —
YET the genuine census was flat across T=32/64/128 (§15.10). Grid noise
shrinks but does not move accuracy ⇒ **s_allocation (per-hop T) is
REFUTED**: the composition residual at hops 0–2 is temporal STRUCTURE
(timing), T-invariant, not grid magnitude. (This also re-reads §15.14's
"~3pp magnitude" term: the iid-injection cost was iid at the genuine-vs-
analytic RMS, which is timing-dominated, not grid.)

**PR32 (swap-target screen, isolated):** origin GELU 0.883 (n=512) →
ANY clamped-ReLU/LIF target installed with NO training craters to chance
(scalar-θ 0.027, per-channel-θ 0.033, 1.5×θ 0.025, +guard 0.031). The
raw swap loss is target-SHAPE-independent — every mappable rectifier
collapses a GELU backbone equally; per-channel θ / guard / clamp width do
not separate them without adaptation. ⇒ the artifact gap is NOT a target-
resolution problem; it is the GELU→clamped-ReLU ADAPTATION, and the only
open question is whether that adaptation is BUDGET-limited (the §15.15
leg was 1200 steps ≈ 0.05 epoch) or a family floor.

**The endgame reduces to two decisive tests, both pre-registered:**
(1) **PR34 (artifact, in flight on peta):** a PROPER multi-epoch ReLU
fine-tune of the origin backbone (LIF value-twin target, KD-to-origin,
full trainset, keep-best analytic). ≥~0.86 ⇒ the artifact gap is budget,
closable with a recipe change (activation-adaptation budget ∝ swap
distance, L-C) — NOT the §15.15 saturation (which was a 1200-step leg).
Plateau ≪0.86 ⇒ clamped-ReLU is the activation-family floor for this
backbone. (2) **PR15/PR8 composed-floor certificate** for the temporal
term once the artifact resolves. Together they answer 87→87: reachable
iff PR34 clears ~0.86 AND the temporal floor ≤ a couple pp.

### 15.17 PR34 — the artifact gap is UNDER-TRAINING, not a floor: a proper
### ReLU fine-tune reaches ≥ origin (2026-07-20)

**The decisive artifact test.** A proper 6-epoch fine-tune of the ORIGIN
backbone to the deployable target (`ClampReLU(z)=clamp(z,0,θ)` — the
T→∞ LIF rate limit, fast, isolates the swap from the grid), KD-to-origin
+ CE, full trainset, keep-best analytic, lr 1e-4: entry (untrained swap)
0.037 → **BEST analytic census 0.8892 (n=2560), ABOVE the GELU origin
0.8678** (+2.1pp; ~0.874 on the pipeline census scale after the +1.5pp
split offset). The trajectory is a clean monotone climb (0.82@300 →
0.90@6300 → plateau 0.906), loss 0.53→0.10.

**This overturns the §15.15 "artifact-training saturation" reading.**
The clamped-ReLU activation family is NOT a floor — it matches-or-exceeds
the GELU origin. The pipeline's Activation Adaptation leaves ~7pp on the
table (0.82 → 0.889). Why the §15.15 AB6 endpoint leg (1200 steps) was
flat is now fully explained: (a) BUDGET — 1200 steps ≈ 0.05 epoch vs the
~9k steps (6 epoch) the swap actually needs (L-C: budget ∝ swap
distance, and the ViT GELU→ReLU swap is a LARGE distance); (b) LR — the
endpoint's default `endpoint_floor_lr`=2e-3 is the §14.7 CRATER regime;
PR34 trains at 1e-4; a crater + keep-best reads as "flat" (revert to
entry). AB6's endpoint had both wounds at once. The saturation law is
RETRACTED for the swap: it was a budget+LR artifact, not information loss.

**Consequence — the artifact axis is OPEN, and it is the biggest lever in
the program.** origin 0.8678 → a properly-trained analytic artifact
≈0.889 (this split) reclaims the entire −4.3pp artifact term and then
some. Generic fix: the conversion recipe must fund AA in proportion to
the swap distance (real multi-epoch budget at a sane genuine-LR), exactly
the L-C principle — not a ViT special case; any large-activation-swap
workload inherits it. PR34 artifact saved (peta pr34_relu_best.pt).

**Next (in flight): AB7** — the full pipeline chain with the AA endpoint
armed at a REAL budget (`aa_endpoint_recovery_steps`≈8000) and a sane LR
(`endpoint_floor_lr`=1e-4), origin-anchored, physics knobs at deploy —
the end-to-end pretrained→deployed number from the better artifact. If
the deployed census jumps from 0.737 toward the artifact minus the
(knob-cured) temporal tax, 87→87 comes into range on the artifact side
and the endgame reduces to the temporal floor alone.

### 15.18 AB7 — the artifact axis does NOT transfer to deployed: analytic↑
### ⇒ genuine↓ (the anti-correlation wall) (2026-07-20)

**AB7 (full chain, AA endpoint 8000 steps @ lr 1e-4, origin-anchored):**
the budget+LR fix WORKED at the step level — the AA endpoint reached its
target (entry 0.8125 → exit 0.859, reached=True in 380 steps; vs AB6's
flat exit=entry), AA step 0.79→0.8345, AQ analytic 0.8477 (best chain
analytic yet). BUT the deployed census is WORSE: genuine **0.6936** (knobs
at deploy, n=2500) vs AB3's 0.7376, with analytic 0.8268 (marginally
above AB3's 0.8244). The 2×2 (locked/knobs) rules out a knob mismatch —
knobs help AB7 too (locked 0.496 → knobs 0.694, +20pp, same as every
artifact).

**The scoreboard, sorted by analytic, exposes the anti-correlation:**

    artifact   analytic   genuine(knobs)   tax
    AB6        0.8144     0.7368           7.8
    AB3        0.8244     0.7376           8.7
    AB7        0.8268     0.6936          13.3   <- best analytic, worst genuine

More value-domain training buys analytic accuracy and PAYS it back (with
interest) in the temporal composition. This is §13.4's trajectory-
dependent residual, now quantified as a monotone anti-correlation:
optimizing the value surrogate moves pre-activations onto configurations
the genuine temporal composition handles WORSE. PR34's 0.889 analytic
would very likely deploy even worse still.

**Consequence — the artifact axis is a MIRAGE for the deployed metric.**
PR34's headline (the clamped-ReLU family reaches ≥origin analytic) is TRUE
and important for understanding, but it does not move R_dep: the deployed
number is bottlenecked entirely by the analytic→genuine temporal
composition, which value-training makes worse. This fully vindicates the
§15.2 deployed-risk principle: only training that passes THROUGH the
genuine composition can raise R_dep; value-artifact quality is not just a
free gauge (§15.10) — beyond a point it is actively HARMFUL. The one
lever that ever raised genuine was genuine train-through (PR22: 0.64→
0.7676) and the physics knobs (0.55→0.74). Both partial; both plateau
~0.74–0.77.

**The endgame question is now sharp and singular:** is the ~0.77 genuine
composition ceiling the S=32 floor, or is there an unfound lever? Two
hard facts bound it: (a) it is S-INVARIANT (S-sweep flat 32/64/128) —
so higher S does NOT help, the tax is timing-structural not grid; (b) it
resists value-training (anti-correlated) and saturates genuine train-
through. The remaining candidates are the deployment-neuron degrees of
freedom that change the TEMPORAL physics itself (firing/reset mode,
threshold convention, boundary) — tested next — and, failing those, the
PR15/PR8 floor certificate to price whether 87→87 needs a chip-model
change rather than a training change.

### 15.19 Temporal-DOF exhausted; the last composability test (2026-07-20)

**PR36 (deployment-neuron temporal DOF, zero-training on AB3+knobs):**
Default reset 0.7617 (both thresholds `<`≡`<=`, comparator inert as
§15.9); **Novena hard-reset WORSE (0.6738)**; guard/dither essential
(dither-off 0.4277). No reset/threshold/boundary setting beats the
Default+dither+guard config. The deployment-neuron degrees of freedom do
not break the ceiling — the temporal tax is not a reset/convention
choice.

**State of the genuine ceiling (~0.74–0.77 at S=32), fully bounded:**
S-invariant (§15.10), reset/threshold-invariant (PR36), anti-correlated
with value-training (§15.18), and it saturates genuine train-through
(§14.7 PR22 0.7676). Every zero-training and value-training lever is
spent. ONE composition test remains untried: genuine train-through (the
only thing that ever RAISED R_dep) starting from AB7's BETTER artifact
basin (analytic 0.827 vs AB3 0.824) — §15.18 showed value-training the
better artifact hurts, but training the GENUINE composition from a better
basin is a distinct experiment (PR35). If it breaks 0.77, the artifact
axis composes with genuine training after all; if it saturates ~0.77, the
S=32 signed-IF temporal floor is confirmed and 87→87 is a chip-model /
higher-S design decision, to be priced by PR15/PR8.

---

## 16. The synchronized-rate exactness theorem: the one mechanism behind
## every residual, and the construction that eliminates it (2026-07-20)

### 16.1 The final harvested datum

PR35 (genuine train-through from AB7's better basin, stopped at 540):
entry 0.7012 → 0.7422@480, flat thereafter — the FIFTH independent
saturation in the same ~0.74–0.77 band (PR22 baseline basin 0.7676;
PR23 1200 steps 0.7688; PR23′ healed 0.7536; fidelity arm 0.7676; PR35
better-artifact basin ~0.742). The ceiling is basin-independent: it is a
property of the COMPOSITION, not of any artifact.

### 16.2 The noise object (why the ceiling exists and why it is invariant)

Under STREAMING execution a hop's neuron fires while it integrates. Per
cycle t it receives charge z_t = Σ_i w_i s_i(t) + b, a random-arrival sum
over input combs. Decompose z_t = z̄ + ε_t. The membrane is a random walk
with a one-sided absorbing barrier (a fire is irreversible; later
negative charge cannot cancel it). The emitted count therefore differs
from the count of the TOTAL charge by a LEVEL-CROSSING statistic — the
number of spurious barrier crossings — whose scale is governed by the
per-cycle noise-to-threshold ratio σ_z/θ, where

    σ_z² = Σ_ij w_i w_j Cov(s_i(t), s_j(t))
         = Σ_i w_i² r_i(1−r_i)   [duty-cycle diagonal]
         + Σ_{i≠j} w_i w_j C_ij  [comb-resonance off-diagonal].

Every empirical property of the residual follows from this ONE object:

- **S-invariance** (§15.10, §15.13): comb duty per cycle is r_i at ANY
  T, so σ_z is T-invariant — more cycles do not reduce per-cycle noise.
  (PR33's 1/T grid RMS measured the VALUE twin, which is not the noise.)
- **Phase dither's +21pp** (§15.11): the encoder anchored every comb at
  cycle 0 — a maximal deterministic C_ij. Dither kills the deterministic
  part; the stochastic resonance of similar-rate (value-correlated)
  channels survives = the measured "token-correlated structure" (§15.14).
- **Guard dose-response / V0=θ/2 catastrophe** (§15.11): V0 shifts the
  barrier's operating point; positive pre-charge feeds crossings.
- **Early-hop concentration** (§15.13, PR31's d_abs/θ decay 0.066→0.0001):
  σ_z/θ is largest where θ is small (stem hops).
- **Anti-correlation with value-training** (§15.18 AB7): value objectives
  sharpen activations/weights, RAISING σ_z and packing pre-activations
  near decision margins — analytic ↑, genuine ↓.
- **Train-through saturation ~0.77** (five basins): training reshapes
  MARGINS against a noise floor it cannot remove; the band is where
  margin-shaping saturates.
- **Novena worse, comparator inert** (PR36); **hop-0 subsume inert,
  chanmean inert, iid costs half of real-d** (§15.14) — all consistent:
  the noise is path-structural, not convention or bias.
- **Tier-0 lossless / mixer τ=4.2**: shallow nets with wide margins sit
  below the noise; depth-12 ViTs do not.

### 16.3 The theorem (elimination by construction)

**Synchronized-rate exactness.** Let a hop run TWO windows: an
integration window in which firing is disabled and the membrane
accumulates the full signed input, V_T = (W·N_in + bT + V0·θ)/θ (a
function of input COUNTS only, since Σ_t W s_t = W Σ_t s_t = W·N_in);
then an input-free emission window of T cycles with fire-and-subtract.
The emitted count is exactly

    N_out = clamp(⌊V_T⌋_strict, 0, T),

the LIF count staircase of the decoded value — for ANY arrival pattern,
any T, any fan-in, any topology. The level-crossing statistic is
identically zero: the genuine temporal composition ≡ the analytic
staircase composition, hop-exactly, hence end-to-end (host ops already
run on decoded values between windows; boundary re-encodes are
count-preserving). This is the A2 kernel identity (`LIFCountStaircase`,
already bit-locked in exact-QAT) PROMOTED from the kernel to the
EXECUTION DISCIPLINE. It is the in-house synchronized schedule the TTFS
family already runs ("latency groups sequential, sim_time = S × groups"),
applied to the rate/LIF family.

**Corollary (the two-phase theorem, exact form).** Under synchronized
execution the value surrogate is not a surrogate: s_W ≡ g_W. Phase-Value
training IS deployed training; §15.18's anti-correlation dissolves; the
§15.2 surrogate-validity criterion is satisfied by identity. The deployed
number equals the analytic number BY CONSTRUCTION, and the whole
remaining program is the VALUE ladder — where the measured ceiling is
0.8892 (PR34, ≥ origin 0.8678) and the chain already reaches AQ-endpoint
0.8477 with the funded swap.

**Costs, stated honestly.** (1) Inference LATENCY becomes (D+1)·T cycles
(ViT-12 @T=32: 416 vs ~44) — but pipelined THROUGHPUT is unchanged (one
inference per T cycles); conversion wall-time IMPROVES ~S× because every
genuine gauge/eval collapses to the analytic forward. (2) The membrane
must hold the full-window signed accumulation (range ≤ Σ|w|·T/θ):
a per-platform capability bit (Loihi-class membranes suffice; platforms
without fire-disable/phased scheduling get a fail-loud capability check).
(3) Guard/dither remain valid knobs for STREAMING deployments where
latency is priced; synchronized is the lossless mode.

### 16.4 The victory plan (SSOT, generic, 15–20 min to SCM)

- **PR37 — the discipline in the walk (tests-first):** a mode-generic
  `lif_execution_discipline: streaming|synchronized` predicate in
  `spiking_semantics` + contract. Under synchronized, the NF genuine
  forward IS the analytic staircase forward (twins merge; property test:
  bit-equality of the per-cycle two-window simulation vs the staircase on
  randomized hops — ties, guards, saturation, signed bias edges).
- **PR38 — HCM two-window cores:** integration [lat, lat+T) with
  fire-disable, emission [lat+T, lat+2T); ChipLatency generalizes
  (lat_k = k·T); nf_scm_parity must read BIT-EXACT per neuron.
- **PR39 — backends:** nevresim + SANA-FE reuse the existing synchronized
  group scheduling (TTFS sync path); Loihi/Novena capability flags,
  fail-loud where unsupported.
- **PR40 — AB8, the gate run:** full chain on peta with synchronized
  discipline + the PR34-grade funded swap (aa_endpoint at epoch-scale
  budget, lr 1e-4) + AQ exact-QAT (whose gauge is now free) + WQ funded
  endpoint → SCM parity. **DoD: pretrained→deployed ≤2pp (PR25) — i.e.
  deployed ≥0.8478, target ≈0.86+ — inside a 15–20-min wall.** The wall
  fits because the slow object (per-cycle genuine reads) no longer exists
  in the loop.
- **PR15/PR8 floor certificate, now provable:** deployed floor ≡ analytic
  floor = staircase quantization only, bounded by the measured 1/T grid
  constants × sensitivities (PR33) — ≲1pp at T=32, and TRAINABLE-through
  by exact-QAT (A2). 87→87−ε follows; ε is the certified quantization
  floor, priced by T.

### 16.5 PR37 landed + validated; the last residual is named: the ENTRY
### composition (2026-07-20)

**PR37 LANDED (gate 8318, typecheck 0):** `lif_execution_discipline:
streaming|synchronized`; the theorem's property tests pass (two-window ≡
strict staircase for arbitrary arrival, order-invariant, integer-tie
exact; streaming ≡ sync only at constant input, bursty counterexample
locked). The sync walk runs a census in 43s (vs ~150s streaming) — the
S-fold gauge speedup is real.

**Empirical validation + the reframe it forced.** Sync census on AB3:
**0.7336 — equal to streaming+dither+guard (0.7376), NOT to model(x)
(0.8244).** Read: (a) the §16 noise object is CONFIRMED AND CLOSED —
sync replaces dither+guard exactly (same number, zero knobs, faster);
(b) the remaining ~9pp was never execution noise — it is shared by every
execution mode. Ablations then acquitted the sync value-path seams (grid
and clamp INERT: all four arms 0.7578@512) and the per-hop divergence
ledger walk-vs-model(x) under sync shows the §14 signature (d_abs/θ
0.066→0.037→0.023→…→0.0001, d_mean +0.016 at hops 0–1) — with the root
measured at last: **hop-0's PRE-ACTIVATION already differs by d_abs
0.21·θ (max 5.5·θ)** between the walk's entry path and the model's input
path. The entire residual is the ENTRY composition — a deterministic
gauge/convention (Type-B family) defect upstream of the first neuron,
compressed by hop-0's clamp to 0.066·θ and echoing through the stem.

**State of the program:** every stochastic/temporal mechanism is now
eliminated BY CONSTRUCTION (sync) or measured-inert; the gap decomposes
as origin 0.8678 → model(x)-analytic 0.8244 (the value ladder — PR34
ceiling 0.889 ≥ origin, budget-funded recipe known) → deployed 0.7336,
where the second arrow is ONE deterministic entry-composition defect.
Next: the finite entry bisect (input_data_scale / ChipInputQuantizer
placement / entry encode input handling — enumerable, each arm seconds
under sync), then the fix at the entry SSOT, then AB8 (sync + funded
swap): the PR25 gate becomes the value ladder alone.

### 16.6 The last defect, caught: entry currency incoherence κ_S=2.64 vs
### per_source=[1.0] (2026-07-20)

**The bisect (first-divergence over all 90 leaf modules, walk vs model,
sync-deterministic):** graph_node_1 (conv_proj, Conv2d, ARMED
output_scale=2.6400) is BIT-EQUAL in both compositions; graph_node_4
(the first consumer ComputeAdapter) receives input model_mean=0.1143 vs
walk_mean=0.0433 — **ratio 0.3789 = 1/2.64 exactly**. Its stored
`per_source_scales = [1.0]`.

**Mechanism.** The producer emits wire = value/κ (κ=2.64). The model's
composed forward decodes correctly because the SNW tuple-transparency
seam (ad7e1251) carries the scale alongside the tensor. The walk's
`rate_of` hand-off passes the bare tensor, so the consumer falls back to
its STORED currency — per_source_scales=[1.0] — and the ×2.64 decode
never happens. This is the §11.2 Currency Coherence Theorem violated at
the entry: κ_S(producer)=2.64 ≠ κ_T(consumer)=1.0 — the B2-class
"fourth self-consistent gauge system" §10.4 predicted, now measured to
four digits. It poisons hop-0's pre-activation (d_abs 0.21·θ), echoes
through the stem (the §14 drift signature, the k-cut hops-0–2 term, the
anti-correlation's fuel), and — critically — the STORED table is what
the chip-side weight fold consumes, so the defect reaches every backend,
in every execution discipline. It was never temporal.

**The fix (one-writer, SSOT):** the σ-install's coherence pass must
STAMP every consumer's per_source_scales from its producers' armed
output_scale (one writer), and `verify_boundary_currency_coherence` must
check the (producer output_scale ↔ consumer per_source_scales) pair —
today it doesn't, which is how 2.64 vs 1.0 survived install. Then the
walk (honest stored-currency reader), the model's tuple path, and the
chip fold agree by construction. Predicted effect: sync census
0.7336 → ≈ model(x) 0.8244; the deployed gate reduces to the value
ladder (PR34 ceiling 0.889 ≥ origin), i.e. PR25 in range. Next: the
stamping fix tests-first → sync re-census → AB8.

### 16.7 CLOSURE: deployed ≡ analytic (2026-07-20)

**The fix landed (gate 8321, tests-first):** (1) `apply_compute_op_scale_
policy` — an ALREADY-armed op refreshes its `per_source_scales` from the
walk and reports its emitted `output_scale`, so consumers decode armed
producers at their true currency (the pre-arm's unity slots were
invisible to the walk — the §10.4 "fourth gauge system", closed);
(2) σ-install re-propagates whenever anything is armed (the pre-arm-only
path previously skipped it); (3) `verify_boundary_currency_coherence`
now checks every (producer emitted gauge ↔ consumer per_source) pair,
fail-loud — the check whose absence let 2.64-vs-1.0 survive install.

**The pre-registered decisive read, on the cached AB3 artifact:**
re-propagation stamps node_4's per_source 1.0000 → 2.6400; the
certificate reads GREEN; and the census (n=2500):

    SYNC-DEPLOYED = 0.8268    model-analytic = 0.8260    Δ = 0.0008

The deployed composition EQUALS the analytic model. The prediction
(0.7336 → ≈0.8244) is confirmed; both reads landed slightly above it
because the repair also coheres the model's own SNW path. The deployed
trajectory of this arc: 0.5527 (streaming, locked) → 0.7376 (physics
knobs) → **0.8268 (one currency stamp; zero training; zero knobs)**.

**What remains for 87→87 is the VALUE LADDER alone: origin 0.8678 →
0.8268 = −4.1pp**, entirely the GELU→ReLU swap + AQ quality — the axis
PR34 measured at ceiling 0.889 (≥ origin) with the funded-swap recipe.
The anti-correlation (§15.18) is dissolved by construction: with
deployed ≡ analytic, value-training IS deployed-training. Next: AB8 —
the chain with the fixed install (currencies stamped natively) + funded
AA at sane LR; then PR38 HCM two-window bit-parity + backends; PR15's
floor certificate is now the trivial statement deployed-floor ≡
analytic-floor.

**Discipline-equivalence certificate (n=2500, repaired AB3):** streaming+
dither+guard **0.8240** ≡ synchronized **0.8268** ≡ analytic **0.8260**
(all within SE). With coherent currencies and a decorrelated encoder the
three compositions agree — the streaming↔sync equivalence holds exactly
as the event-based argument demands; `synchronized` remains a gauge/
verification instrument, not a deployment requirement.

### 16.8 AB8 — the victory-gate run: the fixed pipeline delivers its best
### deployed number natively (2026-07-21)

**The chain (peta, ~26 min compute):** origin 0.8678 cached → AA with the
origin-anchored endpoint at the TRUE origin target (stale envelope cap
dropped): entry 0.8398 → **exit 0.8828** → Shift 0.8404 → σ-install
(currencies stamped natively — the §16.7 fix in-chain; scoped certificate
GREEN; one conv_proj skip) → AQ endpoint entry 0.8750 → **exit 0.8945**
(eval-256, above origin). Along the way the first run's install
fail-loud caught a REAL scope error in the new certificate: 2.64 is also
the imagenet input scale, and input-fed edges correctly hold
per_source=1 (torch hosts consume raw values); the pair law is now
scoped to ARMED-producer edges (traced through structural nodes) —
tests extended, gate 8322.

**Census (n=2500): SYNC-DEPLOYED = 0.8336, analytic = 0.8360** (Δ=0.24pp,
composition free on a FRESH artifact, zero post-hoc repair). Deployed
census arc across the program: −12.8pp → **−3.4pp** vs origin.

**Discipline equivalence (repaired AB3, n=2500):** streaming+knobs
0.8240 ≡ sync 0.8268 ≡ analytic 0.8260 — the event-based equivalence
holds measured; sync is a gauge, not a requirement.

**What separates 0.8336 from 87→87 (−3.4pp):** entirely value-ladder
recipe economics, now visible in one place — the endpoint legs SELECT on
eval-256 reads (SE ±2.8pp; endpoint 0.8945@256 ≈ 0.836 census), so
keep-best optimizes a noisy gauge; plus the Clamp/Shift/AQ descent from
AA's 0.88-class peak. Levers (knobs, not defects): census-graded
endpoint evals (larger eval_subsample for the endpoint legs), retention
between AA→AQ, and AQ budget. PR25 (deployed ≥0.8478) is −1.4pp away
with those knobs; PR38 (HCM two-window bit-parity) + backend capability
flags remain for the cross-simulator certification.

### 16.9 AB9 — PR25 PASSED: pretrained 0.8678 → deployed 0.8592 (−0.86pp)
### (2026-07-21)

**One knob (census-graded endpoint evals, eval_subsample 256→2048) closed
the selection-noise term exactly as diagnosed.** The chain (40.6 min,
one process): AA endpoint entry 0.8114 → exit 0.8687 AT the origin
target, AA step census-grade 0.8642 (−0.36pp — the GELU→ReLU swap is
now near-free); Clamp and Shift exactly flat; AQ entry 0.8556.
**Census (n=2500): SYNC-DEPLOYED 0.8592 ≡ analytic 0.8588 (Δ 0.04pp),
coherence certificate GREEN. PR25 (pretrained→deployed ≤2pp): PASSED
with 1.1pp margin.** Program arc on the gate metric: −12.8pp → −0.86pp.

**PR38 LANDED alongside (gate 8325):** the HCM synchronized reference is
a COUNT-DOMAIN executor (`sync_counts.py`: memb = W·counts + bias·T
(+V0·θ) → the strict `lif_count_staircase`) — bit-equal to two-window
execution by the §16 theorem, arrival/window-free (latch-correct for
level-gapped consumers), locked bit-for-bit against the NF synchronized
walk, and contracted through flow/contract/factory. Single-spike and
recording paths keep the streaming loop.

**In flight:** the WQ → Soft Core Mapping leg on the AB9 artifact
(streaming+knobs physics, HCM-coherent) — the full simulator-verified
deployed number. Remaining to literal 87→87: −0.86pp of value-ladder
polish (AQ funded further / AA→AQ retention) — same knobs, no unknowns.

### 16.10 Metric-of-record correction + the streaming arbiter (2026-07-21)

**Correction (user-caught):** the §16.9 headline (0.8592) was read under
the SYNC gauge, and the AB9 chain's acceptance gauges also ran sync —
the deployed claim was not yet anchored to streaming execution. LAW
(metric of record): every headline deployed number is a STREAMING
pipelined-spike census (fire-during-integrate, per-cycle, signed LIF);
`synchronized` is an internal gauge/verification instrument and an
opt-in mode, never the default deployed claim. SCM certification metrics
likewise read the STREAMING HCM.

**The arbiter (n=2500, AB9 WEIGHT-QUANTIZED artifact, streaming +
dither + guard, currencies repaired):**

    STREAMING genuine census = 0.8580   (origin 0.8678 → −0.98pp)

PR25 (≤2pp) PASSES under the metric of record — post-WQ, real spikes.
Streaming ≈ sync on AB9 too (0.8580 post-WQ vs 0.8592 pre-WQ), the
second measured instance of the discipline equivalence (first: AB3
0.8240/0.8268). The physics record stands as before: every lever
(dither +21pp, guard, currency +9pp) was discovered AND measured on
streaming; the streaming simulators were never altered toward torch —
the one global change (the currency stamp) fixes the chip-side weight
fold. Remaining risk retired; remaining work: streaming-HCM SCM read,
per-neuron ViT parity (both classes), PR39, tier-0 sync locks, §17.

### 16.11 SCM/IR parity debugging state (2026-07-21)

The ViT torch↔sim parity gate at Soft Core Mapping is mid-bisect; the
currency stamp arming MHA-adjacent wrappers exposed two latent IR-side
seams: (1) FIXED — the IR executor passes call-site module kwargs
(need_weights) that the armed wrapper rejected; wrappers now merge
call-site kwargs with constructor-owned winning (gate 8327, unit-locked).
(2) OPEN — a broadcast mismatch (197 tokens vs 256) inside an armed
wrapper on the emitted IR: `broadcast_scale_to_dim(scale, x.shape[-1])`
assumes channel-last, and an IR-side op (column-oriented emission /
transpose region) runs token-last; a CUDA device-side assert follows.
Next: orientation-aware scale broadcast at the IR wrapper (match the
scale to the CHANNEL axis explicitly, not dim -1), then the parity gate.

**Independence note:** the deployed metric of record (streaming census
0.8580 on the WQ artifact, §16.10) does not depend on this gate — it is
the NF streaming walk on the deployed weights. The SCM/IR parity chain
is the remaining CERTIFICATION work: Normalization Fusion is cached
(33 min paid once), mapping/packing re-runs from cache, and the gate now
fails at the named defect rather than silently.

### 16.12 ViT IR parity: executes end-to-end, agreement 0.0156 — a Type-B
### in the armed emission (2026-07-21)

Three first-contact seams fixed and unit-locked (install currency, wrapper
call convention, batch-free constant gather); the deployed IR now EXECUTES
the full ViT and the gate renders its first verdict: **0.0156 ≈ chance**
— the classic Type-B convention-break signature (§3: linear→chance),
same family as the original 0.0115 twin. PRIME SUSPECT: the DOMAIN of
constant/parameter sources through armed wrappers — pos-embed is a
VALUE-domain constant, but an armed consumer's per_source decode treats
its edges as WIRE; if the stamped per-source scale on the constant edge
is non-unity (the one-writer refresh stamps from the walk table, which
has no concept of value-domain parameter edges), the very first add is
wrong and everything downstream is noise. Instrument: the §16.6
first-divergence method, NF-walk vs IR-executor per node (hook both,
find the first diverging op — each iteration minutes from caches).

### 16.13 THE PARITY GATE PASSES: torch↔deployed-sim 0.9922 (2026-07-21)

**The ViT IR certification campaign concludes: agreement 0.9922 over 256
samples (threshold 0.98)** — the deployed simulator provably computes the
NF's function, in the healthy WQ-tie-flip regime. Five first-contact
seams were found, fixed, and unit-locked to get from crash → 0.0156 →
0.9922: (1) install currency one-writer + certificate pair; (2) wrapper
call-site kwargs precedence; (3) fail-loud gather (its interim broadcast
briefly MASKED defect 4 — silent recovery is how Type-Bs hide);
(4) wrapper-owned output_index (the batch-slicing double selection —
the 0.0156 mechanism); (5) cycle-train cache pruning (the 80GB parity
pathology; user-flagged; trains now share value-buffer lifetimes).
Remaining step blocker: the post-parity identity metric OOMs on parity
leftovers — empty_cache without gc.collect() cannot free cyclic
nn.Module tensors; fixed. Costs measured: parity phase 4957s (83 min)
at cuda_peak 68.9GB (the ViT residual topology's live frontier is
genuinely large; a future economy pass can stream it at lower batch).

---

## 17. The certification architecture: spike-count faithfulness (2026-07-21)

**Diagnosis (why argmax parity < 1.0 is structural, not mechanical):** the
old gate compared the deployed sim against the ORIGINAL torch model — two
different floating-point programs; the staircase amplifies ulp-level
membrane differences into count flips near threshold. Perfect parity
between different float programs is unattainable and the wrong target.

**The theorem:** post-WQ, chip-side arithmetic is integer weights ×
integer counts — associative, order-independent, exact in float64/int64.
The float part (LN/attention/softmax) is ONE shared torch implementation
host-side in every backend. Therefore per-neuron window-count equality
between a torch-side IR reference with genuine fire and any controlled
backend is achievable EXACTLY, by construction; and with prediction =
decode(counts) shared, accuracy(oracle) + counts(oracle ≡ backend) ⟹
accuracy(backend) — derived, not re-measured. n=1–2 samples ×
millions of neuron-windows out-powers argmax parity at any n (a Type-B
shows in the first sample, loudly).

**Landed (PR42, gate 8339):** `certification/` top-level module —
`certify_spike_counts` + `SpikeCountCertificate` (typed; exactness
classes exact|counts-export; Loihi = counts-export with documented
±1-count tolerance; fail-loud on unclassified backends/missing keys);
`spike_count_parity_samples` knob (default 2); integration lock: NF sync
walk ≡ HCM count executor per-neuron through the certificate itself.
NEXT: PR43 oracle promotion (torch-IR genuine-fire accuracy read;
model-vs-IR demoted to analytic drift check) → PR44 SCM gate swap +
legacy knob retirement (simulation_batch_count) → PR45 nevresim/SANA-FE
adoption → PR46 budgets + the deployment-time table.

**Old-regime closure (2026-07-21):** attempt 10 reproduced the parity
verdict exactly (0.9922/256 — deterministic, twice-confirmed) and then
hung 79+ min at 0% GPU inside the legacy identity metric (70.6GB still
held despite the gc pass) — killed unread. The identity metric retires
with its regime: its information is subsumed by the streaming census
(0.8580 @ n=2500, the metric of record) and, going forward, by the §17
derived-accuracy rule. Final old-regime baseline: parity 0.9922×2,
oracle accuracy 0.8580, certification wall ~85 min/attempt — the number
PR44's n=2 count-certificates replace.

### 17.1 PR44 LANDED: the alignment layer, the twin edge, and the honest
### placement of the WQ residual (2026-07-21)

**The alignment layer (generic, no name parsing).** IR output provenance
is now complete: `perceptron_index` + `perceptron_output_slice` (whole
output defaulted on the single-core FC path; bank-row slice on bank-backed
conv cores with the conv mappers passing GLOBAL group channel ranges) +
NEW `perceptron_output_column` stamped by the map_fc 2-D recursion
(token/position columns) and the conv position loops. Flat placement =
`column × channels + slice`. `PerceptronCountAssembler` maps any packed
program's stage `output_map` counts back to per-perceptron channel
vectors — drops and reports uncaptured/gap/overlap perceptrons, never
silently compares. Capture is the sanctioned `stage_count_recorder` seam
on the rate flow (fires on the raw clamp counts, pre-decode, both
disciplines). A vacuous certificate (zero windows) raises.

**The measurement that fixed the architecture.** First in-vivo gate run
(t0_05, T=4) FAILED at exact=0.9987: one window, ±1. Diagnosis at the
divergent neuron: the executor's staircase argument was EXACTLY 15.5
(f32 ≡ f64 — clean grid, no tie) vs the walk's 3.9747 normalized — the
two sides compute genuinely different arguments. Layer-wide: |Δarg|
median 0.021 counts, p99 0.12, max 0.22 — the HONEST WQ/chip-grid
residual (model float weights vs core matrices), the same object the
mixer NF↔SCM investigation closed as not-a-bug. Count flips at staircase
boundaries are its inevitable shadow (~0.1–0.5%/window at T=4). ViT
scale (cert6, n=2, 14.5M windows): NF-edge exact=0.9938, max|d|=5,
depth-U-shaped (worst mid-depth p6 0.981) — the residual accumulates
then partially cancels.

**The corrected §17 edge structure:**
- Edge O (oracle accuracy): streaming census on the deployed-grid
  program — measured directly (fast, torch).
- Edge R (model ↔ chip grid): the WQ residual; governed by the EXISTING
  `nf_scm_parity` atol gate + analytic drift report. NOT a count
  certificate — counts legitimately flip at boundaries here.
- Edge C (the certificate, FATAL): identity-mapped IR twin ↔ packed
  program, same core matrices, synchronized discipline —
  `certify_twin_flow_counts`, atol=0. In vivo t0_05: **PASS
  exact=1.000000 over 788 windows**; tiny fixture: exact on both
  disciplines.
- Edge T (transient report): streaming vs synchronized on the packed
  program — the §15/§16 per-cycle transient physics (4.3% windows ±1 at
  T=4), REPORTED with the streaming census accuracy as that cell's
  arbiter; never an atol=0 gate.
- Edge B (PR45): backends (nevresim/SANA-FE/Loihi) certified against the
  HCM executor in the matching discipline — chip-math exact classes.

Gate wiring: `run_spike_count_certificate_gate` in HardCoreMappingStep
(LIF-only, `spike_count_parity_samples` n=2 default, 0 disables), before
the metric read. Full step green in vivo: certificate PASS + transient
report + HCM 0.9809. Gates 8340→8352 all green through the sequence.

### 17.2 cert7 — THE ViT-SCALE TWIN CERTIFICATE PASSES (2026-07-21)

Identity-IR twin vs the packed t2_04 ViT program (NF-cache artifact, T=32,
n=2, synchronized): **PASS exact=1.000000 max|dcount|=0 over 14,524,416
neuron-windows** — every hop aligned, full coverage both sides. The §17
exact edge (Edge C) holds at 86M-parameter scale: packing, scheduling,
per-hop retiming, axon fill, and the count-domain executor introduce ZERO
count deviation between the 1:1 IR program and the deployed packed
program. Combined with the in-vivo t0_05 gate PASS, Edge C is now
demonstrated at both ends of the scale ladder.

Pricing (PR46 datum): the identity-reference flow dominates — ~80-90 min
wall at ViT scale (per-core python loops over ~thousands of single-node
cores; 158GB RSS on peta). Fine as a one-off artifact certificate and at
tier-0 scale (seconds); TOO SLOW as a routine per-run ViT gate in this
naive form. Follow-up: vectorized IR count reference (graph-level
staircase walk over gather plans — no identity mapping/flow machinery) or
cache the identity counts per artifact. The packed-side runs are minutes.

**cert7 transient tail (Edge T at ViT scale):** packed program, streaming
vs synchronized, T=32, n=2: mismatch 1,217,683/14,524,416 neuron-windows
(**8.38%**), **max|dcount| = 30** (of T=32). The per-neuron transient tail
is HEAVY — yet the census is equivalence-grade (AB3 0.8240/0.8268/0.8260;
AB9 0.8580/0.8592): the §15/16 per-cycle physics is large per neuron and
zero-mean at the decision level. This closes the design argument: any
count-tolerance gate on the streaming cell (atol 1, or any fixed atol)
would be both too strict (real physics, not defects) and meaningless
(max 30); the streaming cell's arbiter is the census, full stop. Total
cert7 wall ≈ 2h10m (mapping ~25m; identity sync ~55m dominates; packed
runs minutes-scale each).

### 17.3 PR45 complete: all three backends wired; nevresim refines the
### edge taxonomy again (2026-07-21)

SANA-FE and Loihi: typed `SpikeCountCertificate`s
(`certify_run_records`) now ride their existing fatal first-diff asserts
(per-core in/out + segment output counts; Loihi = counts-export ±1
class). Both are EXACT cells because their runners are timing-aligned to
the HCM reference (SANA-FE's four-part timing fix; Lava replays HCM
segments).

nevresim: the hybrid runner gained the `stage_count_recorder` seam (raw
pre-decode stage counts) and the Simulation step compares them against
the HCM STREAMING flow on identical inputs. First-ever count-level
measurement (t0_05, T=4, n=2): **exact=0.8198, max|d|=2 — 18% of windows
— while decision parity is 1.0.** Mechanism class: nevresim times its
per-cycle program independently (window-edge transients, the same object
as the historical seg-output window-gate seam), so "matching discipline ⇒
exact" is REFUTED between independent per-cycle executors — chip-math
exactness needs a shared arrival schedule, not just shared matrices.
Classification: Edge T′ — transient REPORT with the decision-parity probe
as arbiter (in-vivo green: parity 1.0 + report). The open lever for
promoting nevresim to an exact cell is timing alignment (window gating
aligned to the HCM flow, as done for SANA-FE).

Final §17 cell map: (O) census oracle, measured · (R) model↔grid WQ
residual, atol-gated · (C) identity-twin↔packed sync counts, FATAL
exact=1.0 (t0_05 + ViT 14.5M) · (T) packed streaming↔sync, report ·
(T′) nevresim↔HCM streaming, report · (B) SANA-FE/Loihi↔HCM, FATAL
typed certificates. Gates 8358 green, typecheck 0.

### 17.4 PR46 — the certification time table (2026-07-21, measured)

| Cell | Scale | Cost (measured) | Regime |
| --- | --- | --- | --- |
| OLD: argmax-parity + identity metric | ViT | ~85 min/attempt, 68.9 GB peak, hung twice, non-convergent target | RETIRED |
| (C) twin certificate, sync, n=2 | tier-0 (t0_05) | seconds (in-step; full HCM step green incl. cert + transient report) | routine FATAL gate |
| (C) twin certificate, sync, n=2 | ViT (t2_04) | ≈2h10m one-off — packed mapping build ~25m + identity flow ~55m + packed sync ~15m; identity flow dominates | per-artifact instrument until the vectorized IR reference lands |
| (T) streaming transient report | ViT | rides the same probe (packed streaming ~35m at n=2) | report |
| (B) SANA-FE / Loihi certs | tier-0 | rides the existing sim steps (no added sims) | FATAL typed |
| (T′) nevresim report | tier-0 | rides the existing probe + one packed streaming forward at n | report |
| (O) census oracle (accuracy) | ViT | streaming census n=2500 (the AB9 read) — unchanged, the metric of record | measured |

Bottom line: certification that used to cost ~85 min/attempt and never
converge is now seconds-scale at tier-0 and minutes-scale at ViT, with
the routine per-run gates all riding existing step work.

**The cost lever landed same-day (commit ad765e99):** profiling showed
99.7% of the identity-reference wall was `ChipLatency.calculate()` —
799M recursive `get_delay_for` calls (1.4B `abs`) at per-neuron python
granularity, ~440× redundant per core on identity segments. One
vectorized pass per core in topological order (bit-equal to the
recursive oracle, locked on randomized DAGs; matrix rows past
axon_sources allowed only where weights are zero — the bias-row
contract) measured **336s → 4.3s per ViT identity stage (78×)**: the
identity reference drops ~55 min → ~2–3 min and the ViT twin gate
becomes routine. End-to-end cert8 re-run in flight to reconfirm the
PASS on the vectorized path.
