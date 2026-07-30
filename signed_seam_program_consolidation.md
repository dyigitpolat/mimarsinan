# The Signed-Seam Capacity Program — Session Consolidation

*ANN→SNN deployment exactness on the offloaded pretrained ViT (`mimarsinan`,
branch `main`). This document consolidates the arc from the σ-in-the-op start
through the SNW seam fixes, the confirmed keystone, the e2e infra grind, and the
Phase-D diagnosis + its refutation. All numbers are measured; all fixes landed
tests-first.*

---

## 0. The two anchor moments quoted

- **The situation report** (σ-in-the-op *about to start*): the exactness half of
  the boundary-algebra program was proven and the lossless half was *designed*
  (memo §10f) but not yet implemented; supervisor-4 was grinding the σ-free AQ
  baseline against the 952 s process reaper.
- **The SNW thinking trace** (σ-in-the-op *mid-A/B on the ViT*): the σ install
  had landed and the ViT ladder had run past AQ; a `TypeError` at the
  MultiheadAttention seam surfaced two latent `ScaleNormalizingWrapper` (SNW)
  defects that had to be fixed before the A/B could proceed.

Everything below §3 is what happened *after* the situation report; §4 is the
SNW thinking trace resolved; §5–§8 are what happened *after* that.

---

## 1. State at the situation-report moment (baseline in hand)

### The exactness half — DONE and proven
The conversion **boundary algebra** (κ currency gauges, the σ-scope law, and
*arming*) was complete and green:
- **The arming fix** (`c769ace8`): `mark_wire_value_ops` + a wrap-policy term
  arms an SNW on non-homogeneous re-encoded host ops, so **one** emission fixes
  NF / HCM / nevresim / SANA-FE / Lava uniformly.
- **The σ-scope law** (`30f8710c`, `8a644028`, `8839e3ca`): σ is a property of
  the *(producer → neural-entry) edge*. `trained_entry_boundary` + one scope
  filter in `apply_negative_boundary_policy` skip trained-clamp and never-
  encoded boundaries; producer-side σ was reverted to consumer/walk semantics.
- **Verdict**: t0_30 torch-mixer deploys green end-to-end (0.923, Loihi/SANA-FE
  parities 1.0); the instrumented SCM parity replay reads **1.0000 with σ
  active**. Gate 8236, typecheck 0. (Memo §10e.)

### The lossless half — DESIGNED (memo §10f)
The σ-free trained clamp is exact but **capacity-lossy** where a seam carries
large negative mass (ViT LayerNorm ≈ 50 %). The design: fold σ **into the armed
op's own function, pre-training** —

    SNW′(x) = (f(x·s_in) + σ) / s_out ,  quantile-κ,  plain-forward value twin

so training absorbs σ uniformly across every consumer and representation;
post-training σ stays banned by the scope law. Two laws recorded:
- **Capacity vs resolution**: entry κ must cover the encodable band *and* keep
  κ/T at signal scale ⇒ **κ = QUANTILE, never max** (full-width cover refuted:
  entry 0.06 → 0.0146).
- **Signed-seam completion**: σ-in-the-op supersedes both the deleted AQ σ half
  and the refuted cover.

### The infra reality (unchanged all session)
The shared box (`sura`) kills session-spawned processes at ~952 s. The AQ step
needs ~17 min, so it structurally cannot cache inside one window; the workaround
was an auto-resuming supervisor + a halved tuning budget, with the user's own
tmux as the immune escape.

---

## 2. σ-in-the-op implemented — R1 (`3d146a93` and follow-ups)

**File**: `src/mimarsinan/tuning/orchestration/signed_seam_install.py` (199 LOC).

`install_signed_seam_offsets(model, trainer, pipeline_config, *, quantile=0.99)`
(line 44):
1. Runs an **analytic** `SegmentForwardDriver` with `compute_sample_recorder`
   to capture per-node value samples (reusing the exact recorder machinery in
   `spiking/segment_forward.py`).
2. For each **armed** wire-value op, computes `_signed_seam_quantiles(sample)`
   (line 30) → `(σ, κ)` (quantile magnitudes, not max), stamps
   `node.output_value_offset = σ` and `node.boundary_traffic_scale = κ`.
3. One re-propagation (`compute_per_source_scales` +
   `propagate_boundary_input_scales`) so the wrapper `s_out`, the weight fold,
   and the entry currencies agree on the lifted κ.
4. A **consumer bake** loop classifying each host consumer by shift response
   (§4 below).

**The value twin** — `compute_op_mapper.py` gained class-level defaults for
pickle compatibility (`f2f245be`):
- `is_wire_value_op: bool = False` (line 40), `output_value_offset: Tensor|None
  = None` (line 41).
- `_forward_impl` adds `output_value_offset` after the module call (line 144):
  the plain-forward value twin carries σ so train ≡ deploy.

**The shift-response bake walk** (`e92b051d`): σ propagates differently through
different host ops. In `install_signed_seam_offsets`'s bake loop:
- scalar-shift-**equivariant** ops (`f(v+c)=f(v)+c`) recurse to their consumers;
- shift-**invariant** ops (LayerNorm) absorb σ (it vanishes);
- **bias carriers** (Linear) bake σ directly;
- **sum-like** ops fail loud (they scale c by N).

---

## 3. The SNW seam fixes — the thinking-trace moment resolved

The ViT A/B ran past AQ and died with
`TypeError: ScaleNormalizingWrapper.forward() got an unexpected keyword
argument 'need_weights'`. The thinking trace identified two coupled defects and
the design tension; here is how each resolved.

### 3a. Armed-only stamping + adapter equivariance (`c229fcd7`)
Before the SNW crash, the installer was stamping σ on **unarmed** wire-value ops
(the patch-embed conv, whose input scales are unity so it never gets an SNW).
For an unarmed op the offset would live in the plain forward alone → a
train/deploy split. Fix: the installer now requires the wrap slots
(`node.per_source_scales is not None`, line 94). Also `add` and `getitem`
joined `_SHIFT_EQUIVARIANT_ADAPTER_FNS` (line 177) — a residual `add(a+σ, b)`
passes a single operand's scalar σ through; `getitem` selects it; **`cat` stays
fail-loud** (a partial-slice shift is not bias-compensable downstream).

### 3b. MHA as a σ bias carrier (`b64ea4d2`)
The bake walk reached the real host compute: `nn.MultiheadAttention` fed by the
armed pre-attention LayerNorm. It *is* a bias carrier — the packed
`in_proj_bias`. A scalar input shift σ moves Q,K,V by `σ·in_proj_weight.sum(1)`;
subtracting that from `in_proj_bias` restores the trained function exactly. A
graph probe confirmed the q=k=v edges dedup to **one** consumer, so a single
packed bake is right for self-attention.
`_bake_shift_into_host_bias(module, σ)` (`signed_seam_install.py:155`) now
handles Linear and packed MHA; unpacked/bias-less attention refuses loudly.
Locked by `TestHostBiasCarrierBake` (`f_baked(v+σ) == f(v)`).

### 3c. SNW transparency to `module_kwargs` + tuple return (`ad7e1251`)
The core defect from the thinking trace. The **value twin** already forwarded
`module_kwargs` and selected `output_index` from MHA's `(attn, weights)` tuple;
the **wire twin** (SNW) did neither, so it applied scale/offset to a tuple.

The design tension — *where does output selection happen?* — resolved as the
trace concluded: **SNW owns both, and selects before scale/offset.**
- `ScaleNormalizingWrapper.__init__` (`compute_modules.py:95`) gained
  `module_kwargs` (line 101) and `output_index` (line 102), plain attributes
  (pickle-safe; cached SNWs predate them but the class-level path handles it).
- `SNW.forward` (line 67): `absolute_out = module(*absolute_inputs,
  **self.module_kwargs)`, then `if output_index is not None: absolute_out =
  absolute_out[output_index]`, **then** add `output_offset`, **then** divide by
  `output_scale`. Selection precedes the arithmetic (the trace's key nuance —
  for non-tuple ops `output_index is None`, so byte-identical).
- `compute_op_mapper.py`: `_maybe_wrap_for_scales` passes `module_kwargs` +
  `output_index` into the SNW; `_prepare_inputs` (line 150) extracted;
  `forward_scale_normalized` (line 163) **delegates** selection to the wrapper
  (`if isinstance(module, ScaleNormalizingWrapper): return module(*prepared)`)
  — avoiding the double-selection the trace flagged, because the outer
  `_forward_with_module` would otherwise re-index an already-selected tensor.

**Deployed-emission uniformity** (the trace's last check): the deployed host ops
run the *same* `forward_scale_normalized` composition the IR emits, so fixing
the walk fixes the NF path and the deployed host compute together. The IR
emission geometry unwraps the SNW for op_type/params (`_unwrap_scale_normalizing`)
so only the torch twin uses the wrapper.

Locks: `test_scale_normalizing_wrapper.py::TestModuleCallingConvention`
(kwargs forwarded; tuple selected before scaling; offset after index) and
`test_scale_aware_compute_op.py::…test_wire_twin_matches_value_twin_for_kwargs_tuple_op`
(the mapper's wire twin equals a reference SNW carrying the same kwargs+index).
Full gate **8250** green, typecheck 0.

---

## 4. σ-in-the-op CONFIRMED — the keystone (memo §10g, `b20a7d41`)

With the seam fixed, the ViT A/B read the decisive number at the Activation
Quantization entry:

| t2_04 AQ entry (full acc) | config |
|---|---|
| 0.0146 | full-width cover (κ=max) — refuted, destroys resolution |
| 0.06 | σ-free trained clamp — exact but capacity-amputated |
| **0.5956** | **σ-in-the-op** (post-recovery 0.6046, retention armed) |

A **10× capacity recovery**, inside the predicted 0.3–0.6+ band: the seam loss
was **signed-band amputation, not resolution or training capacity**. Three
implementation laws proven (each fail-loud first): armed-only stamping;
shift-response classification of host consumers; the `f_baked(v+σ)==f(v)` bake
law as tests.

---

## 5. The e2e grind past AQ — infra walls, not algebra (memo §10h, `af11c6cc`)

The σ-armed ladder then hit a sequence of *infrastructure* walls (each cleared;
none was the algebra):

| symptom | cause | lever |
|---|---|---|
| OOM, GPU 0 at 1.1 GiB free | other researchers' jobs saturating the shared GPU | pin to a free GPU (relocated to GPU 1) |
| OOM 16.3 GiB, process holds 93 GiB | `segment_forward` `values` dict holds every layer's `[S=32, batch-512, seq, hidden]` | `deployment_parameters.batch_size` (the **eval** batch, nested block; *not* `tuning_batch_size`) → 64 clears it (0 OOMs) |
| every attempt dies at ~917 s, no traceback, no LIF cache | the ~952 s session reaper; each attempt restarts LIF from the AQ cache | run under the user's own tmux (`t2_04_tmux_run.sh`, immune) |

Within a window the bottleneck is the **genuine spike EVAL**
(`eval_n_batches=39 × S=32` over the whole backbone), not training — reducing
`tuning_budget_scale` (0.5 → 0.2) did not help; every downstream step repeats
that eval, so the ladder cannot chain under the reaper.

**Not a collapse**: the LIF entry `best_full_acc ≈ 0.011` vs analytic ≈ 0.78 is
the genuine spike forward at scale=1.0, which reads chance **by design** — LIF
adaptation is what recovers it. The recovery question stayed *unmeasured*
because adaptation never completed a rung under the reaper.

Two latent bugs the ladder surfaced here (both are §3): MHA bias-carrier and SNW
transparency.

**Audit correction discovered here**: `simulation_step.py:18–20` shows the
deployed accuracy verdict is the **Soft-Core-Mapping identity read**;
nevresim/SANA-FE are *decision-parity probes on a small subsample*, never the
accuracy census. So "parity-certified deployment" is already the architecture —
the expensive census was never being paid; the cost is entirely the in-loop
genuine eval of the *tuning* steps.

---

## 6. Phase D — the LIF spike-capacity audit (memo §10i, `ae7143d2`)

**Probe**: `scripts/_probes/lif_seam_capacity_audit.py` (read-only, analytic-
only, one CIFAR-100 batch on the AQ cache).

LIF semantics (`models/nn/activations/lif.py::_spikes_and_scale`):
`out = rate·scale`, `rate = IFNode(x/scale)` over S steps at threshold 1. So
`activation_scale` is the **max encodable value**, the grid step is `scale/S`,
and the per-neuron resolution is

    eff_levels = S · R_out / activation_scale     (R_out = 99th-pct |LIF output|)

spike-levels (the signal-to-one-spike-quantization-noise ratio).

Measured across the 12 on-chip MLP layers:

    eff_levels:  26 → 18 → 13 → 13 → 8 → 8 → 8 → 5 → 1 → (last 3 silent)

A clean **depth-dependent resolution decay** to 1 spike-level by layer 8; the
last three MLP branches are silent (`R_out ≈ 0` — spiking them to 0 is exact).
Root cause, confirmed by a second probe: `activation_scale / stream_input ≈ 0.7`
at **every** depth — the scale tracks the residual **stream** magnitude (which
accumulates 0.78 → 3.9 with depth, as ViT streams do), but the LIF encodes the
**branch output** (`R_out ≈ 0.3`, flat). The currency references the wrong
tensor, so the mismatch grows with depth and buries the deep signal (the death
cascade). Framed as "the §10f/g quantile-currency law one seam deeper" — the
predicted fix was to recalibrate `activation_scale` to the LIF's own output.

---

## 7. The recalibration fix — PROTOTYPED and REFUTED (memo §10j, `b67e6002`)

Per the "prototype the solution before building the seam" discipline, the fix
was prototyped, not built: `scripts/_probes/lif_output_currency_recal.py`
(CIFAR-100, n=256 on the AQ cache), sweeping both currencies:

| setting | genuine acc |
|---|---|
| analytic (base activation) | **0.816** |
| LIF, original scale | 0.30 |
| LIF `activation_scale` ×{0.5,1,2,4}; pre-act q99/q999/max | ≤ 0.30 |
| LIF `input_activation_scale` ×{1,2,4,8} | ≤ 0.30 |

**Refuted.** No static currency — input or output — recovers the genuine
forward; it plateaus at **0.30** vs analytic **0.816**, and the original scale
is already near-optimal (reducing hard-clamps via `out=rate·scale`; raising
starves resolution). The §10i resolution-decay *pattern* is real, but static
recalibration is not its fix.

**Consequence**: the genuine spiking conversion gap (0.816 → 0.30 best-static)
requires LIF **adaptation** (training) — the documented `[[nf_requires_lif_
adaptation]]` behavior (scale=1.0 = chance, adaptation recovers). There is no
static-currency shortcut around the long run; the prototype-first discipline
prevented building a generic seam for a refuted fix.

**New lead (uncertain)**: the bare LIF install (probe) reads genuine **0.30**,
but the PIPELINE's LIF entry reads **0.011** — a 27× drop that neither
activation scale explains. Caveat: the probe replaces the activation outright
rather than *blending* it as the pipeline does, so 0.30 may partly be an install
artifact. Resolving it means faithfully bisecting the pipeline LIF install
(exact-QAT snap / blend-at-rate-1 / encoding-layer / σ-boundary offsets). If
real, raising the entry floor from 0.011 to 0.30 would give the adaptation a
far better start.

---

## 8. Consolidated verdict & open threads

**Proven, committed, green** (gate 8250, typecheck 0):
- The exactness half (t0_30 bit-exact deploy, parity 1.0).
- σ-in-the-op: ViT AQ entry **0.06 → 0.596** (the "fundamental issue" resolved).
- Three tested downstream fixes: armed-only stamping (+add/getitem
  equivariance), MHA σ bias-carrier, SNW transparency to module_kwargs/tuple.
- The offloaded-ViT VRAM wall (batch_size=64) and the infra ledger.

**Established negatives** (measured, recorded):
- Static LIF currency recalibration does **not** recover the genuine forward
  (0.30 ceiling) — the conversion needs adaptation.
- The full-width κ cover is refuted; post-training σ is banned.

**Open**:
1. Bisect the pipeline LIF-install 0.30 → 0.011 drop (the §10j lead; highest
   potential leverage if real).
2. The genuine conversion needs LIF adaptation — make the long run tractable
   (M2: SE-sized in-loop eval, `n = p(1−p)/SE²`, full census only at
   entry/final) + immune execution (user tmux / xlog1 Slurm when the tunnel is
   restored).
3. Q4 e2e DoD: deployed ≡ analytic within 2·SE + parity ≥ 0.98.

---

## 9. Concrete pointers

**Commits (session, oldest→newest)**: `30f8710c` σ-scope law · `8a644028`
scope filter · `8839e3ca` §10e + t0_30 green · `81b8ce15` fast_lr_scale ·
`10f4a2e6` AQ cover on ViT · `81d0e332` §10f · `3d146a93` **σ-in-the-op (R1)** ·
`f2f245be` mapper class-level defaults · `e92b051d` shift-response bake walk ·
`c229fcd7` armed-only + add/getitem · `b64ea4d2` **MHA bias carrier** ·
`b20a7d41` §10g CONFIRMED · `ad7e1251` **SNW transparency** · `af11c6cc` §10h ·
`ae7143d2` §10i · `b67e6002` §10j REFUTED.

**Memo (SSOT)**: `docs/research/findings/conversion_boundary_algebra.md` §10f
(design) · §10g (confirmed) · §10h (e2e infra) · §10i (Phase-D diagnosis) · §10j
(refutation + lead).

**Implementation**:
- `tuning/orchestration/signed_seam_install.py` — `install_signed_seam_offsets`
  (L44), `_signed_seam_quantiles` (L30), `_bake_shift_into_host_bias` (L155),
  `_SHIFT_EQUIVARIANT_ADAPTER_FNS` (L177), armed-only guard (L94).
- `mapping/support/compute_modules.py` — `ScaleNormalizingWrapper` ctor (L95:
  `module_kwargs`, `output_index`), `forward` (L67: select-before-offset/scale).
- `mapping/mappers/compute_op_mapper.py` — class defaults (L40–41), value-twin
  offset (L144), `_prepare_inputs` (L150), `forward_scale_normalized` delegate
  (L163).
- `models/nn/activations/lif.py::_spikes_and_scale` — the `out=rate·scale`
  semantics that define the eff_levels metric.
- `pipelining/pipeline_steps/verification/simulation_step.py:18–20` — deployed =
  SCM identity read; sim = parity probe (parity-certified deployment).

**Probes**: `scripts/_probes/lif_seam_capacity_audit.py` (resolution audit),
`scripts/_probes/lif_output_currency_recal.py` (the refuted recalibration sweep).

**Infra**: `scratchpad/t2_04_tmux_run.sh` (immune full-budget e2e launch),
`scratchpad/t2_04_resume_aq.json` (batch_size=64, start_step="LIF Adaptation"),
`~/.claude/plans/vit_e2e_measurement_program.md` (the M-phase plan).
