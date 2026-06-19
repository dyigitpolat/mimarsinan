# 54 — Transformation Fixes for Cascaded Single-Spike TTFS (Round-3 synthesis)

GOAL: make the **cascaded single-spike TTFS transformation** convert a continuous ANN
to the **deployed genuine cascade LOSSLESSLY by CONSTRUCTION** (no fine-tuning crutch),
on **deep AND wide** mixed-sign nets — matching the analytical staircase, which in turn
matches the continuous ANN as time-resolution `S` grows.

Other modes already convert losslessly (LIF rate, synchronized ttfs_cycle); generic
factors (weight-quant, chip-mapping, width, depth) do NOT cause meaningful loss there.
The loss is **specific to the cascaded greedy single-spike firing**, and the fix must
live in the cascade dynamics.

All numbers below were re-measured on the shared harness
(`docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/recipe_harness.py`):
`build/genuine_acc/staircase_logits`. COLD = zero fine-tuning. Mixed-sign trained nets,
the hard case, at `d in {3,6,9}` (deep) and width `128/256/512` (wide).

---

## The mechanism (established, re-confirmed)

A cascade neuron fires **ONCE** the cycle its **RUNNING partial sum** crosses `theta` —
on **INCOMPLETE** information. With mixed-sign ReLU weights, the early-arriving (positive)
inputs cross `theta` before the later (negative/cancelling) inputs arrive → a
**premature/wrong fire** → the deep **death cascade** (every measured cold genuine accuracy
is chance, ~0.10, at every depth/width/S). The staircase and LIF rate code consume the
**COMPLETE** weighted sum and are lossless. **Greedy partial-sum firing on mixed signs is
the entire cascaded-specific loss** — it is *not* a capacity, quantization, or decode-shape
limit. This matches Stanojevic et al. 2023 (Nat. Commun., "High-performance deep SNNs with
0.3 spikes/neuron"): the exact ReLU→TTFS map requires firing on the COMPLETE pre-activation,
and naive deep TTFS hits a depth-scaling vanishing/exploding-gradient wall.

Corollary verified across all four directions: the membrane integral is **value-correct at
window end** (`membrane_end / T == z` exactly); the *sole* defect is *when* the single spike
is committed. Every fix that works does so by **deferring the fire until the complete sum has
settled**.

---

## The four directions

| dir | fix | cold lossless? | mechanism addressed | deploy trade |
|---|---|---|---|---|
| **D1** | **Exact-mapping / complete-sum fire** (Stanojevic two-regime; integrate full window, fire on complete `z` at `t_min`) | **YES, bit-exact** | greedy partial-sum firing | latency `T·(depth+1)` (synchronized barrier); no extra spikes; parity-preserving; default-off |
| **D2** | **Two-phase pipelined neuron** (listen→accumulate complete sum, then emit one timing spike) | **YES, `2ph−stair = +0.000`** | greedy partial-sum firing | latency ~`2T`/hop, KEEPS 1/T pipelined throughput (no global barrier); no extra spikes; parity-clean in principle, C++ port pending |
| D3 | negative-weight monotonization / signed channels / value-shift | **NO** (cold + mono stay chance on balanced mixed-sign) | tries to reorder arrivals; cannot foresee cancellations | 2× spikes, 2× fan-in/axons; not parity-preserving |
| D4 | deep+wide scaling of hedged staircase-backward STE (+ per-sample residual diagnosis) | N/A (training crutch, not a construction map) | trains *around* greedy firing | parity-preserving, no extra spikes/latency, but **needs `S` to scale with depth**; hits a hard wall by `d=15` |

### D1 — Exact-mapping complete-sum fire **[WINNER on losslessness]**
Worktree: `wf_08b4088d-804-1`. Src:
`src/mimarsinan/models/nn/activations/ttfs_spiking.py` (TTFSActivation: `complete_sum_fire`
mode + `_complete_sum_step` deferred-fire integrator caching `_complete_value`;
`set_complete_sum_fire`/`reset_state`), `src/mimarsinan/spiking/segment_policy_ttfs.py`
(staggered `wstart = depth·T` windows so `t_min = t_max` of the previous layer; latched
re-encode; value-domain decode; prepare/finalize toggle the node mode), `ARCHITECTURE.md`,
and `tests/unit/spiking/test_segment_forward_driver.py` (bit-exact lossless lock +
default-path-unchanged). Experiment: `.../experiments/exact_mapping_fix.py`.

Re-measured COLD (zero FT):

```
d  S : continuous  staircase  COLD_greedy  COLD_completeSum
d=6 S= 6: cont=0.981  stair=0.698  greedy=0.106  completeSum=0.698
d=6 S= 9: cont=0.981  stair=0.928  greedy=0.106  completeSum=0.928
d=9 S= 6: cont=0.965  stair=0.343  greedy=0.102  completeSum=0.343
d=9 S= 9: cont=0.965  stair=0.814  greedy=0.102  completeSum=0.814
```

`completeSum == staircase` at **every** config (and at higher S the staircase → continuous,
so the conversion is end-to-end lossless as `S` grows). Prior-run logits maxdiff `0.0`;
wide nets (width 256/512, d=6) also `completeSum == staircase == continuous`, maxdiff `0.0`.
Greedy is chance everywhere.

Mechanism: regime 1 integrates the full latched window WITHOUT firing (running sum is
incomplete until cycle `T−1`); at window close it commits the fire on the COMPLETE
normalised `z = (Σ_t W·spike_t)/T + bias` through the exact staircase kernel
(`ttfs_quantized_staircase`), so the downstream re-encode reproduces it losslessly. This is
the Stanojevic two-regime map reconstructed *inside* the cascaded policy. The cost is that
each layer must wait for the previous to complete → `t_min = t_max(prev)` → a synchronized
staggered schedule (`T·(depth+1)` cycles vs pipelined `T+depth`). This is the **same schedule
the already-lossless synchronized `ttfs_cycle` mode uses**, so it is realizable on chip today
(HCM/nevresim/SANA-FE already implement an `S·groups` synchronized timeline).

Why a genuinely pipelined (`T+depth`) cascade *cannot* be made lossless this way: a depth-`d`
neuron's complete sum is only available at the last cycle of its own window, leaving no room
to ramp-encode the fire time — which is exactly why Stanojevic's construction needs the
second post-`t_min` firing regime.

### D2 — Two-phase pipelined neuron **[WINNER on deployability/throughput]**
Worktree: `wf_08b4088d-804-2`. Src: `ttfs_spiking.py` (+80 lines: `set_two_phase` /
`set_two_phase_phase` + `_two_phase_forward` + `_snapshot_emit_value`; default-off, greedy
forward bit-exact unchanged) + 3 unit tests in `tests/unit/models/test_ttfs_spiking_node.py`.
Experiment: `.../experiments/two_phase.py`.

Re-measured COLD (zero FT):

```
 d   S |   cont   stair  greedy  2phase | 2ph-stair
 3   8 |  0.983   0.981   0.102   0.981 |    +0.000
 3  16 |  0.983   0.985   0.102   0.985 |    +0.000
 6   8 |  0.981   0.915   0.106   0.915 |    +0.000
 6  16 |  0.981   0.974   0.106   0.974 |    +0.000
 9   8 |  0.965   0.742   0.102   0.742 |    +0.000
 9  16 |  0.965   0.952   0.102   0.952 |    +0.000
```

`2ph−stair = +0.000` at every config (deep), and prior runs reported `+0.000` on wide nets
(width 128: 0.898; width 256: 0.955). The listen ramp reconstructs the complete sum to float
precision; the snapshotted emit value matches the analytical staircase per-neuron value to
`1e-16`.

Mechanism: LISTEN over the whole window (same ramp integration as greedy, but NEVER fire) →
membrane converges to the COMPLETE sum; snapshot `V = staircase-decode(membrane/T)`; EMIT a
single timing spike at `tau = ceil(T(1−V))`. **Key deployment advantage over D1:** it stays
**pipelined and depth-staggered with NO global barrier** — steady-state throughput is
`1 result / T` cycles (vs synchronized `1 / T·depth`), at the cost of ~`2T`/hop latency
(listen `T` + emit `T`). It is also **fully differentiable** (all params get nonzero grad),
so light FT is available though unnecessary. The companion `input_scale` propagation
(`propagate_boundary_input_scales`, already in src and deployable) is required alongside it.

### D3 — negative-weight monotonization **[does NOT work]**
Worktree: `wf_08b4088d-804-3`. No src change (experiments only). Cold and monotonized both
stay chance on balanced mixed-sign deep nets (d6 S8: stair 0.915 vs cold/mono 0.106; d9 S8:
stair 0.742 vs chance). Monotonization only helps when negatives dominate; in the realistic
balanced regime it cuts single-layer error 0.063→0.050 against a 0.015 grid floor, and the
residual does **not** shrink with depth (it compounds to collapse). Confirms the core
diagnosis: a single-spike fire-time code cannot *locally* foresee future cancellations; any
construction that front-loads a negative shift moves the fire time off `tau(z)`. A sync
barrier would be lossless but *is* the existing synchronized mode. Not viable.

### D4 — deep+wide STE scaling + per-sample residual **[training crutch, depth-limited]**
Worktree: `wf_08b4088d-804-4`. No src change. The hedged staircase-backward STE is lossless
(aggregate AND per-sample) **only while `S` is large enough for the depth**, and the required
`S` grows with depth: `d≤6: S≥8`, `d=9: S≥16`, `d=12: S≥32`, `d=15: S>64` (still fails at
S=64). The break is a **resolution / death-cascade wall, not undertraining** — more
steps/width do not rescue a too-small `S`. This is precisely Stanojevic's depth-scaling
gradient wall.

**Per-sample residual finding (answers the synthesis question):** when `S` is sufficient,
the genuine cascade matches the frozen staircase on **0.93–0.98 of test samples**
(`logit_corr 0.93–0.98`) — only ~2–7% of predictions flip. The residual is **real but small**;
it is **not** an aggregate-only illusion at recoverable depths. **Method trap (important):**
comparing genuine to the *trained flow's own* staircase yields a spurious erratic 0.1–0.87
"residual" because the STE backward constrains only the genuine forward, so the trained-weights
staircase forward drifts off — the genuine cascade still reproduces the *original ANN's*
per-sample decisions at 0.93–0.98. **For D1/D2, the per-sample residual is exactly zero**
(bit-exact / `1e-16`), so the small Dir-4 residual is a property of the *STE crutch*, not an
intrinsic loss of the cascade code under a by-construction fix.

---

## What is PROVEN vs PROTOTYPE

PROVEN (independently re-measured here, plus unit tests):
- The death cascade = greedy partial-sum firing on mixed signs (cold genuine is chance
  everywhere; membrane is value-correct at window end).
- **D1 makes COLD genuine == staircase BIT-EXACTLY** (maxdiff 0.0) on deep (d=9) and wide
  (width 512) mixed-sign nets, zero FT. D1's complete-sum lossless-lock + pipelined-greedy-
  unchanged tests pass (4/4); the greedy default path is byte-identical.
- **D2 makes COLD genuine == staircase to float precision** (`+0.000`, per-neuron `1e-16`)
  on deep and wide nets, zero FT, while preserving pipelined `1/T` throughput.
- D3 does not recover deep balanced mixed-sign nets; D4's STE is lossless only with
  depth-scaled `S` and is a training crutch.

PROTOTYPE / NOT YET PROVEN (the deployment gap):
- Neither D1 nor D2 has been run through the **real `run.py` pipeline** end-to-end, nor
  through the **C++ simulators** (nevresim / SANA-FE) that the NF↔SCM parity gates target.
  D1 reuses the synchronized schedule those backends already realize (lower port risk); D2's
  pipelined listen/emit + per-hop window schedule must be **ported into nevresim/SANA-FE**
  to keep NF↔SCM parity — that backend port is the main open work for D2.
- Lossless claims are on the harness digits task; broader datasets/architectures
  (mmixcore, mlp_mixer multi-segment) are untested for these two modes.
- D1's decode path is grid-quantized/detached (weight-grad zero) — moot for cold-lossless
  conversion, but it is not a usable FT mode; D2 is fully differentiable if FT is ever wanted.

---

## RECOMMENDED transformation fix

**Cherry-pick the construction fixes; retire the STE as the path to losslessness.** The
greedy partial-sum death cascade is curable BY CONSTRUCTION with **zero fine-tuning** by
deferring the fire until the complete window has settled (the Stanojevic complete-sum
principle). Two viable realizations:

1. **D1 (worktree `wf_08b4088d-804-1`) — ship FIRST.** Bit-exact lossless, parity-preserving,
   default-off, and it rides the **already-deployed synchronized schedule** (HCM/nevresim/
   SANA-FE realize it today), so the simulator-parity risk is lowest. Cost is `T·(depth+1)`
   latency. This is the safe, provable productionization target.

2. **D2 (worktree `wf_08b4088d-804-2`) — pursue for throughput.** Same losslessness but
   keeps the cascade's `1/T` pipelined throughput (no global barrier), which is the entire
   point of the cascaded mode over synchronized. The blocker is the C++ listen/emit port +
   NF↔SCM parity gates. If the port lands, D2 strictly dominates D1 (lossless AND
   pipelined-throughput).

Keep the hedged STE only as an **optional, orthogonal** lever for sub-grid-`S` regimes or
robustness margin — NOT as the conversion mechanism. With D1/D2 the conversion is lossless
before any training, so the STE is no longer load-bearing.

**Discard D3** (no recovery; not parity-preserving; doubles spikes).

---

## Concrete next experiments

1. **D1 end-to-end through `run.py`** on a real config (mmixcore / a deep MLP) at deploy-S
   sufficient for the depth; assert deployed genuine == staircase through the SCM/NF↔SCM
   gates (not just the harness). Lowest risk; do this first.
2. **D2 C++ port:** implement listen/emit + per-hop staggered window in nevresim and SANA-FE;
   add NF↔SCM per-neuron parity locks mirroring the existing synchronized/cascaded gates.
   Then re-run the harness lossless lock through the C++ path.
3. **Latency/throughput head-to-head** D1 (`T·(depth+1)`, sync barrier) vs D2 (`~2T`/hop,
   pipelined `1/T`) on the chip cost model, to confirm D2's throughput advantage justifies
   the port.
4. **Depth/width stress past the harness toy:** push D1/D2 to `d=12,15` and multi-segment
   models (mlp_mixer) — verify bit-exactness survives where D4's STE wall (`d≥15`) appears,
   since D1/D2 should have NO depth wall (they consume the complete sum exactly).
5. **Per-sample audit on the real pipeline:** confirm the `1e-16` / bit-exact per-sample
   agreement (vs D4's 0.93–0.98) holds end-to-end, closing the "is the residual a true
   cascade-code loss" question in production.

Literature anchor: Stanojevic et al. 2023 (Nat. Commun., `stanojevic2023highperformance`,
arXiv 2306.08744) — exact ReLU→TTFS map (complete-sum two-regime firing) and the deep-TTFS
gradient wall; Comsa et al. 2021 (`comsa2021temporal`) and Rueckauer et al. 2017
(`rueckauer2017conversion`) for TTFS temporal-coding / ANN→SNN conversion context.
