# Numerical boundary consistency: the cross-stack accuracy spread, measured and explained

**Date:** 2026-07-10.
**Case:** `t0_20_casc_simplemlp_wq_s4` (seed 0, identical config): **0.9621** local
(torch 2.12.0+cu130, RTX PRO 6000 Blackwell) vs **0.9485** and **0.8723** on the
cluster (torch 2.7.1+cu126, H100 NVL MIG 3g.47gb) — a 9 pp spread, including a
7.6 pp spread *between two runs on the same cluster stack*.
**Verdict up front:** the spread is **not** an evaluation-numerics problem and
**not** TF32. It is (1) a missing global seed — the config `seed` is never
applied to torch/numpy/random, so every run trains a *different network* — and
(2) an fp16-autocast metric path that adds a small, stack-dependent residual on
top. The famous "~3 pp cuBLAS reduction-order" comment in `run.py:7-10` does
**not** reproduce on the actual artifacts: fp32 evaluation is flip-free under
every reduction order we could force. All claims below are measured on the
exact artifacts of the local 0.9621 run
(`generated/t0_20_casc_simplemlp_wq_s4_phased_deployment_run/`).

---

## 1. What the reported number actually is

The campaign metric ("Test accuracy" 0.9621) is the full 10 000-sample MNIST
test accuracy of the **torch NF forward** (`trainer.model(x)`), reported at the
Weight Quantization / Quantization Verification steps
(`src/mimarsinan/model_training/basic_trainer_eval.py:113-131`), harvested from
`_GUI_STATE/live_metrics.jsonl` by `scripts/run_tier.py:96-97`. Two properties
matter:

- It runs under `torch.autocast("cuda")` — **fp16 matmuls** — unconditionally
  on CUDA (`basic_trainer_eval.py:15-18`, entered at `:104` and `:116`).
- For the casc cell the model carries the deployed cascade forward override
  (`_SegmentSpikeForward`, installed by the TTFS cycle tuner), i.e. the metric
  already exercises the cycle-accurate threshold compares.

Reproduction on the saved WQ model (10 000 samples): fp16-autocast eval gives
**0.9621** — bit-for-bit the reported number; fp32/fp64/CPU eval all give
**0.9620**. The reported metric is an fp16 artifact even locally.

## 2. Root causes, ranked by measured impact

### RC1 (dominant): the pipeline never seeds the RNGs

- The config registry documents `seed` as "Global torch/numpy seed; identical
  config + seed reproduces the step trajectory", owner
  `PipelineSession/determinism`
  (`src/mimarsinan/config_schema/registry/entries_run.py:34-37`). **No code
  implements that owner.** `grep manual_seed src/` shows the seed reaches only:
  dataset split generators (`data_handling/data_provider.py:67`), conversion
  draws (`tuning/orchestration/conversion_draws.py:31-33`), the rate-axis
  decision generator (`tuning/axes/manager_rate_axis.py:78-86`), and the search
  evaluators (unused in fixed-hw mode). Model init
  (`pipelining/pipeline_steps/config/model_building_step.py:30-31`), training
  shuffles, and dropout all consume torch's **default process RNG**, which
  torch seeds from OS entropy per process (verified: two fresh interpreter
  processes report `torch.initial_seed()` = 1839768257933012063 vs
  8220855561887761215).
- **Paired-run experiment** (same machine, same GPU, same command, config
  = t0_20 with `stop_step: "Pretraining"`, 2 epochs):

  | run | Test acc | Training acc | state-dict sha256 |
  |---|---|---|---|
  | vanilla A | 0.9833 | 0.9830175438596491 | `8a3b286ce4bf5407` |
  | vanilla B | 0.9780 | 0.9828596491228070 | `9f3bce2a4b098729` |
  | seed-pinned A | 0.9793 | 0.9825964912280701 | `fbe0e83e2eb80eae` |
  | seed-pinned B | 0.9793 | 0.9825964912280701 | `fbe0e83e2eb80eae` |

  Two epochs of unseeded pretraining already differ by **0.53 pp**; the full
  casc pipeline then amplifies through Tq=4/S=4/5-bit discretization cliffs and
  ~10 accuracy-branching accept/reject points (`degradation_tolerance` is 0.15
  in the config — a 15 pp corridor): `tuning/orchestration/mbh_gate.py:53`
  (accept vs midpoint retry), `:75-86` (stall + restore), `:110` (best-snapshot
  ratchet); `tuning/orchestration/smooth_adaptation_cycle.py:299`
  (catastrophic rollback), `:376/:382` (paired/marginal rollback vs commit),
  `:427-430` (target streak); predicates in
  `tuning/orchestration/acceptance_sensor.py:59-63,95,99,122-127`. A different
  init therefore selects a *different trajectory through a branching program*,
  not a perturbed endpoint. This alone accounts for same-stack spreads of the
  0.9485-vs-0.8723 magnitude; nothing GPU-specific is needed.
- The seed-pinned wrapper (prototype, ~10 lines, below) yields **bit-identical
  weights and metrics** across runs. Fix validated.

### RC2: fp16 autocast on the metric and training paths

- Sites: eval `basic_trainer_eval.py:15-18` (used by `test()` `:116` and
  `validate*` `:104`); training loop `model_training/basic_trainer.py:220`
  (`with autocast("cuda")`) + `GradScaler` (`:115-214`). The MBH ledger already
  treats fp16 probes as non-gate-grade and disables autocast for its own
  measurements (`tuning/orchestration/mbh_ledger.py:21-32`: "the live probe
  convention is fp16 under CUDA") — the *reported* metric never got the same
  treatment.
- Measured flip exposure of fp16 vs fp64, analytic NF forward, 2048 samples
  (staircase outputs on the S-grid, per perceptron):
  TTFS-FT stage 231/370/2552/12 flips (of 524k/262k/524k/20k values;
  worst 0.49 %), 2/2048 decision flips, accuracy 0.94336 → 0.94287;
  WQ stage 249/296/1748/8 flips, 1/2048 decisions, 0.95508 → 0.95557.
  End-to-end on the deployed cascade forward: 3/10 000 decision flips
  (0.9620 → 0.9621).
- Cross-stack: on the H100 probe the specific fp16 GEMM fingerprint happened to
  be bit-identical to Blackwell (`fp16_sha 0abfb80327a7d1cd` on both), so fp16
  eval residuals are small here — but they are unspecified behavior across
  cuBLASLt versions, and `allow_fp16_reduced_precision_reduction` defaults True
  on both stacks.

### RC3: cross-stack fp32 GEMM bit differences (training only)

Cluster-vs-local numeric probe (identical seeded inputs; CPU input sha256
matches across stacks — seeded init *is* cross-stack reproducible):

| | local | cluster |
|---|---|---|
| python / torch | 3.10.20 / 2.12.0+cu130 | 3.12.3 / 2.7.1+cu126 |
| GPU / cuDNN | RTX PRO 6000 Blackwell / 92000 | H100 NVL MIG 3g.47gb / 90501 |
| fp32 GEMM sha / Σ | `1f1c0081ed6e1ef2` / −3102.1753333117813 | `cc192f860d6cf13d` / −3102.1746335085481 |
| max err vs fp64 | fp32 2.45e-5, tf32 3.64e-2, fp16 5.14e-2 | fp32 2.83e-5, tf32 3.65e-2, fp16 5.14e-2 |

fp32 GEMMs differ at the 7th digit across stacks → gradient trajectories
diverge chaotically even under identical seeds. Consequence: **seeding makes
each stack internally reproducible, but cross-stack equality of a *trained*
model is unattainable**; cross-stack agreement must be defined on a fixed
artifact (train once, evaluate everywhere) — and on a fixed artifact, fp32
evaluation is flip-free (§4).

### Exonerated: TF32, and the run.py reduction-order legend

- **TF32 is OFF for matmul on both stacks** (probe: `matmul.allow_tf32 False`,
  `float32_matmul_precision highest`, `NVIDIA_TF32_OVERRIDE` unset on both).
  The repo pins nothing anywhere (`grep allow_tf32|float32_matmul_precision|
  use_deterministic_algorithms src/ run.py` → only the `CUBLAS_WORKSPACE_CONFIG`
  env default at `run.py:10`).
- **But** `cudnn.allow_tf32` is **True by default on both stacks** — every
  *convolutional* cell (lenet5/deepcnn/mixer tiers) evaluates its NF conv
  layers in TF32 (10-bit mantissa) *today*. Measured TF32 flip exposure on this
  MLP's staircases if it were enabled for matmul: 61-698 flips/layer. This is a
  live hazard for the conv cells of the same class as the casc collapse.
- **Reduction order (`run.py:7-10`, "~3 pp")**: not reproducible. On both the
  TTFS-FT and WQ artifacts, fp32 staircase outputs are identical for: plain
  cuBLAS matmul, K-permuted matmul, 8-chunk split-K accumulation, batch sizes
  1/57/128/10 000, CPU vs GPU, and vs float64 — **0 flips in 1.33 M values per
  stage**. The historical ~3 pp drift was almost certainly the fp16 metric path
  and/or unseeded trajectory variance, misattributed. The comment should be
  rewritten when `run.py` is next touched.

## 3. Discretization-boundary inventory (deployed + verification forwards)

"Grid-guaranteed" = the value feeding the boundary is mathematically constrained
to a representable grid; "raw" = raw float matmul/accumulation output.

| # | Site | Boundary op | Fed by | Grid-guaranteed? |
|---|---|---|---|---|
| 1 | `models/spiking/wire_semantics.py:19,34` (`ttfs_quantized_staircase[_np]`) | `ceil(S·(1−V/θ))` + `<S` fire test | NF: torch fp32 GPU linear output V; SCM: numpy float64 matmul | **raw** (NF); float64-exact post-WQ (SCM, see #12) |
| 2 | `wire_semantics.py:44,54` (strict `<` twins) | `floor(...)+1` | same | same |
| 3 | `wire_semantics.py:61,65` (`floor_staircase[_np]`) | `floor(x·levels)` | pre-activation / rate | raw |
| 4 | `wire_semantics.py:72,79` (`ttfs_spike_time[_np]`) | `round(S·(1−rate))` | clamped rates | encoding inputs on 1/255 grid (MNIST `ToTensor`) — near-grid |
| 5 | `wire_semantics.py:86,94` (`ttfs_grid_quantize[_np]`) | `<S` on #4's output | #4 | yes (integer spike times) |
| 6 | `models/nn/activations/autograd.py:48,65,78,118` (STE quantisers) | floor/ceil/round twins of #1/#3/#4 | torch fp32 GPU matmul | raw |
| 7 | `models/nn/activations/ttfs_spiking.py:152-153,179-181` (cycle-accurate TTFS node) | heaviside on `membrane − 1.0`; `:148` encoding `−1/T` init | torch fp32 GPU matmul + per-cycle ramp accumulation (`spiking/segment_forward.py:126-132`, cycle loop `spiking/segment_policy_ttfs.py:40,255-279`) | **raw** — this is the deployed casc NF metric path |
| 8 | `models/nn/ttfs_cycle_kernels.py:21-23` + `models/spiking/ttfs_cycle_step.py:36-38` | `torch.le/lt(threshold, memb)` | torch matmul in `COMPUTE_DTYPE=float64` (`models/spiking/spiking_config.py:7`; buffers `models/spiking/hybrid/lif_step.py:61-99`) | float64; exact post-WQ |
| 9 | LIF: `models/spiking/lif_core_step.py:22-25` matmul; `models/nn/lif_kernels.py:25` compare, `:27,:30` reset | `lt/le` + subtractive/zero reset | torch float64 (hybrid rail) | float64 |
| 10 | `chip_simulation/ttfs/ttfs_cycle_genuine.py:13,26-28` | `ceil` encode + fire | numpy float64 (`@`, `ttfs_segment.py:163`) | float64; spec for C++ |
| 11 | `chip_simulation/ttfs/ttfs_encoding.py:15,24,37,55,63` | round/rint + cycle compares | numpy float64 rates | yes (integer spike times) |
| 12 | Deployment WQ: `mapping/export/chip_quantize.py:37,59` | `np.round → int8/int16` (dtype `:22`), scale folded into `node.threshold` `:48,:61`, bias `:83-85` | fused float weights × `q_max/max|w|` | output **is** the grid |
| 13 | SCM executor: `chip_simulation/ttfs/ttfs_executor.py:38` (`_CONTRACT_DTYPE=float64`, casts `:82,:98-121,:230-236,:271`) | delegates to #1/#10 | numpy float64 matmul over int weights × k/S inputs | **integer-exact post-WQ**: `S·V = Σ nᵢkᵢ (+ S·b)`, all integers ≪ 2^53 → float64 matmul exact in every summation order |
| 14 | nevresim C++: `signal_t = double` (`nevresim/include/simulator/compute_policy/ttfs_analytical_compute.hpp:18`, `real_valued_compute.hpp:10`, `ttfs_quantized_compute.hpp:28`), weights `double` by default (`code_generation/generate_main.py:96`) | Compare `<=`/`<` | serial C++ double | deterministic (fixed order) + integer-exact post-WQ |
| 15 | Loihi rail: `chip_simulation/subtractive_lif.py:78-85`; decode `chip_simulation/lava_loihi/runner.py:230` | `>=`/`>` compare, subtractive reset; `np.rint` | numpy float voltages | raw (float64, deterministic order) |
| 16 | Recording/encode: `chip_simulation/recording/spike_modes.py:15,19,29,34,38`, `_spike_encoding.py:44-46` | round/compare | torch/numpy rates (float32 at `_spike_encoding.py:30`) | mostly grid-side |
| 17 | Input quantisers: `autograd.py:99,107` (`ChipInputQuantizer`), parity gates `pipelining/core/nf_scm_parity.py:124-139,184-191,251` (`atol=1e-9` / decision agreement) | round + tolerance compares | fp32 normalized inputs | inputs on 1/255 grid |

**Key structural fact:** the SCM/nevresim verification rail (#10-#14) is float64
with fixed (serial or numpy-BLAS) summation and, after weight quantization,
*integer-exact* — platform-invariant by construction. The **only**
platform-sensitive rail is the torch NF rail (#1, #6, #7) whose boundaries eat
raw fp32/fp16 GPU matmul outputs; and the only rail feeding the *reported
metric* is that one, under fp16.

## 4. Near-boundary mass and flip sensitivity (measured)

Analytic NF forward, float64 reference, 2048 test samples; distance =
`|k_raw − round(k_raw)|` of interior values (`0 < r < 1`), `k_raw = S·(1−V/θ)`.

**Interior mass within distance d of a ceil boundary:**

| stage / perceptron | interior frac | <1e-9 | <1e-7 | <1e-6 | <1e-5 | <1e-4 | <1e-3 | min dist |
|---|---|---|---|---|---|---|---|---|
| TTFS-FT p0 | 0.367 | 0 | 0 | 0 | 5.2e-6 | 2.0e-4 | 2.0e-3 | 1.4e-6 |
| TTFS-FT p1 | 0.458 | **1.7e-2** | 3.4e-2 | 5.1e-2 | **6.8e-2** | 6.8e-2 | 8.7e-2 | 2.7e-12 |
| TTFS-FT p2 | 0.372 | 0 | 0 | 0 | 3.1e-5 | 2.2e-4 | 1.9e-3 | 1.3e-6 |
| WQ p0 | 0.346 | 0 | 7.0e-4 | 8.8e-4 | 8.8e-4 | 1.3e-3 | 2.9e-3 | 1.9e-9 |
| WQ p1 | 0.377 | 0 | 3.6e-3 | 3.6e-3 | 3.6e-3 | 3.6e-3 | 3.6e-3 | 5.5e-9 |
| WQ p2 | 0.359 | 5.3e-5 | 4.9e-3 | 8.4e-3 | 8.4e-3 | 8.4e-3 | 8.4e-3 | 2.3e-11 |
| WQ p3 | 0.028 | 0 | 1.8e-3 | 7.0e-3 | 7.0e-3 | 7.0e-3 | 7.0e-3 | 7.4e-8 |

Up to **6.8 %** of a mid-pipeline layer's interior values sit within 1e-5 of a
boundary (an exactly-on-grid cluster produced by grid inputs × near-grid
weights), then an almost **empty band** up to ~1e-4 → the continuum. This
bimodality is what makes an epsilon-snap well-posed (§5d).

**Flip counts vs float64 (same batch, per stage; staircase outputs / decisions):**

| variant | TTFS-FT flips p0/p1/p2/p3 | dec. | WQ flips p0/p1/p2/p3 | dec. |
|---|---|---|---|---|
| fp32 GPU (b128/b57/b10000/b1), CPU fp32 | 0/0/0/0 | 0 | 0/0/0/0 | 0 |
| fp32 K-permuted / split-K(8) | 0/0/0/0 | 0 | 0/0/0/0 | 0 |
| TF32 forced on | 61/108/698/5 | 1 | 62/73/443/2 | 1 |
| fp16 autocast | 231/370/2552/12 | 2 | 249/296/1748/8 | 1 |

Measured fp32 noise amplitude in `k_raw` units (vs float64): max 2.8e-5
(encoding layer, K=784), ≤ 4.9e-6 elsewhere; loose per-neuron Higham bound
`S·γ_{K+2}·(|x|ᵀ|w|+|b|)/θ` evaluates to ≤ 4.1e-3 (p0) / ≤ 4.5e-4 (rest).
The exactly-on-grid cluster survives every fp32 reduction order because its
sums are computed exactly (grid inputs `k/S` × quantized weights; and near-grid
values within ~1e-9 are below fp32 resolution ~2.4e-7, so all orders round to
the same fp32 value). TF32/fp16 break exactness by rounding the *inputs* to
10/11-bit mantissas — that, not accumulation order, is what flips boundaries.

## 5. Fix design (validated where testable; nothing in `src/` was modified)

Priority order by measured impact:

**(a) Implement the missing determinism owner — the actual fix for the 9 pp.**
`PipelineSession` (composition root, `pipelining/session.py:111-131`) applies,
before any step runs: `random.seed(seed)`, `np.random.seed(seed)`,
`torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`,
`torch.use_deterministic_algorithms(True, warn_only=True)`,
`torch.backends.cudnn.benchmark = False` — with `seed` from
`DeploymentPlan.seed` (`pipelining/core/deployment_plan.py:113,175`).
`run.py:10` already provides `CUBLAS_WORKSPACE_CONFIG`. Prototype (wrapper, no
src changes) validated: **bit-identical state dicts and metrics across runs**
(§2 RC1 table). Unit test: build `PipelineSession.from_config` twice with the
same seed → identical `Model Building` state-dict hashes; different seeds →
different. Note: cross-stack bit-equality of *training* remains impossible
(RC3) — the campaign protocol for cross-stack comparison should be
"train once, ship the artifact, evaluate everywhere", which the pipeline's
cache/resume (`start_step`) already supports.

**(b) Metric-grade evaluation: remove fp16 from reported metrics.**
Generalize the existing seam `mbh_ledger._autocast_disabled`
(`tuning/orchestration/mbh_ledger.py:21-27`) into a shared
`metric_grade_eval(device)` context (fp32, autocast off) in
`model_training/`, and use it in `basic_trainer_eval.test()` (`:116`) and the
validation paths (`:104`) instead of `_eval_autocast`. Keep autocast for
*training* throughput; the *measurement* must be fp32. Measured effect: the
3/10 000 fp16 decision flips vanish; the reported number becomes the
fp32 = fp64 = CPU value (0.9620). Cost on t0_20 eval: negligible (MLP).

**(c) Pin fp32 precision explicitly at session init (defensive + conv fix).**
`torch.backends.cuda.matmul.allow_tf32 = False`,
**`torch.backends.cudnn.allow_tf32 = False`** (this one is a live behavior
change: conv cells currently evaluate under TF32 on both stacks),
`torch.set_float32_matmul_precision("highest")`. Measured stake: TF32 flips
61-698 staircase outputs/layer. One place: the same session-init block as (a);
never per-step.

**(d) Epsilon-guarded boundary snap in the wire-op SSOT (cross-stack guarantee
for fixed artifacts).** Change the fire index from `ceil(k_raw)` to
`ceil(k_raw − ε)` in the parity-locked twins
(`wire_semantics.py:19,34,44,54` — both torch and numpy, so NF↔SCM parity is
preserved by construction), with either the per-neuron rigorous bound
`ε_j = S·γ_{K+2}·(|x|ᵀ|w_j| + |b_j|)/θ_j` (one extra abs-matmul; provably ≥ any
fp32 dot-product error, Higham forward-error bound) or the fixed
`ε = 2^-15 ≈ 3.05e-5` (> measured max noise 2.8e-5, inside the measured empty
band). Guarantees: any backend whose evaluation error is < ε produces the same
fire index for every value whose true distance to the boundary exceeds ε; exact
integers are unaffected (`ceil(k − ε) = k`); the SCM/nevresim integer-exact
rail is unchanged post-WQ (its `k_raw` is exact, so the snap is a no-op there —
but the C++ `ttfs_analytical_compute.hpp` compare must adopt the same ε for
pre-WQ cross-sim screens). Cost, measured: legitimate continuum values inside
`(k, k+ε]` get pulled one step down — occupancy of `(1e-5, 1e-4]` is ≤ 3.7e-4
of interior values (WQ p0) and 0 for the other layers; i.e. a semantic
redefinition on a ≤ 0.04 % set in exchange for platform invariance. This is a
*semantics change* and needs the NF↔SCM ratchet tests regenerated — do it only
if cross-stack NF-metric equality on fixed artifacts is a hard requirement;
with (a)+(b)+(c) the measured fp32 residual is already zero on both probed
stacks.

**(e) Ratchet the integer-exactness invariant.** Post-WQ, the verification
rails are exact because `chip_quantize.py:37,59` emits true `int8/int16` and
all SCM arithmetic is float64 over integers < 2^53. Add a unit test that (i)
asserts quantized `core_matrix.dtype` is integral, (ii) bounds
`S·(Σ|nᵢ|·S + S·|b|) < 2^53` per core, (iii) checks
`ttfs_quantized_staircase_np` equals its torch twin on a grid-adversarial
corpus (values at distance {0, ±1e-12, ±1e-7, ±1e-5} from boundaries). Also
rewrite the `run.py:7-10` comment: the observed drift was fp16-metric +
unseeded-trajectory noise, not cuBLAS reduction order (this memo, §2/§4).

## 6. Repro artifacts

Scripts + raw JSON in the session scratchpad
(`.../scratchpad/m3/`): `exp1_eval_variants.py` (variant grid, 10 000 samples),
`exp2_boundary_mass.py` (mass + flips per stage), `exp3_reduction_order.py`
(permuted/split-K), `probe_body.py` (stack fingerprint; cluster run
`slurmech 20260710-021822-e9b1c134`, job 679136), `run_seeded.py` (fix-(a)
prototype), configs `m3_{van,seed}_{a,b}.json`. The seeded wrapper, for
reference:

```python
# before any CUDA context / pipeline import, seed = config["seed"]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

Measured bottom line: with (a) applied, same-stack runs are bit-identical; with
(b)+(c), the reported metric of a fixed artifact is flip-free across every
precision/order/batching/device variant we could construct on both probed
stacks; (d) upgrades that from "measured" to "guaranteed".
