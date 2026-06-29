# MNIST `mlp_mixer_core` Fix Wave — measured outcome (v2)

This is the measured outcome of the **MNIST Mixer Fix Wave**: the repair attempt
on the nine failing `mlp_mixer_core` diagnostic families from the v1 diagnostic
wave. It is a partial closure — one family (quantized TTFS) reaches the fidelity
bar on the honest on-chip axis; two frontiers remain open (LIF accuracy, and a
genuine-cascade collapse on the cascaded cell).

All numbers below are **measured** on the slurmech H100 cluster (6-node pack
wave, 33 jobs across 6 detached allocations), harvested verbatim from each run's
`__target_metric.json` + `cost_record.json` via the remote stager
(`scripts/campaign/slurmech_remote_stage.py`) and the local harvester
(`scripts/campaign/mnist_mixer_local_harvest.py`).

## Two acceptance axes (and why both matter)

The plan's acceptance criterion is hard on three counts:

- `deployed_acc >= 0.97`
- `relative_time < 1.0`, where `relative_time = run_wall_s / fastest_successful_analytical_wall_s`
- `returncode == 0`

`relative_time` is a **pipeline-wall** ratio: it includes the one-time QAT
training cost. It is the right axis for "how long did it take to *produce* this
deployable artifact", and the wrong axis for "how expensive is this artifact to
*run* on-chip". The honest on-chip efficiency axes are read directly from each
run's `cost_record.json`:

- `latency_steps` — deployed spiking-sim latency to a decision (SANA-FE)
- `mj_per_sample` — deployed energy per sample
- `relative_deploy_latency` / `relative_deploy_energy` — those two relative to
  the analytical-control `cost_record` (the same reference column, but measured
  on the on-chip cost rather than the training wall).

Both axes are reported below. Where they disagree, the disagreement *is* the
finding.

## Headline results (measured; full wave, 33/33 runs harvested)

Reference column — analytical TTFS control: **acc 0.980–0.984**, on-chip
**latency 12 steps**, **0.0359 mJ/sample**. Final family counts: **9
VALID_FLAGGED, 1 MEASURED_DEAD, 1 GATE_REJECTED**.

| Family / recipe | acc (min..max) | on-chip lat | on-chip mJ | rel_lat | rel_energy | classification |
|---|---|---|---|---|---|---|
| `mixer_ttfs_quantized_q100_aggressive_timing` | **0.970..0.974** | 13 | 0.0349..0.0365 | 1.08× | **0.97..1.02×** | flag (wall only) |
| `mixer_ttfs_quantized_q100_fast_timing` | **0.970..0.974** | 13 | 0.0359..0.0378 | 1.08× | 1.00..1.05× | flag (wall only) |
| `mixer_sync_ttfs_faithfulness_repair` | 0.955..0.965 | 36 | 0.088..0.093 | 3.0× | 2.45..2.58× | flag (acc) |
| `mixer_sync_ttfs_relaxed_parity` | 0.959..0.965 | — | — | — | — | **GATE_REJECTED** |
| `mixer_lif_genuine_qat` | 0.938..0.968 | 12 | 0.053..0.057 | **1.00×** | 1.48..1.59× | flag (acc) |
| `mixer_lif_fast_stabilized` | 0.944..0.962 | 12 | 0.056..0.062 | 1.00× | 1.57..1.72× | flag (acc) |
| `mixer_lif_genuine_qat_a03` | 0.940..0.962 | 12 | 0.058..0.060 | 1.00× | 1.62..1.66× | flag (acc) |
| `mixer_lif_genuine_qat_a07` | 0.944..0.950 | 12 | 0.052..0.056 | 1.00× | 1.45..1.56× | flag (acc) |
| `mixer_cascaded_genuine_blend_fast` | 0.920..0.934 | 13 | 0.043 | 1.08× | 1.19× | flag (acc) |
| `mixer_cascaded_genuine_qat` | **0.114..0.137** | — | — | — | — | **MEASURED_DEAD** |

## Finding 1 — three of four v1 collapses are fixed; one persists in the genuine cascade

The v1 diagnostic wave had **four families at 0/N `rc=1` hard collapse**:
proxy-refine (weight-quantization retention collapse), controller-baseline
(TTFS-cycle fine-tuning collapse), and both synchronized families (sync-QAT and
sync-relaxed, failing the NF↔SCM + torch↔deployed parity gates).

The v2 wave fixes **three of the four**:

- The synchronized parity collapse is repaired by the genuine cascade —
  `faithfulness_repair` completes `rc=0` at acc ~0.96 (Finding 3).
- The proxy/relaxed path no longer "collapses": `relaxed_parity` is now a
  *principled* gate rejection at healthy accuracy (`GATE_REJECTED`), not an
  accuracy collapse.

But **one collapse persists, and the genuine cascade is what triggers it.** The
cascaded cell's full genuine-QAT recipe (`mixer_cascaded_genuine_qat`,
`ttfs_blend_fast_stabilize_steps=300`) collapses on **all three seeds** at the
**TTFS Cycle Fine-Tuning** step — accuracy falls from ~0.96–0.97 to **0.114,
0.114, 0.137** (chance), tripping the retention guard
(`0.1135 < 0.85 × 0.9641`). This is the *same* failure mode as the v1
`controller_baseline` collapse, and it is correctly classified `MEASURED_DEAD`
(distinct from the `GATE_REJECTED` of `relaxed_parity`). The collapse appears
*after* the full stabilize schedule runs — these are the longest walls in the
wave (1994–2206 s) — so it is not a budget/early-exit artifact; the genuine TTFS
cycle tuning actively destroys the trained model on this cell.

The lighter `mixer_cascaded_genuine_blend_fast` (proxy-then-refine, no full
genuine cascade) **survives** on all three seeds (`rc=0`, acc 0.920–0.934) but
sits below the 0.97 bar. So on the cascaded cell the relationship inverts
relative to the synchronized cell: **the genuine cascade collapses where the
proxy variant survives** (see the cell-dependence note in Finding 3).

## Finding 2 — quantized TTFS meets the fidelity mandate; the wall gate hides it

**All six quantized-TTFS runs (2 recipes × 3 seeds) clear `acc ≥ 0.97`**
(0.970–0.974), against the analytical control's 0.980–0.984 — a ~1pp conversion
gap. More importantly, they deploy at **essentially the same on-chip cost as the
analytical control**: latency 13 vs 12 steps (1.08×), and energy 0.0349–0.0378
mJ vs the control's 0.0359 — i.e. **0.97×–1.05× energy**. The
`aggressive_timing` recipe's best seed (0.0349 mJ) is *strictly more
energy-efficient* than the analytical control at ≥0.97 accuracy.

Yet every one of these runs is **flagged, not passed** — and the only reason is
`failure_reason = slower_than_baseline`. Their pipeline wall (642–711 s) exceeds
the analytical control's wall (501 s) because they pay a one-time QAT training
cost the analytical control never incurs. **This is a `relative_time`-axis
artifact, not a deployment-efficiency result.** On the honest on-chip axis these
recipes are accepted: near-lossless accuracy at ~1.0× latency and ~1.0× energy.

> **Load-bearing axis decision (flagged for the record, not silently changed).**
> The `relative_time < 1.0` gate measures *time-to-produce-the-artifact*, which
> includes one-time QAT. For the conversion-fidelity mandate ("ANN→SNN retains
> accuracy near-losslessly at competitive deployment cost") the operative
> efficiency axis is `relative_deploy_latency` / `relative_deploy_energy` from
> `cost_record.json`. The harvester now emits both; promotion decisions should
> read the on-chip axis. I have **not** removed the `relative_time` gate — it is
> still a meaningful training-cost signal — but the family classification should
> not call a near-lossless ~1.0×-on-chip-cost recipe a failure on training wall
> alone.

## Finding 3 — genuine cascade beats budget loosening (thesis-grade)

`relaxed_parity` and `faithfulness_repair` are the same synchronized cell under
two different repair philosophies:

- `faithfulness_repair` runs the **genuine cascade** (`stabilize_steps=300`):
  all three seeds complete `rc=0` at acc 0.955–0.965.
- `relaxed_parity` **loosens the NF↔SCM per-neuron parity budget** and skips the
  genuine cascade (`stabilize_steps=0`, proxy only): 2/3 seeds are rejected
  `rc=1` at the **torch↔deployed-sim parity gate** (`scm_torch_sim_parity` <
  0.98) despite healthy accuracy (0.960, 0.965).

Loosening the *static* per-neuron parity budget does **not** buy deployment
fidelity: the deployed spiking sim still diverges from the trained torch cascade,
and the independent torch↔sim gate correctly rejects it. Only the genuine
cascade — which actually trains under the spiking dynamics — passes that gate.
This is a clean validation of the central thesis: **fidelity is earned by
training under the target dynamics, not by relaxing the checks that measure it.**

The harvester now classifies this honestly: `relaxed_parity` is
`GATE_REJECTED` (healthy-accuracy `rc=1` fidelity-gate rejection), a distinct
bucket from `MEASURED_DEAD` (accuracy collapse). The two failure modes are not
the same and are no longer conflated.

> **Cell-dependence (do not over-generalize "genuine > proxy").** The genuine
> cascade wins on the *synchronized* cell (faithfulness_repair passes where
> relaxed_parity is gate-rejected) but **loses on the *cascaded* cell**, where
> the full genuine-QAT recipe collapses at TTFS Cycle Fine-Tuning (Finding 1)
> while the proxy-blend variant survives. The correct statement is narrower:
> *loosening the parity checks does not buy fidelity*, and *training under the
> target dynamics earns fidelity where that training is stable*. On the cascaded
> cell the genuine TTFS cycle tuning is **not** stable, and that instability — not
> a parity-budget question — is the open problem (queued as a SOLUTION study).

## Finding 4 — LIF is the remaining frontier (a real, modest gap)

The four LIF recipes (genuine_qat at α ∈ {0.3, 0.5, 0.7}, plus fast_stabilized)
all complete `rc=0` and deploy at **the same latency as the analytical control
(12 steps, 1.00×)**, but at **~1.45–1.72× energy** (more spikes per decision) and
**~1.5–2.5 pp short of the 0.97 bar** (best single seed: genuine_qat 0.968;
family means 0.947–0.956).

The α-sweep shows **α is not the lever**: default α=0.5 (mean 0.956) edges out
α=0.3 (0.950) and α=0.7 (0.947). The LIF gap is not a KD-blend tuning problem;
it is an intrinsic rate-coding conversion gap. This is the one family the fix
wave did **not** close, and it is the correct target for the next study (a
genuine SOLUTION study per the research mandate — candidate levers: longer
stabilize schedules, per-channel LIF threshold calibration, or a higher
timestep budget traded against the energy axis).

## Finding 5 — synchronized firing is faithful but 3× the on-chip latency

`faithfulness_repair` is fidelity-faithful (passes the torch↔sim gate) and lands
at acc ~0.96, but it deploys at **36 steps (3.0× latency) and ~2.5× energy**.
Synchronized firing inherently needs more timesteps to resolve. This is a real
efficiency cost, correctly surfaced by the on-chip axis, and it makes the
cascaded/quantized TTFS paths the preferred deployment route at this accuracy.

## Status & next steps

- **Wave:** complete — 6/6 packs, 33/33 runs harvested. Final counts: 9
  VALID_FLAGGED, 1 MEASURED_DEAD, 1 GATE_REJECTED.
- **Closed by the on-chip axis:** quantized TTFS (≥0.97 at ~1.0× on-chip cost).
- **Open frontiers (two SOLUTION-study targets):**
  1. **LIF accuracy** (~1.5–2.5 pp short; same latency, ~1.5× energy; α is not
     the lever).
  2. **Cascaded genuine-QAT TTFS-cycle collapse** — the genuine cascade falls to
     chance at TTFS Cycle Fine-Tuning (`MEASURED_DEAD`); the proxy-blend variant
     survives but underperforms. Candidate levers overlap the residual-collapse
     fixes already studied (`residual_collapse_composed_fix.md`,
     `deep_residual_lif_deploy_fix.md`): per-mode QAT recipe at the cycle step,
     retention-guarded LR schedule, or freezing the cycle-tuning when the proxy
     is already converged.
- **Tooling landed this wave:** remote artifact stager (13 tests),
  `GATE_REJECTED` honesty split (separates fidelity-gate rejection from accuracy
  collapse), and on-chip deploy-cost axes wired end-to-end through the harvester.

## Round-1 + θ solution-study wave (measured; ec=exit, parity where the gate runs)

Two SOLUTION studies launched in parallel on 6 nodes (cascaded revival + LIF gap),
plus a round-2 θ wave. Numbers below are final exit codes; accuracy is the deployed /
identity-mapped spiking-sim metric (not mid-pipeline torch). Seeds 0–2.

| recipe (mode) | ec | deployed acc | fidelity | verdict |
|---|---|---|---|---|
| cascaded `genuine_blend_fast` | 0 | **0.9396** | parity **0.9961** | faithful, lossy (~3pp short) — the only surviving cascaded path |
| cascaded `genuine_qat` | 1 | collapse | — | controller deep-cascade collapse (rate stalls ~0.887, FT→0.14) |
| cascaded `policy_isolate` | 1 | collapse | — | **regression trigger CONFIRMED**: conversion_policy→controller→collapse (identical to genuine_qat) |
| cascaded `blend_theta` | 1 | — | parity 0.9688<0.98 | per-channel θ verified-unfaithful (see θ finding) |
| cascaded `staircase_ste_theta` | 1 | — | accuracy-retention | fails the cycle retention gate |
| LIF `genuine_qat` (+a03/a07) | 0 | ~0.958–0.964 | gate N/A for LIF | best LIF cluster; α barely moves it |
| LIF `fast_stabilized` | 0 | 0.955–**0.9677** | gate N/A | one seed nearly clears 0.97 |
| LIF `theta_cotrain` | 0 | ~0.958 | identity-sim 0.9537 vs torch 0.9592 | **faithful** (per-channel decode scale; θ doc) |
| LIF `theta_distmatch` | 0 | ~0.943 | gate N/A | distmatch+θ regresses vs θ alone |
| LIF `fine_ladder` / `controller_gradual` | 0 | 0.952–0.958 | gate N/A | no better than fast |
| LIF `q100_distmatch` | 0 | 0.90–0.93 | gate N/A | quantile=1.0 saturation hurts — drop it |

**Cascaded revival conclusion.** The regression is a **controller-path deep-cascade
collapse**: any recipe routing the genuine ramp through `smooth_adaptation_cycle`
(`genuine_qat`, `policy_isolate`) falls to chance as rate→1.0. The fast ladder
(`genuine_blend_fast`) is the *only* path that reaches full genuine spiking, and it is
faithful (parity 0.9961) but lands at 0.94. The open problem is **lifting the fast
ladder above ~0.94 without re-entering the controller collapse** — NOT a θ or
`genuine_qat` problem.

**LIF gap conclusion.** Every faithful LIF lever clusters **0.952–0.968**; none
reliably clears 0.97 (best single seed 0.9677, `fast_stabilized`). `genuine_qat` and
`theta_cotrain` are the strongest *faithful* levers; α-sweep, `fine_ladder`,
`controller_gradual` add nothing; `q100`/`distmatch` regress. The gap is real, small,
and not yet closed by any single round-1 lever — a composition (θ + QAT + best-seed
schedule) is the round-2 hypothesis.

## Round-2 push wave (measured; both hypotheses tested, both NEGATIVE)

The round-2 hypotheses — (a) LIF θ+QAT compose clears 0.97, (b) a finer near-1.0 ramp
or KD-anchoring lifts the cascaded fast ladder above 0.94 — were tested as config-only
compositions of the schema-registered levers (5 recipes × 3 seeds, all `ec=0`).
(First launch was a null result — a pack-feeder path bug, `normalize_job_cmd`
reconstructed a flat config path and ignored the manifest's `--config-dir` subdir; fixed
generically + relaunched as round-2b. Numbers below are the clean relaunch.)

| recipe (mode) | ec | deployed acc (s0/s1/s2) | fidelity | verdict |
|---|---|---|---|---|
| LIF `theta_qat` (θ+QAT KD/CE, α=0.5) | 0 | 0.955 / 0.952 / 0.949 | gate N/A for LIF | **no lift** over θ-alone (~0.958) |
| LIF `theta_qat_a03` (α=0.3) | 0 | 0.951 / 0.958 / 0.958 | gate N/A | **no lift**; lower α marginally better but still <0.97 |
| cascaded `blend_fast_finer` (rates …0.93,0.97,0.99,1.0; 160 steps) | 0 | 0.941 / 0.936 / 0.942 | parity 0.992–1.0 | faithful, **no lift** over anchor |
| cascaded `blend_fast_kd` (+ce_alpha 0.1 KD anchoring) | 0 | 0.934 / 0.937 / 0.941 | parity 0.992–1.0 | faithful, **no lift** |
| cascaded `genuine_blend_fast` (anchor) | 0 | 0.936 / 0.936 / 0.934 | parity 0.988–1.0 | reproduces round-1 ~0.94 |

**Both solution studies plateau below the 0.97 acceptance bar with config-only levers.**

- **LIF** caps at **~0.95–0.96** across every faithful recipe (θ, QAT, θ+QAT compose,
  fine ladder, controller-gradual). Composing θ with genuine-QAT KD/CE does **not** beat
  the best single lever — the levers are redundant, not additive. The remaining gap to
  0.97 is real and small but is **not** a training-schedule problem; it points at the LIF
  decode/quantization itself (time-step count, readout-scale calibration) — a structural
  lever, not a config one.
- **Cascaded** caps at **~0.94 @ parity ≥0.98**. A finer near-1.0 ramp and KD anchoring
  give no measurable lift over the plain fast ladder. The fast ladder is faithful but
  ceilinged; the only known way past it routes the genuine ramp through the controller
  (`smooth_adaptation_cycle`), which **collapses** (round-1). So lifting cascaded above
  0.94 requires fixing the controller deep-cascade collapse
  (`residual_collapse_composed_fix.md`) — a code change, not a recipe.

**Decision point.** Config-only levers are exhausted for both modes. The next levers are
structural and load-bearing — (1) cascaded: fix the controller-path collapse so a longer
genuine ramp is usable; (2) LIF: scale time-steps / recalibrate the per-channel decode
scale; (3) per-neuron threshold groups (#46) to make per-channel θ faithful for TTFS.
Each is a real code/hardware-model change, not a config sweep — deferred to a design call
rather than launched autonomously.
