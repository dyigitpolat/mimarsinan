# 51 — A near-lossless, stable, fast recipe for ANN → cascaded single-spike TTFS

Artifact 50 localized the cascaded gap to a greedy partial-sum FIRING-gain deficit and
called the genuine→staircase residual "structural". **This round overturns the
pessimistic framing of that residual**: with the right training recipe the genuine
cascade reaches the staircase/LIF ceiling — i.e. near-lossless — at depth, stably, and
fast. The cap was **optimisation**, not capacity.

Harness: `experiments/recipe_harness.py` + `recipe_*.py`, benched by
`focused_nearloss.py` (GPU `_DeepMLPBN` on digits, all-data, 3 seeds). A recipe is a
`train(flow, xtr, ytr, xva, yva, S, base, teacher, *, steps, seed)` that fine-tunes the
cold converted ANN through the genuine differentiable cascade.

## The decisive reframing: the cap is OPTIMISATION, not capacity

The genuine cascade uses the SAME weights as a staircase that scores 0.98 — so it is
fully expressive. Two measurements pin the bottleneck on optimisation:

- **Higher S HURTS genuine FT** (opposite of LIF). d=9: kd-FT S=16 0.905 → S=32 0.878 →
  S=64 0.622. More cycles (`n_cycles = S + depth`) = a longer cascade to backprop the
  fire-once surrogate gradient through = a harder optimisation. (Frozen-weight
  calibration is S-INDEPENDENT; FT is S-NEGATIVE.) ⇒ **train at low S (16).**
- The death cascade needs a JOINT correction (artifact 50): greedy/coordinate methods
  stall; gradient FT optimises all layers together but from a cold (chance) start it is
  unstable at depth.

So the recipe must make the deep, cold, fire-once cascade EASY to optimise.

## The winning levers (each targets a distinct bottleneck)

| lever | recipe | what it fixes |
|---|---|---|
| **joint per-channel θ co-training** (θ_lr 5e-2, w_lr 2e-3) | `warm_theta_kd` | θ IS the firing-gain knob (the collapse's root cause); co-adapting it with weights lets the optimiser set the firing regime directly |
| **progressive shallow→deep weight unfreeze** | `progressive_depth` | deep layers train only after the upstream distribution stabilises — eases the deep-cascade optimisation |
| **continuous-teacher KD** (α=0.3) + cosine LR + **all-data** | all | strongest single regulariser; val-split costs ~1.5pp on the toy |
| **combine the above** | `combo` | most stable + matches the staircase ceiling at depth |

LOSERS (documented): `blend_curriculum` (collapses at d=9 → 0.08), `s_anneal` (its
high-S warmup stages hurt), `surrogate_anneal` / `staircase_distill` (unstable at depth).

## Results (genuine deployed-cascade accuracy, 3 seeds, S=16, 1500 steps)

```
            d=6 (cont 0.981, staircase 0.980)     d=9 (cont 0.965, staircase 0.951)
combo              0.967 ± 0.011                          0.949 ± 0.011
warm_theta_kd      0.970 ± 0.013                          0.945 ± 0.016
progressive_depth  0.955 ± 0.012                          0.946 ± 0.012
kd_baseline        0.953 ± 0.025                          0.921 ± 0.021
```

- **At d=9, `combo` reaches 0.949 vs the staircase ceiling 0.951 — the genuine→staircase
  structural gap is closed to ~0.2pp**, with the lowest variance (most stable).
- Since **staircase ≡ LIF bit-exact** (artifact 50), the genuine cascade now matches the
  LIF/lossless ceiling. The only residual to the continuous ANN (~1.4pp at S=16) is the
  T+1-level quantisation floor that LIF shares — i.e. near-lossless w.r.t. the deployable
  code, not a cascade-specific loss.
- This BEATS the prior "fragile dead-end ~0.93" cascaded result.

## Speed (d=9 S=16, genuine deployed accuracy vs FT steps, 2 seeds)

```
steps |         combo |    warm_theta
  150 | 0.818±0.110   | 0.339±0.075
  400 | 0.953±0.007   | 0.840±0.045
```

`combo` reaches the near-lossless ceiling (0.953 vs staircase 0.958) in **400
steps**, with tiny variance (±0.007) — fast AND stable AND near-lossless, all three.
progressive-depth supplies the fast start (0.818 @ 150); θ-cotraining supplies the
near-lossless ceiling (warm_theta alone is slow early, 0.339 @ 150, but reaches
~0.945 by 1500). The combination dominates.

## Recommended recipe & the harness θ fix

`combo` (= θ co-train + progressive-depth + KD, low S, all-data) is the recommendation:
near-lossless at d=6 and d=9, lowest variance. `warm_theta_kd` is the simplest strong
variant (best at d=6).

A prerequisite bug fix (in `recipe_harness.promote_theta_per_channel`): perceptron
`set_activation_scale` only copies `.data` into the existing parameter, so returning a
NEW θ Parameter to the optimiser is a silent no-op (the forward never reads it). The fix
REBINDS `activation_scale` on BOTH the perceptron and its `TTFSActivation` node to the
same trainable param — this is what unlocked `warm_theta_kd` (pre-fix it scored ≈baseline).

## The high-S optimisation WALL (honest limit) — `highs.py`, `swarm_test.py`

"Near-lossless" above is **relative to the staircase/LIF ceiling at a given S**, not the
continuous ANN. The continuous-ANN gap has two parts: the genuine→staircase part (closed
by combo) AND the staircase→ANN **quantisation** part, which only shrinks at HIGHER S. So
to truly match the ANN you must raise S — but **genuine FT hits an optimisation wall at
high S** (the long fire-once cascade's surrogate gradient degrades; grad-clip + 2500 steps
don't fix it). combo + grad-clip, accuracy vs S (gap to the ANN):

```
 d   S |  ANN   ceiling   genuine combo   gap→ANN
 6   8 | 0.981   0.915      0.968          0.013
 6  16 | 0.981   0.974      0.972          0.009   <- d=6 sweet spot (near-98%)
 6  32 | 0.981   0.980      0.946          0.035   <- ceiling rises, genuine FALLS
 9  16 | 0.965   0.952      0.944          0.020   <- d=9 sweet spot
 9  32 | 0.965   0.965      0.911          0.054   <- ceiling == ANN, genuine can't track
 9  48 | 0.965   0.970      0.826          0.139
```

There is a **sweet spot at S≈16**: genuine cascaded peaks at **0.972 (d=6) / 0.944 (d=9)**
and DEGRADES at higher S even though the ceiling there equals the ANN. The S-warm-start
(`recipe_combo_swarm`: train the easy S=16 basin, continue at high S) NARROWS but does NOT
break the wall (d=9 S=32: direct 0.929 → warm 0.942, still 3.3pp under the 0.975 ceiling).

**Honest bottom line:** genuine single-spike cascaded reaches **near-98% at moderate depth
(d=6 S=16, 0.972)** but caps **~2pp below the ANN/LIF at the 9-deep depth (0.944)**; the
residual is a high-S optimisation wall, not yet cracked. True ANN-level at depth still
needs the **synchronized** schedule (the staircase, ≥0.97) or a multi-spike code. Train at
the **S≈16 sweet spot** (the real template's S=4 is far too low).

## Port to the real pipeline — θ-cotrain LANDED (code + tests)

The key lever is ported into the real tuner as an opt-in, default-off flag
`ttfs_theta_cotrain` (cascaded only; mutually exclusive with the per-depth gain ramp):

- `spiking/theta_cotrain.py::promote_activation_scale_per_channel` — rebinds each
  non-encoding perceptron's `activation_scale` to a per-output-channel `requires_grad`
  Parameter on the perceptron AND its nodes (the harness θ-rebind, productionised).
- `TTFSCycleAdaptationTuner._maybe_promote_theta_cotrain` (end of `_after_install_blend`,
  after scalar-θ distmatch/gain calibration) → the model optimiser co-trains θ with the
  weights through the genuine cascade. Stats on `_theta_cotrain_stats`.
- Tests: `tests/unit/spiking/test_theta_cotrain.py` (6) + `tests/unit/tuning/test_theta_cotrain.py`
  (9, incl. end-to-end that θ MOVES during the step and the deployed cascade keeps it).
  Flag-off byte-identical; 104 existing genuine/gain tests still pass.

REMAINING: (1) real-pipeline validation on the 9-deep mmixcore (cascaded ttfs_cycle_based,
`ttfs_theta_cotrain=True`) vs the LIF lossless number; (2) optional: progressive-depth
unfreeze as a second tuner flag (maps awkwardly to the controller — best on the fast
path) for the fast-start; (3) keep deploy S low (higher S hurts genuine FT).
