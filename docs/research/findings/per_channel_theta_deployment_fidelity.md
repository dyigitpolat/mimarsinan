# Per-channel θ co-train is not deployment-faithful under per-perceptron threshold grouping

**Status:** mechanism CONFIRMED by code + measured by the honest NF↔SCM parity gate.
Round-1/2 accuracy harvested (see `mnist_mixer_fix_wave.md`): TTFS θ rejected at parity
0.9688<0.98; LIF θ faithful (identity-sim 0.9537 vs torch 0.9592) but caps ~0.95–0.96
and does **not** compose to a lift with genuine-QAT (θ+QAT round-2b = 0.949–0.958, no
gain over θ-alone ~0.958).

## Question

`ttfs_theta_cotrain` / `lif_theta_cotrain` rebind each non-encoding perceptron's
`activation_scale` to a **per-output-channel** trainable Parameter, so the gradual
blend ramp can co-optimise the firing threshold θ *per channel* (a wide and a narrow
channel of the same perceptron want different thresholds). Prior toy-cascade notes
claimed this is near-lossless. Does it survive the honest deployment-fidelity gate on
the real `mlp_mixer_core`?

## Verdict

**No.** Adding per-channel θ to the parity-clean `genuine_blend_fast` cascaded recipe
drops the decision-level NF↔SCM agreement to **0.9688 < 0.98** → `NfScmParityError`.
The same scalar-free recipe *without* θ passes at parity **0.9961**. The gate is
correct; θ as currently mapped is genuinely unfaithful, so the gate was **not**
weakened to accommodate it (per CLAUDE.md).

## Mechanism (two independent structural anchors)

The analytical reference and the deployed cascade disagree on what θ *is*:

1. **NF reference honors per-channel θ.** `nf_scm_parity.py:_capture_nf_normalized`
   normalises each perceptron's output by its full `activation_scale` tensor —
   `scale.dim() > 0` ⇒ `_channel_broadcast_view(scale, out)` (a distinct divisor per
   channel). So the analytical forward sees per-channel θ.

2. **Deployment scalarises θ.** The deployed path carries a single threshold per
   perceptron, by two compounding facts:
   - `activation_analysis_step.py:220` re-derives `activation_scales` **fresh** as one
     scalar quantile per perceptron (`scale_from_activations(merged, quantile=q)`),
     discarding any per-channel structure the tuner trained.
   - `layout_ir_mapping_finalize.py:127` sets `threshold_group_id = perceptron_index`:
     the deployed hardware model groups **all neurons of a perceptron under one
     threshold group**. Per-neuron thresholds are not representable in the current
     layout — there is exactly one threshold per perceptron.

Per-channel θ (per-neuron threshold) is therefore **structurally collapsed to a scalar
at mapping**. NF normalises per channel; the deployed cores threshold per perceptron;
their argmax boundaries diverge on the channels whose θ departs most from the
per-perceptron quantile → the measured 3pp agreement loss.

## It is mode-DEPENDENT, not mode-independent (round-2 measured)

The round-2 wave **refuted** an earlier prediction that LIF θ would break parity for
the same reason. It does not — and the reason is illuminating:

- **`lif_theta_cotrain` deploys near-losslessly** (ec=0): the genuine identity-mapped
  spiking simulation scores **0.9537 vs torch 0.9592 (~0.6pp)**. Per-channel θ survives
  LIF deployment.
- **The strict NF↔SCM gate does not even run for LIF.** `nf_scm_parity_enabled`
  (`nf_scm_parity.py:62`) returns False unless the mode is cascaded or
  `training_forward_kind() == "analytical_staircase"`; LIF is neither. So a *passing*
  LIF run was never subjected to the decision-level check that caught TTFS θ — its
  faithfulness rests on the genuine-cascade spiking-sim accuracy, which is the honest
  deployed number and is near-lossless.

**Why θ is faithful for LIF but not TTFS:** the two modes use `activation_scale`
differently. In TTFS it is the *firing threshold* — a hardware quantity shared across a
perceptron's neurons (one `threshold_group_id` per perceptron), so per-channel θ cannot
be represented and is scalarized. In LIF it is the *decode/readout scale* applied
per-channel at spike-count readout (output ≈ spikes·scale/steps), which is **not** tied
to the shared threshold group — so per-channel structure is preserved through
deployment. Per-channel θ is thus a **viable faithful lever for LIF and a dead end for
TTFS-cascaded**, on the same hardware model.

## Forward path

- **LIF:** `lif_theta_cotrain` is faithful (per-channel decode scale) and a legitimate
  accuracy lever (~0.958 deployed, near-lossless), but round-2b showed it does **not**
  compose to a lift: θ+genuine-QAT measured 0.949–0.958, no gain over θ-alone. The LIF
  faithful levers are redundant (cap ~0.95–0.96), not additive — closing the last ~1–2pp
  needs a structural lever (time-steps / decode-scale calibration), not another recipe.
- **TTFS-cascaded:** do **not** use per-channel θ (verified unfaithful). Faithful
  per-channel θ would require per-neuron threshold groups — `threshold_group_id` unique
  per neuron — which multiplies threshold memory and changes the core-packing model
  (cores packed assuming shared per-group thresholds). A real hardware-model change, not
  a config fix; recorded as candidate study #46.
- **`genuine_qat` is NOT the cascaded answer.** Round-1 measured cascaded `genuine_qat`
  *collapsing* at the genuine-cascade ramp tail (rate stalls ~0.887, TTFS-Cycle
  fine-tuning drops to 0.14 → accuracy-retention assertion) — the same controller-path
  deep-cascade collapse as `policy_isolate`. It is parity-clean only because it never
  reaches deployment. The cascaded revival's surviving faithful path is the **fast
  ladder** (`genuine_blend_fast`: ec=0, deployed 0.9396 @ parity 0.9961) — the open
  problem is lifting it above ~0.94 without re-entering the controller collapse.
  `genuine_qat` remains a valid faithful lever for **LIF** (no deep-cascade collapse
  there; ec=0 ~0.96).

## Why this matters

The honest NF↔SCM parity gate did exactly its job: it caught an accuracy optimisation
that improves the *analytical* model but is not realisable on the deployed hardware,
*before* it could be reported as a deployed-SNN result. A weaker (accuracy-only)
acceptance test would have mis-reported per-channel θ as a win.
