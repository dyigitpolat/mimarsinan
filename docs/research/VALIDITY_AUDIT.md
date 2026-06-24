# Deployment Validity Audit — On-Chip Parameter Majority

**Constraint (hard).** A valid deployment must run **≥ 50% of total model parameters on
the chip**. A parameter-majority host-side deployment is an **invalid configuration** — the
chip is not the deployment vehicle if the host holds most of the weights.

**Gate.** `src/mimarsinan/mapping/verification/onchip_majority.py`
(`assert_onchip_majority_or_raise`), wired into `SoftCoreMappingStep` after IR pruning,
default-on (`onchip_majority_gate`, floor `onchip_majority_min_fraction=0.5`). Merged `35312a0`.
On-chip = `total_params − host_params`; host = unique host-side ComputeOp params (the offloaded
encoding Linear/Conv, classifier readout, attention), deduped by module identity. 10 dedicated
tests.

## Measured on-chip fraction (real runs, reproduced with the gate's own function)

| family | total params | host params | on-chip % | verdict |
|---|---:|---:|---:|---|
| **deep_mlp d4 (w64)** | 63,390 | 50,895 | **19.7%** | ❌ INVALID |
| **deep_mlp d8 (w64)** | 80,050 | 50,895 | **36.4%** | ❌ INVALID |
| deep_cnn d4 | 98,462 | 1,455 | 98.5% | ✅ VALID |
| deep_cnn d8 | 294,562 | 1,455 | 99.5% | ✅ VALID |
| mlp_mixer (cert vehicle) | 78,039 | 7,695 | 90.1% | ✅ VALID |
| lenet5 | 107,806 | 1,011 | 99.1% | ✅ VALID |

**Why deep_mlp fails:** its 784→64 input encoder is 50,240 params and runs host-side (the
analog→spike encoder is intrinsically off-chip); the narrow w64 body (≈64²·depth) cannot
out-weigh it. `subsume` does not fix this — the encoder is host-side regardless. Deeper is
"less bad" (more on-chip hidden layers) but still <50%. A **wide** deep_mlp (W ≥ 784/(depth−1),
e.g. W≥256) would be valid, but every deep_mlp run in this campaign used w64.

## Consequence

- **deep_mlp is retired as a deployment vehicle.** 70 ledger rows (model=deep_mlp) annotated
  `deployment_validity: INVALID_host_majority`; 17 deep_mlp backlog batches disabled. The
  death-cascade depth law, WS6 breadth, and WS7 escalation results were all on deep_mlp — the
  *phenomena* may be real, but they **cannot be reported as valid on-chip deployments**.
- **deep_cnn is the valid trainable-deep replacement** (98.5–99.5% on-chip, trains deep). The
  deep_cnn depth × {cascaded,synchronized} sweeps now carry the trainable-deep science validly.
- **Valid evidence base:** deep_cnn, mlp_mixer (the 6/9-MET cert vehicle), lenet5.
- Optional future work: re-establish the depth law on a **wide** deep_mlp (W≥256) if a pure-MLP
  point is wanted — but deep_cnn already covers trainable-deep validly.
