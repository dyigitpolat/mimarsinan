# W4 — the first live DeploymentRecord (2026-08-11)

Evidence that the thesis-§2 formalization emits from a real deployment, not a
fixture. Artifact: `artifacts/deployment_record_first_emission_t0_45.json`
(11.9 kB, cell `t0_45_lifs_simplemlp_wq_s4`, CPU run — the *functional* claims
are device-independent). The GPU-solo no-regression leg ran separately and
**reproduced the reference deployed metric exactly: 0.9830**, with the same
record assertions green (seal-load OK, boundary traffic non-None, energy and
`cost_record.json` present) — so the record machinery costs nothing in
deployed accuracy.

## What the record proves

| Claim | Evidence in the artifact |
|---|---|
| Every §2 fragment populates | `identity, schedule, placement, utilization, traffic, timing, energy, accuracy, adaptation` all present, seal-valid on `DeploymentRecord.from_dict` |
| Provenance is structural | 9/9 fragment groups carry `Provenance{kind, producer, step}`: `energy` measured@SANA-FE, `traffic` measured@gate+NoC, `schedule`/`placement`/`utilization` derived@Hard Core Mapping, `identity` declared@DeploymentPlan |
| The sizing SSOT is self-consistent | `params_reloaded = 68 096` params × `weight_bits = 5` ÷ 8 = **42 560 B** = the record's summed `params_bytes`; independently `programming_bits = 340 480` = 42 560 × 8 |
| Traffic is live, not stubbed | boundary reduction from the spike-count gate (node 3, 10 neurons, 6 counts) + SANA-FE NoC census (452 intra-tile packets, 97 input-path, 0 inter-tile — a single-tile mapping) |
| The floorplan is the declared one | `cores_per_tile 4, mesh 6×5, derivation "derived"` — the W1.2 deterministic derivation, recorded |
| Legacy continuity holds AND the dead fields went live | `cost_record.json` v3: `acc_deploy 0.9815`, `mj_per_sample 0.0011245621`, and the formerly dead-by-default `reprogram_passes 1`, `reuse_passes 0`, `params_reloaded 68 096`, `ft_pass_walls` 1 entry |
| No double counting | `timing.latency.note` states SANA-FE `sim_time_s` already contains NoC transport, so no separate NoC latency term is added |

## Reading the pass census honestly

`schedule.pass_count = 0` with `reprogram_passes = 1` is not a contradiction:
`pass_count` is the *scheduled multi-pass* census (0 for an unscheduled program,
byte-for-byte the `LayoutPlan.from_hybrid_mapping` formula the seal cross-checks
against `LayoutVerificationStats.schedule_pass_count`), while `reprogram_passes`
counts *segments that must be programmed* from the IR reuse plan. Two currencies,
two provenances, both recorded — which is exactly why the record keeps provenance
per field instead of collapsing to one number.
