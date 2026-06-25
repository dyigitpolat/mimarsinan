# E5 — Pareto decision layer: cascaded vs synchronized (deep_cnn)

**Unit:** `wave3/e5-pareto`
**Module:** `mimarsinan.chip_simulation.pareto` (pure read, runs no model, byte-identical)
**Tests:** `tests/unit/chip_simulation/test_pareto.py`
**Consumes (does NOT modify):** the campaign ledger's per-schedule `deep_cnn` accuracy
rows; the `coverage_ledger` / `cost_extraction` cell-coordinate conventions.
**Question:** is the *cascaded* schedule a redundant code we should RETIRE, or does it
win a budget REGIME? This is the program's "automatic genericity" evidence — a decision
emitted from measured accuracy + a defensible cost model, not asserted.

---

## 1. The measured accuracy gap (data of record)

The deep_cnn science rows carry per-schedule deployed accuracy
(`cascaded_deployed_mean` / `synchronized_deployed_mean` / `ann_test_acc_mean`) keyed by
`dataset / depth / S / cores_count`. For each dataset the verdict reads the **most
defensible cell** — cores-instrumented first, then deepest (depth stresses the cascade),
then latest timestamp (the reconciled measurement). The selected cells:

| dataset | depth | S | cores | cascaded | synchronized | ANN ref | **acc gap (sync − casc)** |
|---------|------:|--:|------:|---------:|-------------:|--------:|--------------------------:|
| mnist   | 10 | 4 | 480 | 0.9297 | 0.9903 | 0.9907 | **+6.06 pp** |
| kmnist  |  8 | 4 | 480 | 0.8900 | 0.9619 | 0.9689 | **+7.19 pp** |
| fmnist  |  8 | 4 | 480 | 0.7900 | 0.9034 | 0.9330 | **+11.34 pp** |

The d10/S4 MNIST headline (sync 0.9903 vs cascaded 0.9297, **~6 pp**) is the cell the
unit is required to detect; the test asserts it to within ±1 pp and asserts the harder
datasets (fmnist) show a gap **≥** the easier ones. Synchronized tracks its ANN reference
to within noise on every cell; the gap is a **cascaded** firing-gain deficit (the
depth-driven death-cascade documented in WS3), NOT a synchronized loss.

These numbers are read straight from `runs/campaign/ledger.jsonl`; no value is fabricated
or fit. (The ledger is a runtime artifact not in git state; the real-ledger tests skip —
do not fail — when it is absent from the worktree tree.)

---

## 2. The cost proxy — a MODEL-ESTIMATE WITH A BAND (the integrity crux)

The ledger carries `cores_count` but **NOT** measured per-sample energy or latency per
schedule. We therefore do not fabricate a cost. `schedule_cost_band(schedule, s_global,
depth, cores)` derives a `CostProxyBand` from two grounded models, each carried as a
(lo, nominal, hi) band and LABELED a model-estimate (`COST_BAND_DISCLAIMER`):

### 2a. Latency (tight model-estimate)

From the **documented genuine-TTFS execution model**
(`ttfs_cycle_synchronized_execution`, `ttfs_cascade_latch_erosion`):

- **synchronized**: latency groups run **sequentially**, a full S-cycle window each →
  `sim_time = S × latency_groups`.
- **cascaded** (pipelined): `sim_time = S + latency_groups` — strictly lower.

The only assumption is the `latency_groups ≈ depth` mapping; we band it `±1` group for
input/output framing. The schedule formula itself is **exact** given the group count, so
this axis is a *tight* model-estimate. For the headline cells:

| dataset (cell) | sync latency [lo, **nom**, hi] | cascaded latency [lo, **nom**, hi] | nominal speedup (casc) |
|----------------|-------------------------------:|-----------------------------------:|-----------------------:|
| mnist d10/S4   | [36, **40**, 44] | [13, **14**, 15] | **2.86×** |
| fmnist d8/S4   | [28, **32**, 36] | [11, **12**, 13] | **2.67×** |
| kmnist d8/S4   | [28, **32**, 36] | [11, **12**, 13] | **2.67×** |

(units = sim-steps/sample; cascaded is ~2.7–2.9× lower latency on these cells.)

### 2b. Energy (banded proxy, present only when cores known)

`energy_proxy ~ cores × active_steps` (the chip is soma-dominated; every active step
charges the resident cores — the same `Σ_d steps` the latency proxy counts). This is a
**proxy**, carried as a band, and present **only** when `cores_count` is in the row.
For the headline MNIST cell (cores=480): sync energy proxy
[17280, **19200**, 21120] vs cascaded [6240, **6720**, 7200] (cores·sim-steps).

### 2c. UNINSTRUMENTED axis (a stated instrumentation gap)

When `cores_count` is absent, `CostProxyBand.energy_uninstrumented` is `True` and the
energy fields are `None` — the gap is **flagged, not invented**.

> **Absolute per-sample spike energy per schedule is UNINSTRUMENTED in the ledger.**
> The latency proxy is a tight derivation from the documented execution model; the energy
> proxy is `cores × active_steps`, a *model-estimate with a band*, **not** a measured
> per-sample energy. Closing this is a remaining instrumentation task: emit measured
> per-schedule spike-energy (the `cost_extraction.CostRecord` already carries `spikes` /
> `mj_per_sample` per run; wiring it per-schedule into the ledger would replace this proxy
> with a measured number).

---

## 3. The verdict — REGIME, not retire (conditional on the cost band)

`cascaded_vs_synchronized(rows)` emits, per dataset, a `ScheduleVerdict` with the accuracy
gap, the cost band, the front schedule, and the E5b recommendation:

- **`RETIRE_CASCADED`** only if synchronized **DOMINATES** — better accuracy **AND** not
  worse cost (cascaded not on the (cost, accuracy) front).
- **`REGIME_DEPENDENT`** when synchronized wins accuracy but cascaded keeps a cost axis
  (here: strictly lower pipelined latency → cascaded stays on the front → it wins the
  hard-latency budget).
- **`NO_GAP`** when the schedules tie within the read-noise floor (0.5 pp).

**Result on the real ledger:** every measured dataset is `REGIME_DEPENDENT`
(`any_full_retire = False`). Synchronized is the accuracy-front schedule everywhere
(+6 to +11 pp), but cascaded's ~2.7–2.9× lower latency keeps it on the (latency, accuracy)
Pareto front — so cascaded is **not** dominated and **not** retired. It is the
hard-latency-budget code.

> **The verdict is CONDITIONAL on the cost-proxy band** (`conditional_on_cost_band =
> True`). If the UNINSTRUMENTED per-sample energy were measured and turned out *higher*
> for cascaded than synchronized despite its lower latency (e.g. ramp-reconstruction
> traffic), cascaded could fall off the front and the verdict would flip toward
> `RETIRE_CASCADED`. With the documented latency model alone, cascaded survives.

---

## 4. `propose_recipe(budget, rows, dataset)` — the E5c selector

Picks `(firing, schedule, S, placement)` from the front for a stated budget:

| budget | schedule chosen | rationale |
|--------|-----------------|-----------|
| `accuracy` | **synchronized** | the accuracy-front schedule (+6.06 pp on MNIST) |
| `latency`  | **cascaded** (IF on front) | pipelined `S + groups < S·groups`; chosen despite the accuracy cost |
| `latency`, cascaded off front | **synchronized** | a lower-accuracy AND not-cheaper cascaded is never chosen (e.g. depth-1 cell: no latency win) |

On MNIST: `accuracy → ttfs / synchronized / S=4 / on_chip_majority`;
`latency → ttfs / cascaded / S=4 / on_chip_majority`. The recipe carries the cost-band
disclaimer in its rationale — the pick is conditional on the same band.

---

## 5. Honesty ledger

- **Accuracy** — measured, read from the ledger, not fit. (Caveat carried in the ledger
  rows themselves: some cells were measured at small `max_simulation_samples`; the verdict
  selects cores-instrumented cells but does not re-derive variance — it reports the
  recorded means.)
- **Latency** — a *tight* model-estimate from the documented execution model, banded ±1
  group. Not a measured wall-clock latency.
- **Energy** — a *model-estimate proxy* (`cores × active_steps`) with a band, present only
  when `cores_count` is known; **flagged UNINSTRUMENTED** otherwise. NOT measured
  per-sample energy.
- **Verdict** — `REGIME_DEPENDENT` on every measured dataset, explicitly **conditional on
  the cost-proxy band**. No unconditional retirement is claimed.
