# TODO — SANA-FE static / leakage power accounting

**Status:** design agreed, not yet implemented.

## Why

SANA-FE models only **dynamic, per-event energy**: every `energy_*` attribute
in the arch YAML is a cost per event (synapse spike, soma access/update,
soma spike-out, axon-in/out, NoC hop). There is **no static / leakage /
idle-power term** anywhere in `sana_fe/src/*` (verified by grep — only one
unimplemented comment mention in `pipeline.hpp:150`).

For real chips this matters: TrueNorth's idle is ~63 mW of an ~70 mW total
at low spike rates (Akopyan 2015); Loihi 1 idle is ~30-50 mW per chip
(Davies 2018). For sparse / low-rate workloads, static dominates dynamic.
Reporting dynamic-only is a **dynamic-energy lower bound on actual chip
energy**, not the chip energy.

This is a SANA-FE design choice (event-driven architecture, comparative-
tool framing, calibration tractability), not a physical claim. We accept
that and patch it on the mimarsinan side rather than in upstream SANA-FE.

## Design

Add a static term to the preset and accumulate it post-hoc using SANA-FE's
own simulated chip time (`sim_time`).

### Why `sim_time` is the right multiplicand

SANA-FE distinguishes:

| Field | Meaning |
|---|---|
| `total_sim_time`, `ts.sim_time` (`sana_fe/src/chip.cpp:611`, `schedule.cpp:100`) | **Simulated chip wall-clock seconds.** Per-timestep critical path: `max(longest_neuron_chain, longest_message_chain) + timestep_sync_delay`, all derived from the arch-YAML latency constants. Accumulated across timesteps. |
| `wall_time`, `*_wall` | Simulator's own execution time via `std::chrono`. Not what we want. |

SANA-FE's `SpikingChip::get_power()` (`chip.cpp:614`) already uses
`total_energy / total_sim_time` — so the simulator itself treats
`total_sim_time` as the chip's runtime denominator for power. Use it the
same way for static.

### Implementation sketch (~30 lines)

1. **Fix segment-time aggregation first** (`runner.py` around line 141):
   `aggregate_sim_time_s` is currently `max(seg.sim_time_s for seg in segments)`.
   For sequential segments on one inference (each consumes the previous
   one's output) the total chip runtime is the **sum**, not the max. Max
   only makes sense for cross-sample pipelining, which we don't model.
   Either fix in place or add `total_sim_time_s` alongside.

2. **Add `static_power_w_per_core` to the preset typedef**
   (`presets.py:PerEventEnergy`). Default to `0.0` in `LOIHI_PRESET` and
   `TRUENORTH_PRESET` so existing tests/users see no behaviour change
   until they opt in.

3. **Add `static_j` to `SanafeEnergyBreakdown`**
   (`records.py:SanafeEnergyBreakdown`). Keep it separate from the
   four dynamic buckets (`synapse_j`, `dendrite_j`, `soma_j`,
   `network_j`) so consumers can see the split and not be silently
   misled by a single inflated number.

4. **Accumulate in `_run_neural_stage`** (`runner.py`):
   ```python
   n_live_cores = sum(1 for c in hcm.cores if _used_neurons(c) > 0)
   static_j = (preset.static_power_w_per_core
               * n_live_cores
               * seg_sim_time_s)
   ```
   Add to the segment's `SanafeEnergyBreakdown` and roll up to the run
   level alongside the dynamic terms.

5. **Surface in the GUI** (`sanafe-tab.js`, `snapshot/builders.py`): add
   a "Static" bar to the energy decomposition stacked chart so users
   see the split. Don't silently fold into `total_j`.

### Public sticker values to seed

- Loihi 1: ~30-50 mW idle per chip across ~128 cores → ~0.3-0.4 mW per
  core. Davies 2018.
- TrueNorth: ~63 mW idle per chip across 4096 cores → ~15 µW per core.
  Akopyan 2015.

These are rough; cite the source in `presets.py` next to the value.

## Caveats to put in the docstring

1. **Model-bias-consistent, not measured.** Static × `sim_time` inherits
   the same arch-YAML calibration bias as the dynamic energy. They are
   internally consistent (both built from the same per-event latency
   model) but neither is independently grounded.
2. **`sim_time` is a parallel-everything timing lower bound.** Real chips
   with shared buses / serialization points run slower; Loihi's actual
   scheduler doesn't perfectly parallelize. So static × sim_time is also
   a lower bound on real static energy.
3. **Custom hardware proposals** (mimarsinan's whole point) need
   user-supplied `static_power_w_per_core` — there is no defensible
   default for a chip that doesn't exist yet. Document that the
   user-supplied number is what drives the static report.

## Cross-references

- Memory: `~/.claude/projects/-home-yigit-repos-research-stuff-mimarsinan/memory/sanafe_runner_timing_window.md`
- Presets: `src/mimarsinan/chip_simulation/sanafe/presets.py`
- Records: `src/mimarsinan/chip_simulation/sanafe/records.py`
- Runner aggregation: `src/mimarsinan/chip_simulation/sanafe/runner.py` (~L141, `_run_neural_stage`)
- SANA-FE source proofs: `sana_fe/src/chip.cpp:611`, `sana_fe/src/schedule.cpp:100`, `sana_fe/src/pipeline.hpp:150`
