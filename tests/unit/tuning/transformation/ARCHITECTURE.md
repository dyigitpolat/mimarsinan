# tests/unit/tuning/transformation/ -- Controller-validation suite (report Part IV)

Pure-Python, GPU-free validation of the tuning controller against analytic mock
surfaces — the report's IV.2 "crux": prove one `RateScheduler` is stable across
heterogeneous axis profiles before any expensive real run.

## Contents

| File | Purpose |
|------|---------|
| `mock_axis_zoo.py` | Analytic `attempt(target) -> committed` surfaces: `smooth_monotone`, `cliff`, `plateau_then_drop`, `recovery_limited`, `adversarial_timing`, `non_monotone`. |
| `test_controller_invariants.py` | RateScheduler over each archetype: lands within the feasible edge (I1), monotone progress (I2), bounded probes, partial-result on recovery-limited, `last_successful_step` parity. |
| `test_multi_axis.py` | `MultiAxisDriver`/`VectorRateScheduler` (P7): each axis reaches its edge, monotone committed vector, value-domain guard rejects non-value-domain axes. |
| `test_metamorphic.py` | IV.5 relations without ground truth: reducing ε weakly increases committed for a cliff; one-shot never commits past a cliff; larger N shrinks paired SE; tightening k never turns reject→accept; same config → identical trajectory. |
| `test_determinism.py` | IV.8: `rng_snapshot`/`deterministic_rng` restore RNG around a probe; a fixed-seed scripted tuner replays an identical `DecisionTrace` (I6). |
| `test_chaos_rollback.py` | IV.6: forced recovery divergence → exact bitwise restore; `CheckpointGuard.bracket()` restore; non-monotone surface still terminates (dense-grid safe mode is the future V9 handling). |
| `test_axis_conformance.py` | IV.4: spec-14.1 over the ManagerRate family + BlendAxis — set_rate(0)==identity/reversible, idempotent attach, stable descriptor, state round-trip, set_decision_seed no-op. |
| `test_perf_gates.py` | IV.7 (deterministic, CUDA-free): cliff probe-count is log-bounded (no linear rate scan); smooth axis commits in one greedy probe; `CheckpointGuard` `scope='tunable'` skips the frozen backbone (the W6 VRAM lever) while `scope='full'` round-trips bitwise. |

## Remaining (report Part IV)

The CUDA wall-clock / peak-VRAM budgets on the ViT integration probe (the
non-deterministic half of IV.7) are not yet a gate. The paired-gate Monte-Carlo
(IV.3) lives at `tests/unit/tuning/test_paired_sensor_calibration.py`.
