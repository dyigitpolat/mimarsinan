# tests/unit/tuning/transformation/ -- Controller-validation suite (report Part IV)

Pure-Python, GPU-free validation of the tuning controller against analytic mock
surfaces — the report's IV.2 "crux": prove one `RateScheduler` is stable across
heterogeneous axis profiles before any expensive real run.

## Contents

| File | Purpose |
|------|---------|
| `mock_axis_zoo.py` | Analytic `attempt(target) -> committed` surfaces spanning the profile space: `smooth_monotone`, `cliff`, `plateau_then_drop`, `recovery_limited`, `adversarial_timing`, `non_monotone`. |
| `test_controller_invariants.py` | Runs `RateScheduler` over each archetype and asserts: lands within the feasible edge (never commits past it, I1), monotone committed progress (I2), bounded probe count, partial-result on recovery-limited, and `last_successful_step` parity. |

## Remaining (report Part IV, not yet built)

`test_sensor_montecarlo.py` (IV.3 — see `tests/unit/tuning/test_paired_sensor_calibration.py`
for the paired-gate calibration already landed), `test_axis_conformance.py`
(IV.4 — partial coverage in `tests/unit/tuning/test_adaptation_axis_conformance.py`),
`test_metamorphic.py` (IV.5), `test_chaos_rollback.py` (IV.6),
`test_perf_gates.py` (IV.7), `test_determinism.py` (IV.8).
