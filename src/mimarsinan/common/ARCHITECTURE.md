# common/ — Shared leaf utilities: env-var SSOT, file I/O, compiler discovery, reporting, and diagnostics.

`common/` is the near-leaf utility layer that every other mimarsinan module may
depend on. It centralizes the `MIMARSINAN_*` environment-variable contract
(`env.py`), the single sanctioned log-and-degrade seam (`best_effort`), the
`Reporter` protocol used by the deployment pipeline for metrics, and the file
writers that emit simulator inputs and chip code for nevresim runs. New
contributors should treat it as the home for small, dependency-free mechanisms
shared across the pipeline.

## Key files
| File | Purpose |
|---|---|
| `best_effort.py` | `best_effort` context manager: the single sanctioned log-and-continue seam for non-critical telemetry/rendering side work |
| `build_utils.py` | `find_cpp20_compiler`: probe-compiles to discover a working C++20 compiler (clang++ 17-20 with libc++, g++ 11-14, clang++ with libstdc++, plain g++), plus legacy wrappers |
| `dependency_manifest.py` | Reads what `pyproject.toml` declares (base and extra requirements, a distribution's specifier, its `==` pin, its direct reference) so an install pin has one place to live; the SANA-FE pin-drift guard and the dependency-declaration tests read it instead of a script literal |
| `diagnostics.py` | CUDA debugging/profiling helpers: `enable_cuda_debug`, `describe_tensor`, `phase_profiler` (time/RSS/CUDA-peak), `cuda_guard` (synchronize-bracketed blocks) |
| `env.py` | Single source of truth for `MIMARSINAN_*` (and `IMAGENET_ROOT`) environment variables; one call-time accessor per flag so tests can monkeypatch |
| `file_utils.py` | Writers for pipeline artifacts: directory prep, per-sample simulator input files (scalar and spike-train), and chip weights/code emission (`save_weights_and_chip_code`) |
| `instance_memo.py` | `InstanceMemo`: per-instance derived-value memo for unhashable hosts (id-keyed, finalizer-evicted); caches precomputed plans keyed on IR/segment objects |
| `measurement/` | Baseline-agreement SSOT (`baseline.py`): `MetricTolerance` is the ONE definition of "matches" for a measured number against a recorded one (absolute + symmetric relative parts, plus `for_sampled_proportion`, which derives the bound from the record and the sample count it is judged over), and `assert_matches_recorded_baseline` raises `BaselineMismatchError` naming both numbers, the gap, the bound and the likely causes — preprocessing mismatch first, the defect it was built for. Workload-free: the expectation comes from the caller or a recorded artifact |
| `lifecycle/` | Process-lifetime SSOT — the one answer to "nothing may outlive the run that created it". `process_tree.py`: `/proc` enumeration (`iter_descendants`, `process_identity`/`process_is_alive`, pid-reuse-proof) and the classified sweep `reap_processes(members, …)` — SIGTERM/grace/SIGKILL the holders, but *release* resource trackers so their EOF-triggered unlink pass actually runs and named semaphores are not orphaned — with `reap_descendants` as its live-tree-walk membership rule. `exit_contract.py`: `install_exit_contract` routes SIGTERM/SIGINT/SIGHUP/SIGQUIT into the single `exit_process(code, teardown=…)` epilogue, which reaps BEFORE any telemetry teardown and then `os._exit`s, AND stakes out the same teardown out-of-process for the uncatchable kills. `owner.py`: `owner_token`/`owner_bound_initializer` — the die-with-owner watch a spawned worker installs — plus `stamp_cohort`/`cohort_members`, the inherited-environment cohort mark (`MIMARSINAN_COHORT_TOKEN`) that identifies a run's processes from `/proc` alone, after the kernel has re-parented them and including the forkservers and resource trackers CPython spawns itself, which never run any initializer. `cohort_reaper.py`: `spawn_cohort_reaper`/`reap_when_owner_dies` — a detached, stdio-free child that polls the owner and sweeps everything carrying its mark once it is gone, the only defence that survives SIGKILL/OOM-kill of the owner; it is still a child, so the ordinary reap collects it on every clean path. `child_launcher.py`: `run_child` — launches a child in its own session, captures on reader threads, and gates completion on the child's `wait()` instead of pipe EOF held by its cohort |
| `presentation.py` | Display/serialization helpers: `safe_float` (safe conversion for plots and labels) and `layer_key_from_node_name` (grouping key that collapses per-position/per-tile/psum core names into one layer stack) |
| `reporter.py` | `Reporter` protocol for pipeline metric reporting and structured events (`report`/`console_log`/`event`), `DefaultReporter` (throttled console output; events are a GUI concern), and `emit_reporter_event` (event emission tolerant of pre-event reporter implementations) |
| `workload_profile.py` | Workload-profile injection contracts: `DataWorkloadProfile` / `ModelWorkloadProfile` (what a dataset / model architecture may lawfully register), `CalibrationSetPolicy`, `fold_workload_profiles` (explicit config > model > data > frozen default), and the `ResolvedWorkloadProfile` carried by `DeploymentPlan.workload`. A builder registers a `pretrained_weight_sets` tuple (from `pretrained/`); the scalar `pretrained_weight_source` is retired |
| `pretrained/` | The pretrained-weight-set registration contract: `PretrainedWeightSet` (id/label/task/dataset/input_shape/num_classes/source + expected accuracy, licence, preprocessing, `adapts_*`, `model_config_requires`) and the PURE regime predicates over the injected records (`registered_weight_sets`, `applicable_weight_sets`, `weight_set_mismatch`, `legal_weight_set_ids` / `legal_preload_values`, `select_weight_set` / `selected_source` / `derived_weight_set_id`, `preload_unavailable_reason`, `preload_regime_error`, and `unusable_baseline_reason` — why a record's accuracy is NOT an expectation for a metric measured under this config, which is what disarms the preload baseline gate on adapted deploys). `None` = builder not consulted; `()` = registers nothing (disables the regime) |

## Dependencies
- `chip_simulation` — `file_utils.save_weights_and_chip_code` lazily imports `nevresim.connectivity.default_nevresim_connectivity_mode` to pick the chip-code emission mode.
- `code_generation` — the same function lazily imports `mapping_spans_export` (`chip_config_header`, `write_mapping_spans_file`) for runtime-connectivity chip emission.

Both imports are deferred inside the function body; module import time keeps
`common/` a leaf.

## Dependents
- `pipelining` — `best_effort`, `phase_profiler`/`cuda_guard`, env flags (vram probe, resource debug, NF-SCM parity debug), `prepare_containing_directory`, `DefaultReporter`, and `measurement` + `pretrained.unusable_baseline_reason` in `weight_preloading_step` (the preload's recorded-baseline gate).
- `gui` — `best_effort`, env accessors (`runs_root`, `templates_dir`, `gui_no_browser`), `layer_key_from_node_name`.
- `data_handling` — `best_effort`, FFCV/ImageNet/resource-debug env accessors, `DataWorkloadProfile` for the provider registration hook, and `lifecycle.owner` (every multi-worker `DataLoader` carries an owner-bound `worker_init_fn`).
- `chip_simulation` — `find_cpp20_compiler`, `file_utils` input/chip-code writers, `loihi_quiet`, `measurement.MetricTolerance` (the parity record comparison's closeness predicate).
- `visualization` — `layer_key_from_node_name`, `safe_float`.
- `search` — `best_effort`.
- `mapping` — `best_effort`, `cuda_debug_enabled`.
- `tuning` — `best_effort`.
- `model_training` — `vram_probe_enabled`.
- `models` — `cuda_debug_enabled`.
- `code_generation` — `file_utils` writers.

## Exported API
`__init__.py` re-exports:
- `prepare_containing_directory`, `input_to_file`, `save_inputs_to_files`, `save_weights_and_chip_code` — file I/O for pipeline artifacts.
- `find_cpp20_compiler` — C++20 compiler discovery.
- `Reporter`, `DefaultReporter` — metric-reporting protocol and default implementation.

Other symbols (`env` accessors, `best_effort`, diagnostics, `layer_key_from_node_name`, `safe_float`) are imported from their submodules directly.

`lifecycle/__init__.py` re-exports nothing: the frontends (`run.py`, `src/main.py`)
and the launch scripts import `exit_contract`, `owner`, `process_tree`,
`cohort_reaper` and `child_launcher` by name, so which lifecycle concern a call
site depends on stays visible at the import line. `install_exit_contract` is the
one call site for the whole contract — it installs the signal handlers, stamps the
cohort and starts the reaper together, so no frontend can arm half of it.
