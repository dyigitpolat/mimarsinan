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
| `diagnostics.py` | CUDA debugging/profiling helpers: `enable_cuda_debug`, `describe_tensor`, `phase_profiler` (time/RSS/CUDA-peak), `cuda_guard` (synchronize-bracketed blocks) |
| `env.py` | Single source of truth for `MIMARSINAN_*` (and `IMAGENET_ROOT`) environment variables; one call-time accessor per flag so tests can monkeypatch |
| `file_utils.py` | Writers for pipeline artifacts: directory prep, per-sample simulator input files (scalar and spike-train), and chip weights/code emission (`save_weights_and_chip_code`) |
| `instance_memo.py` | `InstanceMemo`: per-instance derived-value memo for unhashable hosts (id-keyed, finalizer-evicted); caches precomputed plans keyed on IR/segment objects |
| `presentation.py` | Display/serialization helpers: `safe_float` (safe conversion for plots and labels) and `layer_key_from_node_name` (grouping key that collapses per-position/per-tile/psum core names into one layer stack) |
| `process_tree.py` | `/proc`-based process-tree enumeration (`iter_descendants`) and exit-path reaping (`reap_descendants`: TERM, grace, KILL) so runs exit promptly with no orphaned worker processes holding inherited pipes |
| `reporter.py` | `Reporter` protocol for pipeline metric reporting and structured events (`report`/`console_log`/`event`), `DefaultReporter` (throttled console output; events are a GUI concern), and `emit_reporter_event` (event emission tolerant of pre-event reporter implementations) |
| `workload_profile.py` | Workload-profile injection contracts: `DataWorkloadProfile` / `ModelWorkloadProfile` (what a dataset / model architecture may lawfully register), `CalibrationSetPolicy`, `fold_workload_profiles` (explicit config > model > data > frozen default), and the `ResolvedWorkloadProfile` carried by `DeploymentPlan.workload`. A builder registers a `pretrained_weight_sets` tuple (from `pretrained/`); the scalar `pretrained_weight_source` is retired |
| `pretrained/` | The pretrained-weight-set registration contract: `PretrainedWeightSet` (id/label/task/dataset/input_shape/num_classes/source + expected accuracy, licence, preprocessing, `adapts_*`, `model_config_requires`) and the PURE regime predicates over the injected records (`registered_weight_sets`, `applicable_weight_sets`, `weight_set_mismatch`, `legal_weight_set_ids` / `legal_preload_values`, `select_weight_set` / `selected_source` / `derived_weight_set_id`, `preload_unavailable_reason`, `preload_regime_error`). `None` = builder not consulted; `()` = registers nothing (disables the regime) |

## Dependencies
- `chip_simulation` — `file_utils.save_weights_and_chip_code` lazily imports `nevresim.connectivity.default_nevresim_connectivity_mode` to pick the chip-code emission mode.
- `code_generation` — the same function lazily imports `mapping_spans_export` (`chip_config_header`, `write_mapping_spans_file`) for runtime-connectivity chip emission.

Both imports are deferred inside the function body; module import time keeps
`common/` a leaf.

## Dependents
- `pipelining` — `best_effort`, `phase_profiler`/`cuda_guard`, env flags (vram probe, resource debug, NF-SCM parity debug), `prepare_containing_directory`, `DefaultReporter`.
- `gui` — `best_effort`, env accessors (`runs_root`, `templates_dir`, `gui_no_browser`), `layer_key_from_node_name`.
- `data_handling` — `best_effort`, FFCV/ImageNet/resource-debug env accessors, `DataWorkloadProfile` for the provider registration hook.
- `chip_simulation` — `find_cpp20_compiler`, `file_utils` input/chip-code writers, `loihi_quiet`.
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
