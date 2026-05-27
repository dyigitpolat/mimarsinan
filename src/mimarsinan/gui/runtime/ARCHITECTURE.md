# gui/runtime/ — Process monitoring runtime

File-based IPC, in-memory collector, and subprocess orchestration for the GUI.

## Layout

| Module / package | Purpose |
|------------------|---------|
| `collector/` | `DataCollector` mixins: steps, metrics, console, read API, WebSocket |
| `persistence/` | `paths`, `store`, `load`, `resource_paths` under `_GUI_STATE/` |
| `process_spawn.py` | `ManagedRun`, `spawn_run`, console pipe reader |
| `process_monitor.py` | Orphan recovery, list/detail APIs, `kill_run` |
| `process_manager.py` | Thin facade over spawn + monitor |
| `active_run_tailers.py` | Poll `live_metrics.jsonl` and `steps.json` |
| `active_run_hub.py` | Ref-counted WS subscription hub |
| `snapshot_executor.py` | Post-step persistence worker queue |
| `run_cache_seed.py` | Edit & continue cache copy helpers |
| `composite_reporter.py` | Multi-reporter dispatch |

## Exported API (`__init__.py`)

`DataCollector`, `ActiveRunHub`, `ProcessManager`, `ManagedRun`

## Dependents

- `gui.handle`, `gui.start`, `gui.server`, `gui.runs`, `run.py --headless`
