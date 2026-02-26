# gui/ -- Browser-Based Pipeline Monitor

A real-time browser-based dashboard that launches with every pipeline run,
providing live monitoring and post-run inspection.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `__init__.py` | `GUIHandle`, `start_gui` | Facade: creates collector, reporter, registers step hooks |
| `data_collector.py` | `DataCollector` | Thread-safe in-memory store; broadcasts updates via WebSocket |
| `reporter.py` | `GUIReporter` | Implements `Reporter` protocol; forwards metrics to `DataCollector` |
| `composite_reporter.py` | `CompositeReporter` | Dispatches to multiple reporters (WandB + GUI) |
| `server.py` | `start_server` | FastAPI + Uvicorn server in a daemon thread |
| `snapshot.py` | `build_step_snapshot` | Pure functions extracting JSON-safe snapshots from pipeline artifacts |

### Frontend (`static/`)

Single-page application using ES modules and Plotly.js. See `static/js/` for
modular visualization components (overview, model, IR graph, hardware, search,
scales tabs).

## Dependencies

- **Internal**: `common.wandb_utils` (`Reporter` protocol), `mapping.ir` (for snapshots), `mapping.spike_source_spans`.
- **External**: `fastapi`, `uvicorn`, `websockets`.

## Dependents

- Entry point (`main.py`) calls `start_gui()` and wraps the pipeline reporter.

## Exported API (\_\_init\_\_.py)

`GUIHandle`, `start_gui`.
