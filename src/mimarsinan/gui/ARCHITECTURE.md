# gui/ -- Browser-Based Pipeline Monitor

A real-time browser-based dashboard that launches with every pipeline run,
providing live monitoring and post-run inspection. When started with `python run.py --ui`,
the server also serves the **configuration wizard** and exposes APIs for data providers,
model types, and config schema; POST `/api/run` starts a pipeline from the wizard.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `__init__.py` | `GUIHandle`, `start_gui` | Facade: creates collector, reporter; optional `start_step` backfills skipped steps from cache for browsing |
| `data_collector.py` | `DataCollector` | Thread-safe in-memory store; broadcasts updates via WebSocket |
| `reporter.py` | `GUIReporter` | Implements `Reporter` protocol; forwards metrics to `DataCollector` |
| `composite_reporter.py` | `CompositeReporter` | Dispatches to multiple reporters (WandB + GUI) |
| `server.py` | `start_server`, `create_app` | FastAPI + Uvicorn server in a daemon thread; optional `run_config_fn` for POST `/api/run` |
| `snapshot.py` | `build_step_snapshot` | Pure functions extracting JSON-safe snapshots; step-specific tabs and new/edited kinds |
| `persistence.py` | `load_persisted_steps`, `save_step_to_persisted` | Load/save step state to `_GUI_STATE/steps.json` for backfill |
| `heatmap_renderer.py` | `render_heatmap_png_data_uri` | Renders weight matrices as PNG data URIs for GUI; no raw matrices sent to frontend |

### Wizard and config APIs (when started with `--ui`)

- `GET /api/data_providers` — list registered data providers (id, label).
- `GET /api/model_types` — list model types (id, label, category).
- `GET /api/model_config_schema/{model_type}` — config fields for dynamic form generation.
- `POST /api/run` — body = full deployment config JSON; creates pipeline, attaches collector, runs in background thread; returns 202.
- `GET /wizard` — serves the deployment configurator wizard (`static/wizard.html`).

### Frontend (`static/`)

Single-page application using ES modules and Plotly.js. See `static/js/` for
modular visualization components (overview, model, IR graph, hardware, search,
scales tabs). The **wizard** (`wizard.html`, `wizard.css`, `js/wizard.js`) is the
deployment configurator: it loads data providers and model types from the API,
builds a config, and submits it via POST `/api/run`; RUN redirects to `/` (monitor).

## Dependencies

- **Internal**: `common.wandb_utils` (`Reporter` protocol), `mapping.ir` (for snapshots), `mapping.spike_source_spans`, `gui.heatmap_renderer` (for snapshot heatmaps).
- **External**: `fastapi`, `uvicorn`, `websockets`, `matplotlib` (heatmap rendering).

## Dependents

- Entry point (`main.py`) calls `start_gui()` and wraps the pipeline reporter.
- `run.py --ui` starts the GUI server with an empty collector and wizard at `/wizard`; POST `/api/run` runs the pipeline in a background thread.

## Exported API (\_\_init\_\_.py)

`GUIHandle`, `start_gui`.
