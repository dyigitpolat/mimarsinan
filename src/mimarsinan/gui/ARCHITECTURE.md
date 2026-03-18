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
| `composite_reporter.py` | `CompositeReporter` | Dispatches to multiple reporters (e.g. default + GUI) |
| `server.py` | `start_server`, `create_app` | FastAPI + Uvicorn server in a daemon thread; optional `run_config_fn` for POST `/api/run` |
| `snapshot/` | `build_step_snapshot`, `snapshot_model`, `snapshot_pruning_layers`, `snapshot_ir_graph`, `snapshot_hard_core_mapping`, `snapshot_search_result`, `snapshot_adaptation_manager` | Package: `helpers.py` (numeric/dict helpers, cache key map), `builders.py` (all snapshot builders); pure functions extracting JSON-safe snapshots; step-specific tabs and new/edited kinds. **Pruning**: `snapshot_pruning_layers(model)` extracts per-layer weight heatmaps with pruning masks (red lines) for the Pruning Adaptation step; `build_step_snapshot` adds `pruning_layers` when step is Pruning Adaptation. Hardware snapshot: per-placement `utilization_frac`, `constituent_count` per core, and when a core is fused, `fused_axon_boundaries` and `fused_component_count` for GUI boundaries and badges. |
| `persistence.py` | `load_persisted_steps`, `save_step_to_persisted`, `save_run_info`, `update_run_status`, `load_run_info`, `append_live_metric`, `load_live_metrics` | Load/save step state to `_GUI_STATE/steps.json` for backfill; run lifecycle metadata in `_GUI_STATE/run_info.json`; streaming metrics in `_GUI_STATE/live_metrics.jsonl` |
| `process_manager.py` | `ProcessManager`, `ManagedRun` | Spawns headless pipeline processes, tracks them via filesystem polling (run_info.json, steps.json, live_metrics.jsonl), provides status/metrics/step detail APIs, kill with SIGTERM→SIGKILL escalation |
| `runs.py` | `list_runs`, `get_run_config`, `get_run_pipeline`, `get_run_step_detail` | Discover and load historical pipeline runs from the generated files directory |
| `templates.py` | `list_templates`, `get_template`, `save_template`, `delete_template` | CRUD for deployment configuration templates saved as JSON files |
| `wizard_config_builder.py` | `build_deployment_config_from_state` | Builds complete deployment config from wizard UI state, applying defaults and presets |
| `heatmap_renderer.py` | `render_heatmap_png_data_uri` | Renders weight matrices as PNG data URIs for GUI; no raw matrices sent to frontend |

### Step Status Persistence Contract

Steps in `_GUI_STATE/steps.json` follow a strict status lifecycle:

1. **`on_step_start`** → writes `status: "running"`, `start_time`, `end_time: null`
2. **`on_step_end`** → writes `status: "completed"`, `end_time`, `target_metric`, full snapshot
3. **Failure** → writes `status: "failed"` (or inferred: `running` + dead process → `failed`)
4. **Fallback rule**: if `status` key is missing but `end_time` is present, status is inferred as `"completed"` (backwards compatibility with older persisted data)

This contract is enforced in `__init__.py` (`GUIHandle.on_step_end`) and consumed by `process_manager.py` (`get_run_detail`, `get_run_step_detail`, `list_active`) and `runs.py` (`get_run_pipeline`, `get_run_step_detail`).

### Process-Based Concurrent Runs

Pipeline runs are executed as **isolated OS processes** via `ProcessManager`:
- `spawn_run()` creates a unique timestamped working directory and launches `run.py --headless`
- IPC is file-based: `run_info.json` (lifecycle), `steps.json` (step state), `live_metrics.jsonl` (streaming metrics)
- On server restart, `_recover_orphaned_runs()` scans for existing run directories with active PIDs
- `kill_run()` sends SIGTERM → waits 3s → escalates to SIGKILL

### Welcome Page (`static/welcome.html`, `static/js/welcome.js`)

Landing page with active runs monitoring (mini pipeline bars, Plotly sparklines, estimated completion), searchable past runs grid, and template management with inline rename.

### Wizard and config APIs (when started with `--ui`)

- `GET /api/data_providers` — list registered data providers (id, label).
- `GET /api/model_types` — list model types (id, label, category).
- `GET /api/model_config_schema/{model_type}` — config fields for dynamic form generation.
- `POST /api/run` — body = full deployment config JSON; creates pipeline, attaches collector, runs in background thread; returns 202.
- `POST /api/pipeline_steps` — body = same deployment config shape as `/api/run`; returns `{"steps": ["Step Name", ...]}` for the pipeline that would be built. Used by the wizard to show a live pipeline preview without running the pipeline.
- `GET /wizard` — serves the deployment configurator wizard (`static/wizard.html`).

### Frontend (`static/`)

Single-page application using ES modules and Plotly.js. See `static/js/` for
modular visualization components (overview, model, IR graph, hardware, search,
scales, pruning tabs). **Pruning tab**: shown for the Pruning Adaptation step; lists layers with per-layer weight heatmaps (red lines for pruned rows/columns, same convention as IR Graph and Hardware) and a layer browser (list + detail panel). **Hardware tab**: shows soft-core and fused hardware-core boundaries
on miniview and detail heatmaps; "Constituents (N)" table with ID, dimensions,
utilization per constituent; clicking a constituent or heatmap region opens
soft-core detail with "Located in" (segment, hard core, region) for two-way
traceability. Snapshot provides per-placement utilization and fused boundaries. The **wizard** (`wizard.html`, `wizard.css`, `js/wizard.js`) is the
deployment configurator: it loads data providers and model types from the API,
builds a config, and submits it via POST `/api/run`; RUN redirects to `/` (monitor).
**Weight Quantization** and **Activation Quantization** are locked (derived) from Float and Spiking Mode: no manual selection in regular deployment. Float ON locks Weight Quant to OFF; Float OFF locks it to ON. Rate-coded or TTFS Quantized locks Activation Quant to ON; plain TTFS locks it to OFF. Rate-coded spiking mode thus forces activation quantization ON; the Cycles field is disabled for non-quantized TTFS (analytical TTFS does not use simulation steps). Target Tq is disabled when activation quantization is off. **Float weights** is a toggle in the **Hardware Configuration** panel (next to Weight Bits): when ON it disables the Weight Bits control and locks Weight Quantization to off in Deployment Mode; pipeline uses vanilla (float) deployment. **Pruning fraction** is a [0–1) range slider with value display; the 0.8–1.0 range
is styled in red and a feasibility warning is shown in that range.

**Pipeline steps bar**: A bar at the top of the wizard (below the header) shows the
ordered list of pipeline steps for the current configuration. It calls POST
`/api/pipeline_steps` with the current config (debounced, e.g. 250 ms) on load and
whenever the user changes options. Steps are rendered as horizontal chips; new steps
animate in (opacity + scale). On loading, the bar shows a subtle loading state; on
API error, the last known step list is kept or a short "Could not load steps" message
is shown.

**Monitor plots** (step-detail metrics tab, scales-tab adaptation, search-tab): legends
are placed outside the plot area to the right (`x: 1.02`, `margin.r: 100`). Accuracy
and Adaptation curves use a fixed vertical axis [0, 1]. A single data point is drawn
as a horizontal line from that point to the right edge. In the step-detail metrics tab,
architecture search metrics (names containing "search") are shown in separate plots
per metric so each keeps its own scale. Search history is rendered as one card and
plot per numeric metric (separate charts per objective).

## Dependencies

- **Internal**: `common.reporter` (Reporter protocol), `mapping.ir` (for snapshots), `mapping.spike_source_spans`, `gui.heatmap_renderer` (for snapshot heatmaps).
- **External**: `fastapi`, `uvicorn`, `websockets`, `matplotlib` (heatmap rendering).

## Dependents

- Entry point (`main.py`) calls `start_gui()` and wraps the pipeline reporter.
- `run.py --ui` starts the GUI server with `ProcessManager` and wizard at `/wizard`; POST `/api/run` spawns headless pipeline processes.
- `run.py --headless <config>` runs a pipeline with file-based monitoring (no GUI server); writes to `_GUI_STATE/`.

## Exported API (\_\_init\_\_.py)

`GUIHandle`, `start_gui`.

## Active Run API Endpoints (when `ProcessManager` available)

- `GET /api/active_runs` — summary of all tracked runs (status, progress, steps, target metrics)
- `GET /api/active_runs/{run_id}/pipeline` — detailed pipeline state for an active run
- `GET /api/active_runs/{run_id}/steps/{step_name}` — step detail with live metrics
- `DELETE /api/active_runs/{run_id}` — terminate a running process

## Historical Run API Endpoints

- `GET /api/runs` — list past runs (optionally with `?include_steps=true`)
- `GET /api/runs/{run_id}/config` — full deployment config of a past run
- `GET /api/runs/{run_id}/pipeline` — pipeline overview of a past run
- `GET /api/runs/{run_id}/steps/{step_name}` — step detail of a past run

## Template API Endpoints

- `GET /api/templates` — list saved templates
- `GET /api/templates/{id}` — get a template's config
- `POST /api/templates` — save a new template (body: `{name, config}`)
- `DELETE /api/templates/{id}` — delete a template
