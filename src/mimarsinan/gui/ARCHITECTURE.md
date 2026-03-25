# gui/ -- Browser-Based Pipeline Monitor

A real-time browser-based dashboard that launches with every pipeline run,
providing live monitoring and post-run inspection. When started with `python run.py --ui`,
the server also serves the **configuration wizard** and exposes APIs for data providers,
model types, and config schema; POST `/api/run` starts a pipeline from the wizard.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `__init__.py` | `GUIHandle`, `start_gui`, `_TeeStream` | Facade: creates collector, reporter; installs `_TeeStream` on `sys.stdout`/`sys.stderr` to capture console output into DataCollector; optional `start_step` backfills skipped steps from cache for browsing |
| `data_collector.py` | `DataCollector`, `MetricEvent`, `ConsoleLogEntry`, `StepRecord` | Thread-safe in-memory store; broadcasts metric, step lifecycle, and console log events via WebSocket |
| `reporter.py` | `GUIReporter` | Implements `Reporter` protocol; forwards metrics to `DataCollector` |
| `composite_reporter.py` | `CompositeReporter` | Dispatches to multiple reporters (e.g. default + GUI) |
| `server.py` | `start_server`, `create_app`, `_gui_entry_url`, `_schedule_open_browser` | FastAPI + Uvicorn server in a daemon thread; optional `run_config_fn` for POST `/api/run`. On startup prints a single welcome URL and opens it via `webbrowser.open` (after a short delay); set `MIMARSINAN_GUI_NO_BROWSER=1` to skip opening a browser. |
| `snapshot/` | `build_step_snapshot`, `snapshot_model`, `snapshot_pruning_layers`, `snapshot_ir_graph`, `snapshot_hard_core_mapping`, `snapshot_search_result`, `snapshot_adaptation_manager` | Package: `helpers.py` (numeric/dict helpers, cache key map), `builders.py` (all snapshot builders); pure functions extracting JSON-safe snapshots; step-specific tabs and new/edited kinds. **Pruning**: `snapshot_pruning_layers(model)` extracts per-layer weight heatmaps with pruning masks (red lines) for the Pruning Adaptation step; `build_step_snapshot` adds `pruning_layers` when step is Pruning Adaptation. Hardware snapshot: per-placement `utilization_frac`, `constituent_count` per core, and when a core is fused, `fused_axon_boundaries` and `fused_component_count` for GUI boundaries and badges. |
| `persistence.py` | `load_persisted_steps`, `save_step_to_persisted`, `write_persisted_steps_replace`, `save_run_info`, `update_run_status`, `load_run_info`, `append_live_metric`, `load_live_metrics`, `append_console_log`, `load_console_logs` | Load/save step state to `_GUI_STATE/steps.json` for backfill; `write_persisted_steps_replace` atomically replaces the whole steps map (used after backfilling skipped steps so the monitor sees completed steps on disk); run lifecycle metadata in `_GUI_STATE/run_info.json`; streaming metrics in `_GUI_STATE/live_metrics.jsonl`; stdout/stderr log lines in `_GUI_STATE/console.jsonl` |
| `process_manager.py` | `ProcessManager`, `ManagedRun`, `_start_console_reader` | Spawns headless pipeline processes with `stdout=PIPE, stderr=PIPE`; background reader threads drain both pipes and write tagged lines to `_GUI_STATE/console.jsonl`; tracks runs via filesystem polling, provides status/metrics/step detail APIs, kill with SIGTERM→SIGKILL escalation |
| `run_cache_seed.py` | `copy_pipeline_cache_from_previous_run`, `copy_steps_json_from_previous_run` | Edit & continue: copies pipeline cache files and optionally `_GUI_STATE/steps.json` from the prior run so `run_from` has cache entries and backfill can restore full step snapshots for the monitor |
| `runs.py` | `list_runs`, `get_run_config`, `get_run_pipeline`, `get_run_step_detail`, `get_run_console_logs` | Discover and load historical pipeline runs from the generated files directory; `get_run_console_logs` reads `console.jsonl` for the console tab |
| `templates.py` | `list_templates`, `get_template`, `save_template`, `delete_template`, `name_and_deployment_from_post_body` | CRUD for deployment configuration templates saved as JSON files. Files are a **flat deployment dict** (same shape as `get_run_config`). `save_template` sets `experiment_name` on the stored copy to the given template display name (so lists and wizard load show the template label, not the originating run name). POST `/api/templates` accepts `{name, config}` or a flat deployment body. |
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
- If the deployment dict includes `_continue_from_run_id` (wizard edit & continue), that key is **not** written to `_RUN_CONFIG/config.json`; `copy_pipeline_cache_from_previous_run` runs first so the new process loads the previous run’s pipeline cache from disk
- IPC is file-based: `run_info.json` (lifecycle), `steps.json` (step state), `live_metrics.jsonl` (streaming metrics)
- On server restart, `_recover_orphaned_runs()` scans for existing run directories with active PIDs
- `kill_run()` sends SIGTERM → waits 3s → escalates to SIGKILL

### Welcome Page (`static/welcome.html`, `static/js/welcome.js`)

Landing page with active runs monitoring (mini pipeline bars, Plotly sparklines, estimated completion), searchable past runs grid, and template management with inline rename.

Active-run cards use **incremental DOM updates**: on each poll only changed fields (status, elapsed, progress, mini-step classes, metrics, ETA) are patched in place. New cards get `w-active-card--enter` (which triggers the `fadeUp` entrance animation); the class is removed on `animationend` so polls never retrigger it. Sparklines are redrawn only when the metric series changes (length + last value fingerprint).

### Wizard and config APIs (when started with `--ui`)

- `GET /api/data_providers` — list registered data providers (id, label).
- `GET /api/model_types` — list model types (id, label, category).
- `GET /api/model_config_schema/{model_type}` — config fields for dynamic form generation.
- `POST /api/run` — body = full deployment config JSON; creates pipeline, attaches collector, runs in background thread; returns 202.
- `POST /api/hw_config_verify` — verifies hardware core config; returns `{feasible, errors, field_errors, packing, stats}`. `stats` is a `LayoutVerificationStats.to_dict()` augmented with `host_side_segment_count` and `layout_preview`; together they provide per-core waste/utilization metrics, layout-derived neural-segment summary (`neural_segment_count`, `segment_latency_min/median/max` where latency means latency groups per segment), host-side segment count (shown as sync barriers), a compact latency-group miniview, and coalescing/splitting counts. When `allow_scheduling=True` and single-pass packing fails, `schedule_info` is included with `num_passes` and `max_cores_per_pass`.
- `POST /api/pipeline_steps` — body = same deployment config shape as `/api/run`; returns `{"steps": ["Step Name", ...], "semantic_groups": ["group_id", ...]}` for the pipeline that would be built. Used by the wizard to show a live pipeline preview without running the pipeline; `semantic_groups` (same length as `steps`) drives per-step colour coding.
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
builds a config, and submits it via POST `/api/run`; RUN redirects to `/monitor?run_id=...` when a run id is returned, or `/` otherwise.
**Mapping stats panel**: after a successful hardware verification, the wizard
renders a compact stats panel (`#hwStatsPanel`) below the validation banner
showing overview cards (cores used, softcores, neural segments, sync barriers),
a compact layout-only miniview (`layout_preview.flow`) with input/host/latency-group/output
stages, health-bar style percentage bars (wasted axons/neurons, param utilization),
per-core min/avg/max breakdowns, and detail rows for latency-groups-per-segment plus
coalescing/splitting summaries. The data comes from the
`"stats"` key returned by `/api/hw_config_verify`. On failure or re-validation,
the panel is hidden so stale numbers are never shown.
**Scheduled Mapping** toggle (`#scheduledMappingToggle`, in **Deployment Mode**) enables hardware core reuse across
sequential passes. When ON, `allow_scheduling` is set in `deployment_parameters` and the
auto-suggest endpoint uses `suggest_hardware_config_scheduled` (exploring the core-count ↔
pass-count tradeoff). The verify endpoint reports `schedule_info` with estimated pass count.

**Weight Quantization** and **Activation Quantization** are locked (derived) from Float and Spiking Mode: no manual selection in regular deployment. Float ON locks Weight Quant to OFF; Float OFF locks it to ON. Rate-coded or TTFS Quantized locks Activation Quant to ON; plain TTFS locks it to OFF. Rate-coded spiking mode thus forces activation quantization ON; the Cycles field is disabled for non-quantized TTFS (analytical TTFS does not use simulation steps). Target Tq is disabled when activation quantization is off. **Float weights** is a toggle in the **Hardware Configuration** panel (next to Weight Bits): when ON it disables the Weight Bits control and locks Weight Quantization to off in Deployment Mode; pipeline uses vanilla (float) deployment. **Pruning fraction** is a [0–1) range slider with value display; the 0.8–1.0 range
is styled in red and a feasibility warning is shown in that range.

**Pipeline steps bar**: Both the wizard header and the monitor's top section render
pipeline steps as a column-bar layout (`static/pipeline-step-bar.css`). Each step is a
vertical column with a tall shiny bar (coloured by semantic group) and the step name as
a diagonal label beneath it (`transform: rotate(45deg)`), mirroring the diagonal tick
labels in the Plotly target-metric chart. Monitor states: `running` → glowing bar +
bold white label; `completed` → dim group colour + grey label; `pending` → near-off +
grey label. Wizard preview uses `psb-list--preview` modifier: all bars shown at
uniform preview intensity, no glow. Semantic groups flow from
`get_pipeline_semantic_group_by_step_name` (backend) → API → `data-group` attribute →
CSS. The wizard bar calls POST `/api/pipeline_steps` (debounced, 250 ms) on load and
on config change; on error the last known list is kept.

**Edit & continue** (`/wizard?run_id=...` from Welcome): `window.__isEditContinueMode`
is set and `document.body` gets class `edit-continue-mode` so the preview bar is
interactive and the hint label is shown. Click handling is delegated on
`#pipelineStepsList` via `data-ec-start-step` (URL-encoded step name); inline
`onclick` with quoted names is not used because it breaks HTML attributes for
names containing spaces. Clicking a step sets `window.__wizardStartStep` (toggle
off by clicking again); the chosen step is passed in the deployment config as
`start_step` for `Pipeline.run_from`.

*New run cache seeding*: `buildConfig()` sets `_continue_from_run_id` to the
Welcome run id (`window.__editContinueSourceRunId`). `ProcessManager.spawn_run`
strips it from the saved `_RUN_CONFIG/config.json` and calls
`copy_pipeline_cache_from_previous_run` and `copy_steps_json_from_previous_run`
so the new working directory has the prior run’s pipeline cache **and** a seed
`steps.json` before `run.py --headless` starts. After `_backfill_skipped_steps`
runs, `_persist_skipped_steps_to_steps_json` replaces `steps.json` with only
the skipped (completed-from-cache) steps so the monitor/APIs match execution —
`on_step_end` only persists steps that actually run in-process, so this extra
write is required for skipped steps.

*Restart-step default*: `init()` concurrently fetches `GET /api/runs/{run_id}/config`
(the deployment JSON) and `GET /api/runs/{run_id}/pipeline`; after
`loadStateFromConfig` the completed step names are stored in `_ecPrevCompleted`
(a persistent `Set`, not cleared). On the first successful `POST /api/pipeline_steps`
response, `updatePipelineStepsBar` picks the first canonical step not in that
completed set as the default `__wizardStartStep` (one-shot via `_ecSuggestionDone`).
If the currently selected step is
absent from the new step list (e.g. after changing config), it is silently
cleared. `update()` is called after any automatic step change to keep the JSON
preview in sync.

*Hardware Auto-suggest*: `loadStateFromConfig` always turns off `#hwAutoSuggestToggle`
and sets `_hwAutoMode = false` when `__isEditContinueMode` is true (for both
`user` and `auto` hardware modes), so core-type inputs are editable immediately
and `done()` never triggers an unwanted `autoFillHardware`. Toggles restored from
JSON use `forced=false` so they remain editable; spiking/hardware dependency
rules still apply via `applySpikingDeps` / `applyHwDeps` after load.

**Monitor connection dot** (`#conn-dot`, `static/js/main.js`): the status dot reflects WebSocket connectivity for in-process runs (no `run_id` param) and HTTP polling health (`pollOk`) for run-id-scoped active and historical runs where WebSocket is not used.

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
- `GET /api/active_runs/{run_id}/console?offset=N` — console log entries from `console.jsonl` (stdout+stderr)
- `DELETE /api/active_runs/{run_id}` — terminate a running process

## Historical Run API Endpoints

- `GET /api/runs` — list past runs (optionally with `?include_steps=true`)
- `GET /api/runs/{run_id}/config` — full deployment config of a past run
- `GET /api/runs/{run_id}/pipeline` — pipeline overview of a past run
- `GET /api/runs/{run_id}/steps/{step_name}` — step detail of a past run
- `GET /api/runs/{run_id}/console?offset=N` — console log entries for a past run

## In-Process API Endpoints

- `GET /api/console_logs?offset=N` — console log entries from in-process DataCollector (WebSocket also pushes `console_log` events)

## Template API Endpoints

- `GET /api/templates` — list saved templates
- `GET /api/templates/{id}` — get a template's config
- `POST /api/templates` — save a new template (body: `{name, config}`)
- `DELETE /api/templates/{id}` — delete a template
