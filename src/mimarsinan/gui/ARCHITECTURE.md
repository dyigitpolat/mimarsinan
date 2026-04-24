# gui/ -- Browser-Based Pipeline Monitor

A real-time browser-based dashboard that launches with every pipeline run,
providing live monitoring and post-run inspection. When started with `python run.py --ui`,
the server also serves the **configuration wizard** and exposes APIs for data providers,
model types, and config schema; POST `/api/run` starts a pipeline from the wizard.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `__init__.py` | `GUIHandle`, `start_gui`, `_TeeStream` | Facade: creates collector, reporter; installs `_TeeStream` on `sys.stdout`/`sys.stderr` to capture console output into DataCollector; optional `start_step` backfills skipped steps from cache for browsing. `GUIHandle` owns a `SnapshotExecutor` that offloads post-step bookkeeping (snapshot persistence, resource materialisation) off the pipeline thread; `on_step_end` calls `target_metric` and `build_step_snapshot` synchronously then hands the rest (`step_completed`, `save_step_to_persisted`, `save_resource_to_disk`) to that executor. `GUIHandle.wait_snapshots_idle` / `shutdown` drain it before process exit. |
| `data_collector.py` | `DataCollector`, `MetricEvent`, `ConsoleLogEntry`, `StepRecord` | Thread-safe in-memory store; broadcasts metric, step lifecycle, console log, and `pipeline_overview` events via WebSocket. Maintains a `snapshot_version` per step; `get_step_detail(name, since_seq=N)` returns a weak HTTP ETag (`W/"{step}-{status}-{version}"`), metrics filtered to `seq > N`, and a `latest_metric_seq` cursor so frontends can poll incrementally. |
| `resources.py` | `ResourceDescriptor`, `ResourceStore` | Step-scoped, thread-safe cache for *lazy* resources (heatmap PNGs, connectivity JSON). `ResourceDescriptor` bundles `(kind, rid, producer, media_type)`; the producer is a zero-arg callable invoked on first fetch and memoised per-step. `clear_step` evicts and bumps a per-step version counter (called on `step_started` to invalidate stale URLs). |
| `snapshot_executor.py` | `SnapshotExecutor` | Single-worker FIFO queue used by `GUIHandle` to decouple heavy post-step I/O (steps.json rewrite, on-disk resource persistence) from the pipeline thread. Exceptions in jobs are logged but never crash the worker. |
| `active_run_stream.py` | `ActiveRunHub` | Ref-counted per-run file tailers that push `live_metrics.jsonl` lines and `steps.json` mtime changes over the `/ws/active_runs/{run_id}` WebSocket to the parent GUI. Replaces the 3 s poll previously used for subprocess-spawned active runs so charts update at ~20 Hz. Stops the tailer threads when the last subscriber disconnects. |
| `reporter.py` | `GUIReporter` | Implements `Reporter` protocol; forwards metrics to `DataCollector` |
| `composite_reporter.py` | `CompositeReporter` | Dispatches to multiple reporters (e.g. default + GUI) |
| `server.py` | `start_server`, `create_app`, `_gui_entry_url`, `_schedule_open_browser` | FastAPI + Uvicorn server in a daemon thread; optional `run_config_fn` for POST `/api/run`. On startup prints a single welcome URL and opens it via `webbrowser.open` (after a short delay); set `MIMARSINAN_GUI_NO_BROWSER=1` to skip opening a browser. |
| `snapshot/` | `build_step_snapshot`, `snapshot_model`, `snapshot_pruning_layers`, `snapshot_ir_graph`, `snapshot_hard_core_mapping`, `snapshot_search_result`, `snapshot_adaptation_manager` | Package: `helpers.py` (numeric/dict helpers, cache key map), `builders.py` (all snapshot builders); builders now return `(summary_dict, list[ResourceDescriptor])`: heavy artefacts (per-layer / per-core / pre-pruning heatmap PNGs, per-stage connectivity span arrays) are **not** embedded in the summary. Each emits `has_<thing>: true` plus a `<thing>_resource: {kind, rid}` hint, and the descriptor list carries a zero-arg producer closure that materialises the bytes / JSON on demand. Producers deep-copy mutable NumPy matrices at creation time so deferred materialisation is immune to later pipeline mutations. `build_step_snapshot` aggregates descriptors across sub-builders and returns `(snapshot, snapshot_key_kinds, resource_descriptors)`. **Pruning**: `snapshot_pruning_layers(model)` extracts per-layer weight heatmaps with pruning masks (red lines) for the Pruning Adaptation step; `build_step_snapshot` adds `pruning_layers` when step is Pruning Adaptation. Hardware snapshot: per-placement `utilization_frac`, `constituent_count` per core, and when a core is fused, `fused_axon_boundaries` and `fused_component_count` for GUI boundaries and badges. |
| `persistence.py` | `load_persisted_steps`, `save_step_to_persisted`, `write_persisted_steps_replace`, `save_resource_to_disk`, `load_resource_from_disk`, `load_persisted_steps_cache_clear`, `save_run_info`, `update_run_status`, `load_run_info`, `append_live_metric`, `load_live_metrics`, `append_console_log`, `load_console_logs` | Load/save step state to `_GUI_STATE/steps.json` for backfill; `load_persisted_steps` is fronted by an LRU cache keyed on `(abspath, mtime_ns, size)` so repeated historical / active-run polls don't re-parse large JSON. `save_step_to_persisted` / `write_persisted_steps_replace` take a per-file lock so the pipeline thread (`on_step_start` → `status=running`) and the snapshot executor thread (`_finalize` → `status=completed`) cannot clobber each other's read-modify-write cycles (previously fast steps like `Model Configuration` / `Model Building` could intermittently end up with no `completed` entry on disk). `_sanitize_path_segment` normalises unsafe characters to `_` rather than rejecting them so step names containing spaces (e.g. `Hard Core Mapping`, `Soft Core Mapping`, `Pruning Adaptation`) round-trip through `_resource_disk_path` without crashing the snapshot executor. Lazy resources materialised by the snapshot executor are persisted under `_GUI_STATE/resources/{step}/{kind}/{rid}.{ext}` so historical and subprocess-spawned active runs can be served from disk. Streaming metrics land in `_GUI_STATE/live_metrics.jsonl`; stdout/stderr log lines in `_GUI_STATE/console.jsonl` |
| `process_manager.py` | `ProcessManager`, `ManagedRun`, `_start_console_reader` | Spawns headless pipeline processes with `stdout=PIPE, stderr=PIPE`; background reader threads drain both pipes and write tagged lines to `_GUI_STATE/console.jsonl`; tracks runs via filesystem polling, provides status/metrics/step detail APIs, kill with SIGTERM→SIGKILL escalation. `get_run_detail` exposes `status` and `error` fields from `run_info.json` so the UI can display error banners on failure. |
| `run_cache_seed.py` | `copy_pipeline_cache_from_previous_run`, `copy_steps_json_from_previous_run` | Edit & continue: copies pipeline cache files and optionally `_GUI_STATE/steps.json` from the prior run so `run_from` has cache entries and backfill can restore full step snapshots for the monitor |
| `runs.py` | `list_runs`, `get_run_config`, `get_run_pipeline`, `get_run_step_detail`, `get_run_console_logs` | Discover and load historical pipeline runs from the generated files directory; `get_run_console_logs` reads `console.jsonl` for the console tab |
| `templates.py` | `list_templates`, `get_template`, `save_template`, `delete_template`, `name_and_deployment_from_post_body` | CRUD for deployment configuration templates saved as JSON files. Files are a **flat deployment dict** (same shape as `get_run_config`). `save_template` sets `experiment_name` on the stored copy to the given template display name (so lists and wizard load show the template label, not the originating run name). POST `/api/templates` accepts `{name, config}` or a flat deployment body. |
| `wizard_config_builder.py` | `build_deployment_config_from_state` | Builds complete deployment config from wizard UI state, applying defaults and presets |
| `heatmap_renderer.py` | `render_heatmap_png_data_uri` | Renders weight matrices as PNG data URIs for GUI; no raw matrices sent to frontend |

### Lazy Resource URL Contract

Heavy artefacts previously inlined as base64 data URIs in step snapshots (core /
weight-bank / pre-pruning heatmap PNGs, per-stage connectivity span arrays) are
now fetched lazily. Snapshot payloads carry only `has_<thing>: true` flags and
a `<thing>_resource: {kind, rid}` hint; the browser resolves those hints against
one of three endpoint prefixes:

| Run kind | Prefix |
|----------|--------|
| live (current process) | `/api/steps/{step}/resources/{kind}/{rid}` |
| historical (completed) | `/api/runs/{run_id}/steps/{step}/resources/{kind}/{rid}` |
| active subprocess      | `/api/active_runs/{run_id}/steps/{step}/resources/{kind}/{rid}` |

`{kind}` is one of the `RESOURCE_KIND_*` constants exported from `snapshot/`:
`ir_core_heatmap`, `ir_core_pre_pruning`, `ir_weight_bank_heatmap`,
`hard_core_heatmap`, `pruning_layer_heatmap`, `connectivity`. The live handler
materialises the resource through the in-memory `ResourceStore`; the historical
and active-run handlers stream the corresponding file from
`_GUI_STATE/resources/…`. A matching `Content-Type` (`image/png` or
`application/json`) is set per-kind; legacy runs that pre-date the split return
404, which the frontend treats as "no thumbnail available".

### Step Detail ETag Contract

`GET /api/steps/{step_name}` returns a weak HTTP ETag of the form
`W/"{step}-{status}-{snapshot_version}"`. `snapshot_version` is bumped on each
step lifecycle transition (start, complete, fail) but **not** on metric
appends — metrics stream over the WebSocket. The endpoint also accepts
`?since_seq=N` and filters `detail.metrics` to `seq > N`; the response includes
`latest_metric_seq` so the client can advance its cursor across polls.

Clients send `If-None-Match` on subsequent polls and receive `304 Not Modified`
when the snapshot is unchanged, sidestepping the entire snapshot payload.
Step-detail polls run on a 30 s watchdog interval; actual updates arrive via
the WebSocket `pipeline_overview` broadcast, which the `DataCollector` emits on
every step lifecycle event.

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

- `GET /api/data_providers` — list registered data providers (`id`, `label`, `supports_preprocessing`).
- `GET /api/data_providers/{id}/metadata?resize_to=&normalize=&interpolation=&datasets_path=` — instantiate the provider with the wizard's preprocessing and report `{input_shape, num_classes, supports_preprocessing}`.  Wizard calls this whenever the data provider or preprocessing drawer changes so it never has to guess input shape from the provider name.
- `GET /api/model_types` — list model types (id, label, category).
- `GET /api/model_config_schema/{model_type}` — config fields for dynamic form generation.
- `POST /api/run` — body = full deployment config JSON; creates pipeline, attaches collector, runs in background thread; returns 202.
- `POST /api/hw_config_verify` — verifies hardware core config; returns `{feasible, errors, field_errors, packing, stats}`. `stats` is a `LayoutVerificationStats.to_dict()` augmented with `host_side_segment_count` and `layout_preview`; together they provide per-core waste/utilization metrics, layout-derived neural-segment summary (`neural_segment_count`, `segment_latency_min/median/max` where latency means latency groups per segment), host-side segment count (shown as sync barriers), a compact latency-group miniview, and coalescing/splitting counts. When `allow_scheduling=True` and single-pass packing fails, `schedule_info` is included with `num_passes` and `max_cores_per_pass`.
- `POST /api/pipeline_steps` — body = same deployment config shape as `/api/run`; returns `{"steps": ["Step Name", ...], "semantic_groups": ["group_id", ...]}` for the pipeline that would be built. Used by the wizard to show a live pipeline preview without running the pipeline; `semantic_groups` (same length as `steps`) drives per-step colour coding.
- `GET /wizard` — serves the deployment configurator wizard (`static/wizard.html`).

### Frontend (`static/`)

Single-page application using ES modules and Plotly.js. See `static/js/` for
modular visualization components (overview, model, IR graph, hardware, search,
scales, pruning, live-search tabs). **`resource-urls.js`** holds the step+run
context and resolves `{kind, rid}` hints to fully-qualified resource URLs for
live / historical / active-subprocess runs; tab modules stay stateless and
import `imgSrcAttr` / `resourceUrl` from it. **Pruning tab**: shown for the Pruning Adaptation step; lists layers with per-layer weight heatmaps (red lines for pruned rows/columns, same convention as IR Graph and Hardware) and a layer browser (list + detail panel). **Hardware tab**: shows soft-core and fused hardware-core boundaries
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
stages, health-bar style percentage bars (wasted axons/neurons, param utilization, fragmentation when present in stats),
per-core min/avg/max breakdowns, and detail rows for latency-groups-per-segment plus
coalescing/splitting summaries. The data comes from the
`"stats"` key returned by `/api/hw_config_verify`. On failure or re-validation,
the panel is hidden so stale numbers are never shown.
**Scheduled Mapping** toggle (`#scheduledMappingToggle`, in **Deployment Mode**) enables hardware core reuse across
sequential passes. When ON, `allow_scheduling` is set in `deployment_parameters` and the
auto-suggest endpoint uses `suggest_hardware_config_scheduled` (exploring the core-count ↔
pass-count tradeoff). The verify endpoint reports `schedule_info` with estimated pass count.

**Weight Quantization** and **Activation Quantization** are locked (derived) from Float and Spiking Mode: no manual selection in regular deployment. Float ON locks Weight Quant to OFF; Float OFF locks it to ON. TTFS Quantized locks Activation Quant to ON; LIF and plain TTFS lock it to OFF (LIF subsumes it via its intrinsic T+1-level output; plain TTFS uses continuous analytical mapping). The Cycles field is disabled for non-quantized TTFS (analytical TTFS does not use simulation steps). Target Tq is disabled when activation quantization is off. **Float weights** is a toggle in the **Hardware Configuration** panel (next to Weight Bits): when ON it disables the Weight Bits control and locks Weight Quantization to off in Deployment Mode; pipeline uses vanilla (float) deployment. A **Loihi Simulation** toggle in the Deployment Mode section enables a post-nevresim Lava-based Loihi-compatible pass (LIF mode only). **Pruning fraction** is a [0–1) range slider with value display; the 0.8–1.0 range
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

*Restart-step default*: `init()` loads `GET /api/wizard/schema` (NAS objective list) **before** fetching run/template config so `loadStateFromConfig` can restore optimization objective chips without a race. It then fetches `GET /api/runs/{run_id}/config` (or template) and `GET /api/runs/{run_id}/pipeline`; after `loadStateFromConfig` the completed step names are stored in `_ecPrevCompleted`
(a persistent `Set`, not cleared). On the first successful `POST /api/pipeline_steps`
response, `updatePipelineStepsBar` picks the first canonical step not in that
completed set as the default `__wizardStartStep` (one-shot via `_ecSuggestionDone`).
If the currently selected step is
absent from the new step list (e.g. after changing config), it is silently
cleared. `update()` is called after any automatic step change to keep the JSON
preview in sync.

*Config hydrate*: `loadStateFromConfig` applies segment values with both `setSegVal` and `handleSegmentChange` where side effects matter (e.g. optimizer toggles Agentic Evolution-specific fields visibility). Saved `arch_search.objectives` is applied via `updateObjectiveCheckboxes(objectives)`. Legacy `accuracy_evaluator` value `direct` is normalized to `fast` for the wizard select.

*Persisted HW search bounds*: `main._parse_deployment_config` merges `platform_constraints.search_space` into `deployment_parameters.arch_search` for the pipeline but uses a **deep copy** of `platform_constraints` for that merge, so the dict serialized to `_RUN_CONFIG/config.json` still contains `search_space`. Edit & Continue (`GET /api/runs/{id}/config`) therefore reloads num core types, core counts, axon/neuron bounds, and max threshold groups for Hardware Search mode.

When `?template_id=` or `?run_id=` is present, `init()` skips the initial `updateSearchVisibility()` so the search banner does not run a hide transition before hydrate (which could leave `#searchSection` stuck hidden). After `loadStateFromConfig`, `done()` does **not** call `autoFillHardware` / `scheduleHwValidation` if either search toggle is active — otherwise fixed-HW auto/verify would overwrite the loaded config and show mapping stats / “Auto-configured” while the Search Strategy panel should stay authoritative.

*Hardware Auto-suggest*: `loadStateFromConfig` always turns off `#hwAutoSuggestToggle`
and sets `_hwAutoMode = false` when `__isEditContinueMode` is true (for both
`user` and `auto` hardware modes), so core-type inputs are editable immediately
and `done()` never triggers an unwanted `autoFillHardware`. Toggles restored from
JSON use `forced=false` so they remain editable; spiking/hardware dependency
rules still apply via `applySpikingDeps` / `applyHwDeps` after load.

**Monitor connection dot** (`#conn-dot`, `static/js/main.js`): the status dot reflects WebSocket connectivity for in-process runs (no `run_id` param) and for subprocess-spawned active runs (which now use `/ws/active_runs/{run_id}`); historical (finished) runs continue to fall back to HTTP polling health (`pollOk`).

**Realtime update path** (`main.js`, `step-detail.js`): `refreshPipeline` is downgraded from a 5 s poll to a 30 s watchdog — live pipeline overview is driven by the `pipeline_overview` WebSocket message (`applyPipelineOverviewFromWS`). Subprocess-spawned active runs (`?run_id=…` with an alive child) use a dedicated `/ws/active_runs/{run_id}` channel (`connectActiveRunWebSocket`) whose events are emitted by `ActiveRunHub` tailing the child's `live_metrics.jsonl` and `steps.json`. `scheduleStepDetailRefresh` uses a **leading-edge throttle** (200 ms cooldown): the first call in a window fires the REST refresh *immediately* and coalesces any further calls into a single trailing refresh, so step transitions no longer wait out a 200 ms debounce before fetching. On every step switch (via `step_started` or an auto-follow-triggered `pipeline_overview`) the panel is swapped to a lightweight **loading placeholder** (`showStepDetailLoading`) so the user sees the new step name/"loading…" badge instantly instead of staring at the previous step's DOM. `refreshStepDetail` is ETag-aware: it sends `If-None-Match` (seeded from the last response's `ETag` header) and `?since_seq=N` (the last `latest_metric_seq` it saw). A `304 Not Modified` keeps the DOM unchanged and only re-renders charts. On step switch the cached ETag is bypassed so the panel rebuilds. Live chart updates use a `requestAnimationFrame` coalescer (replacing the old 500 ms `setTimeout`) and prefer `Plotly.extendTraces` for incremental redraws — each chart tracks its trace names and last point counts so only new metric points are appended. Structural changes (new trace, run reset) fall back to `Plotly.react`. **First-metric scaffold recovery** (`updateLiveCharts`): when a metric arrives for the selected step but the metrics-tab chart containers don't exist yet — e.g. the last full render found empty buffers and emitted a "No metrics recorded" placeholder, or a brand-new metric group just appeared — `updateLiveCharts` client-side-rebuilds the metrics tab from the live buffers via `renderMetricsTab` (no REST fetch), preventing the "metrics only appear when I click the tab" freeze. Lazy heatmap `<img>`s are emitted with `loading="lazy" decoding="async"`. Connectivity spans for the hardware tab are fetched on first click via `/api/.../resources/connectivity/seg/{idx}` and memoised in a per-session `Map` keyed by the resolved URL (see `resource-urls.js`, `hardware-tab.js`).

**Monitor plots** (step-detail metrics tab, scales-tab adaptation, search-tab): legends
are placed outside the plot area to the right (`x: 1.02`, `margin.r: 100`). Accuracy
and Adaptation curves use a fixed vertical axis [0, 1]. A single data point is drawn
as a horizontal line from that point to the right edge. In the step-detail metrics tab,
architecture search metrics (names containing "search") are shown in separate plots
per metric so each keeps its own scale. Search history is rendered as one card and
plot per numeric metric (separate charts per objective).

**Live Search tab** (`static/js/search-live.js`, `static/search-live.css`): a cyberpunk-
themed real-time tracker for `AgentEvolveOptimizer` architecture search. The optimizer
emits structured `search_event` JSON via `reporter("search_event", json.dumps(event))`.
These events flow through `DataCollector` → WebSocket → `main.js` → `search-live.js`.
`main.js` buffers parsed events in `state.searchEvents[stepName]`; `step-detail.js`
adds a "Live Search" tab when search events are present; `renderLiveSearchTab` calls
`initSearchLive` and `replaySearchEvents` using **`state.searchEvents[stepName]`** as
the source of truth (falls back to `detail._searchEvents`). **`detachSearchLive`** runs
when switching to any other tab so `handleSearchEvent` does not update a detached DOM;
reopening Live Search replays from `state.searchEvents`. A **global Pareto strip**
(`#sl-pareto-strip`) below the header updates on each `generation_complete` with the
latest `pareto_front` (per-generation Pareto subpanels were removed). Each generation
card uses a two-column layout (`sl-gen-layout`): **main** (~2fr) holds subpanels for
reasoning (collapsible), candidates (grid with health bars), and **Constraints &
insights** with two collapsible sections; when a generation completes, those sections
receive content and **`sl-collapse-open`** so text is visible. Constraint and
performance text is rendered via **`renderMarkdown`** (`util.js`) into `.sl-md` using **marked** (CommonMark + GFM) and **DOMPurify** sanitization; both load as **pinned jsDelivr ESM URLs** from `util.js` (no import map required).
**Diagnostics** (~1fr, `sl-gen-diagnostics`) lists `llm_trace` events in call order;
clicking a row shows structured request sections and response fields (no JSON string
dumps); long text previews show a truncation hint when applicable. Header stats show
gen counter, valid/failed counts, Pareto size, and elapsed time.

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
- `GET /api/active_runs/{run_id}/steps/{step_name}/resources/{kind}/{rid}` — serve a lazy resource written to the subprocess's `_GUI_STATE/resources/…` directory by its snapshot executor

## Historical Run API Endpoints

- `GET /api/runs` — list past runs (optionally with `?include_steps=true`)
- `GET /api/runs/{run_id}/config` — full deployment config of a past run
- `GET /api/runs/{run_id}/pipeline` — pipeline overview of a past run
- `GET /api/runs/{run_id}/steps/{step_name}` — step detail of a past run
- `GET /api/runs/{run_id}/console?offset=N` — console log entries for a past run
- `GET /api/runs/{run_id}/steps/{step_name}/resources/{kind}/{rid}` — serve a lazy heatmap PNG or connectivity JSON from the run's on-disk resource cache (legacy runs that predate the split return 404)

## In-Process API Endpoints

- `GET /api/console_logs?offset=N` — console log entries from in-process DataCollector (WebSocket also pushes `console_log` events)
- `GET /api/steps/{step_name}` — returns step detail with a weak ETag. Accepts `If-None-Match` (→ `304 Not Modified` on unchanged snapshot) and `?since_seq=N` (filters `metrics` to `seq > N`; response includes `latest_metric_seq`)
- `GET /api/steps/{step_name}/resources/{kind}/{rid}` — serve a lazy heatmap PNG or connectivity JSON from the live `ResourceStore`. Producers are invoked exactly once per step on first demand and memoised until the next `step_started`

## Template API Endpoints

- `GET /api/templates` — list saved templates
- `GET /api/templates/{id}` — get a template's config
- `POST /api/templates` — save a new template (body: `{name, config}`)
- `DELETE /api/templates/{id}` — delete a template
