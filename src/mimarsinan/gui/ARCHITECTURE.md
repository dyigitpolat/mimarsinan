# gui/ — Browser-based pipeline monitor, run manager, and configuration workbench

Real-time dashboard served alongside every pipeline run: a FastAPI/uvicorn server
(daemon thread) streams step lifecycle, metrics, and console output to a Plotly
single-page frontend, and (with `run.py --ui`) also serves the deployment
configuration workbench and spawns headless pipeline subprocesses. Central
abstractions: `GUIHandle` (pipeline hooks that build step snapshots and persist
them), the thread-safe `DataCollector` (in-memory state + WebSocket broadcast),
and `ResourceDescriptor`/`ResourceStore` (lazy, step-scoped heavy artefacts —
heatmap PNGs, connectivity JSON — materialised on first HTTP fetch). The SPA
assets (HTML/CSS/ES-module JS) live in the non-package `static/` directory;
third-party runtime assets (Plotly, fonts) are vendored under `static/vendor/`
so the GUI works offline — a unit ratchet (`test_static_offline.py`) rejects
any external `src`/`href`/`url()` reference in static assets. The configurator
frontend (`static/js/wizard/`) renders ENTIRELY from `GET /api/config_schema`
(the config-key registry) — the HTML is layout chrome with zero field
knowledge, the draft state IS the config document (explicit keys only), and
derivation runs exclusively server-side via `POST /api/config/resolve`
(derived chips with WHY, keyed inline errors with rule-prescribed remedies,
diff-vs-defaults template view, loud unrecognized-keys tray, the emitted
document the review pane shows verbatim). The configurator is a CAD-style
WORKBENCH (`static/js/wizard/workbench.js`): a left section rail with free
navigation (Workload · Co-Design · Deployment semantics · Tuning & Budgets ·
Review & Launch — sections host whole registry concern GROUPS, the taxonomy
is the placement, with per-section error badges) and a persistent sticky
right live rail (resolve verdict, the honest vertical pipeline-assembly list
re-rendered from every resolve, a compact mapping summary mirroring the
Co-Design panel, and Launch with strong disabled/blocked semantics). The
flagship Co-Design section shows model architecture, the core grid, and the
full mapping-performance panel on one screen so both can be tuned against
live mapping feedback ("Suggest hardware" included). Field rendering is
schema-driven and generic: relevance predicates control field EXISTENCE,
`promote_when` makes mode-defining knobs primary in their mode, bounded
numerics render as slider+numeric combos, small enums as segmented buttons,
and every empty control states what empty resolves to (`empty_means`).
Policy-derived keys (simulator enables, core maxima) render as status/chips,
never as knobs. A fresh draft is seeded from `GET /api/config/starter`.
`scripts/wizard_screenshots.py` (dev-only, browser-driven) captures the
review evidence set: per-section shots, the Co-Design interaction sequence,
per-mode switches, the template flow, and the error/remedy flow.

## Key files
| File | Purpose |
|---|---|
| `exports.py` | Flat public re-export surface (`GUIHandle`, `start_gui`, `backfill_skipped_steps`, `DataCollector`, `to_json_safe`) consumed by `__init__.py`. |
| `handle.py` | `GUIHandle` facade: step start/end and metric/event hooks, stdio tee, snapshot build, synchronous status writes plus async resource persistence via `SnapshotExecutor`. Records each step's honest `metric_kind` (`measured`/`carried`) and gate `verdict` from the step's own declaration — a carried value is never persisted as a measurement. |
| `heatmap_renderer.py` | Matplotlib rendering of weight matrices to PNG bytes / data URIs, with red pruned-row/column overlays. |
| `json_util.py` | `to_json_safe` recursive JSON coercion (NaN/Inf → `None`, numpy → lists/scalars, fallback `str`). |
| `reporter.py` | `GUIReporter` implementing the `Reporter` protocol; forwards metrics to the `DataCollector`. |
| `resources.py` | `ResourceDescriptor` (kind, rid, producer, media_type) and thread-safe `ResourceStore`: lazy once-only materialisation, per-step eviction and version counter for ETags. |
| `runs.py` | Discovery and loading of historical runs from the generated-files root: run list, config, pipeline overview, step detail (with disk rebuild fallback), console logs, resume-step suggestion. |
| `start.py` | `start_gui` bootstrap (collector + resource store + server + handle) and `backfill_skipped_steps` for edit-and-continue: replays cached steps into the collector and rewrites `steps.json`. |
| `tee_stream.py` | `TeeStream`: line-buffered stdout/stderr tee that forwards complete lines to the console-log callback while writing through to the original stream. |
| `templates.py` | CRUD for saved deployment-config templates (JSON files under the templates dir), persisted minimally through the wizard config builder. |
| `runtime/` | Runtime machinery: `DataCollector` (collector/), the structured pipeline-event vocabulary (`events.py`: `PipelineEvent` + kinds mirroring the console `[TAG]`s one-to-one, transported via `reporter.event`, persisted to `events.jsonl`, WS-broadcast as `{"type":"event"}` frames), on-disk persistence of `steps.json`/metrics/events/console/resources (persistence/), subprocess run management (`ProcessManager`, spawn/monitor), `ActiveRunHub` jsonl tailers for active-run WebSockets, `CompositeReporter`, `SnapshotExecutor`, and run-cache seeding. |
| `server/` | FastAPI app factory and uvicorn startup (`app.py`) plus route modules: pipeline/runs/templates/console APIs, lazy-resource endpoints, wizard and config-schema APIs, and hardware layout verification; `json_safe.py` provides the sanitising JSON response class. |
| `snapshot/` | Pure per-artifact snapshot builders returning `(summary, ResourceDescriptor list)`: model, IR graph, hardware mapping, adaptation, pruning, search, and SANA-FE snapshots, `RESOURCE_KIND_*` constants, disk-based snapshot rebuild for legacy runs, and the best-effort console `[TAG]` parser (`console_events.py`) that backfills events for runs recorded before `events.jsonl`. |
| `viewmodel/` | Pure, I/O-free view-models (parsed run artifacts in, chart-ready JSON out; unit-tested against synthetic streams): `overview_vm` (measured points + verdict markers — a carried metric NEVER plots), `step_metrics_vm` (the one metric-categorization rule table), `events_vm` (per-kind display hints + annotation lanes), `staircase_vm` (the D-hat ratchet staircase; raises on a falling ratchet), `gantt_vm` (step timeline + endpoint step-budget ledger + artifact/total wall split), `a6_vm` (install-resolution gauge cards). |
| `wizard/` | Configuration workbench application layer: `emit.py` (explicit-keys-only config emission — the ONE builder used by Deploy, templates, and the representability test; unknown keys preserved and reported, never dropped; non-declarable derived keys — the ConversionPolicy-owned sim enables, `activation_quantization` — are removed), `build_deployment_config_from_state` (thin alias over emit), `schema_api.py` (`/api/config_schema` payload: serialized registry + recipe/preprocessing/NAS sub-schemas; `/api/config/resolve` payload: resolution + live step preview), `starter.py` + `starter_baseline.json` (the fresh-state contract: `GET /api/config/starter` serves the packaged baseline DOCUMENT — the lenet5 vehicle, the only tier-0 family green in all five modes, with a fresh experiment name and no pinned derived mode keys — pinned resolvable/emittable/mappable per mode switch by `test_wizard_starter.py`; workload facts live in the document, never in framework code), wizard schema surfaces (model types, NAS, temporal allocation, pipeline steps), and state validation. |

## Dependencies
- `common` — `best_effort` error scoping, env-derived paths (`runs_root`, `templates_dir`, `gui_no_browser`), `layer_key` helpers for snapshots.
- `config_schema` — deployment defaults, `validate_deployment_config`, `namespaced_schema` exposure metadata for the wizard builder, `display_view` structured config views.
- `mapping` — IR types and spike-source span compression for snapshots; layout verification service, request types, and hardware-config suggesters behind `/api/hw_config_verify` and auto-suggest (lazy imports).
- `models` — `builders.wizard_schema` model-type schemas for the wizard form.
- `pipelining` — deployment pipeline step specs and semantic groups for step previews; model registry for model-type/config-schema APIs (lazy imports).
- `data_handling` — `BasicDataProviderFactory` for the data-provider listing/metadata endpoints (lazy import); `preprocessing` normalization/interpolation option surfaces for the wizard schema payload.
- `search` — `ALL_OBJECTIVES` / `ACCURACY_OBJECTIVE_NAME` for the wizard NAS schema.
- `tuning` — `S_ALLOCATION_MODES` for the wizard temporal-allocation schema.

## Dependents
- `pipelining` — `session.py` uses `CompositeReporter`; `architecture_search_helpers` uses `to_json_safe`.
- `config_schema` — `display_view` lazily imports the wizard config builder and NAS schema.
- `search` — compilagent `backend_tools` imports `to_json_safe`.
- Entry point `run.py` (repo root) — starts the GUI server, `ProcessManager`, and headless-mode collector/persistence.

## Exported API
`__init__.py` re-exports from `exports.py`:
- `GUIHandle` — pipeline hook facade attached to a running pipeline.
- `start_gui` — start the dashboard server and return a `GUIHandle`.
- `backfill_skipped_steps` — replay cached steps when resuming mid-pipeline.
- `DataCollector` — thread-safe in-memory run state store (from `runtime.collector`).
- `to_json_safe` — JSON-safe payload coercion.
