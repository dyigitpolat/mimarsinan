# scripts/campaign/ -- Autonomous Research Campaign Daemons

Long-lived daemons that keep the GPU queue full and the research loop turning
without a human in the loop. They produce/consume `scripts/gpu/gpu_queue.py` jobs.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `scheduler.py` | `Scheduler`, `instantiate`, `onchip_precheck`, `_classify_cfg_validity`, `capacity_precheck`, `_estimate_cfg_capacity`, `set_path`, `get_path`, `existing_ids` | FILLS the queue from a declarative `backlog.json`: instantiates each batch's config grid, dedupes against everything enqueued/run, refills to a high-watermark. Before enqueuing, runs the TIERED validity pre-check (`onchip_precheck` → `classify_validity`, both params AND macs): only an INVALID model (`min(param,mac)` on-chip below `deployment_parameters.onchip_min_fraction`, default 0.20) is SKIPPED/logged `invalid_host_majority` so it never claims a GPU. VALID and VALID_FLAGGED are ADMITTED; a flagged job (below the 0.50 majority `onchip_majority_fraction`) is logged with its named `research_gap_ops`/`placement_fixable_ops` as transferable-tuning evidence. It THEN runs the E4 capacity pre-check (`capacity_precheck` → `_estimate_cfg_capacity` → `estimate_cores_needed`): a config whose STATIC hard-core lower bound exceeds its declared core budget is SKIPPED/logged `capacity_exceeded` (naming `cores_needed`/`cores_available`/`overflowing_segment`) so an unplaceable net (VGG16@224 on a 1000-core budget) never claims a GPU. Gated by `onchip_majority_gate` / `capacity_gate`; model-build / IR / classification / estimate failures are NON-FATAL — the job is enqueued anyway. |
| `director.py` | research director | GROWS the backlog from ledger findings + FLAGS uncovered runs. |
| `research_loop.py` | research-loop primitives | Enqueue/wait/results/ledger helpers for research workflows. |
| `coverage_report.py` | `main` (CLI) | **Frontier E1 CLI** — prints the hypervolume coverage report over `runs/campaign/ledger.jsonl` (`--ledger`): the tier tally (VALID / VALID_FLAGGED / INVALID), the coverage FRACTION against a claimed sub-product (`--vehicle` / `--dataset` / `--sync` / `--firing` / `--S` pin axes), the named UNTESTED frontier, the RESEARCH-GAP and PLACEMENT-FIXABLE frontiers, and (`--axes`) the orthogonal-vs-interacting axis classification. `--json` emits the machine-readable report. Delegates to `mimarsinan.chip_simulation.coverage_ledger`. |

## Dependencies

- **Internal (lazy)**: `mimarsinan.mapping.verification.onchip_fraction.classify_validity`, `mimarsinan.pipelining.core.registry.model_registry.ModelRegistry`, `mimarsinan.data_handling.data_provider_factory.BasicDataProviderFactory` — imported only inside `_classify_cfg_validity` so importing the scheduler stays light for the daemon. The E4 capacity pre-check `_estimate_cfg_capacity` additionally (lazily) imports `mimarsinan.config_schema.runtime.build_flat_pipeline_config`, `mimarsinan.torch_mapping.convert_torch_model`, `mimarsinan.mapping.ir_mapping_class.IRMapping`, and `mimarsinan.mapping.verification.capacity.estimate_cores_needed` to build the IR and run the static core-count bound. `coverage_report.py` imports `mimarsinan.chip_simulation.coverage_ledger` (the E1 coverage API).
- **External**: `scripts/gpu/gpu_queue.GpuQueue`.

## Dependents

- `campaign_runner` (in `scripts/gpu/`) DRAINS the queue the scheduler fills.
