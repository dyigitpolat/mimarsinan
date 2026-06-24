# scripts/campaign/ -- Autonomous Research Campaign Daemons

Long-lived daemons that keep the GPU queue full and the research loop turning
without a human in the loop. They produce/consume `scripts/gpu/gpu_queue.py` jobs.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `scheduler.py` | `Scheduler`, `instantiate`, `onchip_precheck`, `_classify_cfg_validity`, `set_path`, `get_path`, `existing_ids` | FILLS the queue from a declarative `backlog.json`: instantiates each batch's config grid, dedupes against everything enqueued/run, refills to a high-watermark. Before enqueuing, runs the TIERED validity pre-check (`onchip_precheck` → `classify_validity`, both params AND macs): only an INVALID model (`min(param,mac)` on-chip below `deployment_parameters.onchip_min_fraction`, default 0.20) is SKIPPED/logged `invalid_host_majority` so it never claims a GPU. VALID and VALID_FLAGGED are ADMITTED; a flagged job (below the 0.50 majority `onchip_majority_fraction`) is logged with its named `research_gap_ops`/`placement_fixable_ops` as transferable-tuning evidence. Gated by `onchip_majority_gate`; model-build/classification failures are NON-FATAL — the job is enqueued anyway. |
| `director.py` | research director | GROWS the backlog from ledger findings + FLAGS uncovered runs. |
| `research_loop.py` | research-loop primitives | Enqueue/wait/results/ledger helpers for research workflows. |

## Dependencies

- **Internal (lazy)**: `mimarsinan.mapping.verification.onchip_fraction.classify_validity`, `mimarsinan.pipelining.core.registry.model_registry.ModelRegistry`, `mimarsinan.data_handling.data_provider_factory.BasicDataProviderFactory` — imported only inside `_classify_cfg_validity` so importing the scheduler stays light for the daemon.
- **External**: `scripts/gpu/gpu_queue.GpuQueue`.

## Dependents

- `campaign_runner` (in `scripts/gpu/`) DRAINS the queue the scheduler fills.
