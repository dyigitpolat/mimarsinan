# scripts/campaign/ -- Autonomous Research Campaign Daemons

Long-lived daemons that keep the GPU queue full and the research loop turning
without a human in the loop. They produce/consume `scripts/gpu/gpu_queue.py` jobs.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `scheduler.py` | `Scheduler`, `instantiate`, `onchip_precheck`, `_estimate_cfg_onchip_fraction`, `set_path`, `get_path`, `existing_ids` | FILLS the queue from a declarative `backlog.json`: instantiates each batch's config grid, dedupes against everything enqueued/run, refills to a high-watermark. Before enqueuing, runs the on-chip-majority pre-check (`onchip_precheck`): a host-majority model (static on-chip fraction below `deployment_parameters.onchip_majority_min_fraction`, default 0.5; gated by `onchip_majority_gate`) is SKIPPED and logged `invalid_host_majority` so it never claims a GPU. Model-build/estimation failures are NON-FATAL — the job is enqueued anyway. |
| `director.py` | research director | GROWS the backlog from ledger findings + FLAGS uncovered runs. |
| `research_loop.py` | research-loop primitives | Enqueue/wait/results/ledger helpers for research workflows. |

## Dependencies

- **Internal (lazy)**: `mimarsinan.mapping.verification.onchip_fraction.estimate_onchip_fraction`, `mimarsinan.pipelining.core.registry.model_registry.ModelRegistry`, `mimarsinan.data_handling.data_provider_factory.BasicDataProviderFactory` — imported only inside `_estimate_cfg_onchip_fraction` so importing the scheduler stays light for the daemon.
- **External**: `scripts/gpu/gpu_queue.GpuQueue`.

## Dependents

- `campaign_runner` (in `scripts/gpu/`) DRAINS the queue the scheduler fills.
