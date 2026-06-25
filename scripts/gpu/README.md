# GPU scheduling for parallel, isolated work (minimal)

Shared queueing infra so many parallel jobs (across processes / git worktrees on
one host) saturate the **dynamically-changing** set of free GPUs without ever
hard-coding GPU ids.

Two job classes:

| mode | for | predicate | held |
|------|-----|-----------|------|
| `free` | profiling / wall-clock (AC5, re-freeze) | `mem_free/total ≥ 80%` **and** `util < 5%` **and** unleased | exclusive |
| `fit` | correctness-only (accuracy, crash-sweeps) | `mem_free − pool_fit_reservations ≥ need_mb`; **util ignored** | shared |

A single `flock` makes the snapshot→pick→claim atomic, so two jobs never grab the
same free GPU. External users' usage is seen via `nvidia-smi`; leases only stop
**our** pool from double-booking. Dead leases (owning pid gone) are pruned on every
read — no daemon, no heartbeat.

## Files
- `gpu_lease.py` — the core: `query_nvidia_smi`, `choose(mode, need_mb, …)` (pure
  pick), `acquire`/`release`, `acquire_blocking` (grab a freed GPU the instant it
  appears, so nothing idles). Lease dir: `$MIM_GPU_LEASE_DIR` or `/dev/shm/mim_gpu_leases_<uid>`.
- `gpu_dispatch.py` — drains a job manifest across GPUs, keeping them saturated.
  `python gpu_dispatch.py --manifest jobs.json --results out.json --logdir d`.
  Manifest: `[{"id","mode","need_mb","cmd":[...]|str,"cwd"?,"env"?}, …]`.
- `bootstrap_worktree.sh` — symlink the gitignored runtime deps (venv, datasets,
  build, nevresim, spikingjelly) from the main checkout into a worktree.
- `deploy_imagenet_snn.py` (F4 capstone) — load the trained ResNet-50 checkpoint
  (`runs/imagenet/resnet50.pt`, `{model state_dict, val_top1}`) → deploy it as a LIF
  SNN through the real pipeline (`pretrained_bridge.deploy_and_eval`) on an
  ImageNet-val SUBSET → record ANN top-1, deployed top-1, validity tier
  (`classify_validity`) + a cost record (`extract_cost_record`) → append a
  campaign-shaped ledger row to `runs/campaign/ledger.jsonl` via the campaign
  `ledger-append` convention. CLI:
  `env/bin/python scripts/gpu/deploy_imagenet_snn.py --checkpoint runs/imagenet/resnet50.pt --num-eval 256 --T 4`.
  The real run is a SUPERVISED step (slow LIF sim → subset is honest); the unit
  tests (`test_deploy_imagenet_snn.py`) exercise the wiring on a tiny CPU stand-in
  (no ImageNet, no GPU).

Tests: `tests/unit/gpu/` (no real GPU needed — snapshots are injected).
`scripts/gpu/test_deploy_imagenet_snn.py` covers the F4 deploy harness in-place.
