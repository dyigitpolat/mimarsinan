# Reproduction Scripts

Persisted drivers for checkpoint measurements. All require a GPU and the project virtualenv.

## Prerequisites

```bash
cd /path/to/mimarsinan   # project root (parent of docs/)
source env/bin/activate
chmod +x docs/checkpoint_research/repro/*.sh
```

`MIMARSINAN_DISABLE_FFCV=1` is set automatically by `common.sh`.

## Scripts

| Script | Variants | Expected runtime | Output reference |
|---|---|---|---|
| `run_single_smoke.sh` | 1 cell (CIFAR-10 d4 sync) | ~8 min | sanity check |
| `run_cifar_baseline_grid.sh` | 8 cells | ~1–2 GPU-hours | `data/00_cifar_baseline_grid.jsonl` |
| `run_ttfs_T_sweep.sh` | T ∈ {8,16,32,64} | ~30 min each | `data/01_ttfs_T_sweep.jsonl` |
| `run_alpha_q_sweep.sh` | α, q, LIF | ~30 min each | `data/02_ttfs_alpha_q_sweep.jsonl` |
| `run_budget_sweep.sh` | budget × epochs | ~45–90 min each | `data/03_budget_sweep.jsonl` |

## Usage

From repo root:

```bash
./docs/checkpoint_research/repro/run_single_smoke.sh
```

Variant configs are written to `repro/generated/` (gitignored pattern: ephemeral run artifacts).

## Reading results

After each run:

```bash
# Deployed metric
cat generated/checkpoint_repro_<label>_phased_deployment_run/__target_metric.json

# ANN metric (from log)
grep "Activation Analysis" generated/checkpoint_repro_<label>_phased_deployment_run/*.log
```

See [data/SCHEMA.md](../data/SCHEMA.md) for JSONL field definitions.

## Base config

[`base_configs/cifar10_d4_synchronized.json`](base_configs/cifar10_d4_synchronized.json) — derived from `templates/mnist_deep_cnn_d8_cascaded.json` adapted for CIFAR-10 d4 synchronized TTFS. Variant scripts patch this via `make_variant_config` in `common.sh`.
