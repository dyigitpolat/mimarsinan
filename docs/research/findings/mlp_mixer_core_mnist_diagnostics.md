# MNIST `mlp_mixer_core` Diagnostics

This checkpoint prepares the MNIST `mlp_mixer_core` closure campaign without
launching GPU work.

## Acceptance

Acceptance is hard for every diagnostic row:

- `deployed_acc >= 0.97`
- `relative_time < 1.0`, where `relative_time = run_wall_s / fastest_successful_baseline_wall_s`
- `returncode == 0`

Rows that improve accuracy but miss timing, or run quickly but stay below 97%,
remain diagnostic results rather than accepted cells.

## Prepared Cells

The manifest covers:

- LIF diagnostic: `templates/mnist_mmixcore_matrix_1_lif_rate.json`
- TTFS-cycle synchronized diagnostic: `templates/mnist_mmixcore_matrix_7_ttfs_cycle_synchronized.json`
- TTFS-cycle cascaded diagnostic: `templates/mnist_mmixcore_matrix_6_ttfs_cycle_cascaded.json`
- Analytical TTFS control: `templates/mnist_mmixcore_matrix_4_ttfs_analytical.json`
- TTFS-quantized control: `templates/mnist_mmixcore_matrix_5_ttfs_quantized_offload.json`

Each row carries planned per-step metric/timing slots for Activation Analysis,
Activation Adaptation, Clamp, Shift, AQ, LIF/TTFS-cycle tuning, WQ, NormFusion,
SCM/HCM, and simulation, plus proxy-vs-genuine probe metadata where required.

## Recipe Presets

The registry defines:

- `mixer_lif_fast_minimal`
- `mixer_lif_fast_stabilized`
- `mixer_sync_ttfs_qat_minimal`
- `mixer_cascaded_proxy_then_refine_minimal`
- `mixer_cascaded_genuine_blend_fast`
- `mixer_controller_baseline`

Budgets are recorded separately as ramp, recovery, stabilization, evaluation,
and wall-time ceilings so later Pareto analysis can compare accuracy against
relative runtime rather than raw end-to-end wall time alone.

## Queue Preparation

Generate queue jobs without enqueuing or launching:

```bash
env/bin/python scripts/campaign/coverage_breadth.py queue-manifest-mnist-mixer \
  --out runs/campaign/mnist_mixer_queue_manifest.json \
  --config-dir experiments/campaign \
  --seeds 0,1,2
```

Then enqueue through the existing research-loop queue interface:

```bash
env/bin/python scripts/campaign/research_loop.py enqueue \
  runs/campaign/mnist_mixer_queue_manifest.json
```

Alternatively, generate a default-off scheduler backlog file:

```bash
env/bin/python scripts/campaign/coverage_breadth.py generate-mnist-mixer \
  --out runs/campaign/backlog_mnist_mixer_diagnostics.json \
  --seeds 0,1,2
```
