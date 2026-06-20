# MMIXCORE + MNIST — 9 templates, full config-matrix coverage

Nine wizard-constructible deployment templates (`mnist_mmixcore_matrix_1..9_*.json`) that
together exercise every option of the deployment configuration matrix
(`docs/DESIGN_GOALS_and_refactoring_vectors.md` §2), all on the **mmixcore** platform
(two core types: 256ax/512neu ×60 + 512ax/256neu ×60, coalescing + neuron-splitting
enabled) and **MNIST** (`mlp_mixer_core`). Generator: `experiments/gen_mmixcore_matrix_templates.py`.

Constructibility is proven, not asserted: each passes `validate_wizard_state` (the wizard's
own validator) and `build_flat_pipeline_config` (the wizard build path), and they are picked
up by `test_namespaced_schema.py::test_template_config_roundtrips`.

## The 9 templates

| T | spiking_mode | firing | schedule | encoding | thr | quant | pruning | sync-pts | bias | mode | backends |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lif | Default | — | subsume | `<` | WQ | — | — | on_chip | phased | nevresim·sanafe·hcm |
| 2 | lif | **Novena** | — | **offload** | `<` | WQ | — | — | on_chip | phased | **loihi**·nevresim·hcm |
| 3 | lif | Default | — | subsume | `<` | WQ | **on (0.5)** | **on** | on_chip | phased | nevresim·sanafe·hcm |
| 4 | ttfs | TTFS | — | subsume | `<=` | WQ | — | — | on_chip | phased | nevresim·sanafe·hcm |
| 5 | **ttfs_quantized** | TTFS | — | **offload** | `<=` | WQ+**AQ** | — | — | on_chip | phased | nevresim·sanafe·hcm |
| 6 | **ttfs_cycle_based** | TTFS | **cascaded** | subsume | `<=` | WQ | — | — | on_chip | phased | nevresim·sanafe·hcm |
| 7 | ttfs_cycle_based | TTFS | **synchronized** | subsume | `<=` | WQ | — | — | on_chip | phased | sanafe·hcm¹ |
| 8 | ttfs_cycle_based | TTFS | cascaded | **offload** | `<=` | WQ | — | **on** | **param_encoded** | phased | nevresim·sanafe·hcm |
| 9 | ttfs | TTFS | — | subsume | `<=` | **none** | — | — | on_chip | **vanilla** | nevresim·sanafe·hcm |

¹ nevresim is auto-disabled for synchronized TTFS (no synchronized nevresim backend) — a
derived rule, not a manual omission.

## Axis coverage (every value appears ≥1×)

| Axis | Values covered | Templates |
|---|---|---|
| **Firing mode** | lif · ttfs · ttfs_quantized · ttfs_cycle_based | 1-3 · 4,9 · 5 · 6-8 |
| **LIF firing variant** | Default · Novena | 1,3 · 2 |
| **Sync mode** | cascaded · synchronized | 6,8 · 7 |
| **Encoding** | subsume · offload | 1,3,4,6,7,9 · 2,5,8 |
| **Thresholding** | `<` · `<=` | 1-3 · 4-9 |
| **Weight quant** | on · off | 1-8 · 9 |
| **Activation quant** | derived-on (ttfs_quantized) | 5 |
| **Pruning** | on · off | 3 · rest |
| **Mapping: coalescing** | allowed | all (mmixcore) |
| **Mapping: neuron-splitting** | allowed | all (mmixcore) |
| **Mapping: sync-points** (`allow_scheduling`) | on · off | 3,8 · rest |
| **Bias mode** (derived from `has_bias`) | on_chip · param_encoded | 1-7,9 · 8 |
| **Backend** | nevresim · SANA-FE · Loihi/Lava · HCM/SCM | most · most · 2 · all |
| **Pipeline mode** (derived) | phased · vanilla | 1-8 · 9 |

## Notes on derived axes (resolved, never declared raw)

- **activation_quantization** is *derived* — forced on only for `ttfs_quantized` (T5);
  off for the cycle-based / LIF tuned modes.
- **bias mode** is *derived* from per-core `has_bias` (`on_chip` vs `param_encoded`) — T8
  sets `has_bias=false` to exercise `param_encoded`.
- **pipeline_mode** is *derived* — `vanilla` when no quantization (T9), else `phased`.
- **nevresim availability** is *derived* — disabled for synchronized TTFS (T7).
- **Mapping strategy** (coalesce/split/sync-point) is *derived* from the model shape ×
  the capability gates; the templates declare the gates (`allow_*`), the framework derives
  the per-layer strategy. Lava/Loihi is LIF-only, so it appears only on a LIF template (T2).

Spike-generation is `TTFS` for the TTFS family and `Uniform` for LIF (derived from the
firing mode); the matrix's stochastic / spike-train spike-gen options are not standard for
mmixcore deployment and are intentionally out of this set.
