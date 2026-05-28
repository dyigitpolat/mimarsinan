# Firing modes — cross-backend contract

Behavior is modeled as **orthogonal dimensions** (reset, comparison, spike encoding), not combined policy permutations.

| Dimension | Config key | Values | Concern |
|-----------|------------|--------|---------|
| Reset | `firing_mode` | `Default`, `Novena` | Membrane reset on fire |
| Comparison | `thresholding_mode` | `<`, `<=` | Threshold comparator |
| Input encoding | `spike_generation_mode` | Uniform, … | Segment input spikes |
| Dynamics | `spiking_mode` | lif, ttfs, … | Neuron model path |

## Reset (`firing_mode`)

| `firing_mode` | Membrane reset on fire | nevresim reset policy | Training (`LIFActivation`) | SANA-FE `reset_mode` | Lava |
|---------------|------------------------|----------------------|----------------------------|----------------------|------|
| `Default` | Subtractive (`v -= θ`) | `SubtractiveReset` | `v_reset=None` | `soft` | `zero_reset=False` |
| `Novena` | Zero-reset (`v = 0`) | `ZeroReset` | `v_reset=0.0` | `hard` | `zero_reset=True` |
| `TTFS` | Analytical TTFS path | N/A (use `spiking_mode`) | N/A | N/A | N/A (Lava LIF-only) |

## Comparison (`thresholding_mode`)

| `thresholding_mode` | Comparator | HCM / Lava / SANA-FE | nevresim compare policy |
|---------------------|------------|----------------------|-------------------------|
| `<` | strict (`v > θ`) | yes | `StrictCompare` |
| `<=` | inclusive (`v >= θ`) | yes | `InclusiveCompare` |

Nevresim codegen composes LIF firing as `LIFirePolicy<ResetPolicy, ComparePolicy>` (see `code_generation/generate_main.py`).

## Spike encoding (`spike_generation_mode`)

Lava and SANA-FE inject segment inputs via `NeuralBehaviorConfig.encode_segment_input()` → `recording._spike_encoding.encode_segment_input`. HCM uses torch `spike_modes.to_spikes` per cycle. Nevresim uses compile-time `{Mode}SpikeGenerator` templates.

Parse deployment config once via `chip_simulation.behavior_config.NeuralBehaviorConfig.from_deployment_config` or `pipelining.core.simulation_factory.build_neural_behavior_config`.

LIF-only on Lava: `spiking_mode` must be `lif`. TTFS spike encoding uses dedicated paths, not `encode_segment_input`.
