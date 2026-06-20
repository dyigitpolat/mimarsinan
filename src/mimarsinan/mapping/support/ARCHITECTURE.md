# mapping/support/ — Mapping support utilities

Shared helpers used across mapping, packing, and the hybrid build (geometry,
activation scales, bias bakes, scheduling).

## Files

| File | Exports | Role |
|------|---------|------|
| `activation_scales.py` | `compute_node_output_scales`, `compute_node_input_scales` | IR-graph activation scale tables for TTFS / ComputeOp rescaling. |
| `compute_modules.py` | host-op payload classes | Host-side ComputeOp payload classes. |
| `core_geometry.py` | `used_neurons`, … | Occupied axon/neuron counts for HardCore-style objects. |
| `ir_source_spans.py` / `spike_source_spans.py` | span builders | Source span tables for segment I/O. |
| `neg_shift_bias.py` | `apply_negative_value_shifts`, `calibration_forward_for_mode`, `apply_negative_shift_bias`, `negative_shifts_from_min`, `transfer_negative_shifts_to_ir`, `propagate_negative_shifts_to_hybrid` | **Negative-value shift**: calibrate per-ComputeOp output minima via the mode's NF forward (`calibration_forward_for_mode` validates the mode then delegates to the `SpikingModePolicy.calibration_forward()` SSOT — LIF chip-aligned, TTFS segment driver, or analytical driver; still fails loud for unsupported modes such as `rate`), derive `s = max(0, −min)`, bake consumer perceptron bias `B' = B − W·s` once (idempotent), tag `_negative_shift` on the mapper op, and carry it mapper→IR (`{name}_col{i}` leading-dim splits get their per-column row)→hybrid `node_output_shifts`. Fails loud on ComputeOp→ComputeOp, multi-input concat, and a baked subsumed encoder feeding a ComputeOp. |
| `per_source_scales.py` | `compute_per_source_scales` | Per-source activation scales for branching architectures. Thin driver over the V6 polymorphic walk (`mappers/scale_propagation.py`); per-kind decisions live on each `Mapper.propagate_source_scale`, not an isinstance chain. |
| `scale_broadcast.py` | broadcast helpers | Shared scale-vector broadcast for mapper and IR paths. |
| `shape_probe.py` | `probe_module_io_shapes` | Zeros-tensor shape inference for host ComputeOps; the dummy follows the module's parameter dtype. |
| `ttfs_bias.py` | `apply_ttfs_quantization_bias_compensation` | Bake TTFS QuantizeDecorator shift into perceptron bias pre-mapping. |
| `schedule/` | partitioners | Segment schedule budgeting and splitting. |

## Dependents

- `pipelining/pipeline_steps/mapping/soft_core_mapping_step.py` — runs the
  negative-shift calibration pre-mapping and transfers shifts to the IR.
- `pipelining/core/simulation_factory.py` — propagates shifts IR→hybrid.
- `spiking/segment_forward.py` (NF) and the hybrid/TTFS executors consume the
  resulting `_negative_shift` / `node_output_shifts`.
