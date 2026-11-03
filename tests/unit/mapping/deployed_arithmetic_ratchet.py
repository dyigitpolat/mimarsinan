"""[U-DAR] T-7's classified table: every float dtype literal remaining on an executor path (keyword default, cast or
``dtype=`` keyword in ``tests/unit/mapping/test_deployed_arithmetic.EXECUTOR_MODULES``), with its count, class and
reason. Shrink it as literals move to the record (``mapping/ir/deployment_dtype.py``); never grow it unreviewed."""

from __future__ import annotations

RECORDER = "recorder: the value is written to a record or a count capture, never fed back into the evaluation"
ENCODER = "encoder: 0-or-1 spikes or integer counts, exact in this dtype"
COUNTERFACTUAL = "declared counterfactual: the H8 arm's float32 value executor"
DEPARTURE = "a named departure from the record (reason in the entry)"
KERNEL = "kernel internal: a float64 temporary inside one kernel, between the record's seam dtypes (exact widening)"
NOT_DEPLOYED = "not an evaluation of the deployed program (shape inference, an analysis's own twin or metric)"
CLASSES = (RECORDER, ENCODER, COUNTERFACTUAL, DEPARTURE, KERNEL, NOT_DEPLOYED)

#: (module under src/mimarsinan, enclosing function, literal) -> (count, class, reason)
CLASSIFIED = {
    ('certification/randomized/executors.py', '_gathered_output', 'dtype= np.float64'):
        (1, RECORDER, "the randomized certificate's captures, widened exactly to float64 for the comparison (never fed back)"),
    ('certification/randomized/executors.py', '_timing_nodes', 'dtype= np.float64'):
        (1, RECORDER, "the randomized certificate's captures, widened exactly to float64 for the comparison (never fed back)"),
    ('certification/randomized/executors.py', 'input_tensor', 'dtype= torch.float64'):
        (1, KERNEL, 'the midpoint rate (k + 1/2) / T computed exactly in float64 before the declared input cast'),
    ('certification/randomized/executors.py', 'per_neuron', 'cast .double()'):
        (1, RECORDER, "the randomized certificate's captures, widened exactly to float64 for the comparison (never fed back)"),
    ('certification/randomized/executors.py', 'per_neuron', 'dtype= torch.float64'):
        (1, RECORDER, "the randomized certificate's captures, widened exactly to float64 for the comparison (never fed back)"),
    ('certification/randomized/executors.py', 'run_nevresim_nodes.record', 'dtype= np.float64'):
        (1, RECORDER, "the randomized certificate's captures, widened exactly to float64 for the comparison (never fed back)"),
    ('certification/randomized/executors.py', 'run_nodes', 'cast .double()'):
        (3, RECORDER, "the randomized certificate's captures, widened exactly to float64 for the comparison (never fed back)"),
    ('certification/randomized/executors.py', 'run_nodes.record', 'cast .double()'):
        (1, RECORDER, "the randomized certificate's captures, widened exactly to float64 for the comparison (never fed back)"),
    ('chip_simulation/lava_loihi/core_lava.py', 'LavaCoreMixin._run_core_lava', 'dtype= np.float32'):
        (2, ENCODER, "pad and tail blocks of 0-or-1 input spikes, exact in float32, cast to the record's core dtype before the Lava process"),
    ('chip_simulation/lava_loihi/runner.py', 'LavaLoihiRunner.run._on_neural', 'dtype= np.float32'):
        (1, DEPARTURE, "Lava takes its segment inputs at float32 (the record's boundary buffers are float64); its encoder derives the input trains from them; measured value-equal on the gate programs (report, departures)"),
    ('chip_simulation/odin_fpga/segment.py', 'segment_record', 'dtype= np.float32'):
        (1, RECORDER, "the ODIN segment record's rate and count fields"),
    ('chip_simulation/sanafe/runner/neural_stage.py', 'SanafeNeuralStageMixin._run_neural_stage', 'cast .astype(np.float32)'):
        (2, ENCODER, "SANA-FE's input spike raster (0-or-1 values) cast through float32 for the simulator and back"),
    ('chip_simulation/sanafe/runner/neural_stage.py', 'SanafeNeuralStageMixin._run_neural_stage', 'cast .astype(np.float64)'):
        (2, ENCODER, "SANA-FE's input spike raster (0-or-1 values) cast through float32 for the simulator and back"),
    ('chip_simulation/sanafe/runner/neural_stage_record.py', 'SanafeNeuralStageRecordMixin._finalize_neural_stage_record', 'cast .astype(np.float32)'):
        (1, RECORDER, "the segment's input rates written to the SANA-FE stage record"),
    ('chip_simulation/simulation_runner/carry.py', 'assemble_input_train', 'dtype= np.float64'):
        (1, ENCODER, 'the host-encoded input train (0-or-1 spikes) a SpikeTrain-mode segment reads'),
    ('chip_simulation/simulation_runner/flat.py', 'SimulationFlatMixin._run_flat_mapping', 'dtype= np.float64'):
        (1, KERNEL, 'the membrane-readout decode (C2, lif_membrane_readout, no row of record arms it): counts plus membrane terms m_T/theta, exact in float64, read by argmax'),
    ('chip_simulation/simulation_runner/hybrid.py', 'SimulationHybridMixin._assemble_segment_input_np', 'dtype= np.float32'):
        (1, DEPARTURE, "nevresim takes its segment inputs at float32 (the record's boundary buffers are float64); its C++ generator reads llround(rate * T) per input, so a count-grid rate keeps its count; measured value-equal on the gate programs (report, departures)"),
    ('chip_simulation/simulation_runner/hybrid.py', 'SimulationHybridMixin._get_compute_op_output_size', 'dtype= torch.float32'):
        (2, NOT_DEPLOYED, 'shape inference: a dummy zero batch through one host op to read its output width, never a deployed value'),
    ('chip_simulation/simulation_runner/membrane_probe.py', 'stash_membrane_corrections', 'dtype= np.float64'):
        (1, KERNEL, 'the membrane-readout decode (C2, lif_membrane_readout, no row of record arms it): counts plus membrane terms m_T/theta, exact in float64, read by argmax'),
    ('chip_simulation/simulation_runner/segment_run.py', 'window_counts_from_records', 'dtype= np.float64'):
        (2, KERNEL, 'window counts from the record build: integer counts and the float32 segment input widened exactly to float64'),
    ('chip_simulation/ttfs/float_membrane.py', '_exact_window_membrane', 'cast .astype(np.float64)'):
        (2, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/float_membrane.py', '_exact_window_membrane', 'dtype= np.float64'):
        (1, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/float_membrane.py', 'float_membrane', 'dtype= np.float64'):
        (3, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/membrane_lattice.py', 'lattice_membrane', 'cast .astype(np.float64)'):
        (2, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/membrane_lattice.py', 'lattice_membrane', 'dtype= np.float64'):
        (3, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/segment_arrays.py', 'SegmentTtfsArrays.integral_neurons', 'dtype= np.float64'):
        (1, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/segment_arrays.py', 'SegmentTtfsArrays.integral_neurons._int', 'dtype= np.float64'):
        (1, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/segment_arrays.py', '_segment_ttfs_arrays_from_mapping_uncached', 'dtype= np.float64'):
        (5, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/ttfs_cycle_genuine.py', 'decode_spike_time_to_activation', 'cast .astype(np.float64)'):
        (1, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/ttfs_cycle_genuine.py', 'encode_activation_to_spike_time', 'dtype= np.float64'):
        (1, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/ttfs_cycle_genuine.py', 'run_ttfs_cycle_genuine_layers', 'dtype= np.float64'):
        (3, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/ttfs_cycle_genuine.py', 'ttfs_cycle_fire_step', 'dtype= np.float64'):
        (1, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/ttfs_encoding.py', 'ttfs_latched_spike_train', 'dtype= np.float64'):
        (2, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/ttfs_encoding.py', 'ttfs_single_spike_train', 'dtype= np.float64'):
        (2, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/ttfs_executor.py', 'TtfsAnalyticalExecutor.membrane_voltages', 'dtype= np.float64'):
        (1, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/ttfs_executor.py', 'TtfsAnalyticalExecutor.run_segment', 'cast .astype(np.float64)'):
        (1, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/ttfs_executor.py', 'TtfsAnalyticalExecutor.run_segment', 'dtype= np.float64'):
        (3, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/ttfs_executor.py', '_segment_record_from_neural_result', 'dtype= np.float64'):
        (2, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/ttfs_recorder.py', 'normalize_core_output_activation', 'dtype= np.float64'):
        (2, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/ttfs_segment.py', '_gather_segment_output', 'dtype= np.float64'):
        (1, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/ttfs_segment.py', '_run_ttfs_segment_ordered', 'cast .astype(np.float64)'):
        (1, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/ttfs_segment.py', '_run_ttfs_segment_ordered', 'dtype= np.float64'):
        (3, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('chip_simulation/ttfs/ttfs_segment.py', 'gather_segment_ttfs_output_from_cores', 'dtype= np.float64'):
        (3, KERNEL, "the analytical TTFS executor's closed form (real-valued membranes, spike times and their encodings) at float64 between the record's float64 boundary buffers"),
    ('mapping/ir/types.py', 'NeuralCore.execute', 'dtype= torch.float32'):
        (3, NOT_DEPLOYED, "the IR's own float32 interpreter of a core (tests only: its ReLU and clamp are no deployed core's), pinned by tests/unit/mapping/test_onchip_attention.py"),
    ('mapping/pruning/liveness_transfer/constant_core.py', 'derive_core_constants', 'cast .astype(np.float64)'):
        (1, KERNEL, 'the constant lattice holds Python floats (float64): constants read from and written back to it, and the carriability check against them'),
    ('mapping/pruning/liveness_transfer/constant_core.py', 'derive_core_constants', 'dtype= np.float64'):
        (1, KERNEL, 'the constant lattice holds Python floats (float64): constants read from and written back to it, and the carriability check against them'),
    ('mapping/pruning/liveness_transfer/constant_transfer.py', '_derive_constant_outputs', 'cast .to(torch.float64)'):
        (2, KERNEL, 'the constant lattice holds Python floats (float64): constants read from and written back to it, and the carriability check against them'),
    ('mapping/pruning/liveness_transfer/constant_transfer.py', '_derive_constant_outputs', 'dtype= np.float64'):
        (1, KERNEL, 'the constant lattice holds Python floats (float64): constants read from and written back to it, and the carriability check against them'),
    ('mapping/pruning/liveness_transfer/constant_transfer.py', '_run', 'dtype= np.float64'):
        (1, KERNEL, 'the constant lattice holds Python floats (float64): constants read from and written back to it, and the carriability check against them'),
    ('mapping/pruning/liveness_transfer/transfer_registry.py', '_derive_index_bijection', 'dtype= torch.float64'):
        (1, NOT_DEPLOYED, 'the index-bijection probe of a liveness transfer: a structural analysis (which input reaches which output), no deployed value'),
    ('models/spiking/hybrid/membrane_readout.py', 'apply_membrane_corrections_numpy', 'dtype= np.float64'):
        (2, KERNEL, 'the membrane-readout decode (C2, lif_membrane_readout, no row of record arms it): counts plus membrane terms m_T/theta, exact in float64, read by argmax'),
    ('models/spiking/hybrid/rate_forward.py', 'HybridRateForwardMixin._forward_rate._on_compute_rate', 'cast .to(torch.float32)'):
        (1, RECORDER, 'the compute_outputs recorder capture (verify M-2: rate_forward.py:223), read by tests and records only'),
    ('models/spiking/hybrid/rate_forward.py', 'HybridRateForwardMixin._forward_rate._on_neural_rate', 'cast .to(torch.float32)'):
        (1, DEPARTURE, "the HCM recorder's segment input rates at float32 (the record's boundary is float64): Lava's twin reads them (run_segments_from_reference, the Loihi step) to re-encode its inputs; the Loihi spike-parity certificate compares Lava's counts with the HCM's exactly and refuses on any difference, so a float32 rounding that moved a spike refuses by name, never moves a row (sd18 gate: parity passes on BASE and the unit)"),
    ('pipelining/core/gates/value_gates.py', '_assert_aq_grid_parity', 'cast .double()'):
        (1, RECORDER, "the AQ R-edge's decision-agreement fraction (a metric of the gate)"),
    ('spiking/count_grid.py', 'call_in_fp64', 'cast .to(torch.float64)'):
        (2, KERNEL, 'the count-grid twin (verify O-1): a raw-gauge buffer decoded to exact window counts and the op evaluated on them in float64 (call_in_fp64)'),
    ('spiking/count_grid.py', 'decode_window_counts', 'cast .to(torch.float64)'):
        (1, KERNEL, 'the count-grid twin (verify O-1): a raw-gauge buffer decoded to exact window counts and the op evaluated on them in float64 (call_in_fp64)'),
    ('spiking/segment_boundary.py', 'boundary_normalization_scales', 'dtype= np.float64'):
        (2, KERNEL, "the wire divisors, computed in float64; each use casts them to its buffer's dtype"),
    ('torch_mapping/value_parity/comparison.py', 'relative_error', 'cast .double()'):
        (2, NOT_DEPLOYED, "the value parity's relative-error metric"),
    ('torch_mapping/value_parity/witness.py', '_uniform', 'dtype= torch.float64'):
        (1, NOT_DEPLOYED, "the value parity witness: the model's own float64 twin and its seeded probe inputs (an analysis's exact twin)"),
    ('torch_mapping/value_parity/witness.py', 'fp64_program', 'cast .to(torch.float64)'):
        (1, NOT_DEPLOYED, "the value parity witness: the model's own float64 twin and its seeded probe inputs (an analysis's exact twin)"),
    ('torch_mapping/value_parity/witness.py', 'fp64_program', 'dtype= torch.float64'):
        (1, NOT_DEPLOYED, "the value parity witness: the model's own float64 twin and its seeded probe inputs (an analysis's exact twin)"),
    ('torch_mapping/value_parity/witness.py', 'fp64_snapshot', 'cast .to(torch.float64)'):
        (1, NOT_DEPLOYED, "the value parity witness: the model's own float64 twin and its seeded probe inputs (an analysis's exact twin)"),
    ('torch_mapping/value_parity/witness.py', 'probe_inputs', 'dtype= torch.float64'):
        (4, NOT_DEPLOYED, "the value parity witness: the model's own float64 twin and its seeded probe inputs (an analysis's exact twin)"),
    ('torch_mapping/value_parity/witness.py', 'scaled_probe_inputs', 'dtype= torch.float64'):
        (2, NOT_DEPLOYED, "the value parity witness: the model's own float64 twin and its seeded probe inputs (an analysis's exact twin)"),
}
