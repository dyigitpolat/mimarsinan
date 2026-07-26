"""ValueHybridCoreFlow: run a HybridHardCoreMapping in the value domain."""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    assemble_segment_input_torch,
    execute_compute_op_torch,
    gather_final_output_torch,
    store_segment_output_torch,
)
from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import run_hybrid_stages
from mimarsinan.chip_simulation.value_run.value_execution import (
    run_neural_segment_values,
)

_RAW_INPUT_NODE_ID = -2


class ValueHybridCoreFlow(nn.Module):
    """The value-domain deployed program: affine core segments + host ComputeOps.

    ``stage_count_recorder`` is the shared observable seam (node-granular value
    captures via ``certification.count_alignment.flow_node_counts``); the
    ``lif_execution_synchronized`` attribute is accepted and ignored so the
    capture helpers work verbatim.
    """

    def __init__(self, hybrid_mapping, device="cpu",
                 dtype: torch.dtype = torch.float32,
                 activation_bits: "int | None" = None):
        super().__init__()
        self.hybrid_mapping = hybrid_mapping
        self.execution_device = torch.device(device)
        self.value_dtype = dtype
        self.activation_bits = activation_bits
        self.stage_count_recorder = None
        self.lif_execution_synchronized = False
        self._fp64_ops: Dict[int, nn.Module] = {}

    def _run_compute_stage(self, op, x_flat, state_buffer):
        original = op.params.get("module")
        if self.value_dtype == torch.float64 and original is not None:
            # Host modules are stored fp32; certification runs need the whole
            # program in fp64, so the op briefly executes a double twin.
            double = self._fp64_ops.get(int(op.id))
            if double is None:
                double = copy.deepcopy(original).double()
                self._fp64_ops[int(op.id)] = double
            op.params["module"] = double
            try:
                return execute_compute_op_torch(
                    op, x_flat, state_buffer,
                    in_scale=1.0, out_scale=1.0, output_dtype=self.value_dtype,
                    gather_dtype=self.value_dtype,
                )
            finally:
                op.params["module"] = original
        return execute_compute_op_torch(
            op, x_flat, state_buffer,
            in_scale=1.0, out_scale=1.0, output_dtype=self.value_dtype,
            gather_dtype=self.value_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x_flat = x.reshape(batch, -1).to(self.execution_device, self.value_dtype)
        state_buffer: Dict[int, torch.Tensor] = {_RAW_INPUT_NODE_ID: x_flat}

        def on_neural(_index, stage, buf):
            seg_input = assemble_segment_input_torch(
                stage.input_map, buf, batch, self.execution_device, self.value_dtype
            )
            seg_output = run_neural_segment_values(
                stage.hard_core_mapping, seg_input,
                activation_bits=self.activation_bits,
            )
            recorder = self.stage_count_recorder
            if recorder is not None:
                recorder(stage, seg_output)
            store_segment_output_torch(stage.output_map, buf, seg_output)

        def on_compute(_index, stage, buf):
            op = stage.compute_op
            buf[int(op.id)] = self._run_compute_stage(op, x_flat, buf)

        run_hybrid_stages(
            self.hybrid_mapping, state_buffer,
            on_neural=on_neural, on_compute=on_compute,
        )
        return gather_final_output_torch(
            self.hybrid_mapping.output_sources, state_buffer, x_flat,
            batch, self.execution_device, self.value_dtype,
        )
