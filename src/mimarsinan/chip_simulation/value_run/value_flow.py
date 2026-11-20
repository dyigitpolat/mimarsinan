"""ValueHybridCoreFlow: run a HybridHardCoreMapping in the value domain."""

from __future__ import annotations

import time
from typing import Dict

import torch
import torch.nn as nn

from mimarsinan.mapping.support.wavefront_stages import SCHEDULE_CORE_ORDINALS
from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    assemble_segment_input_torch,
    decref_consumers,
    decref_op_consumers,
    gather_final_output_torch,
    input_slot_runs,
    scatter_rows,
    store_segment_output_torch,
)
from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import run_hybrid_stages
from mimarsinan.chip_simulation.value_run import operand_payloads
from mimarsinan.chip_simulation.value_run.host_ops import execute_host_op
from mimarsinan.chip_simulation.value_run.value_execution import (
    ValueSegmentScope,
    run_neural_segment_values,
)
from mimarsinan.mapping.ir.dynamic_bank import stage_payload_sources
from mimarsinan.mapping.packing.executed_reads import consumer_counts


def _consumer_counts(hybrid_mapping) -> Dict[int, int]:
    """{node_id: downstream reads} for state-buffer pruning: ``executed_reads.consumer_counts`` (every stage's slices,
    host-op sources, program outputs and N2's payload writes), memoized as the spiking flows memoize it."""
    cached = getattr(hybrid_mapping, "_consumer_counts_cache", None)
    if cached is not None:
        return cached
    counts = consumer_counts(hybrid_mapping)
    hybrid_mapping._consumer_counts_cache = counts
    return counts


def _payload_sources(hybrid_mapping, stage) -> list:
    return stage_payload_sources(stage, getattr(hybrid_mapping, "dynamic_bank_bindings", None) or {})


_RAW_INPUT_NODE_ID = -2


class ValueHybridCoreFlow(nn.Module):
    """The value-domain deployed program: affine core segments + host ComputeOps.

    ``stage_count_recorder`` is the shared observable seam (node-granular value
    captures via ``certification.count_alignment.flow_node_counts``); the
    ``lif_execution_synchronized`` attribute is accepted and ignored so the
    capture helpers work verbatim.

    [U-DAR] ``device`` and ``dtype`` are declared, never defaulted: the deployed
    program's are the record's (``VALUE_ARITHMETIC.device``, ``.core_dtype``:
    the CPU, float64); any other dtype is a declared counterfactual executor.
    """

    def __init__(self, hybrid_mapping, *, device, dtype: torch.dtype):
        super().__init__()
        self.hybrid_mapping = hybrid_mapping
        self.execution_device = torch.device(device)
        self.value_dtype = dtype
        self.stage_count_recorder = None
        # N2: called (stage, core index, operand block, programmed payload) for every operand block written.
        self.payload_recorder = None
        self.lif_execution_synchronized = False
        self._fp64_ops: Dict[int, nn.Module] = {}
        # Static gather plans, shared across forwards (weights are NOT:
        # they stay chain-scoped, [wsm V3]).
        self._plan_cache: Dict = {}

    def _run_compute_stage(self, op, x_flat, state_buffer):
        # Host modules are stored fp32; the deployed value program runs whole in
        # the record's float64, so the op briefly executes its float64 twin.
        return execute_host_op(
            op, x_flat, state_buffer, dtype=self.value_dtype, twins=self._fp64_ops,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._by_input_slot(x, retain_state=False)[0]

    def run_retaining_state(self, x: torch.Tensor):
        """``(output, state buffer of every node, flat input)``: the forward without state-buffer pruning."""
        return self._by_input_slot(x, retain_state=True)

    def _by_input_slot(self, x: torch.Tensor, *, retain_state: bool):
        """[thru:w2] A program of input slots runs once per slot on its rows (``input_slot_runs``), the output, the
        retained state and the flat input put back in batch order; any other program runs once."""
        runs = input_slot_runs(self.hybrid_mapping.stages, int(x.shape[0]))
        if runs == [(None, None)]:
            return self._execute(x, retain_state=retain_state)
        batch = int(x.shape[0])
        parts = [(rows, self._execute(x[rows], retain_state=retain_state, input_slot=b)) for b, rows in runs]
        output = scatter_rows([(rows, run[0]) for rows, run in parts], batch)
        if not retain_state:
            return output, None, None
        state = {node: scatter_rows([(rows, run[1][node]) for rows, run in parts], batch) for node in parts[0][1][1]}
        return output, state, scatter_rows([(rows, run[2]) for rows, run in parts], batch)

    def _execute(self, x: torch.Tensor, *, retain_state: bool, input_slot=None):
        batch = x.shape[0]
        x_flat = x.reshape(batch, -1).to(self.execution_device, self.value_dtype)
        state_buffer: Dict[int, torch.Tensor] = {_RAW_INPUT_NODE_ID: x_flat}
        # [F1] refcounted state-buffer pruning (the spiking flows' discipline):
        # without it every node's activations live until the forward ends.
        remaining = {} if retain_state else dict(_consumer_counts(self.hybrid_mapping))
        # [wsm V3] forward-local scope: weight tensors live per residency
        # chain — a non-resident stage heads a new chain (freeing the previous
        # one) and resident passes alias the head's uploaded tensors.
        scope = ValueSegmentScope(plan_cache=self._plan_cache)
        spent = {"assemble": 0.0, "neural": 0.0, "store": 0.0, "compute": 0.0}
        t_fwd = time.perf_counter()

        def on_neural(_index, stage, buf):
            if not getattr(stage, "schedule_weights_resident", False):
                scope.begin_chain(stage.hard_core_mapping)
            assert scope.head_hcm is not None, (
                "schedule_weights_resident stage arrived before any chain head"
            )
            _t = time.perf_counter()
            seg_input = assemble_segment_input_torch(
                stage.input_map, buf, batch, self.execution_device, self.value_dtype
            )
            spent["assemble"] += time.perf_counter() - _t
            _t = time.perf_counter()
            bindings = getattr(self.hybrid_mapping, "dynamic_bank_bindings", None)
            payloads = operand_payloads.materialize(stage, buf, bindings) if bindings else None
            payload_hook = self.payload_recorder
            seg_output = run_neural_segment_values(
                stage.hard_core_mapping, seg_input,
                resident_head=scope.head_hcm, scope=scope, payloads=payloads, bindings=bindings,
                ordinals=getattr(stage, SCHEDULE_CORE_ORDINALS, None),
                payload_recorder=None if payload_hook is None
                else (lambda *args, s=stage, hook=payload_hook: hook(s, *args)),
            )
            spent["neural"] += time.perf_counter() - _t
            recorder = self.stage_count_recorder
            if recorder is not None:
                recorder(stage, seg_output)
            _t = time.perf_counter()
            store_segment_output_torch(stage.output_map, buf, seg_output)
            spent["store"] += time.perf_counter() - _t
            decref_consumers(
                buf, remaining,
                (int(s.node_id) for s in stage.input_map
                 if s.node_id is not None and int(s.node_id) >= 0),
            )
            payload_sources = _payload_sources(self.hybrid_mapping, stage) if bindings else []
            if payload_sources:
                decref_consumers(buf, remaining, payload_sources)

        def on_compute(_index, stage, buf):
            op = stage.compute_op
            _t = time.perf_counter()
            buf[int(op.id)] = self._run_compute_stage(op, x_flat, buf)
            spent["compute"] += time.perf_counter() - _t
            decref_op_consumers(buf, remaining, op)

        def on_other_slot(_index, stage, _buf):
            # [thru:w2] another input slot's stage: the weights it loads stay loaded for this slot's resident stages
            if stage.kind == "neural" and not getattr(stage, "schedule_weights_resident", False):
                scope.begin_chain(stage.hard_core_mapping)

        run_hybrid_stages(
            self.hybrid_mapping, state_buffer,
            on_neural=on_neural, on_compute=on_compute,
            input_slot=input_slot, on_other_slot=on_other_slot,
        )
        _wall = time.perf_counter() - t_fwd
        if _wall > 5.0:   # slow forwards only; tests stay silent
            print(f"[ValueFlow] forward={_wall:.1f}s batch={batch} "
                  f"assemble={spent['assemble']:.1f}s neural={spent['neural']:.1f}s "
                  f"store={spent['store']:.1f}s compute={spent['compute']:.1f}s",
                  flush=True)
        output = gather_final_output_torch(
            self.hybrid_mapping.output_sources, state_buffer, x_flat,
            batch, self.execution_device, self.value_dtype,
        )
        return output, state_buffer, x_flat
