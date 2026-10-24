"""Value-domain neural-segment kernel: one affine pass (or integer law) per core, dependency order."""

from __future__ import annotations

import heapq
from typing import Any, Dict, Iterable, List, Sequence

import torch

from mimarsinan.chip_simulation.value_run.operand_blocks import fill_operand_blocks, operand_blocks, static_part
from mimarsinan.chip_simulation.value_run.value_law import prepare_core_law, run_core_law
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore.matrix_placement import (
    same_core_matrix_payloads,
)
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
from mimarsinan.models.nn.activations.value_quantizer import quantize_to_value_grid
from mimarsinan.models.spiking.signal_spans import SpanFillPlan


class ValueWalkOrderError(ToolchainRefusal, RuntimeError):
    """A segment whose cores read each other in a cycle, so no walk order exists."""


def dependency_order(sources: Sequence[Iterable[int]], latencies: Sequence[int]) -> List[int]:
    """The walk order of a segment's cores: every core after the cores it reads (``sources[i]``), ties broken
    by stored latency then index (Kahn's algorithm on a ``(latency, index)`` heap).

    Equal to the latency order wherever that order is already topological (the smallest key is always ready),
    so a pass-local latency written on the shared mapping can never reorder a walk past a producer (U1 rule R6).
    Structure-agnostic, so every walk over a segment's cores can read this one contract.
    """
    reads = [{int(s) for s in core_sources} for core_sources in sources]
    readers: List[List[int]] = [[] for _ in reads]
    for consumer, producers in enumerate(reads):
        for producer in producers:
            readers[producer].append(consumer)
    pending = [len(producers) for producers in reads]
    ready = [(int(latencies[i]), i) for i in range(len(reads)) if pending[i] == 0]
    heapq.heapify(ready)
    order: List[int] = []
    while ready:
        _latency, index = heapq.heappop(ready)
        order.append(index)
        for consumer in readers[index]:
            pending[consumer] -= 1
            if pending[consumer] == 0:
                heapq.heappush(ready, (int(latencies[consumer]), consumer))
    if len(order) != len(reads):
        raise ValueWalkOrderError(f"{len(reads) - len(order)} cores of a segment read each other in a cycle")
    return order


def segment_walk_order(hcm) -> List[int]:
    """The value walk's order over a packed segment: :func:`dependency_order` on its core-to-core reads."""
    sources = [{int(s.src_core) for s in core.get_axon_source_spans() if s.kind == "core"} for core in hcm.cores]
    return dependency_order(sources, [int(core.latency or 0) for core in hcm.cores])


class _PreparedValueSegment:
    """Per-(mapping, dtype, device) tensors: weights, biases, gather plans, order.

    ``resident_from`` aliases the residency-chain head's uploaded weight and
    bias tensors ([wsm V3]: ``schedule_weights_resident`` passes reuse the
    programmed bank content per ordinal — verified at build by the
    placement-geometry SUBSET check — so re-upload is pure waste)."""

    def __init__(
        self, hcm, device: torch.device, dtype: torch.dtype,
        resident_from: "_PreparedValueSegment | None" = None,
        upload_memo: "dict | None" = None,
        plan_cache: "dict | None" = None,
    ) -> None:
        ensure_core_latencies(hcm)
        self.order: List[int] = segment_walk_order(hcm)
        if resident_from is not None:
            assert len(hcm.cores) <= len(resident_from.weights)
            self.weights = resident_from.weights[: len(hcm.cores)]
            self.biases = resident_from.biases[: len(hcm.cores)]
        else:
            self.weights = [
                _upload(core, dtype, device, upload_memo)
                for core in hcm.cores
            ]
            self.biases = [
                None if (bias := getattr(core, "hardware_bias", None)) is None
                else torch.as_tensor(bias, dtype=dtype, device=device).reshape(1, -1)
                for core in hcm.cores
            ]
        self.thresholds = []
        self.plans = []
        self.entry_transforms = []
        # Gather plans are static index tensors, pure in (core spans, device):
        # rebuilding them per forward cost 18.4 s of a 26.3 s preparation
        # against 0.3 s of matmuls. Weights stay chain-scoped ([wsm V3]).
        plans = plan_cache if plan_cache is not None else {}
        for core in hcm.cores:
            self.thresholds.append(float(core.threshold))
            key = (id(core), str(device))
            plan = plans.get(key)
            if plan is None:
                plan = SpanFillPlan(core.get_axon_source_spans(), device)
                plans[key] = plan
            self.plans.append(plan)
            # [mvm AQ] the boundary grid snaps ENTRY cores only (no upstream
            # core sources); the plan applies it to the columns it owns.
            grid = getattr(core, "boundary_grid", None)
            armed = (
                grid if (grid is not None and grid.armed
                         and not plan.has_upstream_core_sources) else None
            )
            self.entry_transforms.append(
                None if armed is None
                else (lambda t, g=armed: quantize_to_value_grid(t, g.scale, g.bits))
            )
        out_key = (id(hcm), "outputs", str(device))
        output_plan = plans.get(out_key)
        if output_plan is None:
            spans = (
                hcm.get_output_source_spans()
                if hasattr(hcm, "get_output_source_spans")
                else compress_spike_sources(list(hcm.output_sources.flatten()))
            )
            output_plan = SpanFillPlan(spans, device)
            plans[out_key] = output_plan
        self.output_plan = output_plan
        self.output_size = int(len(hcm.output_sources.flatten()))
        self.axon_counts = [int(w.shape[0]) for w in self.weights]
        # [E283] law-carrying cores run the integer law in fp64 on their codes.
        self.laws = [
            prepare_core_law(core, weight, index)
            for index, (core, weight) in enumerate(zip(hcm.cores, self.weights))
        ]
        # [N2] per core its operand blocks and its static grid without them (filled on first use).
        self.operand: "List[list] | None" = None
        self.operand_static: List[Any] = []


def ensure_core_latencies(hcm) -> None:
    """Latency tiers break the value walk's dependency-order ties."""
    if any(core.latency is None for core in hcm.cores):
        ChipLatency(hcm).calculate()


def _upload(core, dtype, device, memo: "dict | None"):
    """Device tensor for the core's weight grid; cores resolving to
    byte-identical grids share one tensor ([F2] — content-keyed, so cores
    whose padded grid is now a transient composite still upload once).
    The memo retains the payloads and re-checks them by identity, so a
    freed-and-reallocated array can never alias through a stale key."""
    if memo is None:
        return torch.as_tensor(core.get_core_matrix(), dtype=dtype, device=device)
    key = (core.core_matrix_key(), dtype, str(device))
    payloads = core.core_matrix_payloads()
    hit = memo.get(key)
    if hit is not None and same_core_matrix_payloads(hit[0], payloads):
        return hit[1]
    tensor = torch.as_tensor(core.get_core_matrix(), dtype=dtype, device=device)
    memo[key] = (payloads, tensor)
    return tensor


class ValueSegmentScope:
    """Forward-local owner of uploaded weight/bias tensor lifetime.

    The lifetime unit is the residency CHAIN — one non-resident head stage
    plus its consecutive ``schedule_weights_resident`` passes ([wsm V3]):
    ``begin_chain`` frees the previous chain, and the scope dying with the
    forward frees the last one, so peak residency is one chain, never the
    sum over segments."""

    def __init__(self, plan_cache: "Dict | None" = None) -> None:
        self.head_hcm: Any = None
        # Static gather plans outlive the forward; weights do not.
        self.plan_cache: Dict = plan_cache if plan_cache is not None else {}
        self._prepared: Dict = {}
        # [F2] content-keyed weight-upload memo: cores resolving to one
        # byte-identical grid share ONE device tensor within the chain.
        self._upload_memo: Dict = {}

    def begin_chain(self, hcm) -> None:
        """Drop the previous chain's tensors BEFORE the new head uploads."""
        self._prepared.clear()
        self._upload_memo.clear()
        self.head_hcm = hcm

    def prepared(
        self, hcm, device: torch.device, dtype: torch.dtype, resident_head=None,
    ) -> _PreparedValueSegment:
        by_key = self._prepared.setdefault(hcm, {})
        key = (str(device), dtype)
        prepared = by_key.get(key)
        if prepared is None:
            head = (
                None if resident_head is None or resident_head is hcm
                else self.prepared(resident_head, device, dtype)
            )
            prepared = _PreparedValueSegment(
                hcm, device, dtype, resident_from=head,
                upload_memo=self._upload_memo, plan_cache=self.plan_cache,
            )
            by_key[key] = prepared
        return prepared


def _operand_cores(prepared: _PreparedValueSegment, hcm, bindings) -> List[list]:
    """Per core its operand blocks (N2), the static grids cleared of the banks' placeholders, cached per segment."""
    cached = prepared.operand
    if cached is None:
        cached = operand_blocks(hcm, bindings)
        prepared.operand_static = [None if not blocks else static_part(prepared.weights[i], blocks)
                                   for i, blocks in enumerate(cached)]
        prepared.operand = cached
    return cached


def run_neural_segment_values(
    hcm, seg_input: torch.Tensor, resident_head=None,
    scope: "ValueSegmentScope | None" = None, payloads=None, bindings=None, payload_recorder=None,
) -> torch.Tensor:
    """Execute one packed segment in the value domain: y_core = (x @ W + b) / theta.

    ``theta`` is the weight-quantization dequant scale stamped by
    ``quantize_ir_graph`` (1.0 on float programs); always-on axons read 1.0
    (the param-encoded bias row); fused cores execute as one wide dot product.
    Entry cores carrying a ``boundary_grid`` snap their input-sourced
    columns onto it with the same formula the model-side
    ``ValueGridQuantizer`` applies. A core carrying a ``readout_law`` runs
    that law instead, in fp64 on its input codes (E283).
    """
    device, dtype = seg_input.device, seg_input.dtype
    if scope is None:
        scope = ValueSegmentScope()  # standalone call: single-call lifetime
    prepared = scope.prepared(hcm, device, dtype, resident_head=resident_head)
    batch = seg_input.shape[0]

    operand = _operand_cores(prepared, hcm, bindings) if bindings else None
    buffers: Dict[int, torch.Tensor] = {}
    for idx in prepared.order:
        signal = torch.empty(
            batch, prepared.axon_counts[idx], device=device, dtype=dtype
        )
        prepared.plans[idx].apply(
            signal, input_spikes=seg_input, buffers=buffers, on_value=1.0,
            input_transform=prepared.entry_transforms[idx],
        )
        weight = None
        if operand is not None and operand[idx]:
            weight = fill_operand_blocks(prepared.operand_static[idx], operand[idx], payloads or {})
            if payload_recorder is not None:
                for block in operand[idx]:
                    payload_recorder(idx, block, weight[:, block.axon_offset: block.axon_offset + block.axons,
                                                       block.neuron_offset: block.neuron_offset + block.neurons])
        law = prepared.laws[idx]
        if law is not None:
            buffers[idx] = run_core_law(signal, law, weight).to(dtype)
            continue
        out = signal @ prepared.weights[idx] if weight is None else torch.bmm(signal.unsqueeze(1), weight).squeeze(1)
        bias = prepared.biases[idx]
        if bias is not None:
            out = out + bias
        theta = prepared.thresholds[idx]
        if theta != 1.0:
            out = out / theta
        buffers[idx] = out

    seg_output = torch.empty(
        batch, prepared.output_size, device=device, dtype=dtype
    )
    prepared.output_plan.apply(
        seg_output, input_spikes=seg_input, buffers=buffers, on_value=1.0
    )
    return seg_output
