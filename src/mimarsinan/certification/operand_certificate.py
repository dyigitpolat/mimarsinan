"""N2's operand certificate (design §3.2): every edge of a program with dynamic operand banks at atol 0, the
payload edge over every written cell within its payload width, and a nonzero census that keeps it non-vacuous."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from mimarsinan.certification.count_alignment import flow_node_counts
from mimarsinan.certification.host_condition import OUTSIDE_SCOPE, HostConditionCertificate, host_kernel_replay
from mimarsinan.chip_simulation.deployed_host import DEPLOYED_HOST_DEVICE, DEPLOYED_VALUE_DTYPE
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import GoldenTrace, run_golden
from mimarsinan.chip_simulation.value_run.operand_blocks import operand_blocks
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.capability.op_kinds import op_kind
from mimarsinan.mapping.support.host_kernel_evaluation import forward_of, twin_as_program
from mimarsinan.models.nn.activations.integer_law import grid_levels

#: Samples the whole golden replays alone against their place in the batch (the batch-of-one edge: a structural spot
#: check of the first two; the whole golden costs about 11 s per call on ViT-Lite-7/4, fixed overhead, so every sample
#: of a 512-input certificate would add about 1.6 h).
BATCH_OF_ONE_SAMPLES = 2
BATCH_OF_ONE = "batch of one (golden alone/in the batch)"
#: [W49] The host-kernel edge: every sampled input alone through every host kernel the golden runs, and the batch
#: once more (``certification.host_condition``), at atol 0: Theorem 1's host condition on this program.
HOST_ALONE = "host kernels alone (every sample)"


class OperandCertificateError(ToolchainRefusal, ValueError):
    """A program the operand certificate does not describe: one without dynamic operand banks."""


@dataclass(frozen=True)
class PayloadCellCertificate:
    """N2's payload edge: every cell the executor programs into an operand core against the golden's codes, each
    within its bank's ``payload_bits`` (``beyond_width`` of the mismatches lie outside it)."""

    samples: int
    banks: int
    cells_compared: int
    mismatched: int
    nonzero_payload_codes: int
    passed: bool
    beyond_width: int = 0


def certify_payload_cells(flow, golden_program, samples: torch.Tensor) -> PayloadCellCertificate:
    """Run ``flow`` with its payload recorder on and compare each programmed operand block, per sample, with the
    payload codes the int64 golden recovers on ``golden_program``; a payload cached across samples fails here, and
    so does a cell beyond its bank's ``payload_bits`` grid."""
    trace = GoldenTrace()
    run_golden(golden_program, samples.detach().cpu().double().numpy(), trace=trace)
    bindings = golden_program.dynamic_bank_bindings
    tally = {"compared": 0, "mismatched": 0, "beyond_width": 0}
    banks = set()

    def record(_stage, _core, block, programmed):
        want = trace.payloads[block.bank][:, slice(*block.bank_axons), slice(*block.bank_neurons)]
        got = programmed.detach().cpu().double().numpy()
        wide = np.abs(got) > grid_levels(bindings[block.bank].payload_bits)
        tally["compared"] += int(got.size)
        tally["mismatched"] += int(np.count_nonzero((got != want) | wide))
        tally["beyond_width"] += int(np.count_nonzero(wide))
        banks.add(block.bank)

    flow.payload_recorder = record
    try:
        with torch.no_grad():
            flow(samples)
    finally:
        flow.payload_recorder = None
    nonzero = int(sum(np.count_nonzero(codes) for codes in trace.payloads.values()))
    compared, mismatched = tally["compared"], tally["mismatched"]
    return PayloadCellCertificate(samples=int(samples.shape[0]), banks=len(banks), cells_compared=compared,
                                  mismatched=mismatched, nonzero_payload_codes=nonzero,
                                  passed=compared > 0 and mismatched == 0 and nonzero > 0,
                                  beyond_width=tally["beyond_width"])


@dataclass(frozen=True)
class EdgeResult:
    name: str
    compared: int
    mismatched: int
    max_abs_delta: float


def _edge(name: str, got, want) -> EdgeResult:
    got = torch.as_tensor(np.asarray(got, dtype=np.float64)).reshape(-1)
    want = torch.as_tensor(np.asarray(want, dtype=np.float64)).reshape(-1)
    if got.shape != want.shape:
        return EdgeResult(name, int(want.numel()), int(want.numel()), float("inf"))
    delta = (got - want).abs()  # a NaN on either side is a mismatch: equality, not a positive difference
    return EdgeResult(name, int(want.numel()), int((~(got == want)).sum()),
                      float(delta.max()) if delta.numel() else 0.0)


@dataclass(frozen=True)
class OperandProgramCertificate:
    """N2 design §3.2 at atol 0: R (NF / SCM), C (SCM / HCM, per neuron window), golden on both programs, the host
    ops replayed bitwise segment by segment, the golden alone against the batch, the payload edge, and the nonzero
    census that keeps it non-vacuous (payload codes, operand-core outputs, host-op outputs by op kind, outputs)."""

    edges: tuple
    payload: PayloadCellCertificate
    nonzero_census: dict
    passed: bool
    host_condition: Optional[HostConditionCertificate] = None

    def summary(self) -> str:
        parts = [f"{e.name}: {e.mismatched}/{e.compared}" for e in self.edges]
        parts.append(f"payload: {self.payload.mismatched}/{self.payload.cells_compared}")
        host = f"; host condition {self.host_condition.verdict()}" if self.host_condition is not None else ""
        return "; ".join(parts) + f"; nonzero {self.nonzero_census}{host}; passed={self.passed}"


def _cat(parts) -> np.ndarray:
    return np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in parts]) if parts else np.zeros(0)


def _golden_node_values(identity_mapping, trace: GoldenTrace) -> dict:
    """{IR core id: the golden's output values} of every core of the identity program (one hard core each)."""
    out = {}
    for stage in identity_mapping.stages:
        if stage.kind != "neural":
            continue
        segment = stage.hard_core_mapping
        for core, placements in zip(segment.cores, segment.soft_core_placements_per_hard_core):
            codes = trace.core_codes.get(id(core))
            if codes is not None and placements:
                out[int(placements[0]["ir_node_id"])] = codes * np.asarray(core.readout_law.output_step)
    return out


def _flow_outputs(mapping, samples):
    flow = ValueHybridCoreFlow(mapping, device=DEPLOYED_HOST_DEVICE, dtype=DEPLOYED_VALUE_DTYPE)  # [U-HDA]
    with torch.no_grad():
        out, state, _flat = flow.run_retaining_state(samples)
    return flow, out, state


def _batch_of_one_edge(identity_mapping, samples: torch.Tensor, codes: np.ndarray, trace: GoldenTrace) -> EdgeResult:
    """The golden on the first ``BATCH_OF_ONE_SAMPLES`` samples alone against the same samples in the batch, every
    output code and host value: a structural spot check of design §3.2's per-sample replay. Every sample alone through
    the host kernels is the host-kernel edge (``HOST_ALONE``), the part a batch-dependent host op moves."""
    got, want = [], []
    for i in range(min(BATCH_OF_ONE_SAMPLES, int(samples.shape[0]))):
        alone = GoldenTrace()
        alone_codes, _steps = run_golden(identity_mapping, samples[i:i + 1].numpy(), trace=alone)
        got.append(alone_codes)
        want.append(codes[i])
        for op_id, values in alone.host_values.items():
            got.append(values)
            want.append(trace.host_values[op_id][i])
    return _edge(BATCH_OF_ONE, _cat(got), _cat(want))


def _host_alone_edge(host_condition) -> EdgeResult:
    """The host-kernel edge from the replay's certificate: every output element of every host kernel call compared,
    an element mismatched where it differed alone or on the repeat; one device and dtype per kernel required."""
    kernels = host_condition.kernels
    compared = sum(k.elements for k in kernels)
    mismatched = sum(max(k.batch_dependent, k.nondeterministic) for k in kernels)
    mismatched += sum(k.elements for k in host_condition.several_placements)
    return EdgeResult(HOST_ALONE, compared, mismatched, max((k.max_abs_delta for k in kernels), default=0.0))


def judged_edges(edges, host_condition: HostConditionCertificate) -> tuple:
    """[W49] The edges whose mismatches fail the program certificate. One rule for the two edges that read the host
    condition (a sample's host values a function of that sample alone): ``HOST_ALONE`` is the condition's own reading,
    never a failure of the program, carried by name (``OperandProgramCertificate.host_condition``, and Hard Core
    Mapping's ``host_condition`` record, whose replay sees the same calls). ``BATCH_OF_ONE`` fails the program unless
    the replay names its cause: where a host kernel breaks the condition (status ``outside_scope``: batch-dependent,
    nondeterministic, or on several devices or dtypes), its mismatch is that condition's reading, carried by name;
    where none does, the mismatch is a defect of the program (batch dependence outside every host kernel) and fails
    it, as before the host-kernel replay."""
    carried = {HOST_ALONE} | ({BATCH_OF_ONE} if host_condition.status == OUTSIDE_SCOPE else set())
    return tuple(e for e in edges if e.name not in carried)


def certify_operand_program(nf_model, identity_mapping, packed_mapping, samples: torch.Tensor
                            ) -> OperandProgramCertificate:
    """Every edge of §3.2 on ``samples`` (fp64): exact codes, exact host values, exact payload cells. A program
    without dynamic operand banks refuses by name (its payload edge and operand census would be empty)."""
    bindings = getattr(identity_mapping, "dynamic_bank_bindings", {}) or {}
    if not bindings:
        raise OperandCertificateError("no operand banks: the operand certificate needs dynamic operand banks; "
                                      "certify a program without them by certify_twin_flow_values")
    samples = samples.detach().cpu().double()
    with torch.no_grad():
        nf = twin_as_program(identity_mapping, forward_of, nf_model, samples)  # [U-PS] as the program evaluates
    scm_flow, scm, scm_state = _flow_outputs(identity_mapping, samples)
    hcm_flow, hcm, _state = _flow_outputs(packed_mapping, samples)
    trace = GoldenTrace()
    with host_kernel_replay() as replay:  # [W49] every sampled input alone through every host kernel the golden runs
        codes, steps = run_golden(identity_mapping, samples.numpy(), trace=trace)
    packed_codes, packed_steps = run_golden(packed_mapping, samples.numpy())
    reference, packed = flow_node_counts(scm_flow, samples), flow_node_counts(hcm_flow, samples)
    common = sorted(set(reference) & set(packed))
    carried = sorted(set(packed) - set(reference))
    golden_nodes = _golden_node_values(identity_mapping, trace)
    edges = [
        _edge("R (NF/SCM)", nf, scm), _edge("golden/SCM", codes * steps, scm),
        _edge("golden/HCM", packed_codes * packed_steps, hcm),
        _edge("C (SCM/HCM windows)", _cat([packed[n] for n in common]), _cat([reference[n] for n in common])),
        _edge("C carried (HCM/golden windows)", _cat([packed[n] for n in carried if n in golden_nodes]),
              _cat([golden_nodes[n] for n in carried if n in golden_nodes])),
    ]
    twin_ok = not (set(reference) - set(packed)) and all(n in golden_nodes for n in carried)
    host = [s.compute_op for s in identity_mapping.stages if s.kind == "compute"]
    host_got = [scm_state[int(op.id)].numpy() for op in host if int(op.id) in scm_state]
    host_want = [trace.host_values[int(op.id)] for op in host if int(op.id) in scm_state]
    edges.append(_edge("host ops (bitwise)", np.concatenate([h.reshape(-1) for h in host_got]) if host_got else [],
                       np.concatenate([h.reshape(-1) for h in host_want]) if host_want else []))
    edges.append(_batch_of_one_edge(identity_mapping, samples, codes, trace))
    host_condition = replay.certificate(program_host_ops=len(host))
    edges.append(_host_alone_edge(host_condition))
    payload = certify_payload_cells(hcm_flow, identity_mapping, samples)
    operand_cores = [core for s in identity_mapping.stages if s.kind == "neural"
                     for core, blocks in zip(s.hard_core_mapping.cores,
                                             operand_blocks(s.hard_core_mapping, bindings)) if blocks]
    by_kind: dict = {}
    for op in host:
        by_kind[op_kind(op)] = by_kind.get(op_kind(op), 0) + int(np.count_nonzero(trace.host_values[int(op.id)]))
    census = {
        "payload_codes": payload.nonzero_payload_codes,
        "operand_core_outputs": int(sum(np.count_nonzero(trace.core_codes.get(id(c), 0)) for c in operand_cores)),
        "host_op_outputs": dict(sorted(by_kind.items())),
        "outputs": int(np.count_nonzero(codes)),
    }
    required = {"R (NF/SCM)", "golden/SCM", "golden/HCM", "C (SCM/HCM windows)", BATCH_OF_ONE}
    passed = (payload.passed and twin_ok and census["operand_core_outputs"] > 0 and census["outputs"] > 0
              and all(e.mismatched == 0 for e in judged_edges(edges, host_condition))
              and all(e.compared > 0 for e in edges if e.name in required))
    return OperandProgramCertificate(tuple(edges), payload, census, passed, host_condition)
