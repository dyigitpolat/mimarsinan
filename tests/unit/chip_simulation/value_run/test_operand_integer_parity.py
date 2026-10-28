"""N2 design §3.2: a two-head, two-layer attention block in both forms (softmax on the host between the products;
ReLU attention with its followers absorbed into the S-core law), at a core budget below its tiles (multi-pass):
NF = SCM = HCM = int64 golden at atol 0 on every output code, the payload edge at atol 0 on every written cell,
and the host-op edge bitwise per host segment. Each mutant is caught: swapped operand roles, one payload code
off by one, the head index of one bank permuted, the softmax replayed in float32 in the golden."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from mimarsinan.certification import operand_certificate
from mimarsinan.certification.host_condition import ESTABLISHED, OUTSIDE_SCOPE
from mimarsinan.certification.operand_certificate import OperandCertificateError, certify_operand_program
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.chip_simulation.value_run import host_ops, int64_golden, operand_payloads, value_flow
from mimarsinan.chip_simulation.value_run.int64_golden import evaluate, run_golden
from mimarsinan.mapping.capability.op_kinds import op_kind
from mimarsinan.mapping.mappers.operand import OperandContractionMapper
from mimarsinan.mapping.mappers.operand import operand_mapper as operand_mapper_module
from unit.chip_simulation.value_run.integer_programs import assert_exact
from mimarsinan.mapping.support.operand_form import AT_STATIONARY, B_STATIONARY
from unit.chip_simulation.value_run.operand_programs import attention_program, image_inputs

FORMS = ("softmax", "relu")
GEOMETRY = dict(axons=64, neurons=64, count=3, passes=128)


@pytest.fixture(scope="module", params=FORMS)
def program(request):
    return attention_program(request.param)


def _hcm(program):
    return program.packed(**GEOMETRY)


def test_the_packed_program_is_multi_pass(program):
    mapping = program.packed_mapping(**GEOMETRY)
    passes = {}
    for stage in mapping.stages:
        if stage.kind == "neural":
            passes[stage.schedule_segment_index] = passes.get(stage.schedule_segment_index, 0) + 1
    assert max(passes.values()) >= 2


def test_nf_scm_hcm_and_the_golden_agree_at_atol_0(program):
    x = image_inputs(16)
    with torch.no_grad():
        nf = program.nf()(x)
        scm = program.identity()(x)
        hcm = _hcm(program)(x)
    assert_exact(scm, nf)
    assert_exact(hcm, scm)
    golden, steps = int64_golden.run_golden(program.identity_mapping(), x.numpy())
    assert_exact(torch.as_tensor(golden * steps), scm)
    golden_packed, steps_packed = int64_golden.run_golden(program.packed_mapping(**GEOMETRY), x.numpy())
    assert_exact(torch.as_tensor(golden_packed * steps_packed), hcm)


def test_the_certificate_holds_with_its_denominators_and_nonzero_census(program):
    x = image_inputs(16)
    certificate = certify_operand_program(program.nf(), program.identity_mapping(),
                                          program.packed_mapping(**GEOMETRY), x)
    assert certificate.passed, certificate.summary()
    for edge in certificate.edges:
        assert edge.mismatched == 0, edge
        assert edge.compared > 0 or edge.name.startswith("C carried"), edge  # carried windows exist only where a pass cuts a chain
    assert certificate.payload.cells_compared > 0 and certificate.payload.mismatched == 0
    census = certificate.nonzero_census
    assert census["payload_codes"] > 0 and census["operand_core_outputs"] > 0
    assert census["host_op_outputs"].get("payload_write", 0) > 0
    host = [s.compute_op.op_type for s in program.identity_mapping().stages if s.kind == "compute"]
    if any("softmax" in str(op).lower() for op in host):
        assert census["host_op_outputs"]["softmax"] > 0


def test_the_relu_form_has_no_host_op_between_its_products():
    program = attention_program("relu")
    mapping = program.identity_mapping()
    kinds = [s.compute_op.op_type if s.kind == "compute" else "neural" for s in mapping.stages]
    assert "softmax" not in " ".join(kinds).lower()
    scores, values = [m for m in program.model.get_mapper_repr().execution_order()
                      if isinstance(m, OperandContractionMapper)][:2]
    assert scores.follower_kinds == ("scale", "relu", "scale")


def _caught(program, x) -> bool:
    """A mutant is caught when the certificate fails or a check refuses it by name; it never passes silently."""
    try:
        return not certify_operand_program(program.nf(), program.identity_mapping(),
                                           program.packed_mapping(**GEOMETRY), x).passed
    except Exception as exc:  # noqa: BLE001 -- only a named refusal (through the forward's wrapper) counts
        cause = exc
        while cause is not None and not isinstance(cause, ToolchainRefusal):
            cause = cause.__cause__
        if cause is None:
            raise
        return True


def test_swapped_operand_roles_are_caught(monkeypatch):
    original = OperandContractionMapper.ir_operands

    def swapped(self, a_ir, b_ir):
        stream, payload = original(self, a_ir, b_ir)
        if stream.shape == np.swapaxes(payload, -1, -2).shape:  # QK^T: stream K, write Q^T, read S^T as S
            return np.swapaxes(payload, -1, -2), np.swapaxes(stream, -1, -2)
        return stream, payload

    monkeypatch.setattr(OperandContractionMapper, "ir_operands", swapped)
    assert _caught(attention_program("softmax"), image_inputs(8))


def test_one_payload_code_off_by_one_is_caught(monkeypatch):
    program = attention_program("softmax")
    original = operand_payloads.materialize

    def off_by_one(stage, state_buffer, bindings, sample_slice=slice(None)):
        fresh = original(stage, state_buffer, bindings, sample_slice)
        if fresh:
            bank = min(fresh)
            fresh[bank] = fresh[bank].clone()
            fresh[bank][0, 0, 0] += 1.0
        return fresh

    monkeypatch.setattr(operand_payloads, "materialize", off_by_one)
    certificate = certify_operand_program(program.nf(), program.identity_mapping(),
                                          program.packed_mapping(**GEOMETRY), image_inputs(8))
    assert not certificate.passed and certificate.payload.mismatched >= 1


def test_the_head_index_of_one_bank_permuted_is_caught():
    program = attention_program("softmax")
    bindings = program.ir.dynamic_bank_bindings
    first, second = sorted(bindings)[:2]
    a, b = bindings[first], bindings[second]
    bindings[first] = dataclasses.replace(a, payload_offset=b.payload_offset, batch_index=b.batch_index)
    bindings[second] = dataclasses.replace(b, payload_offset=a.payload_offset, batch_index=a.batch_index)
    assert _caught(program, image_inputs(8))


#: A witness every bank of which reaches a compared output code in both forms (head width 32, depth 1). The default
#: block's ReLU form hides its banks behind proj's always-on rows (zeroing any bank changes no code), so the bank
#: mutants run here as well.
SENSITIVE = dict(dim=64, heads=2, depth=1)
SENSITIVE_GEOMETRY = dict(axons=256, neurons=256, count=4, passes=128)  # three segments in two passes


@pytest.fixture(scope="module", params=FORMS)
def sensitive(request):
    return attention_program(request.param, block=SENSITIVE)


def test_every_bank_of_the_sensitive_witness_reaches_a_compared_code(sensitive, monkeypatch):
    x = image_inputs(16)
    clean, _steps = int64_golden.run_golden(sensitive.identity_mapping(), x.numpy())
    original = int64_golden.bank_payload
    for bank in sorted(sensitive.ir.dynamic_bank_bindings):
        def zeroed(values, binding, bank=bank):
            codes = original(values, binding)
            return torch.zeros_like(codes) if binding.bank_id == bank else codes

        monkeypatch.setattr(int64_golden, "bank_payload", zeroed)
        codes, _steps = int64_golden.run_golden(sensitive.identity_mapping(), x.numpy())
        assert np.count_nonzero(codes != clean) > 0, f"bank {bank} reaches no compared code"
    monkeypatch.setattr(int64_golden, "bank_payload", original)
    packed = sensitive.packed_mapping(**SENSITIVE_GEOMETRY)
    assert max(sum(1 for s in packed.stages if s.kind == "neural" and s.schedule_segment_index == i)
               for i in {s.schedule_segment_index for s in packed.stages if s.kind == "neural"}) >= 2
    certificate = certify_operand_program(sensitive.nf(), sensitive.identity_mapping(), packed, x)
    assert certificate.passed, certificate.summary()


def test_two_value_heads_permuted_on_grid_are_caught(sensitive):
    """The head-permutation mutant on the PV banks (offset, head index and payload step together, so the swapped
    payload stays on its grid) of both forms."""
    values = [m for m in sensitive.model.get_mapper_repr().execution_order()
              if isinstance(m, OperandContractionMapper)][1]
    indices = {op.perceptron_index for op in values.banks}
    bindings = sensitive.ir.dynamic_bank_bindings
    pv = sorted(b for b in bindings if any(n.weight_bank_id == b and n.perceptron_index in indices
                                          for n in sensitive.ir.get_neural_cores()))
    first, second = pv[:2]
    a, b = bindings[first], bindings[second]
    try:
        bindings[first] = dataclasses.replace(a, payload_offset=b.payload_offset, batch_index=b.batch_index,
                                              payload_step=b.payload_step)
        bindings[second] = dataclasses.replace(b, payload_offset=a.payload_offset, batch_index=a.batch_index,
                                               payload_step=a.payload_step)
        certificate = certify_operand_program(sensitive.nf(), sensitive.identity_mapping(),
                                              sensitive.packed_mapping(**SENSITIVE_GEOMETRY), image_inputs(8))
        edges = {e.name: e for e in certificate.edges}
        assert not certificate.passed and edges["R (NF/SCM)"].mismatched > 0, certificate.summary()
    finally:
        bindings[first], bindings[second] = a, b


def test_the_softmax_replayed_in_float32_in_the_golden_is_caught(monkeypatch):
    program = attention_program("softmax")
    original = int64_golden.execute_host_op

    def float32_softmax(op, x_flat, state, *, dtype, twins):
        if "softmax" in str(op.op_type).lower():
            out = original(op, x_flat.float(), {k: v.float() for k, v in state.items()}, dtype=torch.float32,
                           twins={})
            return out.double()
        return original(op, x_flat, state, dtype=dtype, twins=twins)

    monkeypatch.setattr(int64_golden, "execute_host_op", float32_softmax)
    assert _caught(program, image_inputs(8))


def test_the_absorbed_scale_rides_in_the_s_core_law_not_on_the_host():
    program = attention_program("softmax")
    mapping = program.identity_mapping()
    host = [s.compute_op for s in mapping.stages if s.kind == "compute"]
    scores = [m for m in program.model.get_mapper_repr().execution_order()
              if isinstance(m, OperandContractionMapper)][0]
    assert scores.follower_kinds == ("scale",)
    assert not any(getattr(getattr(op.params.get("module"), "fn", None), "__name__", "") == "mul" for op in host)
    assert operand_mapper_module is not None and evaluate is not None


@pytest.mark.parametrize("form", FORMS)
def test_the_absorbed_law_is_the_traced_followers_within_one_code(form):
    """Exactness cannot see a law that drops an absorbed follower (every program reads the same law), so this pins
    the absorption: each bank's stamped codes equal the float path (product, followers as traced, snapped by its
    readout grid) within one code, the declared rounding of the ratio into M0; ReLU banks have no negative code."""
    program = attention_program(form)
    model = program.nf()
    x = image_inputs(8)
    for mapper in [m for m in model.get_mapper_repr().execution_order() if isinstance(m, OperandContractionMapper)]:
        inputs = []
        original = mapper.forward
        mapper.forward = lambda value, f=original: inputs.append(value) or f(value)
        stamps = [op.stamp for op in mapper.banks]
        try:
            with torch.no_grad():
                model(x)
                captured = [original(inputs[0])]
                for op in mapper.banks:
                    op.stamp = None
                captured.append(original(inputs[0]))
        finally:
            del mapper.forward
            for op, stamp in zip(mapper.banks, stamps):
                op.stamp = stamp
        form = mapper.lowering.form
        inverse = [1 + form.out_perm.index(i) for i in range(len(form.out_perm))]
        law, snapped = (t.permute(0, *inverse).reshape(t.shape[0], form.banks, form.m, form.n) for t in captured)
        steps = torch.as_tensor([float(s.law["output_step"][0]) for s in stamps], dtype=torch.float64)
        codes_law = torch.round(law / steps.reshape(1, -1, 1, 1))
        codes_float = torch.round(snapped / steps.reshape(1, -1, 1, 1))
        assert float((codes_law - codes_float).abs().max()) <= 1.0, (mapper.name, mapper.follower_kinds)
        assert np.count_nonzero(codes_law.numpy())
        if mapper.relu:
            assert float(codes_law.min()) >= 0.0


@pytest.mark.parametrize("form", FORMS)
def test_without_consumer_groups_the_re_snapped_program_also_certifies_at_atol_0(form):
    """design §2.3: with ``readout_ranges`` absent each payload is re-snapped onto its bank's own grid by the payload
    write (K and V rounded twice); the program is still a composition of integer laws and certifies at atol 0."""
    from unit.chip_simulation.value_run.operand_programs import operand_platform
    program = attention_program(form, platform=operand_platform(grouped=False))
    assert all(op.payload_quantizer is not None for m in program.model.get_mapper_repr().execution_order()
               if isinstance(m, OperandContractionMapper) for op in m.banks)
    certificate = certify_operand_program(program.nf(), program.identity_mapping(),
                                          program.packed_mapping(**GEOMETRY), image_inputs(16))
    assert certificate.passed, certificate.summary()


def test_a_host_op_that_depends_on_batch_composition_is_caught(monkeypatch):
    """Design §3.2: the host edge compares golden and executor on one batch, so a host op whose fp64 output moves
    with the batch it runs in (here: softmax shifted by 1e-9 inside any batch of several samples, in both the
    executor and the golden) passes it; the golden alone against the batch catches it."""
    program = attention_program("softmax")
    original = host_ops.execute_host_op

    def batch_dependent(op, x_flat, state, *, dtype, twins):
        out = original(op, x_flat, state, dtype=dtype, twins=twins)
        return out + 1e-9 if op_kind(op) == "softmax" and out.shape[0] > 1 else out

    for module in (int64_golden, value_flow):
        monkeypatch.setattr(module, "execute_host_op", batch_dependent)
    certificate = certify_operand_program(program.nf(), program.identity_mapping(),
                                          program.packed_mapping(**GEOMETRY), image_inputs(8))
    edges = {e.name: e for e in certificate.edges}
    assert edges["host ops (bitwise)"].mismatched == 0
    assert edges[operand_certificate.BATCH_OF_ONE].mismatched > 0 and not certificate.passed


def _softmax_kernel_moved(monkeypatch, moved):
    """Patch the host kernel itself (``ComputeOp._exec_module``, what the host-kernel replay replays): the softmax's
    output moves by 1e-9 on the rows ``moved(batch)`` selects (None: none) inside the batch it runs in."""
    from mimarsinan.mapping.ir import ComputeOp

    kernel = ComputeOp._exec_module

    def patched(self, x):
        out = kernel(self, x)
        rows = moved(int(out.shape[0]))
        if op_kind(self) == "softmax" and rows is not None:
            out = out.clone()
            out[rows] = out[rows] + 1e-9
        return out

    monkeypatch.setattr(ComputeOp, "_exec_module", patched)


def _late_rows(batch):
    return slice(2, None) if batch > 2 else None


def test_every_sample_alone_through_the_host_kernels_catches_what_the_spot_check_misses(monkeypatch):
    """[W49, review M3] A host kernel (the softmax) whose output moves only for samples 2 and later of a batch of
    several: the whole-golden spot check of the first two samples alone passes it; the host-kernel edge, every sampled
    input alone through every host kernel the golden runs, catches it. That edge reads Theorem 1's host condition, so
    the kernel is named and the program is outside the per-sample certified scope by name; the program's own edges
    hold, and the certificate is not failed by it (no unnamed stop)."""
    _softmax_kernel_moved(monkeypatch, _late_rows)
    program = attention_program("softmax")
    certificate = certify_operand_program(program.nf(), program.identity_mapping(),
                                          program.packed_mapping(**GEOMETRY), image_inputs(8))
    edges = {e.name: e for e in certificate.edges}
    assert edges[operand_certificate.BATCH_OF_ONE].mismatched == 0   # the first two samples alone: unchanged
    host_alone = edges[operand_certificate.HOST_ALONE]
    assert host_alone.compared > 0 and host_alone.mismatched > 0
    assert certificate.host_condition.status == OUTSIDE_SCOPE
    assert {k.kind for k in certificate.host_condition.batch_dependent} == {"softmax"}
    assert certificate.passed, certificate.summary()
    assert "host condition outside_scope" in certificate.summary()


def test_one_rule_reads_both_batch_dependence_edges_where_the_replay_names_the_kernel(monkeypatch):
    """[review M3] The rule for both batch-dependence edges (the whole golden's batch-of-one spot check and the
    host-kernel edge): where the host-kernel replay names a kernel that breaks the host condition, their mismatches
    are readings of that condition, carried by name, not failures of the program. A softmax that moves every row of a
    batch of several moves both edges."""
    _softmax_kernel_moved(monkeypatch, lambda batch: slice(None) if batch > 1 else None)
    program = attention_program("softmax")
    certificate = certify_operand_program(program.nf(), program.identity_mapping(),
                                          program.packed_mapping(**GEOMETRY), image_inputs(8))
    edges = {e.name: e for e in certificate.edges}
    assert edges[operand_certificate.BATCH_OF_ONE].mismatched > 0
    assert edges[operand_certificate.HOST_ALONE].mismatched > 0
    assert certificate.host_condition.status == OUTSIDE_SCOPE
    assert certificate.passed, certificate.summary()


def test_a_batch_of_one_mismatch_no_host_kernel_explains_still_fails_the_program(monkeypatch):
    """[review M3] The rule's other side: batch dependence outside every host kernel (in the evaluator around it, both
    in the executor and the golden) leaves the replay naming no kernel, the host condition established, so the
    batch-of-one edge's mismatch is a defect of the program and fails the certificate, as at RC0."""
    program = attention_program("softmax")
    original = host_ops.execute_host_op

    def batch_dependent(op, x_flat, state, *, dtype, twins):
        out = original(op, x_flat, state, dtype=dtype, twins=twins)
        return out + 1e-9 if op_kind(op) == "softmax" and out.shape[0] > 1 else out

    for module in (int64_golden, value_flow):
        monkeypatch.setattr(module, "execute_host_op", batch_dependent)
    certificate = certify_operand_program(program.nf(), program.identity_mapping(),
                                          program.packed_mapping(**GEOMETRY), image_inputs(8))
    edges = {e.name: e for e in certificate.edges}
    assert certificate.host_condition.status == ESTABLISHED
    assert edges[operand_certificate.BATCH_OF_ONE].mismatched > 0 and not certificate.passed


def test_the_host_kernel_edge_reads_the_host_condition_established_on_the_program_as_built():
    program = attention_program("softmax")
    certificate = certify_operand_program(program.nf(), program.identity_mapping(),
                                          program.packed_mapping(**GEOMETRY), image_inputs(8))
    edges = {e.name: e for e in certificate.edges}
    assert edges[operand_certificate.HOST_ALONE].compared > 0 and edges[operand_certificate.HOST_ALONE].mismatched == 0
    assert certificate.host_condition.status == ESTABLISHED
    assert certificate.passed, certificate.summary()


def test_through_the_value_twin_gate_a_batch_dependent_host_kernel_is_named_not_a_failure(monkeypatch, capsys):
    """[review M3] Hard Core Mapping's value twin gate on an operand program: a batch-dependent host kernel does not stop
    the run with an unnamed failure. The gate passes on the program's own edges, and the host condition read around it
    (as Hard Core Mapping's replay reads it) names the kernel outside the per-sample certified scope, in its record."""
    from types import SimpleNamespace

    from mimarsinan.deployment_record.schema import CertificateRecord
    from mimarsinan.pipelining.core.gates import value_gates
    from mimarsinan.pipelining.core.gates.host_condition_gate import host_condition_records, host_kernel_replay

    _softmax_kernel_moved(monkeypatch, _late_rows)
    program = attention_program("softmax")
    monkeypatch.setattr(value_gates, "value_certificate_gate_armed", lambda pipeline: True)
    monkeypatch.setattr(value_gates, "_certificate_samples", lambda pipeline, model, n: image_inputs(n))
    packed = program.packed_mapping(**GEOMETRY)
    with host_kernel_replay() as replay:
        certificate = value_gates.run_value_twin_certificate_gate(
            SimpleNamespace(config={"value_parity_samples": 8}), program.model, program.ir, packed)
    assert certificate.passed and certificate.host_condition.status == OUTSIDE_SCOPE
    (row,) = host_condition_records(replay, packed)
    record = CertificateRecord.from_dict(row)
    detail = json.loads(record.detail)
    assert not record.passed and detail["verdict"].startswith("outside_scope: ")
    assert any(k["kind"] == "softmax" and k["batch_dependent"] > 0 for k in detail["kernels"])
    out = capsys.readouterr().out
    assert "[ValueTwinCertificate] PASS operand program" in out and "host condition outside_scope" in out


def test_the_operand_certificate_refuses_a_program_without_operand_banks():
    from unit.chip_simulation.value_run.operand_programs import build_operand_program, operand_platform
    from mimarsinan.models.families.vit_lite import ViTLiteCCT
    torch.manual_seed(3)
    twin = ViTLiteCCT(num_classes=5, img_size=8, patch=4, dim=8, depth=1, heads=2, mlp_ratio=2.0).eval()
    program = build_operand_program(twin, (3, 8, 8), num_classes=5,
                                    platform=operand_platform(cores=[{"max_axons": 256, "max_neurons": 256,
                                                                      "count": 64}], grouped=False))
    assert not program.ir.dynamic_bank_bindings
    with pytest.raises(OperandCertificateError, match="no operand banks"):
        certify_operand_program(program.nf(), program.identity_mapping(), program.packed_mapping(**GEOMETRY),
                                image_inputs(4))


def test_an_edge_counts_a_nan_on_either_side_as_a_mismatch():
    edge = operand_certificate._edge("e", [float("nan"), 1.0, 2.0], [1.0, 1.0, float("nan")])
    assert edge.mismatched == 2 and edge.compared == 3
    assert operand_certificate._edge("e", [float("nan")], [float("nan")]).mismatched == 1


def _forced(stationary):
    return lambda lowering: 0.0 if lowering.stationary == stationary else 1.0


@pytest.mark.parametrize("form", FORMS)
def test_wherever_the_choice_reads_cost_only_both_lowerings_give_identical_codes(form):
    """Design §2.2's decision class at a shape with fewer tokens than the head width (5 tokens, head width 32), where
    a price alone would lower PV A^T stationary: each product forced either way where its choice is cost-only, the
    others on their declared default, and the output codes are byte-identical. Before the fix the softmax form's PV
    read cost-only, and forcing it A^T (P^T re-snapped on a payload grid of another calibration seam than P's entry
    grid) changed 21 of 320 output codes at this shape (fix round, tests first)."""
    block = dict(dim=64, heads=2, depth=1)
    programs = {s: attention_program(form, block=block, price=_forced(s)) for s in (B_STATIONARY, AT_STATIONARY)}
    at = programs[AT_STATIONARY].report.lowered
    assert any(p.stationary == AT_STATIONARY for p in at)
    for product in at:
        assert product.stationary == AT_STATIONARY or "function-bearing" in product.reason, product
    x = image_inputs(64)
    out = {s: run_golden(p.identity_mapping(), x.numpy()) for s, p in programs.items()}
    codes_b, steps_b = out[B_STATIONARY]
    codes_at, steps_at = out[AT_STATIONARY]
    assert np.array_equal(codes_b, codes_at) and np.array_equal(steps_b, steps_at)
