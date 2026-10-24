"""N2 design §2.1-§2.4: every Contraction of the operand form lowers to one payload write and one operand bank per
batch index, consumed by one ordinary core instance per streamed vector; the stationary operand is chosen by price
where that choice is cost-only; a spec outside the form keeps G1's host op, named; no operand core type declared
leaves the IR byte-identical to G1's."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.ir import ComputeOp, NeuralCore
from mimarsinan.mapping.ir.dynamic_bank import validate_bindings
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.mapping.mappers.operand import OperandContractionMapper
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.support.contraction import Contraction
from mimarsinan.deployment_record.cost.absolute.writes import price_write_terms
from mimarsinan.deployment_record.quantities.write_law import write_policy_of
from mimarsinan.mapping.support.operand_lowering import (
    AT_STATIONARY,
    B_STATIONARY,
    OPERAND_FORM_OUTSIDE,
    Lowering,
    OperandDeclarationError,
    lower_contractions,
    lowering_price,
    operand_rule_form,
    stationary_operand,
)
from mimarsinan.mapping.support.operand_price import declared_physics, operand_region_census
from mimarsinan.models.families.vit_lite import ViTLiteCCT
from mimarsinan.torch_mapping.converter import convert_torch_model

T, F_IN = 6, 8
OPERAND_CORES = [{"max_axons": 64, "max_neurons": 64, "count": 8},
                 {"max_axons": 64, "max_neurons": 64, "count": 4, "operand": {"payload_bits": 8}}]
PLATFORM = {"cores": OPERAND_CORES, "activation_bits": 8, "readout_ranges": "per_consumer_group",
            "readout_law": {"adc_bits": 32, "range": "calibrated", "mult_bits": 16}}


class Bilinear(nn.Module):
    """(x W1)(x W2)^T through ``product`` (G1's witness), its head reading the flattened product."""

    def __init__(self, product, width_b: int = 5, out: int = T):
        super().__init__()
        self.a, self.b = nn.Linear(F_IN, 5), nn.Linear(F_IN, width_b)
        self.head = nn.Linear(T * out, 10)
        self.product = product

    def forward(self, x):
        return self.head(torch.flatten(self.product(self.a(x), self.b(x)), 1))


#: G1's seven spellings (``test_contraction.PRODUCTS``) and the multi-head ``@`` below: eight.
PRODUCTS = {
    "torch_bmm": lambda a, b: torch.bmm(a, b.transpose(1, 2)),
    "torch_matmul": lambda a, b: torch.matmul(a, b.transpose(1, 2)),
    "operator_matmul": lambda a, b: a @ b.transpose(1, 2),
    "method_matmul": lambda a, b: a.matmul(b.transpose(1, 2)),
    "method_bmm": lambda a, b: a.bmm(b.transpose(1, 2)),
    "einsum": lambda a, b: torch.einsum("bik,bjk->bij", a, b),
    "einsum_list": lambda a, b: torch.einsum("bik,bjk->bij", [a, b]),
}


class Heads(nn.Module):
    """Two-head scores ``q @ k^T`` per head: the eighth spelling, two batch indices."""

    def __init__(self):
        super().__init__()
        self.q, self.k = nn.Linear(F_IN, 8), nn.Linear(F_IN, 8)
        self.head = nn.Linear(2 * T * T, 10)

    def forward(self, x):
        b = x.shape[0]
        q = self.q(x).view(b, T, 2, 4).transpose(1, 2)
        k = self.k(x).view(b, T, 2, 4).transpose(1, 2)
        return self.head(torch.flatten(q @ k.transpose(-2, -1), 1))


class Gram(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(F_IN, 5)
        self.head = nn.Linear(T * T, 10)

    def forward(self, x):
        y = self.a(x)
        return self.head(torch.flatten(torch.matmul(y, y.transpose(1, 2)), 1))


class LinearAttention(nn.Module):
    """phi(Q) (phi(K)^T V): two products, the second streaming through the first's output."""

    def __init__(self):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(F_IN, 4), nn.Linear(F_IN, 4), nn.Linear(F_IN, 4)
        self.head = nn.Linear(T * 4, 10)

    def forward(self, x):
        q, k = F.elu(self.q(x)) + 1.0, F.elu(self.k(x)) + 1.0
        return self.head(torch.flatten(q @ (k.transpose(1, 2) @ self.v(x)), 1))


class Outside(nn.Module):
    """``bik,bjk->bi`` sums over two axis sets: outside the operand form."""

    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(F_IN, 5), nn.Linear(F_IN, 5)
        self.head = nn.Linear(T, 10)

    def forward(self, x):
        return self.head(torch.einsum("bik,bjk->bi", self.a(x), self.b(x)))


def _flow(factory, shape=(T, F_IN)):
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        return convert_torch_model(factory().eval(), shape, 10, device="cpu", packaging=MVM_PACKAGING).eval()


def _ir(flow):
    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    with contextlib.redirect_stdout(io.StringIO()):
        return IRMapping(q_max=127, firing_mode="Default").map(repr_)


def _writes(ir):
    return [n for n in ir.nodes if isinstance(n, ComputeOp) and n.op_type == "OperandWrite"]


def _contraction_ops(ir):
    return [n for n in ir.nodes if isinstance(n, ComputeOp) and n.op_type == "Contraction"]


@pytest.mark.parametrize("key", sorted(PRODUCTS))
def test_each_spelling_lowers_to_one_write_and_one_bank(key):
    flow = _flow(lambda: Bilinear(PRODUCTS[key]))
    report = lower_contractions(flow.get_mapper_repr(), PLATFORM)
    assert len(report.lowered) == 1 and not report.host
    assert [type(m) for m in flow.get_mapper_repr().execution_order()].count(OperandContractionMapper) == 1
    ir = _ir(flow)
    assert len(_writes(ir)) == 1 and not _contraction_ops(ir)
    assert len(ir.dynamic_bank_bindings) == 1
    binding = next(iter(ir.dynamic_bank_bindings.values()))
    instances = [n for n in ir.get_neural_cores() if n.weight_bank_id == binding.bank_id]
    assert binding.payload_shape == (5, T) and len(instances) == T  # B = K^T stationary on the tie, m = T rows
    validate_bindings(ir, packaging=MVM_PACKAGING)


def test_the_multi_head_product_takes_one_bank_per_head():
    flow = _flow(Heads)
    report = lower_contractions(flow.get_mapper_repr(), PLATFORM)
    assert len(report.lowered) == 1 and report.lowered[0].banks == 2
    ir = _ir(flow)
    assert len(_writes(ir)) == 1
    assert sorted(b.batch_index for b in ir.dynamic_bank_bindings.values()) == [(0,), (1,)]
    assert all(b.payload_shape == (4, T) for b in ir.dynamic_bank_bindings.values())


def test_a_gram_product_lowers_by_the_same_rule():
    flow = _flow(Gram)
    assert len(lower_contractions(flow.get_mapper_repr(), PLATFORM).lowered) == 1
    assert len(_ir(flow).dynamic_bank_bindings) == 1


def test_linear_attention_lowers_both_products_and_streams_through_the_first():
    flow = _flow(LinearAttention)
    report = lower_contractions(flow.get_mapper_repr(), PLATFORM)
    assert len(report.lowered) == 2 and not report.host
    ir = _ir(flow)
    assert len(_writes(ir)) == 2 and len(ir.dynamic_bank_bindings) == 2
    validate_bindings(ir, packaging=MVM_PACKAGING)


def test_a_spec_outside_the_form_keeps_its_host_op_and_is_named():
    flow = _flow(Outside)
    report = lower_contractions(flow.get_mapper_repr(), PLATFORM)
    assert not report.lowered and len(report.host) == 1
    assert report.host[0].reason == OPERAND_FORM_OUTSIDE
    assert len(_contraction_ops(_ir(flow))) == 1


def test_the_form_reads_batch_axes_and_operand_layouts():
    payload = next(m.module for m in _flow(Heads).get_mapper_repr().execution_order()
                   if isinstance(m, ComputeOpMapper) and isinstance(m.module, Contraction))
    form = operand_rule_form(payload)
    assert form is not None and form.batch == (2,) and (form.m, form.k, form.n) == (T, 4, T)
    assert operand_rule_form(Contraction(torch.einsum, "bik,bjk->bi", ((T, 5), (T, 5)), 1)) is None
    assert operand_rule_form(Contraction(torch.einsum, "bii,bik->bik", ((5, 5), (5, 3)), 1)) is None


def _asymmetric_form():
    """m = 6, k = 5, n = 2: B stationary writes 10 cells and streams 30 elements, A^T writes 30 and streams 10."""
    payload = Contraction(torch.bmm, "bmk,bkn->bmn", ((6, 5), (5, 2)), 60)
    form = operand_rule_form(payload)
    assert form is not None
    return form


def _platform(e_write: float, e_dma: float, payload_bits: int = 8, write_source: str = "host"):
    cores = [{"max_axons": 64, "max_neurons": 64, "count": 4,
              "operand": {"payload_bits": payload_bits, "write_source": write_source}}]
    note = {"note": "test declaration: relative prices only"}
    return {**PLATFORM, "cores": cores, "platform_physics_overrides": {
        "e_write_per_cell": {"nominal": e_write, **note}, "write_verify_iterations": {"nominal": 1.0, **note},
        "e_dma_per_byte": {"nominal": e_dma, **note}}}


def test_two_declarations_whose_prices_flip_flip_the_choice_at_equal_bits():
    form = _asymmetric_form()
    cheap_transfer, _ = stationary_operand(form, lowering_price(_platform(1.0, 0.0)), cost_only=True)
    dear_transfer, reason = stationary_operand(form, lowering_price(_platform(1.0, 10.0)), cost_only=True)
    assert cheap_transfer.stationary == B_STATIONARY and dear_transfer.stationary == AT_STATIONARY
    assert dear_transfer.cells == 30 and "cheaper" in reason


def test_the_choice_never_flips_when_payload_bits_differ_and_says_why():
    form = _asymmetric_form()
    for platform in (_platform(1.0, 0.0, payload_bits=4), _platform(1.0, 10.0, payload_bits=4)):
        lowering, reason = stationary_operand(form, lowering_price(platform), cost_only=False)
        assert lowering.stationary == B_STATIONARY
        assert "function-bearing" in reason and "A^T" in reason


def test_the_lowering_price_is_g3s_write_energy_of_the_placed_regions_plus_the_injected_stream():
    """Design §2.2 by G3's one write law: the regions the packer places (every streamed instance holds its whole
    bank) priced by ``price_write_terms``, and the streamed vectors at ``e_dma_per_byte``."""
    form, platform = _asymmetric_form(), _platform(1.0, 0.1)
    price, physics = lowering_price(platform), declared_physics(platform)
    policy = write_policy_of({**platform, "inference_batch_K": 1})
    assert physics is not None and policy is not None
    per_byte = physics.band("e_dma_per_byte").nominal  # SI (J per byte), the band every pricer reads
    for stationary in (B_STATIONARY, AT_STATIONARY):
        lowering = Lowering(form, stationary)
        census = operand_region_census(lowering)
        assert census.passes[0].written_cells_producer == 6 * 5 * 2  # m k n cells under either lowering
        energy = price_write_terms(census, physics, policy).energy  # mJ
        assert energy is not None and energy.nominal > 0
        expected = energy.nominal * 1e-3 + lowering.streamed_elements * 1.0 * per_byte  # 8-bit values: 1 byte each
        assert price(lowering) == pytest.approx(expected, rel=1e-12)


def test_the_placed_writes_tie_so_the_injected_stream_orders_the_lowerings():
    """Per placed instance both lowerings write m k n cells, so a dear cell write never favours the smaller bank;
    with a transfer energy the lowering streaming fewer vectors wins, and without one the tie rule decides."""
    form = _asymmetric_form()
    chosen, reason = stationary_operand(form, lowering_price(_platform(1.0, 0.1)), cost_only=True)
    assert chosen.stationary == AT_STATIONARY and "cheaper" in reason
    plain, _ = stationary_operand(form, lowering_price(_platform(1.0, 0.0)), cost_only=True)
    assert plain.stationary == B_STATIONARY


def test_the_lowering_price_charges_the_payload_transfer_only_for_host_sourced_writes():
    lowering = Lowering(_asymmetric_form(), B_STATIONARY)
    platform = _platform(1.0, 1.0)
    physics = declared_physics(platform)
    assert physics is not None
    host = lowering_price(platform)(lowering)
    on_chip = lowering_price(_platform(1.0, 1.0, write_source="on_chip"))(lowering)
    # 60 placed cells of 8 bits (1 byte each) cross from the host at e_dma_per_byte (SI, J per byte)
    assert host - on_chip == pytest.approx(60 * 1.0 * physics.band("e_dma_per_byte").nominal, rel=1e-9)
    assert on_chip > 0


def test_the_pipelines_resolved_physics_reaches_the_lowering_price():
    """The resolved platform a pipeline step reads carries the physics as ``platform_physics_resolved`` only."""
    platform = _platform(1.0, 0.1)
    physics = declared_physics(platform)
    assert physics is not None
    resolved = {k: v for k, v in platform.items() if k != "platform_physics_overrides"}
    resolved["platform_physics_resolved"] = physics.to_dict()
    lowering = Lowering(_asymmetric_form(), AT_STATIONARY)
    assert lowering_price(resolved)(lowering) == lowering_price(platform)(lowering) > 0


def test_the_platform_resolver_hands_the_lowering_price_the_declared_physics():
    """Pipeline level: Operand Lowering reads the resolved platform, which carries the physics only as
    ``platform_physics_resolved``; its price equals the price of the declaration it was resolved from."""
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    declared = {"cores": [{"max_axons": 64, "max_neurons": 64, "count": 4,
                           "operand": {"payload_bits": 8, "write_source": "host"}}],
                "weight_bits": 8, "activation_bits": 8, "platform_physics_profile": "generic_estimated_22nm"}
    resolved = build_platform_constraints_resolved(declared)
    assert "platform_physics_resolved" in resolved and "platform_physics_profile" not in resolved
    for stationary in (B_STATIONARY, AT_STATIONARY):
        lowering = Lowering(_asymmetric_form(), stationary)
        assert lowering_price(resolved)(lowering) == pytest.approx(lowering_price(declared)(lowering), rel=1e-12)
        assert lowering_price(resolved)(lowering) > 0


def _vit(attention: str):
    return _flow(lambda: ViTLiteCCT(num_classes=5, img_size=8, patch=4, dim=8, depth=1, heads=2, mlp_ratio=2.0,
                                    attention=attention), shape=(3, 8, 8))


@pytest.mark.parametrize("attention", ["softmax", "relu"])
def test_a_host_fed_operand_keeps_the_declared_default_whatever_the_price(attention):
    """Design §2.2: the choice is cost-only only where both operands are core-fed on their producers' groups. In
    the softmax form P is a host value (streamed on its entry grid, written on a re-snap grid), so PV keeps B."""
    def favour_at(lowering):
        return 0.0 if lowering.stationary == AT_STATIONARY else 1.0

    scores, values = lower_contractions(_vit(attention).get_mapper_repr(), PLATFORM, price=favour_at).lowered
    assert scores.stationary == AT_STATIONARY and "cost-only" in scores.reason
    if attention == "softmax":
        assert values.stationary == B_STATIONARY
        assert "function-bearing" in values.reason and "host value" in values.reason
    else:
        assert values.stationary == AT_STATIONARY and "cost-only" in values.reason


class HostKeys(nn.Module):
    """``a(x) phi(b(x))^T``: the stationary operand is a host value (phi on the host)."""

    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(F_IN, 5), nn.Linear(F_IN, 5)
        self.head = nn.Linear(T * T, 10)

    def forward(self, x):
        return self.head(torch.flatten(self.a(x) @ (F.elu(self.b(x)) + 1.0).transpose(1, 2), 1))


def test_write_source_on_chip_refuses_where_a_bank_re_snaps():
    """Design §2.8: on-chip write-back computes the certified codes only where every payload write copies codes."""
    cores = [{"max_axons": 64, "max_neurons": 64, "count": 4,
              "operand": {"payload_bits": 8, "write_source": "on_chip"}}]
    on_chip = {**PLATFORM, "cores": cores}
    ungrouped = {k: v for k, v in on_chip.items() if k != "readout_ranges"}
    with pytest.raises(OperandDeclarationError, match="on_chip.*no consumer groups"):
        lower_contractions(_flow(lambda: Bilinear(PRODUCTS["torch_bmm"])).get_mapper_repr(), ungrouped)
    with pytest.raises(OperandDeclarationError, match="host-produced stationary operand"):
        lower_contractions(_flow(HostKeys).get_mapper_repr(), on_chip)
    report = lower_contractions(_flow(lambda: Bilinear(PRODUCTS["torch_bmm"])).get_mapper_repr(), on_chip)
    assert len(report.lowered) == 1


def test_ties_go_to_fewer_cells_then_to_b_stationary():
    square = operand_rule_form(Contraction(torch.bmm, "bmk,bkn->bmn", ((4, 3), (3, 4)), 48))
    assert square is not None
    choice, _ = stationary_operand(square, lambda lowering: 1.0, cost_only=True)
    assert choice.stationary == B_STATIONARY
    pv = operand_rule_form(Contraction(torch.bmm, "bmk,bkn->bmn", ((5, 5), (5, 4)), 100))
    assert pv is not None
    choice, _ = stationary_operand(pv, lambda lowering: 1.0, cost_only=True)
    assert choice.stationary == B_STATIONARY and choice.cells == 20
    assert Lowering(pv, AT_STATIONARY).cells == 25


def test_vit_lite_lowers_both_products_of_every_block():
    flow = _flow(lambda: ViTLiteCCT(num_classes=5, img_size=8, patch=4, dim=8, depth=2, heads=2, mlp_ratio=2.0),
                 shape=(3, 8, 8))
    report = lower_contractions(flow.get_mapper_repr(), PLATFORM)
    assert len(report.lowered) == 4 and all(p.banks == 2 for p in report.lowered)
    ir = _ir(flow)
    assert len(_writes(ir)) == 4 and len(ir.dynamic_bank_bindings) == 8
    assert not _contraction_ops(ir)


def _ir_digest(ir) -> str:
    """Every node's kind, id, name, op type, wiring and matrices, the banks and the outputs (no build token)."""
    h = hashlib.sha256()
    for node in ir.nodes:
        h.update(f"{type(node).__name__}|{node.id}|{node.name}|{getattr(node, 'op_type', '')}".encode())
        h.update(repr([(s.node_id, s.index) for s in node.input_sources.flatten()]).encode())
        if isinstance(node, NeuralCore):
            matrix = node.get_core_matrix(ir)
            h.update(np.ascontiguousarray(matrix).tobytes() + repr(matrix.shape).encode())
            h.update(f"{node.weight_bank_id}|{node.weight_row_slice}|{node.perceptron_index}".encode())
    for bank_id, bank in sorted(ir.weight_banks.items()):
        h.update(f"bank{bank_id}".encode() + np.ascontiguousarray(bank.core_matrix).tobytes())
    h.update(repr([(s.node_id, s.index) for s in ir.output_sources.flatten()]).encode())
    return h.hexdigest()


@pytest.mark.parametrize("factory", [lambda: Bilinear(PRODUCTS["torch_bmm"]), Heads, LinearAttention])
def test_without_an_operand_core_type_the_ir_is_byte_identical(factory):
    flow = _flow(factory)
    reference = copy.deepcopy(flow)
    report = lower_contractions(flow.get_mapper_repr(), {**PLATFORM, "cores": OPERAND_CORES[:1]})
    assert not report.lowered and not report.host
    ir, base = _ir(flow), _ir(reference)
    assert _ir_digest(ir) == _ir_digest(base) and ir.dynamic_bank_bindings == {}
