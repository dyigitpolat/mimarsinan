"""Static extents are proved, never sampled (review round 4).

A shape value is a conversion-time literal only where a symbolic-batch shape pass (FakeTensors under a ShapeEnv,
the batch a symbol) proves its extent a concrete integer for every batch of at least 2, with no guard on the batch
before it and no data-dependent size upstream, and a seeded batch-1 trace agrees. Sampling three batch sizes read
a data-dependent count (``x[x == 1.0].shape[0]``, 0 in every random trace) and a batch slice (``y[6:]``) as static
and folded them; now they run on the host, the w0-base form, exact at every batch. The decision reads no random
state, so the probe's witness conversion folds the literals the returned conversion folds; a difference is refused.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.torch_mapping import converter
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.batch_layout import BATCH_SIZE, BatchLayouts
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.value_parity import relative_error
from mimarsinan.torch_mapping.value_parity.verdict import WitnessProgramMismatchError

from .harness import CONTRACTS, EXACT, PACKAGINGS, assert_converted_exact, convert, native, quietly


class Extent(nn.Module):
    """A shape value of ``x`` or ``y = fc(x)`` that no conversion-time literal can stand for."""

    FORMS = {
        "mask_count": lambda x, y: y * x[x == 1.0].shape[0] + y,
        "nonzero_count": lambda x, y: y * torch.nonzero(x == 1.0).shape[0] + y,
        "sample0_count": lambda x, y: y * x[0][x[0] > -0.7].shape[0] + y,
        "from6": lambda x, y: y * y[6:].shape[0] + y,
        "step6": lambda x, y: y * y[::6].shape[0] + y,
        "to_minus6": lambda x, y: y * y[:-6].shape[0] + y,
        "from3": lambda x, y: y * y[3:].shape[0] + y,
        "from4": lambda x, y: y * y[4:].shape[0] + y,
        "step4": lambda x, y: y * y[::4].shape[0] + y,
        "head2": lambda x, y: y * y[:2].shape[0] + y,
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.FORMS[self.how](x, self.fc(x))


def host_ops(flow) -> list:
    return [m.name for m in flow.get_mapper_repr().execution_order()
            if isinstance(m, ComputeOpMapper) and m.name != "fc"]


# ----------------------------------------------------------------------------- data-dependent extents
# The w0-base host-op structure of the two counts (exact there at every batch and input).
DATA_DEPENDENT = {
    "mask_count": ["eq", "getitem", "getattr_1", "getitem_1", "mul", "add"],
    "nonzero_count": ["eq", "nonzero", "getattr_1", "getitem", "mul", "add"],
}
# The host op whose value has no per-sample row (stage 2 refuses it by name): operator, FX node.
DATA_DEPENDENT_HOST_VALUE = {"mask_count": ("getitem", "getitem"), "nonzero_count": ("nonzero", "nonzero")}
EXTENT_BATCHES = (1, 2, 3, 6, 7)


def saturated_and_integer_inputs(batch: int):
    """U(0, 1) images with 0, 1 or 3 entries per sample saturated at exactly 1.0, and integer-valued inputs."""
    inputs = []
    for saturated in (0, 1, 3):
        x = torch.rand(batch, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(10 * batch + saturated))
        x[:, :saturated] = 1.0
        inputs.append(x)
    integers = torch.randint(0, 3, (batch, 8), generator=torch.Generator().manual_seed(50 + batch))
    return inputs + [integers.double()]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(DATA_DEPENDENT))
def test_a_data_dependent_extent_is_evaluated_at_run_time(how, contract):
    """Folded to the literal 0 before (every random trace has no exact 1.0): wrong by 0.5-0.95 on saturated inputs."""
    model = native(lambda: Extent(how))
    flow = convert(model, (8,), contract)
    for batch in EXTENT_BATCHES:
        for x in saturated_and_integer_inputs(batch):
            with torch.no_grad():
                err = relative_error(flow(x), model(x)).relative_error
            assert err <= EXACT, f"{how} {contract} batch {batch}: relative error {err:.3e}"


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(DATA_DEPENDENT))
def test_a_data_dependent_extent_keeps_the_w0_base_host_ops(how, contract):
    assert host_ops(convert(native(lambda: Extent(how)), (8,), contract)) == DATA_DEPENDENT[how]


@pytest.mark.parametrize("how", sorted(DATA_DEPENDENT))
def test_a_data_dependent_host_value_is_refused_by_name_at_stage_2(how):
    operator, node = DATA_DEPENDENT_HOST_VALUE[how]
    model = native(lambda: Extent(how))
    with pytest.raises(ToolchainRefusal, match="identity value program") as excinfo:
        quietly(convert_torch_model, model, (8,), 10, packaging=PACKAGINGS["mvm"], value_program_stage=True)
    message = str(excinfo.value)
    assert "HostValueBatchLayoutError" in message and f"host op {operator} (FX node {node!r})" in message


# ----------------------------------------------------------------------------- no global random state
SAMPLE0_HOST_OPS = ["getitem", "getitem_1", "gt", "getitem_2", "getattr_1", "getitem_3", "mul", "add"]


@pytest.mark.parametrize("seed", range(40))
def test_the_conversion_reads_no_global_random_state(seed):
    """Under 8 of 40 global seeds the sample-0 count was folded (ShapeProp's unseeded draw agreed), wrong."""
    model = native(lambda: Extent("sample0_count"))
    torch.manual_seed(seed)
    flow = quietly(convert_torch_model, copy.deepcopy(model), (8,), 10, packaging=PACKAGINGS["spiking"]).eval()
    assert host_ops(flow) == SAMPLE0_HOST_OPS and flow.shape_literals == {}
    for batch in (1, 2, 3, 6):
        x = torch.randn(batch, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(1000 + batch))
        with torch.no_grad():
            assert relative_error(flow(x), model(x)).relative_error <= EXACT


class StaticLiteral(nn.Module):
    """``y * y.size(1)`` and ``y * y.shape[-1]``: two static shape values folded into literals."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        y = self.fc(x)
        return y * y.size(1) + y * y.shape[-1]


def _recording_conversions(monkeypatch, edit=None) -> list:
    """Record every flow ``trace_and_convert`` returns (the returned conversion first, then the witness)."""
    flows, original = [], converter.trace_and_convert

    def recording(*args, **kwargs):
        traced = original(*args, **kwargs)
        flows.append(traced.flow)
        if edit is not None:
            edit(len(flows), traced.flow)
        return traced

    monkeypatch.setattr(converter, "trace_and_convert", recording)
    return flows


@pytest.mark.parametrize("factory,literals", [
    (StaticLiteral, {"size": ("size", "6"), "getitem": ("getitem", "6")}),
    (lambda: Extent("sample0_count"), {}),
], ids=["static_literals", "sample0_count"])
def test_the_witness_conversion_folds_the_returned_conversions_literals(monkeypatch, factory, literals):
    flows = _recording_conversions(monkeypatch)
    torch.manual_seed(4)
    convert(native(factory), (8,), "spiking")
    assert len(flows) == 2 and flows[0].shape_literals == flows[1].shape_literals == literals


def _diverge_the_witness(call: int, flow) -> None:
    if call == 2:
        flow.shape_literals = {**flow.shape_literals, "size": ("size", "7")}


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_witness_folding_other_literals_is_refused_by_name(monkeypatch, contract):
    _recording_conversions(monkeypatch, _diverge_the_witness)
    with pytest.raises(WitnessProgramMismatchError) as excinfo:
        convert(native(StaticLiteral), (8,), contract)
    message = str(excinfo.value)
    assert "size (FX node 'size'): 6 in the returned conversion, 7 in the witness" in message


def test_a_permissive_conversion_with_a_diverging_witness_is_not_certified(monkeypatch):
    from mimarsinan.torch_mapping.value_parity.certificate import UnverifiedConversionWarning

    _recording_conversions(monkeypatch, _diverge_the_witness)
    with pytest.warns(UnverifiedConversionWarning, match="NOT certified"):
        flow = quietly(convert_torch_model, native(StaticLiteral), (8,), 10, strict=False)
    assert not flow.conversion_certificate.certified
    assert "FX node 'size'" in flow.conversion_certificate.summary()


# ----------------------------------------------------------------------------- batch slices
SLICE_BATCHES = (1, 2, 3, 6, 7, 8, 13)
BATCH_SLICE_HOST_OPS = ["getitem", "getattr_1", "getitem_1", "mul", "add"]
BATCH_SLICES = ["from6", "step6", "to_minus6", "from3", "from4", "step4", "head2"]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", BATCH_SLICES)
def test_a_batch_slice_extent_is_evaluated_at_run_time(how, contract):
    """``y[6:]``, ``y[::6]``, ``y[:-6]`` were folded (wrong from batch 7); ``y[3:]``, ``y[4:]``, ``y[::4]`` refused."""
    flow = assert_converted_exact(native(lambda: Extent(how)), (8,), contract, batches=SLICE_BATCHES)
    assert host_ops(flow) == BATCH_SLICE_HOST_OPS


# ----------------------------------------------------------------------------- the proofs themselves
class _Proofs(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)
        self.p = nn.Parameter(torch.randn(4, 6))

    def forward(self, x):
        y = self.fc(x)
        head = y[:2]
        count = x[x > 0.5]
        rows = x[x[:, 0] > -100.0]
        tail = y[6:]
        late = y.T
        return y + head.sum() + count.sum() + rows.sum() + tail.sum() + late.sum() + (2 * self.p).sum()


class _UnguardedProofs(_Proofs):
    """``_Proofs`` without ``y[6:]``, the one node requesting a guard on the batch symbol."""

    def forward(self, x):
        y = self.fc(x)
        head = y[:2]
        count = x[x > 0.5]
        rows = x[x[:, 0] > -100.0]
        late = y.T
        return y + head.sum() + count.sum() + rows.sum() + late.sum() + (2 * self.p).sum()


def _proofs(seed: int = 0, model_class=_Proofs):
    """The model drawn once; ``seed`` sets the global random state its trace (ShapeProp's draw) starts from."""
    torch.manual_seed(0)
    model = model_class().double().eval()
    torch.manual_seed(seed)
    gm = trace_model(model, (8,))
    return gm, BatchLayouts(gm), {n.name: n for n in gm.graph.nodes}


ROWS = "getitem_3"  # x[x[:, 0] > -100.0]


class TestProofs:
    """On ``_UnguardedProofs`` (round 7: a guard request anywhere leaves the whole program unproved, so the proof
    units run on the program without ``y[6:]``; ``_Proofs`` keeps it for the guard and violation units)."""

    def test_a_proved_axis_is_static_and_every_other_axis_runs_at_run_time(self):
        _, layouts, nodes = _proofs(model_class=_UnguardedProofs)
        assert layouts.layout(nodes["fc"]) == (BATCH_SIZE, 6)
        assert layouts.layout(nodes["gt"]) == (BATCH_SIZE, 8)

    def test_an_extent_static_from_batch_2_but_not_at_batch_1_is_batch_dependent(self):
        """``y[:2]`` has ``min(2, B)`` rows: 1 at batch 1, 2 above. [restated in review round 14: round 7 gave it the
        layout ``(BATCH_SIZE, 6)``; its length is no affine form of B, an opaque value, so no proof covers it]"""
        _, layouts, nodes = _proofs(model_class=_UnguardedProofs)
        assert layouts.layout(nodes["getitem"]) is None
        assert layouts.shape_decision(nodes["getitem"]).node is nodes["getitem"]

    def test_a_data_dependent_value_has_no_static_layout(self):
        for model_class in (_Proofs, _UnguardedProofs):
            _, layouts, nodes = _proofs(model_class=model_class)
            assert layouts.layout(nodes["getitem_1"]) is None and layouts.layout(nodes[ROWS]) is None

    def test_an_opaque_slice_of_the_batch_leaves_only_what_it_reaches_unproved(self):
        """``y[6:]`` has ``max(B - 6, 0)`` rows, no affine form of B: an opaque value. [restated in review round 14:
        its guard ``B >= 6`` left no value of the program proved in rounds 7 to 13; no fold rests on a guard now, and
        only the values computed from the opaque slice are unproved]"""
        _, layouts, nodes = _proofs()
        assert layouts.layout(nodes["fc"]) == (BATCH_SIZE, 6) and layouts.layout(nodes["mul"]) == (4, 6)
        assert layouts.layout(nodes["getattr_1"]) == (6, BATCH_SIZE) and layouts.layout(nodes["getitem_4"]) is None
        assert layouts.shape_decision(nodes["getitem_4"]).node is nodes["getitem_4"]
        assert layouts.proof_statement().endswith("(first: getitem, FX node 'getitem')")
        _, twin, names = _proofs(model_class=_UnguardedProofs)
        assert twin.layout(names["mul"]) == (4, 6) and twin.layout(names["sum_5"]) == ()
        assert twin.layout(names["getattr_1"]) == (6, BATCH_SIZE)

    def test_a_violation_is_observed_at_a_traced_batch_or_proved_for_every_batch(self):
        """``x[x[:, 0] > -100]`` keeps every row at every traced batch; only the proof sees its data-dependent axis."""
        _, layouts, nodes = _proofs()
        assert layouts.violation(nodes["fc"]) is None
        assert layouts.violation(nodes["getitem_1"]).startswith("axis 0 is not the batch axis ((")
        assert layouts.violation(nodes[ROWS]) == (
            "axis 0 is not the batch axis ((u1, 8) for every batch B >= 1, u1 a data-dependent size)"  # [restated in
            # review round 14: "B >= 2"; the domain holds every batch from 1]
        )

    def test_the_layouts_read_no_global_random_state(self):
        before = torch.random.get_rng_state()
        layouts = []
        for seed in (0, 1, 2):
            gm, found, nodes = _proofs(seed)
            state = torch.random.get_rng_state()
            layouts.append({name: (found.layout(node), found.violation(node)) for name, node in nodes.items()})
            assert torch.equal(torch.random.get_rng_state(), state)
        assert layouts[0] == layouts[1] == layouts[2]
        torch.random.set_rng_state(before)
