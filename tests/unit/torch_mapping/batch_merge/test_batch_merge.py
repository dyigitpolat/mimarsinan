"""The batch-merge push-forward (coverage W7): a reshape merging the batch into a window axis, ``(B * n, ...)``, read
only by operations independent along that axis until a reshape splits it back, is rewritten batch-leading.

On the base every such reshape was refused by name (``BatchAxisLayoutError``: "axis 0 is not the batch axis"):
torchvision's window partition ``reshape(B * num_windows, N, C)`` in Swin's attention, after leaf-call inlining. Now
the merged value is carried split, ``(B, n, ...)``, with every reader's axes shifted (``fx_rewrites/batch_merge``), and
the program converts exactly, both stages. A merged value read by an operation that mixes the merged axis (a
reduction, a scan, a flip, a roll, a softmax or a normalization along it, a concatenation or slice of it, a contraction
over it, a random draw), merged or split window-major, broadcast against a batch-leading value, or returned, keeps the
traced reshape and is refused by name as before.
"""

from __future__ import annotations

import pytest
import torch
import torch.fx
import torch.fx as fx

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.torch_mapping.converter_handlers.refusals import BatchAxisLayoutError
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import program_purity
from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityError
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from batch_merge import models as M
from operand_completeness.harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_value_program_exact,
    convert,
    keep_opaque,
    native,
    worst_error,
)

SHAPE = M.SHAPE
BATCHES = (1, 2, 3, 5, 9)

CONVERTED = {"win_attn": M.WinAttn, "win_attn_modules": M.WinAttnModules, "win_mask": M.WinMask,
             "multiplicity": M.Multiplicity, "flatten_merge": M.FlattenMerge, "token_slice": M.TokenSlice,
             "size_split": M.SizeSplit, "cosine": M.Cosine, "matrix_left": M.MatrixLeft}

MIXING = {"sum0": M.Sum0, "softmax0": M.Softmax0, "flip0": M.Flip0, "cumsum0": M.Cumsum0, "roll0": M.Roll0,
          "roll0_method": M.Roll0Method, "normalize0": M.Normalize0, "cat0": M.Cat0, "slice0": M.Slice0,
          "size_scale": M.SizeScale, "contract0": M.Contract0, "transpose_matmul": M.TransposeMatmul,
          "dropout_train": M.DropoutTrain, "window_major": M.WindowMajor, "split_window_major": M.SplitWindowMajor,
          "merged_output": M.MergedOutput, "max0": M.MaxDim0, "min0": M.MinDim0, "unbind0": M.Unbind0,
          "select0": M.Select0, "narrow0": M.Narrow0, "sort0": M.Sort0, "logsumexp0": M.LogSumExp0,
          "softmax_module0": M.SoftmaxModule0, "named_pointwise_helper": M.NamedPointwiseHelper,
          "named_dropout_helper": M.NamedDropoutHelper}

CONSERVATIVE = {"left_broadcast": M.LeftBroadcast, "expand_token": M.ExpandToken, "batch_operand": M.BatchOperand,
                "dtype_view": M.DtypeView, "tensor_start_narrow": M.TensorStartNarrow}


def _normalized(factory) -> fx.GraphModule:
    gm = trace_model(native(factory), SHAPE)
    return normalize_fx_graph(gm, program_purity(gm).foreign_modules)


def _activations(gm: fx.GraphModule) -> list:
    """Every node with a tensor value computed from the input."""
    found, reached = [], set()
    for node in gm.graph.nodes:
        if node.op == "placeholder" or any(arg in reached for arg in node.all_input_nodes):
            reached.add(node)
            if hasattr(node.meta.get("tensor_meta"), "shape") and node.op != "output":
                found.append(node)
    return found


@pytest.mark.parametrize("name", sorted(CONVERTED))
def test_a_merged_region_is_rewritten_batch_leading(name):
    """After the rewrite every activation of the program carries the batch on axis 0 (extent 1 at the trace batch)."""
    gm = _normalized(CONVERTED[name])
    merged = [(n.name, tuple(n.meta["tensor_meta"].shape)) for n in _activations(gm)
              if n.meta["tensor_meta"].shape and n.meta["tensor_meta"].shape[0] != 1]
    assert not merged, merged


@pytest.mark.parametrize("name", sorted(CONVERTED))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_merged_region_converts_exactly_at_every_batch(name, contract):
    flow = assert_converted_exact(native(CONVERTED[name]), SHAPE, contract, batches=BATCHES)
    assert flow.shape_impurity is None


@pytest.mark.parametrize("name", sorted(CONVERTED))
def test_the_value_program_of_a_merged_region_is_exact(name):
    assert_value_program_exact(native(CONVERTED[name]), SHAPE, batches=(2, 3, 9))


@pytest.mark.parametrize("name", sorted(MIXING))
def test_a_merged_value_read_by_an_op_mixing_the_merged_axis_is_still_refused_by_name(name):
    with pytest.raises(ToolchainRefusal) as refused:
        convert(native(MIXING[name]), SHAPE, "mvm")
    assert isinstance(refused.value, (BatchAxisLayoutError, RepresentabilityError)), refused.value
    if isinstance(refused.value, BatchAxisLayoutError):
        assert "is not the batch axis" in str(refused.value) or "carries the batch" in str(refused.value), \
            refused.value


@pytest.mark.parametrize("name", sorted(CONSERVATIVE))
def test_a_form_outside_the_rules_is_refused_by_name_or_exact(name):
    """Exact natively and outside the rewrite's rules: refused by name, or converted exactly; never wrong."""
    model = native(CONSERVATIVE[name])
    try:
        flow = convert(model, SHAPE, "mvm")
    except ToolchainRefusal:
        return
    assert worst_error(flow, model, SHAPE, batches=BATCHES) <= 1e-12


@torch.fx.wrap
def opaque_scale(t):
    keep_opaque(t)
    return t * 2.0


class ImpureWinAttn(M.WinAttn):
    """The window attention in an impure program: no static extent is proved, so nothing is rewritten."""

    def forward(self, x):
        return super().forward(opaque_scale(x))


def test_an_impure_program_keeps_the_merged_reshape_refused_by_name():
    gm = trace_model(native(ImpureWinAttn), SHAPE)
    assert not program_purity(gm).pure
    with pytest.raises(BatchAxisLayoutError, match="axis 0 is not the batch axis"):
        convert(native(ImpureWinAttn), SHAPE, "mvm")


@pytest.mark.parametrize("name", sorted(CONVERTED))
def test_the_rewrite_keeps_every_value_of_the_native_graph(name):
    """The normalized graph computes the native model's output at several batches (the rewrite alone, no mapper)."""
    model, gm = native(CONVERTED[name]), _normalized(CONVERTED[name])
    for batch in (1, 2, 7):
        x = torch.randn(batch, *SHAPE, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            got, want = gm(x), model(x)
        assert (got - want).abs().max() <= 1e-12 * max(1.0, float(want.abs().max())), (name, batch)


class Dropout(torch.nn.Dropout):
    """A user class of torch's dropout's name whose call flips axis 0."""

    def forward(self, input):
        return input.flip(0)


class Gelu(torch.nn.GELU):
    def forward(self, input):
        return input - input.mean(0, keepdim=True)


@pytest.mark.parametrize("module, kind", [
    (torch.nn.GELU(), "pointwise"), (torch.nn.ReLU(), "pointwise"), (torch.nn.Dropout().eval(), "identity"),
    (torch.nn.Identity(), "identity"), (torch.nn.Softmax(dim=-1), "fiber"), (torch.nn.Linear(2, 2), "linear"),
    (torch.nn.Dropout().train(), None), (torch.nn.GLU(), None), (torch.nn.PReLU(), None), (torch.nn.RReLU(), None),
    (Dropout().eval(), None), (Gelu(), None)], ids=lambda value: type(value).__name__ if value is not None else "-")
def test_a_called_module_takes_its_axis_rule_by_its_exact_torch_class(module, kind):
    """A subclass (here one of torch's dropout's name) may override ``forward``: only torch's own class has a rule;
    ``nn.GLU`` halves an axis, ``nn.PReLU`` has a weight and ``nn.RReLU`` draws a slope: none is pointwise. (Imported
    here: the module is new in coverage W7, and on the base this test fails alone, the file still collected.)"""
    from mimarsinan.torch_mapping.fx_rewrites.batch_merge.axes import module_kind

    assert module_kind(module) == kind
