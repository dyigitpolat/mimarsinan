"""Unknown tensor call_methods must FAIL LOUD at conversion, never silently drop.

The regression this pins: ``x.softmax(dim=-1)`` (method form) used to hit the
converter's silent identity fallthrough — the op vanished from the program and
the converted flow diverged from the native model with no error. Unknown
call_methods are now rejected by the representability analyzer AND the
converter; the deliberate structural set stays as an explicit allowlist.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import (
    check_representability,
    convert_torch_model,
)
from mimarsinan.torch_mapping.representability_analyzer import (
    CONVERTIBLE_CALL_METHODS,
    RepresentabilityError,
)


class SoftmaxMethodModel(nn.Module):
    """Linear -> .softmax(dim=-1) (METHOD form) -> Linear."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        return self.fc2(self.fc1(x).softmax(dim=-1))


class ClampMethodModel(nn.Module):
    """Linear -> .clamp(min=0) (a compute method with no conversion rule)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(12, 10)

    def forward(self, x):
        return self.fc(x).clamp(min=0.0)


class NoRuleMethodModel(nn.Module):
    """Linear -> .half() (a method with no torch function of its name, so no conversion rule) -> Linear."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        return self.fc2(self.fc1(x).half().float())


def _assert_converts_exactly(model, shape):
    torch.manual_seed(0)
    flow = convert_torch_model(model.double().eval(), shape, num_classes=10, packaging=MVM_PACKAGING)
    x = torch.randn(5, *shape, dtype=torch.float64)
    with torch.no_grad():
        assert torch.equal(flow(x), model(x))


class TestUnknownCallMethodFailsLoud:
    """[changed in Wave 1 G1: ``.softmax`` and ``.clamp`` have a torch function of their name, so the method rule
    converts them as the host op of the method itself; the refusal is pinned on a method with none, ``.half``]"""

    def test_softmax_method_converts_exactly(self):
        _assert_converts_exactly(SoftmaxMethodModel(), (12,))

    def test_softmax_method_is_representable_by_the_method_rule(self):
        report = check_representability(SoftmaxMethodModel().eval(), (12,), packaging=MVM_PACKAGING)
        assert report.is_representable
        assert [o.node_name for o in report.supported_ops if o.module_type == "softmax"]

    def test_clamp_method_converts_exactly(self):
        _assert_converts_exactly(ClampMethodModel(), (12,))

    def test_a_method_with_no_rule_raises_at_conversion_naming_the_method(self):
        with pytest.raises(RepresentabilityError, match="half"):
            convert_torch_model(NoRuleMethodModel().eval(), (12,), num_classes=10, packaging=MVM_PACKAGING)
        report = check_representability(NoRuleMethodModel().eval(), (12,), packaging=MVM_PACKAGING)
        [op] = [o for o in report.unsupported_ops if o.module_type == "half"]
        assert op.op_type == "call_method" and op.node_name and "half" in (op.reason or "")

    def test_converter_layer_raises_without_analyzer_screening(self):
        """The converter itself must refuse unknown methods (defense in depth:
        it is reachable without the analyzer's screening)."""
        from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
        from mimarsinan.torch_mapping.mapper_graph_converter import (
            MapperGraphConverter,
            UnsupportedCallMethodError,
        )
        from mimarsinan.torch_mapping.representability_analyzer import (
            RepresentabilityReport,
        )
        from mimarsinan.torch_mapping.torch_graph_tracer import trace_model

        model = NoRuleMethodModel().eval()
        gm = normalize_fx_graph(trace_model(model, (12,)))
        converter = MapperGraphConverter(gm, (12,), packaging=MVM_PACKAGING)
        permissive = RepresentabilityReport(is_representable=True)
        with pytest.raises(UnsupportedCallMethodError, match="half"):
            converter.convert(permissive)


class AllowlistedMethodsModel(nn.Module):
    """Exercises the structural call_method allowlist end to end:
    size, contiguous, view, reshape, transpose, permute, unsqueeze, squeeze,
    add, mean, flatten — all as METHODS."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.head = nn.Linear(8, 4)

    def forward(self, x):  # x: (B, 4, 8)
        b = x.size(0)
        y = x.contiguous().view(b, 4, 8).reshape(b, 4, 8)
        y = y.transpose(1, 2).permute(0, 2, 1)      # net identity: back to (B, 4, 8)
        y = y.unsqueeze(-1).squeeze(-1)             # net identity
        y = self.fc(y)
        y = y.add(x)                                # residual via METHOD add
        y = y.mean(1)                               # (B, 8)
        y = y.flatten(1)
        return self.head(y)


class ExpandCatModel(nn.Module):
    """cls-token style ``Parameter.expand`` + cat: the deliberate expand rule."""

    def __init__(self):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, 8))
        self.head = nn.Linear(8, 4)

    def forward(self, x):  # x: (B, 3, 8)
        tok = self.token.expand(x.shape[0], -1, -1)
        y = torch.cat((tok, x), dim=1)
        return self.head(y[:, 0])


class TestAllowlistedMethodsStillConvert:
    def test_structural_methods_convert_and_match_native(self):
        torch.manual_seed(0)
        model = AllowlistedMethodsModel().eval()
        flow = convert_torch_model(
            model, (4, 8), num_classes=4, packaging=MVM_PACKAGING
        )
        x = torch.randn(6, 4, 8)
        with torch.no_grad():
            want = model(x)
            got = flow(x)
        torch.testing.assert_close(got.reshape(want.shape), want,
                                   atol=1e-6, rtol=1e-5)

    def test_expand_cat_converts_and_matches_native(self):
        torch.manual_seed(1)
        model = ExpandCatModel().eval()
        flow = convert_torch_model(
            model, (3, 8), num_classes=4, packaging=MVM_PACKAGING
        )
        x = torch.randn(5, 3, 8)
        with torch.no_grad():
            want = model(x)
            got = flow(x)
        torch.testing.assert_close(got.reshape(want.shape), want,
                                   atol=1e-6, rtol=1e-5)

    def test_allowlist_is_the_deliberate_structural_set(self):
        # The allowlist is an explicit contract: additions must be conscious.
        assert CONVERTIBLE_CALL_METHODS == frozenset({
            "view", "reshape", "flatten", "contiguous",
            "permute", "transpose", "mean",
            "size", "dim", "unsqueeze", "squeeze", "expand",
            "add", "__add__",
        })
