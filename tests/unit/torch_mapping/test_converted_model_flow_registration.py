"""Regression: every nn.Module in the mapper graph must be a registered submodule.

If a module is referenced by ``_mapper_repr._exec_order`` but not in
``ConvertedModelFlow._modules``, PyTorch's base ``nn.Module._apply`` won't
walk it, ``.to(device)`` silently leaves it behind, and the warmup forward
can emit the ``"module is not installed as a submodule"`` warning. This test
builds a few representative models and checks the invariant.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from mimarsinan.torch_mapping.converter import convert_torch_model


class _TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        return self.fc(self.flat(self.pool(self.relu(self.conv(x)))))


def _all_graph_modules(flow):
    ids = set()
    flow.get_mapper_repr()._ensure_exec_graph()
    for node in flow.get_mapper_repr()._exec_order:
        if isinstance(node, nn.Module):
            ids.add(id(node))
        p = getattr(node, "perceptron", None)
        if isinstance(p, nn.Module):
            ids.add(id(p))
    return ids


class TestRegistration:
    def test_all_graph_modules_are_registered(self):
        m = _TinyCNN()
        m.eval()
        flow = convert_torch_model(m, (3, 8, 8), 10, device="cpu")

        registered = {id(mod) for mod in flow.modules()}
        graph_ids = _all_graph_modules(flow)
        missing = graph_ids - registered
        assert not missing, f"{len(missing)} graph modules not registered as submodules"

    def test_params_cover_graph(self):
        m = _TinyCNN()
        m.eval()
        flow = convert_torch_model(m, (3, 8, 8), 10, device="cpu")
        registered_params = {id(p) for p in flow.parameters()}

        graph_param_ids = set()
        flow.get_mapper_repr()._ensure_exec_graph()
        for node in flow.get_mapper_repr()._exec_order:
            for child in [node, getattr(node, "perceptron", None)]:
                if isinstance(child, nn.Module):
                    for p in child.parameters(recurse=True):
                        graph_param_ids.add(id(p))
        missing = graph_param_ids - registered_params
        assert not missing, f"{len(missing)} graph params not in flow.parameters()"

    def test_warmup_no_submodule_warning(self):
        m = _TinyCNN()
        m.eval()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            convert_torch_model(m, (3, 8, 8), 10, device="cpu")

        submodule_warnings = [
            w for w in caught if "not installed as a submodule" in str(w.message)
        ]
        assert not submodule_warnings, (
            f"warmup produced {len(submodule_warnings)} submodule warning(s): "
            f"{[str(w.message) for w in submodule_warnings]}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
class TestRegistrationCuda:
    def test_to_cuda_moves_everything(self):
        m = _TinyCNN()
        m.eval()
        flow = convert_torch_model(m, (3, 8, 8), 10, device="cpu")
        flow = flow.to("cuda")

        for name, p in flow.named_parameters():
            assert p.device.type == "cuda", f"parameter {name} is on {p.device}"
        for name, b in flow.named_buffers():
            assert b.device.type == "cuda", f"buffer {name} is on {b.device}"
