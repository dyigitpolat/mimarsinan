"""Pins the "single source of truth" invariant: the softcores produced by
the shape-only ``LayoutIRMapping`` path are byte-identical to those produced
by the full ``IRMapping`` path that the pipeline uses."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests/integration/fixtures/cifar_vit_wizard_body.json"


def _tiny_vit_like() -> nn.Module:
    """A small but ViT-flavored nn.Module that exercises Linear/LayerNorm/
    residual paths without paying ImageNet ViT-B/16's wall-clock cost."""

    class TinyBlock(nn.Module):
        def __init__(self, dim: int = 32, mlp_dim: int = 64) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, dim)
            self.ln2 = nn.LayerNorm(dim)
            self.fc2 = nn.Linear(dim, mlp_dim)
            self.fc3 = nn.Linear(mlp_dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.fc1(self.ln1(x))
            x = x + self.fc3(torch.relu(self.fc2(self.ln2(x))))
            return x

    class TinyViTLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(16, 32)
            self.block = TinyBlock()
            self.head = nn.Linear(32, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            x = self.block(x)
            return self.head(x)

    return TinyViTLike()


def _build_model_repr(model: nn.Module, input_shape):
    from mimarsinan.torch_mapping.converter import convert_torch_model

    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, *input_shape))
    flow = convert_torch_model(
        model, input_shape=input_shape, num_classes=4, device="cpu",
    )
    repr_ = flow.get_mapper_repr()
    if hasattr(repr_, "assign_perceptron_indices"):
        repr_.assign_perceptron_indices()
    return repr_


def test_layout_softcores_identical_between_view_path_and_full_ir_path() -> None:
    """``LayoutIRMapping`` (view path) and ``IRMapping`` (full IR path) must
    produce the same list of softcores -- otherwise the wizard's layout
    estimate would lie to the user about what the pipeline will actually
    build."""
    from mimarsinan.mapping.ir_mapping import IRMapping
    from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping

    model = _tiny_vit_like()
    repr_ = _build_model_repr(model, input_shape=(16,))

    layout = LayoutIRMapping(
        max_axons=4096, max_neurons=4096,
        allow_coalescing=False, hardware_bias=True,
    )
    layout_softcores = layout.collect_layout_softcores(repr_)

    repr_full = _build_model_repr(_tiny_vit_like(), input_shape=(16,))
    irm = IRMapping(
        q_max=1, firing_mode="Default",
        max_axons=4096, max_neurons=4096,
        allow_coalescing=False, hardware_bias=True,
    )
    irm.map(repr_full)
    full_softcores = irm.layout_softcores

    assert len(layout_softcores) == len(full_softcores), (
        f"softcore count drift: layout={len(layout_softcores)} "
        f"full={len(full_softcores)}"
    )
    for i, (a, b) in enumerate(zip(layout_softcores, full_softcores)):
        assert a.input_count == b.input_count, f"input_count drift @ {i}"
        assert a.output_count == b.output_count, f"output_count drift @ {i}"
        assert a.threshold_group_id == b.threshold_group_id, (
            f"threshold_group_id drift @ {i}"
        )
        assert a.latency_tag == b.latency_tag, f"latency_tag drift @ {i}"
        assert a.segment_id == b.segment_id, f"segment_id drift @ {i}"


@pytest.mark.slow
def test_cifar_vit_layout_under_3_seconds() -> None:
    """Performance regression guard: the full CIFAR-ViT layout call must run
    in under 3 seconds. Pre-fix it took ~28 seconds."""
    if os.environ.get("MIMARSINAN_SKIP_PERF") == "1":
        pytest.skip("MIMARSINAN_SKIP_PERF=1")

    from mimarsinan.gui.server import _get_layout_result_from_request

    body = json.loads(_FIXTURE.read_text())

    t0 = time.perf_counter()
    result = _get_layout_result_from_request(body)
    elapsed = time.perf_counter() - t0

    assert result.feasible
    assert elapsed < 3.0, (
        f"CIFAR-ViT layout took {elapsed:.2f}s; target < 3.0s. "
        "Regression in the shape-only path."
    )
