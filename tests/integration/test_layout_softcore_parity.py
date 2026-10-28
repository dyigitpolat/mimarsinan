"""Pins the "single source of truth" invariant: the softcores produced by
the shape-only ``LayoutIRMapping`` path are byte-identical to those produced
by the full ``IRMapping`` path that the pipeline uses."""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests/integration/fixtures/cifar_vit_wizard_body.json"


@dataclass
class _Tally:
    inside: int = 0
    outside: int = 0
    inside_wall_s: float = 0.0


class _ClockSpy:
    """Counts calls of module-level functions apart inside and outside a timed region, so a speed guard can show
    what its clock measures."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch
        self._timing = False
        self.calls: dict[str, _Tally] = {}

    def watch(self, module: object, name: str) -> None:
        real = getattr(module, name)
        tally = self.calls.setdefault(name, _Tally())

        def counted(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return real(*args, **kwargs)
            finally:
                if self._timing:
                    tally.inside += 1
                    tally.inside_wall_s += time.perf_counter() - t0
                else:
                    tally.outside += 1

        self._monkeypatch.setattr(module, name, counted)

    @contextlib.contextmanager
    def timed_region(self):
        self._timing = True
        try:
            yield
        finally:
            self._timing = False


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
    from mimarsinan.mapping.ir_mapping_class import IRMapping
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
        assert a.residency_class_id == b.residency_class_id, (
            f"residency_class_id drift @ {i}"
        )
        assert a.latency_tag == b.latency_tag, f"latency_tag drift @ {i}"
        assert a.segment_id == b.segment_id, f"segment_id drift @ {i}"


def _time_cifar_vit_layout_step(monkeypatch: pytest.MonkeyPatch):
    """The wizard layout route on the CIFAR-ViT body with its conversion done first. The route's service, fresh,
    converts the model once (``get_model_repr``, which runs the value-parity certificate) before the clock starts; the
    timed call is the route itself, which takes the converted representation from the service and runs the layout step
    on it (the shape-only softcore layout and its verification).

    Returns ``(result, elapsed_s, spy)``; the spy counts the conversions and the layout steps on either side of the
    clock.
    """
    from mimarsinan.gui.server.routes_layout import get_layout_result_from_request
    from mimarsinan.mapping.verification import layout_mapping_service, wizard_layout_verify
    from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest

    spy = _ClockSpy(monkeypatch)
    spy.watch(wizard_layout_verify, "convert_torch_model")
    spy.watch(layout_mapping_service, "verify_soft_core_mapping")
    service = layout_mapping_service.LayoutMappingService()
    monkeypatch.setattr(layout_mapping_service, "DEFAULT_LAYOUT_MAPPING_SERVICE", service)
    body = json.loads(_FIXTURE.read_text())
    service.get_model_repr(LayoutMappingRequest.from_wizard_body(body))

    with spy.timed_region():
        t0 = time.perf_counter()
        result = get_layout_result_from_request(body)
        elapsed = time.perf_counter() - t0
    return result, elapsed, spy


@pytest.mark.slow
def test_cifar_vit_layout_under_3_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Performance regression guard of the shape-only layout step: on the CIFAR-ViT wizard body, the layout call that
    follows the conversion (the softcore layout of the converted representation and its verification) must run in
    under 3 seconds. Pre-fix it took ~28 seconds. The conversion and its value-parity certificate run once before the
    clock starts; the route still runs them, and they keep their own tests."""
    if os.environ.get("MIMARSINAN_SKIP_PERF") == "1":
        pytest.skip("MIMARSINAN_SKIP_PERF=1")

    result, elapsed, spy = _time_cifar_vit_layout_step(monkeypatch)
    conversion = spy.calls["convert_torch_model"]
    layout = spy.calls["verify_soft_core_mapping"]

    assert conversion.inside == 0, (
        f"the timed region ran the torch conversion {conversion.inside} time(s) "
        f"({conversion.inside_wall_s:.2f} s of {elapsed:.2f} s): the guard measures the "
        "conversion and its certificate, not the layout step"
    )
    assert conversion.outside == 1, (
        f"the setup ran the torch conversion {conversion.outside} time(s), expected once: "
        "the spy no longer sees the route's conversion"
    )
    assert layout.inside == 1, (
        f"the timed region ran the layout step {layout.inside} time(s), expected once"
    )
    assert result.feasible, result.error
    assert elapsed < 3.0, (
        f"CIFAR-ViT layout step took {elapsed:.2f}s; target < 3.0s. "
        "Regression in the shape-only path."
    )
