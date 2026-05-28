"""Warmup failures during conversion must propagate by default (no silent prints)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.torch_mapping.conversion_probe import (
    ConversionProbeError,
    ProbeResult,
    probe_forward,
)


class _AlwaysRaisingFlow(nn.Module):
    def __init__(self, message: str = "boom"):
        super().__init__()
        self.message = message

    def forward(self, x):  # noqa: ARG002 - intentionally ignores input
        raise RuntimeError(
            f"[ModelRepresentation] forward failed at node "
            f"ComputeOpMapper(name='fake_add') --- {self.message}"
        )


class TestStrictProbe:
    def test_strict_reraises_with_failing_node_name(self):
        flow = _AlwaysRaisingFlow()
        with pytest.raises(ConversionProbeError) as excinfo:
            probe_forward(flow, (4,), device="cpu", batch=1, strict=True)
        assert "fake_add" in str(excinfo.value)

    def test_permissive_returns_diagnostic(self):
        flow = _AlwaysRaisingFlow()
        result = probe_forward(flow, (4,), device="cpu", batch=1, strict=False)
        assert isinstance(result, ProbeResult)
        assert result.ok is False
        assert result.error is not None
        assert result.failing_node_name == "fake_add"
        message = result.format()
        assert "fake_add" in message

    def test_success_returns_ok_result(self):
        flow = nn.Linear(4, 2)
        result = probe_forward(flow, (4,), device="cpu", batch=1, strict=True)
        assert isinstance(result, ProbeResult)
        assert result.ok is True
        assert result.error is None
        assert result.failing_node_name is None


class TestConverterDefaultsStrict:
    def test_convert_torch_model_propagates_warmup_failure(self, monkeypatch):
        """Strict-mode probe failure must surface as ``ConversionProbeError``."""
        from mimarsinan.torch_mapping import conversion_probe
        from mimarsinan.torch_mapping.converter import convert_torch_model

        def failing_probe(
            flow, input_shape, device, *, batch=1, strict=True, context="",
        ):
            if strict:
                raise ConversionProbeError(
                    "[convert_torch_model] simulated failure at "
                    "node ComputeOpMapper(name='bad_node')"
                )
            return ProbeResult(
                ok=False,
                error=RuntimeError("simulated"),
                failing_node_name="bad_node",
                context=context,
            )

        monkeypatch.setattr(conversion_probe, "probe_forward", failing_probe)

        class _OK(nn.Module):
            def __init__(self):
                super().__init__()
                self.l = nn.Linear(8, 4)

            def forward(self, x):
                return torch.relu(self.l(x))

        with pytest.raises(ConversionProbeError) as excinfo:
            convert_torch_model(_OK(), (8,), num_classes=4, device="cpu")
        assert "bad_node" in str(excinfo.value)
