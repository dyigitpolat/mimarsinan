"""Public API for converting native PyTorch models to a mimarsinan ConvertedModelFlow."""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn

from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.representability_analyzer import (
    RepresentabilityAnalyzer,
    RepresentabilityReport,
    RepresentabilityError,
)
from mimarsinan.torch_mapping.mapper_graph_converter import MapperGraphConverter
from mimarsinan.torch_mapping.converted_model_flow import ConvertedModelFlow
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph


def check_representability(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Union[torch.device, str] = "cpu",
) -> RepresentabilityReport:
    """Trace a native PyTorch model and classify every op as supported, absorbable, or unsupported."""
    gm = trace_model(model, input_shape, device=device)
    gm = normalize_fx_graph(gm)
    analyzer = RepresentabilityAnalyzer(gm)
    return analyzer.analyze()


def convert_torch_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_classes: int,
    device: Union[torch.device, str] = "cpu",
    Tq: int = 64,
    *,
    strict: bool = True,
    encoding_layer_placement: str = "subsume",
) -> ConvertedModelFlow:
    """Convert a trained native PyTorch model to a ``ConvertedModelFlow``.

    ``Tq`` is unused (kept for API compatibility). With ``strict=True`` a failing
    warmup forward raises ``ConversionProbeError``; otherwise a known-broken flow is returned.
    """
    # Imported lazily so tests can monkeypatch conversion_probe.probe_forward at call time.
    from mimarsinan.torch_mapping.conversion_probe import probe_forward

    device = torch.device(device)

    gm = trace_model(model, input_shape, device=device)
    gm = normalize_fx_graph(gm)

    analyzer = RepresentabilityAnalyzer(gm)
    report = analyzer.analyze()

    if not report.is_representable:
        raise RepresentabilityError(report)

    converter = MapperGraphConverter(gm, input_shape)
    mapper_repr = converter.convert(report)

    flow = ConvertedModelFlow(device, mapper_repr)
    mark_encoding_layers(flow.get_mapper_repr(), placement=encoding_layer_placement)

    flow = flow.to(device)

    result = probe_forward(
        flow, input_shape, device=device, batch=1,
        strict=strict, context="convert_torch_model",
    )
    if not result.ok:
        print(f"[convert_torch_model] {result.format()}")

    return flow
