"""
Public API for converting native PyTorch models to mimarsinan PerceptronFlow (ConvertedModelFlow).
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn

from mimarsinan.torch_mapping.torch_graph_tracer import trace_model, TracingError
from mimarsinan.torch_mapping.representability_analyzer import (
    RepresentabilityAnalyzer,
    RepresentabilityReport,
    RepresentabilityError,
)
from mimarsinan.torch_mapping.mapper_graph_converter import MapperGraphConverter
from mimarsinan.torch_mapping.converted_model_flow import ConvertedModelFlow
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers


def check_representability(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Union[torch.device, str] = "cpu",
) -> RepresentabilityReport:
    """Check whether a native PyTorch model can be represented in mimarsinan IR.

    This traces the model, walks the FX graph, and classifies every
    operation as supported, absorbable, or unsupported.

    Args:
        model: The model to check.
        input_shape: Input shape without batch dim, e.g. ``(3, 32, 32)``.
        device: Device for the tracing forward pass.

    Returns:
        A ``RepresentabilityReport`` describing what is and isn't supported.
    """
    gm = trace_model(model, input_shape, device=device)
    from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
    gm = normalize_fx_graph(gm)
    analyzer = RepresentabilityAnalyzer(gm)
    return analyzer.analyze()


def convert_torch_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_classes: int,
    device: Union[torch.device, str] = "cpu",
    Tq: int = 64,
) -> ConvertedModelFlow:
    """Convert a trained native PyTorch model to a ``ConvertedModelFlow``.

    Steps:
        1. Trace the model with ``torch.fx``.
        2. Validate representability.
        3. Convert to a Mapper DAG with Perceptron wrappers.
        4. Transfer trained weights.
        5. Build ``ConvertedModelFlow`` and mark encoding-layer perceptrons.

    Args:
        model: A trained ``nn.Module``.
        input_shape: Input shape without batch, e.g. ``(3, 32, 32)``.
        num_classes: Number of output classes.
        device: Device for the tracing/warmup passes.
        Tq: Unused (kept for API compatibility); input quantization is not applied here.

    Returns:
        A ``ConvertedModelFlow`` ready for the adaptation / quantization pipeline.

    Raises:
        TracingError: If the model cannot be symbolically traced.
        RepresentabilityError: If the model contains unsupported operations.
    """
    device = torch.device(device)

    gm = trace_model(model, input_shape, device=device)

    from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
    gm = normalize_fx_graph(gm)

    analyzer = RepresentabilityAnalyzer(gm)
    report = analyzer.analyze()

    if not report.is_representable:
        raise RepresentabilityError(report)

    converter = MapperGraphConverter(gm, input_shape)
    mapper_repr = converter.convert(report)

    flow = ConvertedModelFlow(device, mapper_repr)
    mark_encoding_layers(flow.get_mapper_repr())

    flow = flow.to(device)

    # Warmup forward pass to initialise any lazy modules (e.g. LazyBatchNorm1d
    # inside Conv2DPerceptronMapper).
    import os
    debug = os.environ.get("MIMARSINAN_CUDA_DEBUG") == "1"
    try:
        flow.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape, device=device)
            _ = flow(dummy)
    except Exception as e:
        if debug:
            raise
        print(
            f"[convert_torch_model] Warmup forward failed (non-fatal): "
            f"{type(e).__name__}: {e}"
        )

    return flow
