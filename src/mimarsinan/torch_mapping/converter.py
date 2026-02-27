"""
Public API for converting native PyTorch models to mimarsinan Supermodels.
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
from mimarsinan.models.supermodel import Supermodel
from mimarsinan.models.preprocessing.input_cq import InputCQ


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
    analyzer = RepresentabilityAnalyzer(gm)
    return analyzer.analyze()


def convert_torch_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_classes: int,
    device: Union[torch.device, str] = "cpu",
    Tq: int = 64,
) -> Supermodel:
    """Convert a trained native PyTorch model to a mimarsinan ``Supermodel``.

    Steps:
        1. Trace the model with ``torch.fx``.
        2. Validate representability.
        3. Convert to a Mapper DAG with Perceptron wrappers.
        4. Transfer trained weights.
        5. Wrap in ``ConvertedModelFlow`` + ``Supermodel``.

    Args:
        model: A trained ``nn.Module``.
        input_shape: Input shape without batch, e.g. ``(3, 32, 32)``.
        num_classes: Number of output classes.
        device: Device for the tracing/warmup passes.
        Tq: Input quantization levels (passed to ``InputCQ`` preprocessor).

    Returns:
        A ``Supermodel`` ready for the adaptation / quantization pipeline.

    Raises:
        TracingError: If the model cannot be symbolically traced.
        RepresentabilityError: If the model contains unsupported operations.
    """
    device = torch.device(device)

    gm = trace_model(model, input_shape, device=device)

    analyzer = RepresentabilityAnalyzer(gm)
    report = analyzer.analyze()

    if not report.is_representable:
        raise RepresentabilityError(report)

    converter = MapperGraphConverter(gm, input_shape)
    mapper_repr = converter.convert(report)

    flow = ConvertedModelFlow(device, mapper_repr)

    preprocessor = InputCQ(Tq)
    supermodel = Supermodel(
        device=device,
        input_shape=input_shape,
        num_classes=num_classes,
        preprocessor=preprocessor,
        perceptron_flow=flow,
        Tq=Tq,
    )

    supermodel = supermodel.to(device)

    # Warmup forward pass to initialise any lazy modules (e.g. LazyBatchNorm1d
    # inside Conv2DPerceptronMapper).
    try:
        supermodel.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape, device=device)
            _ = supermodel(dummy)
    except Exception as e:
        print(f"[convert_torch_model] Warmup forward failed (non-fatal): {e}")

    return supermodel
