"""Legacy re-export facade for mapping types; import from the concrete modules in new code."""

from mimarsinan.code_generation.cpp_chip_model import *
from mimarsinan.models.nn.layers import *
from mimarsinan.mapping.packing.softcore import *
from mimarsinan.transformations.weight_quantization import *

from mimarsinan.mapping.ir import IRSource

import numpy as np
import torch


def _get_ir_types():
    return IRSource

def _create_ir_input_source(idx):
    IRSource = _get_ir_types()
    return IRSource(node_id=-2, index=idx)

from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.export.chip_export import hard_cores_to_chip
from mimarsinan.mapping.mappers.base import Mapper
from mimarsinan.mapping.mappers.structural import (
    ConcatMapper,
    EinopsRearrangeMapper,
    InputMapper,
    PermuteMapper,
    ReshapeMapper,
    StackMapper,
    SubscriptMapper,
)
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.module_mapper import ModuleMapper
from mimarsinan.mapping.mappers.leading_dim import (
    Ensure2DMapper,
    MergeLeadingDimsMapper,
    SplitLeadingDimMapper,
)
from mimarsinan.mapping.mappers.conv1d_mapper import Conv1DPerceptronMapper
from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper

__all__ = [
    "ModelRepresentation",
    "hard_cores_to_chip",
    "Mapper",
    "ConcatMapper",
    "EinopsRearrangeMapper",
    "InputMapper",
    "PermuteMapper",
    "ReshapeMapper",
    "StackMapper",
    "SubscriptMapper",
    "PerceptronMapper",
    "ComputeOpMapper",
    "ModuleMapper",
    "Ensure2DMapper",
    "MergeLeadingDimsMapper",
    "SplitLeadingDimMapper",
    "Conv1DPerceptronMapper",
    "Conv2DPerceptronMapper",
]
