from mimarsinan.code_generation.cpp_chip_model import *
from mimarsinan.models.layers import *
from mimarsinan.mapping.softcore_mapping import *
from mimarsinan.transformations.weight_quantization import *

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

import einops
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

# Import IR types (late binding to avoid circular imports)
def _get_ir_types():
    from mimarsinan.mapping.ir import IRSource
    return IRSource

def _create_ir_input_source(idx):
    IRSource = _get_ir_types()
    return IRSource(node_id=-2, index=idx)

# Re-export from cycle-break modules (ir imports soft_core_mapper lazily; no cycle).
from mimarsinan.mapping.soft_core_mapper import SoftCoreMapping, map_mm
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.chip_export import (
    generate_core_weights,
    generate_core_connection_info,
    to_numpy,
    hard_cores_to_chip,
)
from mimarsinan.mapping.mappers.base import Mapper
from mimarsinan.mapping.mappers.structural import (
    AddMapper,
    ConcatMapper,
    DelayMapper,
    EinopsRearrangeMapper,
    InputMapper,
    ReshapeMapper,
    StackMapper,
    SubscriptMapper,
)
from mimarsinan.mapping.mappers.perceptron import ModuleMapper, PerceptronMapper
from mimarsinan.mapping.mappers.leading_dim import (
    Ensure2DMapper,
    MergeLeadingDimsMapper,
    SplitLeadingDimMapper,
)
from mimarsinan.mapping.mappers.pooling import (
    AdaptiveAvgPool2DMapper,
    AvgPool2DMapper,
    MaxPool2DMapper,
)
from mimarsinan.mapping.mappers.conv import (
    Conv1DMapper,
    Conv1DPerceptronMapper,
    Conv2DMapper,
    Conv2DPerceptronMapper,
)
from mimarsinan.mapping.mappers.transformer import (
    ConstantAddMapper,
    ConstantPrependMapper,
    DropoutMapper,
    GELUMapper,
    LayerNormMapper,
    MultiHeadAttentionComputeMapper,
    SelectMapper,
)
