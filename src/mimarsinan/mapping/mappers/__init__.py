"""Mapper hierarchy: base, structural, perceptron, leading-dim, conv mappers.

Non-neural ops collapse onto a single :class:`ComputeOpMapper` carrying any
``nn.Module`` (see :mod:`mimarsinan.mapping.compute_modules` for the small
``Add`` / ``Mean`` / ``Select`` / ``ConstantAdd`` / ``ConstantPrepend`` /
``ScaleNormalizingWrapper`` wrappers used for ops that are not already
canonical torch modules).
"""

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
from mimarsinan.mapping.mappers.perceptron import (
    ComputeOpMapper,
    ModuleMapper,
    PerceptronMapper,
)
from mimarsinan.mapping.mappers.leading_dim import (
    Ensure2DMapper,
    MergeLeadingDimsMapper,
    SplitLeadingDimMapper,
)
from mimarsinan.mapping.mappers.conv import (
    Conv1DPerceptronMapper,
    Conv2DPerceptronMapper,
)

__all__ = [
    "Mapper",
    "InputMapper",
    "ReshapeMapper",
    "EinopsRearrangeMapper",
    "StackMapper",
    "ConcatMapper",
    "SubscriptMapper",
    "PermuteMapper",
    "PerceptronMapper",
    "ComputeOpMapper",
    "ModuleMapper",
    "MergeLeadingDimsMapper",
    "SplitLeadingDimMapper",
    "Ensure2DMapper",
    "Conv2DPerceptronMapper",
    "Conv1DPerceptronMapper",
]
