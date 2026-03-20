"""Mixin handlers for MapperGraphConverter. Each mixin provides _convert_* methods for a category of ops."""

from mimarsinan.torch_mapping.converter_handlers.linear_mixin import LinearConvertMixin
from mimarsinan.torch_mapping.converter_handlers.conv_mixin import ConvConvertMixin
from mimarsinan.torch_mapping.converter_handlers.structural_mixin import StructuralConvertMixin

__all__ = [
    "LinearConvertMixin",
    "ConvConvertMixin",
    "StructuralConvertMixin",
]
