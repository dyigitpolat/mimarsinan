"""The affine batch domain (``forms.py``: its values, ``operations.py``: its operations) and the declared batch rules
of the tensor kernels (``kernels.py``, ``layouts.py``): the toolchain's own shape proof (review round 14)."""

from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.forms import (
    BATCH,
    Affine,
    Opaque,
    affine,
    describe,
    form,
    meaning,
    opaque_in,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.operations import evaluate

__all__ = ["Affine", "BATCH", "Opaque", "affine", "describe", "evaluate", "form", "meaning", "opaque_in"]
