"""Convolution mappers: perceptron-style (shared-weight)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.mappers.base import Mapper, resolve_activation_type
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer


def _chunk_sizes(total: int, chunk: int):
    assert chunk > 0
    sizes = []
    remaining = int(total)
    while remaining > 0:
        sizes.append(min(chunk, remaining))
        remaining -= sizes[-1]
    return sizes

