"""Shared constants for the SANA-FE runner."""

from __future__ import annotations

import numpy as np

from mimarsinan.mapping.ir.deployment_dtype import SPIKING_ARITHMETIC

_RAW_INPUT_NODE_ID = -2

# [U-DAR] The record's spiking boundary dtype (the HCM's float64; float32 drifts ±1 spike at rate-encoding boundaries).
_COMPUTE_DTYPE: np.dtype = SPIKING_ARITHMETIC.boundary_np
