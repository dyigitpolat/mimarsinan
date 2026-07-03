"""Shared constants for the SANA-FE runner."""

from __future__ import annotations

import numpy as np

_RAW_INPUT_NODE_ID = -2

# float64 matches HCM; float32 drifts ±1 spike at rate-encoding boundaries.
_COMPUTE_DTYPE: np.dtype = np.dtype(np.float64)
