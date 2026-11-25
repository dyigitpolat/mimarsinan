""
from __future__ import annotations

import numpy as np


def log_axis(lo, hi, n, exact=()):
    ""
    v = list(np.exp(np.linspace(np.log(lo), np.log(hi), n)))
    v[0], v[-1] = lo, hi
    return sorted(set([float(x) for x in v] + [float(x) for x in exact]))
