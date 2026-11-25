""
from __future__ import annotations

from collections import OrderedDict

import numpy as np


class PeriodRuleError(Exception):
    ""

    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code = code


def bounded(period, occupancies, rel_tol=0.0):
    ""
    out = np.asarray(period, dtype=float)
    if np.any(~np.isfinite(out)) or np.any(out < 0):
        raise PeriodRuleError("PERIOD", "a period is negative or not finite")
    if not rel_tol >= 0:
        raise PeriodRuleError("TOLERANCE", f"the rounding tolerance {rel_tol!r} is not a non-negative number")
    binds = OrderedDict()
    for name, occ in occupancies.items():
        o = np.asarray(occ, dtype=float)
        if np.any(~np.isfinite(o)) or np.any(o < 0):
            raise PeriodRuleError("OCCUPANCY", f"{name}: an occupancy is negative or not finite")
        hi = o > out * (1.0 + rel_tol)
        out = np.where(hi, o, out)
        for k in binds:
            binds[k] = binds[k] & ~hi
        binds[name] = hi
    return out, binds


def engaged(serial, period, occupancy, rel_tol=0.0, name="resource"):
    ""
    s = np.asarray(serial, dtype=float)
    if np.any(~np.isfinite(s)) or np.any(s < 0):
        raise PeriodRuleError("OCCUPANCY", "a serial time is negative or not finite")
    p = np.asarray(period, dtype=float)
    o = np.asarray(occupancy, dtype=float)
    if np.any(o > p * (1.0 + rel_tol)):
        raise PeriodRuleError("OVERRUN", f"{name}: its occupancy exceeds the period it is given (the period must be "
                                         "bounded by every resource's occupancy first, bounded())")
    return np.minimum(s, p)
