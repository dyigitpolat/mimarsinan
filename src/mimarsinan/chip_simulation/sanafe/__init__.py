"""SANA-FE backend integration.

Public surface exposed to the rest of mimarsinan.  Heavy SANA-FE imports
are gated behind lazy ``_sanafe()`` calls inside submodules so importing
``mimarsinan.chip_simulation.sanafe`` does **not** require the GPL-3.0
SANA-FE Python package to be installed.
"""

from __future__ import annotations

from .records import (
    SanafeCoreRecord,
    SanafeEnergyBreakdown,
    SanafeRunRecord,
    SanafeSegmentRecord,
    SanafeTileRecord,
)

__all__ = [
    "SanafeCoreRecord",
    "SanafeEnergyBreakdown",
    "SanafeRunRecord",
    "SanafeSegmentRecord",
    "SanafeTileRecord",
]
