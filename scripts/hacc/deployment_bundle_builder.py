#!/usr/bin/env python3
"""The repository-side bundle freezer, bound to the RTL COSIMULATION witness.

The freezing itself lives in ``mimarsinan.chip_simulation.odin_hacc`` so the
pipeline's export step and this script cannot drift into two bundle formats;
what this file adds is the WITNESS choice — the committed fixture's counts are
measured on the vendored ODIN core, which needs an RTL simulator a board node
does not have, and that is exactly why the fixture is committed rather than
built at package time.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mimarsinan.chip_simulation.odin_hacc.freeze import (  # noqa: E402
    CAPTURE_SCHEMA as CAPTURE_SCHEMA,
    BundleRefusal as BundleRefusal,
    build_bundle as _build_bundle,
)
from mimarsinan.chip_simulation.odin_hacc.pass_build import (  # noqa: E402
    PassBuild as PassBuild,
    pass_mapping as pass_mapping,
    used_neurons as used_neurons,
)
from mimarsinan.chip_simulation.odin_hacc.witness import (  # noqa: E402
    COSIM_DERIVATION as COSIM_DERIVATION,
    CosimWitness,
)


def build_bundle(**kwargs):
    """Freeze one network on the RTL cosimulation witness."""
    return _build_bundle(witness=CosimWitness(), **kwargs)
