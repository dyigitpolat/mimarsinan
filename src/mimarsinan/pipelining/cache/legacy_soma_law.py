"""Resolve the soma point of chip-aligned NF forwards cached before the soma axes existed.

LIF Adaptation pickles its chip-aligned NF forward into the step cache as the model's instance ``forward`` (every LIF
run: ``cycle_accurate_lif_forward`` defaults on). Since the soma axes, that forward carries the run's resolved point
and refuses at call time without one: it never re-defaults on read, because an assumed point would become the whole
run's physics. A forward cached by a tree that predates the axes therefore deserializes with no point, and every such
run directory would be unresumable although its pickle was written at the default point.

This runs at cache load, beside the legacy placement (``DeploymentPipeline._initialize_config``), the one moment the
run's resolved point is known and no step has run. Each such forward is bound to the run's law when that law is a
default point, and refused by name when the run declares any other point
(``ChipAlignedNFForward.bind_pre_axes_soma_law``). A no-op for a fresh run and for every forward written with a point.
"""

from __future__ import annotations

from typing import List

import torch

from mimarsinan.chip_simulation.soma_law import SomaLaw


def resolve_cached_forward_soma_laws(cache, *, soma_law: SomaLaw) -> List[str]:
    """Bind every cached pre-axes NF forward to ``soma_law`` (a default point) or refuse it by name.

    Returns the cache keys whose forward was bound, sorted. Every module of a cached torch model is searched, since an
    instance ``forward`` may sit on a submodule; objects that are not torch modules hold no installed forward.
    """
    bound = set()
    for key in list(cache.keys()):
        entry = cache.get(key)
        if not isinstance(entry, torch.nn.Module):
            continue
        for module in entry.modules():
            bind = getattr(module.__dict__.get("forward"), "bind_pre_axes_soma_law", None)
            if callable(bind) and bind(soma_law, entry=key):
                bound.add(key)
    return sorted(bound)
