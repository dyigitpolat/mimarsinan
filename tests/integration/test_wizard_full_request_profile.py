"""End-to-end wizard ``/api/hw_config_verify`` profile -- layout + packing.

The user-visible wizard wall time includes both the layout pass *and* the
``verify_hardware_config`` packing call.  Layout alone is captured by
``test_wizard_layout_profile.py``; this harness measures the full route
handler logic the FastAPI server runs inside ``_run`` in server.py:649.
"""

from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import time
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests/integration/fixtures/cifar_vit_wizard_body.json"
_PROFILES_DIR = _REPO_ROOT / "build/profiles"


def _label() -> str:
    return os.environ.get("MIMARSINAN_PROFILE_LABEL", "before")


def _dump_top(stats: pstats.Stats, path: Path, n: int = 30) -> None:
    buf = io.StringIO()
    stats.stream = buf
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(n)
    path.write_text(buf.getvalue())


def _verify_request(body: dict) -> dict:
    """Run the same logic as server.api_hw_config_verify._run, minus FastAPI."""
    from mimarsinan.gui.server import _get_layout_result_from_request
    from mimarsinan.mapping.mapping_verifier import verify_hardware_config
    from mimarsinan.mapping.platform_constraints import (
        resolve_platform_mapping_params,
    )

    core_types = body.get("core_types", [])
    mr = dict(body.get("model_repr_json", {}))

    if core_types:
        pmap = resolve_platform_mapping_params(
            core_types,
            allow_coalescing=bool(body.get("allow_coalescing", False)),
        )
        tile_max_ax = pmap.effective_max_axons
        tile_max_neu = pmap.effective_max_neurons
        mr["allow_coalescing"] = pmap.allow_coalescing
        mr["hardware_bias"] = pmap.hardware_bias
    else:
        tile_max_ax = int(mr.get("max_axons", 1024))
        tile_max_neu = int(mr.get("max_neurons", 1024))

    mr["max_axons"] = max(int(mr.get("max_axons", 1024)), 4096)
    mr["max_neurons"] = max(int(mr.get("max_neurons", 1024)), 4096)

    layout_result = _get_layout_result_from_request(
        mr,
        tiling_max_axons=tile_max_ax,
        tiling_max_neurons=tile_max_neu,
    )
    softcores = layout_result.softcores
    result = verify_hardware_config(
        softcores,
        core_types,
        allow_neuron_splitting=bool(body.get("allow_neuron_splitting", False)),
        allow_coalescing=bool(body.get("allow_coalescing", False)),
        allow_scheduling=bool(body.get("allow_scheduling", False)),
    )
    return {"layout": layout_result, "verify": result}


def _wizard_body() -> dict:
    """Wrap the layout-only fixture into a full /api/hw_config_verify body."""
    inner = json.loads(_FIXTURE.read_text())
    return {
        "model_repr_json": inner,
        "core_types": [
            {"max_axons": 3072, "max_neurons": 768, "count": 69},
            {"max_axons": 768, "max_neurons": 3072, "count": 69},
        ],
        "allow_coalescing": False,
        "allow_neuron_splitting": False,
        "allow_scheduling": False,
    }


@pytest.mark.slow
def test_profile_cifar_vit_full_request() -> None:
    body = _wizard_body()
    _PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    label = _label()
    prof_path = _PROFILES_DIR / f"cifar_vit_full_{label}.prof"
    txt_path = _PROFILES_DIR / f"cifar_vit_full_{label}.txt"

    profiler = cProfile.Profile()
    wall_start = time.perf_counter()
    profiler.enable()
    out = _verify_request(body)
    profiler.disable()
    wall_seconds = time.perf_counter() - wall_start

    profiler.dump_stats(str(prof_path))
    stats = pstats.Stats(profiler)
    _dump_top(stats, txt_path)

    feasible = bool(out["verify"]["feasible"])
    softcores = len(out["layout"].softcores)
    print(
        f"\n[wizard-full-profile] label={label} wall={wall_seconds:.3f}s "
        f"softcores={softcores} feasible={feasible}"
    )
    print(f"[wizard-full-profile] dumped {txt_path}")
