"""cProfile harness for the wizard layout-mapping path.

Run explicitly: ``pytest -m slow tests/integration/test_wizard_layout_profile.py -s``.

Outputs (under ``build/profiles/``):

- ``cifar_vit_layout_<label>.prof`` -- binary cProfile dump (gitignored)
- ``cifar_vit_layout_<label>.txt`` -- top-30 by cumulative time (checked in)

``<label>`` defaults to ``before`` and is overridable via
``MIMARSINAN_PROFILE_LABEL`` so successive phases (after_phase2, after_phase4,
after_final) write distinct artefacts to the same dir.
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


@pytest.mark.slow
def test_profile_cifar_vit_layout_first_call() -> None:
    """Profile a cold-cache wizard layout call for the CIFAR-ViT preset."""
    from mimarsinan.gui.server import _get_layout_result_from_request

    body = json.loads(_FIXTURE.read_text())
    _PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    label = _label()
    prof_path = _PROFILES_DIR / f"cifar_vit_layout_{label}.prof"
    txt_path = _PROFILES_DIR / f"cifar_vit_layout_{label}.txt"

    profiler = cProfile.Profile()
    wall_start = time.perf_counter()
    profiler.enable()
    result = _get_layout_result_from_request(body)
    profiler.disable()
    wall_seconds = time.perf_counter() - wall_start

    profiler.dump_stats(str(prof_path))
    stats = pstats.Stats(profiler)
    _dump_top(stats, txt_path)

    print(
        f"\n[layout-profile] label={label} wall={wall_seconds:.3f}s "
        f"softcores={len(result.softcores)} "
        f"feasible={result.feasible}"
    )
    print(f"[layout-profile] dumped {prof_path}")
    print(f"[layout-profile] dumped {txt_path}")

    assert result.feasible, f"Layout verification not feasible: {result.error}"
    assert result.softcores, "Expected at least one softcore from CIFAR-ViT"
