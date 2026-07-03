"""Introspection and search-space helpers for MimarsinanLayoutBackend."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Dict

from compilagent import PassEvent

from mimarsinan.search.search_space_description import SearchSpaceDescription


def platform_to_json_safe(pcfg: Mapping[str, Any]) -> Dict[str, Any]:
    from mimarsinan.gui.json_util import to_json_safe

    return to_json_safe(dict(pcfg))


def description_for(problem: Any) -> SearchSpaceDescription:
    """Reconstruct the SearchSpaceDescription from the live problem."""
    return SearchSpaceDescription(
        search_mode=str(getattr(problem, "search_mode", "joint")),
        arch_options=tuple(
            (str(k), tuple(v)) for k, v in getattr(problem, "arch_options", ())
        ),
        num_core_types=int(getattr(problem, "num_core_types", 1)),
        core_axons_bounds=tuple(getattr(problem, "core_axons_bounds", (64, 1024))),
        core_neurons_bounds=tuple(
            getattr(problem, "core_neurons_bounds", (64, 1024))
        ),
        core_count_bounds=tuple(getattr(problem, "core_count_bounds", (50, 500))),
        target_tq=int(getattr(problem, "target_tq", 32)),
        weight_bits=int(8),
    )


def fire_pass(callback: Any, stage: str, name: str, started_at: float) -> None:
    try:
        callback(
            PassEvent(
                stage=stage,
                name=name,
                duration_ms=(time.perf_counter() - started_at) * 1000.0,
            )
        )
    except Exception:
        pass


def list_introspection_tools(backend: Any):
    from ..tools import build_introspection_tools

    return build_introspection_tools(backend)
