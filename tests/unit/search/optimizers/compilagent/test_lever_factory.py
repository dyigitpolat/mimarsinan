"""Tests for ``levers_from_description``.

The function delegates to ``SearchSpaceDescription.to_compilagent_levers``
(already covered by ``test_search_space_description.py``); this suite
pins the surface contract — a ``Lever`` per searchable variable, the
expected ``backend_id`` propagated, and graceful handling of an empty
search space.
"""

from __future__ import annotations

from compilagent import EnumChoice, IntFreeform, Lever

from mimarsinan.search.optimizers.compilagent.lever_factory import (
    levers_from_description,
)
from mimarsinan.search.search_space_description import SearchSpaceDescription


def _make_description(search_mode: str = "joint") -> SearchSpaceDescription:
    return SearchSpaceDescription.from_arch_search(
        search_mode=search_mode,
        arch_options=(("activation", ("ReLU", "GELU")),),
        arch_cfg={
            "num_core_types": 2,
            "core_axons_bounds": [64, 1024],
            "core_neurons_bounds": [64, 1024],
            "core_count_bounds": [50, 500],
        },
        target_tq=32,
    )


def test_one_lever_per_variable():
    d = _make_description()
    levers = levers_from_description(d, workload_id="w", backend_id="mimarsinan_layout")
    # 1 arch option + 3 dims * 2 core types = 7
    assert len(levers) == 7
    assert all(isinstance(lv, Lever) for lv in levers)


def test_arch_levers_use_enum_choice():
    d = _make_description()
    levers = levers_from_description(d, workload_id="w", backend_id="b")
    arch = [lv for lv in levers if lv.target_kind == "arch"]
    assert len(arch) == 1
    assert isinstance(arch[0].range, EnumChoice)
    assert set(arch[0].range.candidates) == {"ReLU", "GELU"}


def test_hw_levers_use_intfreeform_with_correct_selectors():
    d = _make_description()
    levers = levers_from_description(d, workload_id="w", backend_id="b")
    hw = [lv for lv in levers if lv.target_kind == "hw.core"]
    selectors = sorted(lv.target_selector for lv in hw)
    expected = sorted(f"{i}.{dim}" for i in (0, 1) for dim in ("max_axons", "max_neurons", "count"))
    assert selectors == expected
    for lv in hw:
        assert isinstance(lv.range, IntFreeform)


def test_backend_id_is_propagated():
    d = _make_description()
    levers = levers_from_description(d, workload_id="w", backend_id="picked-id")
    assert all(lv.backend_id == "picked-id" for lv in levers)


def test_empty_search_returns_empty_tuple():
    d = SearchSpaceDescription.from_arch_search(
        search_mode="joint",
        arch_options=(),
        arch_cfg={
            "num_core_types": 0,
            "core_axons_bounds": [64, 64],
            "core_neurons_bounds": [64, 64],
            "core_count_bounds": [10, 10],
        },
        target_tq=8,
    )
    levers = levers_from_description(d, workload_id="w", backend_id="b")
    assert levers == ()
