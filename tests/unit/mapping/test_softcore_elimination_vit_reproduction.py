"""[W6b] Reproduction: the cached cifar_vit_leaf IR must yield the paper table.

The headline table was first computed by hand from the cached IR of a real
trained run (`papers/structured_elimination_aaai/research_artifacts/tables/
softcore_elimination.md`). This test pins the shipped report against those
numbers, so a regression in the naming, the (R-r)(C-c) arithmetic or the W3c
intersection rule is caught against a REAL 975-softcore mapping rather than a
toy graph.

Two geometries are asserted, and the difference between them is the point:

- ``as_stored`` measures each core against the geometry the stored IR carries,
  which is exactly what the hand computation did -> it reproduces the
  reference row for row, including ``head`` at 0.0% over 58x10;
- ``pre_elimination`` (the DEFAULT, and what the soft-core seam emits) also
  restores the pre-compaction geometry of owned matrices from
  ``pre_pruning_*_mask``. Only ``head`` moves: IR compaction had already
  dropped the 134 axon rows it lost, so measuring against 58x10 credits the
  method with none of them. Its 192x10 crossbar is 69.8% eliminated, which
  moves the program denominator by 1,340 cells out of 67.7M and leaves every
  other row, the surviving-cell total and the physical-bank view identical.

Marked ``slow``: it loads a ~300 MB artifact that lives outside the repo.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from mimarsinan.mapping.softcore_elimination import (
    GEOMETRY_AS_STORED,
    report_from_pruned_ir_graph,
)

# The run directory lives in the main checkout, beside this worktree; the test
# never writes to it.
_CANDIDATES = (
    Path(__file__).resolve().parents[3]
    / "generated/bc2_cifar_vit_leaf_phased_deployment_run"
    / "Soft Core Mapping.ir_graph.pickle",
    Path(
        "/home/yigit/repos/research_stuff/mimarsinan/generated/"
        "bc2_cifar_vit_leaf_phased_deployment_run/Soft Core Mapping.ir_graph.pickle"
    ),
)


def _cached_ir_path() -> Path | None:
    return next((p for p in _CANDIDATES if p.exists()), None)


_IR_PATH = _cached_ir_path()
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        _IR_PATH is None,
        reason="cached cifar_vit_leaf IR graph is not available here",
    ),
]

# (group, instances, axons, neurons, cells, surviving) from the reference table.
_REFERENCE_ROWS = (
    ("blocks_*_fc1", 455, 192, 384, 33_546_240, 2_597_270),
    ("blocks_*_fc2", 455, 384, 192, 33_546_240, 5_465_590),
    ("patch_embed", 64, 48, 192, 589_824, 294_912),
    ("head", 1, 58, 10, 580, 580),
)
_REFERENCE_TOTAL_CELLS = 67_682_884
_REFERENCE_TOTAL_SURVIVING = 8_358_352
_REFERENCE_BANK_CELLS_BEFORE = 1_041_408
_REFERENCE_BANK_CELLS_AFTER = 128_652


@pytest.fixture(scope="module")
def vit_graph():
    assert _IR_PATH is not None
    with open(_IR_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def as_stored_report(vit_graph):
    return report_from_pruned_ir_graph(vit_graph, geometry=GEOMETRY_AS_STORED)


@pytest.fixture(scope="module")
def default_report(vit_graph):
    return report_from_pruned_ir_graph(vit_graph)


class TestReferenceTableIsReproduced:
    @pytest.mark.parametrize(
        "group,instances,axons,neurons,cells,surviving", _REFERENCE_ROWS
    )
    def test_group_row(
        self, as_stored_report, group, instances, axons, neurons, cells, surviving
    ):
        row = as_stored_report.group(group)
        assert row.instances == instances
        assert (row.axons, row.neurons) == (axons, neurons)
        assert row.cells == cells
        assert row.surviving == surviving

    @pytest.mark.parametrize(
        "group,percent",
        [
            ("blocks_*_fc1", 92.3),
            ("blocks_*_fc2", 83.7),
            ("patch_embed", 50.0),
            ("head", 0.0),
        ],
    )
    def test_group_percentage(self, as_stored_report, group, percent):
        row = as_stored_report.group(group)
        assert round(100.0 * row.eliminated_fraction, 1) == percent

    def test_program_total(self, as_stored_report):
        total = as_stored_report.total
        assert total.instances == 975
        assert total.cells == _REFERENCE_TOTAL_CELLS
        assert total.surviving == _REFERENCE_TOTAL_SURVIVING
        assert round(100.0 * total.eliminated_fraction, 1) == 87.7

    def test_physical_bank_view(self, as_stored_report):
        storage = as_stored_report.storage_total
        assert storage.banks == 15
        assert storage.bank_cells_before == _REFERENCE_BANK_CELLS_BEFORE
        assert storage.bank_cells_after == _REFERENCE_BANK_CELLS_AFTER
        assert round(100.0 * storage.bank_eliminated_fraction, 1) == 87.6

    def test_the_two_views_disagree_by_the_sharing_factor(self, as_stored_report):
        """65 crossbars share one fc1 bank, so the as-mapped view counts its
        cells 65 times — the physical view is not a rescaling of it."""
        as_mapped = as_stored_report.total
        physical = as_stored_report.storage_total
        assert as_mapped.cells > 60 * physical.cells_before
        assert as_mapped.surviving > 60 * physical.cells_after


class TestDefaultGeometryOnlyCorrectsTheCompactedOwnedCore:
    def test_bank_backed_groups_are_identical(self, default_report, as_stored_report):
        for group in ("blocks_*_fc1", "blocks_*_fc2", "patch_embed"):
            assert default_report.group(group) == as_stored_report.group(group)

    def test_head_is_measured_against_its_pre_elimination_crossbar(
        self, default_report
    ):
        head = default_report.group("head")
        assert (head.axons, head.neurons) == (192, 10)
        assert head.rows_eliminated == 134
        assert head.cols_eliminated == 0, "model-output logits stay exempt"
        assert (head.cells, head.surviving) == (1_920, 580)

    def test_totals_move_only_by_the_head_rows(
        self, default_report, as_stored_report
    ):
        assert default_report.total.cells == _REFERENCE_TOTAL_CELLS + 1_340
        assert default_report.total.surviving == _REFERENCE_TOTAL_SURVIVING
        assert round(100.0 * default_report.total.eliminated_fraction, 1) == 87.7
        assert (
            default_report.storage_total.bank_cells_after
            == as_stored_report.storage_total.bank_cells_after
        )
