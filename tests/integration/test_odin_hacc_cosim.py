"""[ODIN P8] the SAME tiny bundle, frozen on the vendored RTL instead of the twin.

The unit loop-closer (``tests/unit/chip_simulation/test_odin_hacc_export.py``)
proves the plumbing with the cycle-accurate twin as the witness. This gate swaps
ONE object — ``CosimWitness`` for ``TwinWitness`` — so every frozen count is
measured on the byte-identical vendored ODIN core executing the exporter's own
sequencer program, and then runs the SHIPPED executor against those measured
answers. Three implementations must agree at zero difference: the HCM torch
reference (the export step's own gate), the cycle-accurate twin (the freezer's
first golden gate) and the RTL.

The witness seam is the whole reason this costs one object rather than a second
freezer: the bundle, its four golden gates, its seal and its replay format are
the same code either way.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from integration.odin_hacc_harness import (
    CLASSES,
    ODIN_LAW,
    TIMESTEPS,
    one_hot,
    two_core_mapping,
)
from integration.odin_rtl_harness import require_simulator, timed

from mimarsinan.chip_simulation.odin_hacc.artifact import (
    kernel_table,
    render_bundle,
)
from mimarsinan.chip_simulation.odin_hacc.freeze import build_bundle
from mimarsinan.chip_simulation.odin_hacc.witness import (
    COSIM_DERIVATION,
    CosimWitness,
    TwinWitness,
)
from mimarsinan.chip_simulation.odin_rtl.toolchain import available_engine
from mimarsinan.mapping.latency.chip import ChipLatency

pytestmark = [pytest.mark.slow, pytest.mark.integration]

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "scripts" / "hacc" / "package"
BUNDLE_MODULE = (
    REPO / "src" / "mimarsinan" / "chip_simulation" / "odin_deployment_bundle.py")
HOST_FILES = (
    "odin_board_driver.py", "fake_pyxrt_for_selftest.py",
    "odin_deployment_executor.py",
)

#: One entry raster per class: line ``k`` fires in every cycle, so the readout
#: is one-hot on ``k`` and the true label is ``k``.
LABELS = tuple(range(CLASSES))


def _rasters():
    return [
        [[int(value) for value in one_hot(label)[0]] for _cycle in range(TIMESTEPS)]
        for label in LABELS
    ]


def _freeze(witness):
    mapping = two_core_mapping()
    return build_bundle(
        name="odin_hacc_micro_cosim",
        title="HACC NUS - ODIN Deployment: the tiny classifier, measured on RTL",
        description=(
            "The unit loop-closer's two-core classifier, frozen with the "
            "vendored ODIN core as the witness."),
        mapping=mapping,
        rasters=_rasters(),
        labels=list(LABELS),
        simulation_length=TIMESTEPS,
        chip_latency=int(ChipLatency(mapping).calculate()),
        soma_law=ODIN_LAW,
        weight_bits=4,
        effective_max_axons=7,
        weight_sign_granularity="per_axon",
        membrane_init=0,
        readout_core=1,
        certification=list(LABELS),
        provenance={"derivation": witness.derivation,
                    "generator": "tests/integration/test_odin_hacc_cosim.py",
                    "cosim_engine": available_engine()},
        kernel_table=kernel_table(),
        model={"name": "odin_hacc_micro", "classes": CLASSES,
               "timesteps": TIMESTEPS, "samples": len(LABELS), "synthetic": True},
        witness=witness,
    )


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """The bundle, with every count MEASURED on the vendored RTL."""
    require_simulator()
    with timed("P8 cosim-witnessed bundle"):
        document, capture = _freeze(CosimWitness())
    root = tmp_path_factory.mktemp("odin_hacc_cosim")
    (root / "host").mkdir()
    (root / "deployment").mkdir()
    for name in HOST_FILES:
        shutil.copyfile(PACKAGE / "host" / name, root / "host" / name)
    shutil.copyfile(BUNDLE_MODULE, root / "host" / "odin_deployment_bundle.py")
    (root / "deployment" / "bundle.json").write_text(
        render_bundle(document), encoding="utf-8")
    (root / "deployment" / "replay.json").write_text(
        render_bundle(capture), encoding="utf-8")
    return document, capture, root


@pytest.fixture(scope="module")
def executed(measured):
    _document, _capture, root = measured
    proc = subprocess.run(
        [sys.executable, "host/odin_deployment_executor.py",
         "--xclbin", "/hw-test/no-such.xclbin",
         "--fake-pyxrt", "host/fake_pyxrt_for_selftest.py",
         "--bundle", "deployment/bundle.json",
         "--replay", "deployment/replay.json",
         "--results", "results/deploy"],
        cwd=str(root), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    print(proc.stdout)
    return json.loads(
        (root / "results" / "deploy" / "deployment_report.json").read_text())


class TestTheRtlFreezesTheSameBundleTheTwinDoes:
    def test_the_measured_counts_are_the_twins_at_every_cycle(self, measured):
        """The freezer's first golden gate already refuses otherwise; this
        asserts the two documents are the SAME document, not merely compatible."""
        document, _capture, _root = measured
        twin, _twin_capture = _freeze(TwinWitness())
        assert document["cores"] == twin["cores"]
        assert document["certification"] == twin["certification"]
        assert document["expected"] == twin["expected"]

    def test_the_provenance_names_the_rtl_and_the_engine(self, measured):
        document, _capture, _root = measured
        assert document["provenance"]["derivation"] == COSIM_DERIVATION
        assert document["provenance"]["cosim_engine"]

    def test_the_replay_carries_the_rtls_own_device_cycles(self, measured):
        _document, capture, _root = measured
        cycles = [int(run["device_cycles"]) for run in capture["runs"]]
        assert cycles and min(cycles) > 0, (
            "a cosim-witnessed replay must carry the free-running counter of "
            "the run that produced it; zero would mean nothing executed")


class TestTheShippedExecutorRunsTheRtlWitnessedBundle:
    def test_the_campaign_is_perfect_and_every_certificate_is_green(self, executed):
        assert executed["accuracy"] == 1.0
        assert executed["passed"] is True
        assert executed["certificates"]
        for row in executed["certificates"]:
            assert row["passed"] is True, row["certificate_line"]

    def test_the_task_is_not_answerable_by_a_constant(self, executed):
        predicted = {int(row["predicted"]) for row in executed["readout"]}
        assert len(predicted) == CLASSES, predicted
