"""[ODIN P8] the LOOP-CLOSER: a real model, the real export step, the real executor.

A tiny two-core classifier goes through the PRODUCTION export step, becomes a
sealed deployment bundle, and is then executed by the SHIPPED board executor
against the fake pyxrt — one process, one exit code, exactly as the sbatch job
runs it. What passes here is the whole chain: the HCM-vs-twin gate at export,
the four golden gates of the freezer, the host-mediated transcode of pass 2 from
pass 1's counts, the per-pass certificates and the accuracy accumulator.

Nothing here touches silicon. The point is that every JOINT holds; the board run
is what measures the network.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from integration.odin_hacc_harness import CLASSES, hybrid_program, prepare_step

from mimarsinan.chip_simulation import odin_deployment_bundle as bundle
from mimarsinan.chip_simulation.odin_hacc.artifact import render_bundle
from mimarsinan.chip_simulation.odin_hacc.freeze import BundleRefusal
from mimarsinan.chip_simulation.odin_hacc.program_freeze import (
    OdinHaccExportRefusal,
    readout_core_of,
)
from mimarsinan.chip_simulation.odin_hacc.witness import TwinWitness
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridStage
from mimarsinan.pipelining.pipeline_steps.verification import (
    odin_hacc_deployment_step as step_module,
)
from mimarsinan.pipelining.pipeline_steps.verification.odin_hacc_deployment_step import (
    OdinHaccDeploymentStep,
)

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "scripts" / "hacc" / "package"
CHIP_SIM = REPO / "src" / "mimarsinan" / "chip_simulation"
VERBATIM_MODULES = (
    CHIP_SIM / "odin_deployment_bundle.py",
    CHIP_SIM / "odin_deployment_encoding.py",
)
HOST_FILES = (
    "odin_board_driver.py", "fake_pyxrt_for_selftest.py",
    "odin_deployment_executor.py",
)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """One run of the REAL export step on the tiny classifier."""
    monkeypatch = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("odin_hacc_export")
    try:
        pipeline, step = prepare_step(
            monkeypatch, OdinHaccDeploymentStep, working_directory=str(root))
        step.process()
    finally:
        monkeypatch.undo()
    stats = pipeline.cache["OdinHaccDeploymentStep.odin_hacc_deployment_bundle"]
    return step.document, stats


@pytest.fixture(scope="module")
def staged(tmp_path_factory, exported):
    """The package's host/ beside the freshly exported bundle and its replay."""
    _document, stats = exported
    root = tmp_path_factory.mktemp("odin_hacc_pkg")
    (root / "host").mkdir()
    (root / "deployment").mkdir()
    for name in HOST_FILES:
        shutil.copyfile(PACKAGE / "host" / name, root / "host" / name)
    for module in VERBATIM_MODULES:
        shutil.copyfile(module, root / "host" / module.name)
    shutil.copyfile(stats["paths"]["bundle"], root / "deployment" / "bundle.json")
    shutil.copyfile(stats["paths"]["capture"], root / "deployment" / "replay.json")
    return root


def run_executor(staged: Path, *args: str):
    """The executor as the sbatch job runs it: one process, one exit code."""
    return subprocess.run(
        [sys.executable, "host/odin_deployment_executor.py",
         "--xclbin", "/selftest/no-such.xclbin",
         "--fake-pyxrt", "host/fake_pyxrt_for_selftest.py",
         "--bundle", "deployment/bundle.json",
         "--replay", "deployment/replay.json",
         "--results", "results/deploy", *args],
        cwd=str(staged), capture_output=True, text=True)


@pytest.fixture(scope="module")
def executed(staged):
    proc = run_executor(staged)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(
        (staged / "results" / "deploy" / "deployment_report.json").read_text())
    return proc, report


class TestTheExportedBundleRunsOnTheShippedExecutor:
    def test_the_campaign_is_perfect_on_its_trivial_task(self, executed):
        _proc, report = executed
        assert report["accuracy"] == 1.0
        assert report["correct"] == len(report["readout"])
        assert report["passed"] is True

    def test_every_per_pass_certificate_is_green(self, executed):
        _proc, report = executed
        assert report["certificates"], "no pass was certified at all"
        for row in report["certificates"]:
            assert row["passed"] is True, row["certificate_line"]
            assert "max|dcount|=0" in row["certificate_line"], row

    def test_the_readout_reproduced_the_frozen_scores_exactly(self, executed):
        _proc, report = executed
        for row in report["readout"]:
            assert row["matches_frozen"] is True, row
            assert row["scores"] == row["expected_scores"], row

    def test_the_task_is_not_answerable_by_a_constant(self, executed):
        _proc, report = executed
        predicted = {int(row["predicted"]) for row in report["readout"]}
        assert len(predicted) == CLASSES, predicted

    def test_the_second_pass_was_built_from_the_first_pass_counts(self, exported):
        document, _stats = exported
        consumer = [plan for plan in document["cores"] if plan["core"] == 1][0]
        producers = {int(kind) for kind, _index in consumer["routes"] if kind >= 0}
        assert producers == {0}, consumer["routes"]
        assert document["pass_order"] == [0, 1]


class TestTheExportRefusesRatherThanShipsAWrongExpectation:
    def test_a_witness_that_disagrees_with_the_twin_refuses(
        self, tmp_path, monkeypatch,
    ):
        class _Perturbed(TwinWitness):
            """One neuron, one cycle, one spike too many."""

            def measure(self, build, traces):
                measured = super().measure(build, traces)
                counts = dict(measured.counts)
                counts[(0, 0, 0, 0)] = counts.get((0, 0, 0, 0), 0) + 1
                return type(measured)(
                    counts=counts, device_cycles=measured.device_cycles)

        pipeline, step = prepare_step(
            monkeypatch, OdinHaccDeploymentStep,
            working_directory=str(tmp_path))
        monkeypatch.setattr(step_module, "TwinWitness", _Perturbed)
        with pytest.raises(BundleRefusal, match="cycle-accurate twin"):
            step.process()
        assert not list(tmp_path.glob("odin_hacc/*.json"))
        assert "odin_hacc_deployment_bundle" not in pipeline.cache

    def test_a_provenance_that_names_another_producer_refuses(self, exported):
        from mimarsinan.chip_simulation.odin_hacc.freeze import build_bundle
        from mimarsinan.chip_simulation.odin_hacc.witness import COSIM_DERIVATION

        with pytest.raises(BundleRefusal, match="did not produce it"):
            build_bundle(
                name="x", title="x", description="x", mapping=None, rasters=[[]],
                labels=[0], simulation_length=1, chip_latency=0, soma_law=None,
                weight_bits=4, effective_max_axons=7,
                weight_sign_granularity="per_axon", membrane_init=0,
                readout_core=0, certification=[], kernel_table={}, model={},
                provenance={"derivation": COSIM_DERIVATION},
                witness=TwinWitness())


class TestTheFreezeRefusesAReadoutThatIsNotTheClassifier:
    """A host-readout vehicle would freeze argmax over TRUNK features.

    ``argmax`` over a segment's own neurons is always a well-formed number.
    What makes it the network's prediction is that the segment IS the last
    stage and that its readout core carries exactly the classes — so both are
    asked of the mapping rather than assumed.
    """

    def test_a_host_stage_after_the_segment_refuses(self):
        program = hybrid_program()
        segment = program.stages[0].hard_core_mapping
        program.stages.append(
            HybridStage(kind="compute", name="host_readout"))
        with pytest.raises(OdinHaccExportRefusal, match="not the LAST stage"):
            readout_core_of(program, segment, classes=CLASSES)

    def test_the_intact_program_still_resolves_its_readout_core(self):
        program = hybrid_program()
        segment = program.stages[0].hard_core_mapping
        assert readout_core_of(program, segment, classes=CLASSES) == 1

    def test_a_readout_core_wider_than_the_class_count_refuses(self):
        program = hybrid_program()
        segment = program.stages[0].hard_core_mapping
        # The readout core keeps its three wires but declares five USED
        # neurons: the bundle scores every one of them.
        segment.cores[1].neurons_per_core = 5
        segment.cores[1].available_neurons = 0
        with pytest.raises(OdinHaccExportRefusal, match="used neuron"):
            readout_core_of(program, segment, classes=CLASSES)

    def test_a_program_whose_output_width_is_not_the_class_count_refuses(
            self, tmp_path, monkeypatch):
        """End to end, through the REAL step: the mapping and the model disagree."""
        _pipeline, step = prepare_step(
            monkeypatch, OdinHaccDeploymentStep, working_directory=str(tmp_path),
            config_overrides={"num_classes": CLASSES + 1})
        with pytest.raises(OdinHaccExportRefusal, match="class\\(es\\)"):
            step.process()
        assert not list(tmp_path.glob("odin_hacc/*.json"))


class TestTheBundleIsSealedAndDeterministic:
    def test_the_bundle_seals_to_its_own_bytes(self, exported):
        document, stats = exported
        loaded = bundle.load_bundle(stats["paths"]["bundle"])
        assert loaded["self_hash"] == document["self_hash"]
        assert loaded["schema"] == bundle.SCHEMA

    def test_the_capture_replay_is_bound_to_this_bundle(self, exported):
        document, stats = exported
        replay = json.loads(Path(stats["paths"]["capture"]).read_text())
        assert replay["bundle_self_hash"] == document["self_hash"]
        stamped = {k: v for k, v in replay.items() if k != "self_hash"}
        assert bundle.self_hash(stamped) == replay["self_hash"]

    def test_a_second_export_of_the_same_inputs_is_byte_identical(
        self, exported, tmp_path, monkeypatch,
    ):
        document, _stats = exported
        _pipeline, step = prepare_step(
            monkeypatch, OdinHaccDeploymentStep, working_directory=str(tmp_path))
        step.process()
        assert render_bundle(step.document) == render_bundle(document)

    def test_the_declared_campaign_is_what_the_bundle_ships(self, exported):
        document, stats = exported
        assert stats["samples"] == len(document["samples"]) == 6
        assert stats["certification_samples"] == 3
        assert bundle.certification_samples(document) == (0, 1, 2)

    def test_the_kernel_table_is_the_shipped_driver_protocol(self, exported):
        document, _stats = exported
        from mimarsinan.chip_simulation.odin_fpga import kernel_registers

        assert document["kernel"]["name"] == kernel_registers.KERNEL_NAME
        assert document["kernel"]["kernel_args"] == kernel_registers.KERNEL_ARGS

    def test_the_provenance_names_the_twin_and_not_a_device(self, exported):
        document, _stats = exported
        provenance = document["provenance"]
        assert provenance["derivation"] == TwinWitness.derivation
        assert "NOT a silicon measurement" in provenance["derivation"]
        assert provenance["agreement_arms"] == ["hcm"]
