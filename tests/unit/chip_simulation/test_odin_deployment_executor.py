"""[ODIN P7a/P8] the board executor: host-mediated passes, certified and timed.

STRUCTURE, NOT SILICON. The fake pyxrt replays the per-pass capture images the
committed cosimulation recorded, keyed by the sha256 of the stimulus that earned
each one — so a host that builds the WRONG stimulus is answered with no verdict
rather than with some other pass's counts. What passes here is the transcode,
the stimulus arithmetic, the certificate, the accuracy and every typed refusal;
never a card.

The executor lives in the package and imports its two siblings by name, so every
test drives a STAGED host/ directory: exactly the three files the packager puts
there, plus the committed bundle and its replay.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mimarsinan.chip_simulation import odin_deployment_bundle as bundle

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "scripts" / "hacc" / "package"
BUNDLE_MODULE = (
    REPO / "src" / "mimarsinan" / "chip_simulation" / "odin_deployment_bundle.py")
BUNDLE_NAME = "nc1_two_core_passes.json"
REPLAY_NAME = "nc1_two_core_passes_capture.json"

HOST_FILES = (
    "odin_board_driver.py", "fake_pyxrt_for_selftest.py",
    "odin_deployment_executor.py", "render_die_map.py",
)


@pytest.fixture(scope="module")
def staged(tmp_path_factory) -> Path:
    """The package's host/ + deployment/, exactly as make_package assembles it."""
    root = tmp_path_factory.mktemp("odin_deploy_pkg")
    (root / "host").mkdir()
    (root / "deployment").mkdir()
    for name in HOST_FILES:
        shutil.copyfile(PACKAGE / "host" / name, root / "host" / name)
    shutil.copyfile(BUNDLE_MODULE, root / "host" / "odin_deployment_bundle.py")
    for name in (BUNDLE_NAME, REPLAY_NAME):
        shutil.copyfile(PACKAGE / "deployment" / name, root / "deployment" / name)
    return root


def run_executor(staged: Path, *args: str, results: str = "results/deploy"):
    """The executor as the sbatch job runs it: one process, one exit code."""
    return subprocess.run(
        [sys.executable, "host/odin_deployment_executor.py",
         "--xclbin", "/selftest/no-such.xclbin",
         "--fake-pyxrt", "host/fake_pyxrt_for_selftest.py",
         "--bundle", f"deployment/{BUNDLE_NAME}",
         "--replay", f"deployment/{REPLAY_NAME}",
         "--results", results, *args],
        cwd=str(staged), capture_output=True, text=True)


def reseal(path: Path, mutate) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(
        json.dumps(bundle.seal(document), sort_keys=True,
                   separators=(",", ":")) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def clean_run(staged):
    completed = run_executor(staged)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(
        (staged / "results" / "deploy" / "deployment_report.json")
        .read_text(encoding="utf-8"))
    return completed, report


class TestTheCleanCampaign:

    def test_it_certifies_every_pass_of_every_certified_sample(self, clean_run,
                                                               staged):
        _completed, report = clean_run
        document = bundle.load_bundle(str(staged / "deployment" / BUNDLE_NAME))
        certified = bundle.certification_samples(document)
        assert len(report["certificates"]) == (
            len(certified) * len(document["pass_order"]))
        assert all(row["passed"] for row in report["certificates"])
        assert all(row["certificate"]["exact_match_fraction"] == 1.0
                   for row in report["certificates"])

    def test_the_certificate_line_is_the_house_format_and_the_exact_class(
            self, clean_run):
        _completed, report = clean_run
        line = report["certificates"][0]["certificate_line"]
        assert line.startswith("[SpikeCountCertificate] ")
        assert "[odin_fpga/exact]" in line and "max|dcount|=0" in line

    def test_the_final_readout_of_every_shipped_sample_matches_the_frozen_one(
            self, clean_run):
        _completed, report = clean_run
        assert report["readout"]
        assert all(row["matches_frozen"] for row in report["readout"])

    def test_accuracy_is_accumulated_from_the_labels_not_asserted(self, clean_run):
        _completed, report = clean_run
        correct = sum(1 for row in report["readout"] if row["correct"])
        assert report["correct"] == correct
        assert report["accuracy"] == pytest.approx(correct / len(report["readout"]))
        assert 0.0 < report["accuracy"] < 1.0

    def test_each_core_is_programmed_exactly_once(self, clean_run, staged):
        _completed, report = clean_run
        document = bundle.load_bundle(str(staged / "deployment" / BUNDLE_NAME))
        programmed = [row["core"] for row in report["per_core"]]
        assert programmed == list(document["pass_order"])
        assert len(programmed) == len(set(programmed))

    def test_it_ran_one_pass_per_core_per_sample(self, clean_run, staged):
        _completed, report = clean_run
        document = bundle.load_bundle(str(staged / "deployment" / BUNDLE_NAME))
        assert report["walls"]["passes"] == (
            len(document["pass_order"]) * len(report["campaign"]["samples_run"]))

    def test_the_report_names_the_bundle_bytes_it_executed(self, clean_run, staged):
        _completed, report = clean_run
        document = bundle.load_bundle(str(staged / "deployment" / BUNDLE_NAME))
        assert report["bundle"]["self_hash"] == document["self_hash"]
        assert "run_cosim" in report["bundle"]["provenance"]["derivation"]


class TestEveryDistinctStageIsTimed:

    EXPECTED = ("bo_write_s", "sync_s", "run_s", "readback_s", "decode_s",
                "transcode_s", "pass_total_s")

    def test_every_stage_carries_percentiles_and_a_total(self, clean_run):
        _completed, report = clean_run
        for field in self.EXPECTED:
            stage = report["walls"]["per_stage"][field]
            assert set(stage) == {"p50", "p90", "p99", "min", "max", "mean",
                                  "total"}
            assert stage["min"] <= stage["p50"] <= stage["p90"] <= stage["max"]
            assert stage["total"] > 0.0

    def test_the_per_sample_and_per_core_walls_are_reported_separately(
            self, clean_run):
        _completed, report = clean_run
        assert report["walls"]["sample_total_s"]["total"] > 0.0
        assert report["walls"]["core_program_s"]["total"] > 0.0
        assert all("core_program_s" in row and "program_bo_write_s" in row
                   and "program_sync_s" in row
                   for row in report["per_core"])

    def test_a_pass_total_is_at_least_the_stages_it_contains(self, clean_run,
                                                             staged):
        _completed, _report = clean_run
        rows = (staged / "results" / "deploy" / "deployment_samples.tsv"
                ).read_text(encoding="utf-8").splitlines()
        header = rows[0].split("\t")
        assert header[:2] == ["sample", "core"]
        for line in rows[1:]:
            row = dict(zip(header, line.split("\t")))
            parts = sum(float(row[field]) for field in
                        ("bo_write_s", "sync_s", "run_s", "readback_s",
                         "decode_s", "transcode_s"))
            assert float(row["pass_total_s"]) >= parts * 0.99

    def test_the_tsv_carries_one_row_per_pass(self, clean_run, staged):
        _completed, report = clean_run
        rows = (staged / "results" / "deploy" / "deployment_samples.tsv"
                ).read_text(encoding="utf-8").splitlines()
        assert len(rows) - 1 == report["walls"]["passes"] == report["tsv_rows"]


class TestTheSampleBound:

    def test_the_default_names_its_own_source(self, clean_run):
        _completed, report = clean_run
        assert "packaged default" in report["campaign"]["sample_bound_source"]

    def test_the_command_line_narrows_the_campaign(self, staged):
        completed = run_executor(staged, "--samples", "2",
                                 results="results/two")
        assert completed.returncode == 0, completed.stdout + completed.stderr
        report = json.loads((staged / "results" / "two" /
                             "deployment_report.json").read_text())
        assert report["campaign"]["samples_run"] == [0, 1]
        assert "command line" in report["campaign"]["sample_bound_source"]

    def test_the_environment_bound_is_recorded_as_such(self, staged, tmp_path):
        completed = subprocess.run(
            [sys.executable, "host/odin_deployment_executor.py",
             "--xclbin", "x", "--fake-pyxrt", "host/fake_pyxrt_for_selftest.py",
             "--bundle", f"deployment/{BUNDLE_NAME}",
             "--replay", f"deployment/{REPLAY_NAME}",
             "--results", str(tmp_path / "env")],
            cwd=str(staged), capture_output=True, text=True,
            env={**dict(__import__("os").environ), "ODIN_DEPLOY_SAMPLES": "1"})
        assert completed.returncode == 0, completed.stdout + completed.stderr
        report = json.loads(
            (tmp_path / "env" / "deployment_report.json").read_text())
        assert report["campaign"]["samples_run"] == [0]
        assert "ODIN_DEPLOY_SAMPLES=1" in report["campaign"]["sample_bound_source"]

    def test_a_campaign_of_zero_samples_refuses(self, staged, tmp_path):
        completed = run_executor(staged, "--samples", "0",
                                 results=str(tmp_path / "none"))
        assert completed.returncode == 2
        assert "not a result" in completed.stderr


class TestTheMutations:
    """Each of these is a way the evidence or the device could be wrong."""

    def test_tampered_expected_counts_go_red_instead_of_quiet(self, staged,
                                                              tmp_path):
        work = tmp_path / "counts"
        shutil.copytree(staged, work, ignore=shutil.ignore_patterns("results"))
        reseal(work / "deployment" / BUNDLE_NAME,
               lambda doc: doc["certification"]["windows"]["0"]["1"][0]
               .__setitem__(0, doc["certification"]["windows"]["0"]["1"][0][0] + 1))
        document = json.loads(
            (work / "deployment" / BUNDLE_NAME).read_text())
        reseal(work / "deployment" / REPLAY_NAME,
               lambda doc: doc.__setitem__("bundle_self_hash",
                                           document["self_hash"]))
        completed = run_executor(work, results="results/mutant")
        assert completed.returncode == 1, completed.stdout + completed.stderr
        assert "FAIL exact=" in completed.stdout
        report = json.loads((work / "results" / "mutant" /
                             "deployment_report.json").read_text())
        assert report["passed"] is False
        assert any(not row["passed"] for row in report["certificates"])

    def test_a_tampered_self_hash_refuses_in_the_fixture_corrupt_class(
            self, staged, tmp_path):
        work = tmp_path / "seal"
        shutil.copytree(staged, work, ignore=shutil.ignore_patterns("results"))
        target = work / "deployment" / BUNDLE_NAME
        target.write_text(
            target.read_text(encoding="utf-8").replace('"NC=1', '"nc=1', 1),
            encoding="utf-8")
        completed = run_executor(work, results="results/seal")
        assert completed.returncode == 2
        assert "OdinBundleCorrupt" in completed.stderr
        assert "frozen evidence" in completed.stderr

    def test_the_corrupt_class_is_the_drivers_own(self, staged):
        sys.path.insert(0, str(staged / "host"))
        try:
            for name in ("odin_board_driver", "odin_deployment_bundle",
                         "odin_deployment_executor"):
                sys.modules.pop(name, None)
            import odin_board_driver as driver
            import odin_deployment_executor as executor
        finally:
            sys.path.pop(0)
        assert issubclass(executor.OdinBundleCorrupt, driver.OdinFixtureCorrupt)
        assert issubclass(executor.OdinTranscodeDiverged, driver.OdinDriverError)

    def test_a_damaged_program_payload_refuses_before_the_card_is_written(
            self, staged, tmp_path):
        work = tmp_path / "payload"
        shutil.copytree(staged, work, ignore=shutil.ignore_patterns("results"))
        reseal(work / "deployment" / BUNDLE_NAME,
               lambda doc: doc["cores"][0]["program"].__setitem__("bytes", 17))
        document = json.loads((work / "deployment" / BUNDLE_NAME).read_text())
        reseal(work / "deployment" / REPLAY_NAME,
               lambda doc: doc.__setitem__("bundle_self_hash",
                                           document["self_hash"]))
        completed = run_executor(work, results="results/payload")
        assert completed.returncode == 2
        assert "OdinBundleCorrupt" in completed.stderr

    def test_a_replay_frozen_for_another_bundle_refuses(self, staged, tmp_path):
        work = tmp_path / "replay"
        shutil.copytree(staged, work, ignore=shutil.ignore_patterns("results"))
        reseal(work / "deployment" / REPLAY_NAME,
               lambda doc: doc.__setitem__("bundle_self_hash", "sha256:0"))
        completed = run_executor(work, results="results/replay")
        assert completed.returncode == 2
        assert "certify a run nobody made" in completed.stderr

    def test_a_device_that_emits_one_extra_spike_fails_the_certificate(
            self, staged, tmp_path):
        work = tmp_path / "extra"
        shutil.copytree(staged, work, ignore=shutil.ignore_patterns("results"))

        def add_spike(doc):
            run = min(doc["runs"], key=lambda entry: (entry["core"],
                                                      entry["sample"]))
            run["events"].append(list(run["events"][0]))

        reseal(work / "deployment" / REPLAY_NAME, add_spike)
        completed = run_executor(work, results="results/extra")
        assert completed.returncode in (1, 2), completed.stdout + completed.stderr
        assert ("FAIL exact=" in completed.stdout
                or "OdinTranscodeDiverged" in completed.stderr)

    def test_a_pass_order_that_runs_a_consumer_first_refuses(self, staged,
                                                            tmp_path):
        work = tmp_path / "order"
        shutil.copytree(staged, work, ignore=shutil.ignore_patterns("results"))
        reseal(work / "deployment" / BUNDLE_NAME,
               lambda doc: doc.__setitem__(
                   "pass_order", list(reversed(doc["pass_order"]))))
        document = json.loads((work / "deployment" / BUNDLE_NAME).read_text())
        reseal(work / "deployment" / REPLAY_NAME,
               lambda doc: doc.__setitem__("bundle_self_hash",
                                           document["self_hash"]))
        completed = run_executor(work, results="results/order")
        assert completed.returncode == 2
        assert "before core 0" in completed.stderr

    def test_a_bundle_built_for_another_kernel_refuses(self, staged, tmp_path):
        work = tmp_path / "kernel"
        shutil.copytree(staged, work, ignore=shutil.ignore_patterns("results"))
        reseal(work / "deployment" / BUNDLE_NAME,
               lambda doc: doc["kernel"].__setitem__("kernel_args", 5))
        document = json.loads((work / "deployment" / BUNDLE_NAME).read_text())
        reseal(work / "deployment" / REPLAY_NAME,
               lambda doc: doc.__setitem__("bundle_self_hash",
                                           document["self_hash"]))
        completed = run_executor(work, results="results/kernel")
        assert completed.returncode == 2
        assert "different kernel" in completed.stderr


@pytest.fixture(scope="module")
def placement(tmp_path_factory) -> Path:
    """A placement CSV shaped like the mining Tcl's, with a shell to stay grey."""
    path = tmp_path_factory.mktemp("die") / "placement.csv"
    rows = ["name,class,site,x,y"]
    for index in range(400):
        kind = ("odin_core" if index % 3 == 0
                else "sequencer" if index % 3 == 1 else "shell")
        rows.append(f"cell_{index},{kind},SLICE_X{index}Y{index},"
                    f"{index % 40},{index % 37}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


class TestTheDieMapRenderer:

    def _render(self, staged: Path, placement: Path, out: Path, fmt: str):
        return subprocess.run(
            [sys.executable, str(staged / "host" / "render_die_map.py"),
             "--csv", str(placement), "--out", str(out), "--format", fmt],
            capture_output=True, text=True)

    def test_the_stdlib_path_writes_rects_with_no_dependency(self, staged,
                                                             placement, tmp_path):
        out = tmp_path / "die.svg"
        completed = self._render(staged, placement, out, "svg")
        assert completed.returncode == 0, completed.stderr
        text = out.read_text(encoding="utf-8")
        assert text.startswith("<svg") and "</svg>" in text
        assert text.count("<rect") > 10
        assert "#9aa0a6" in text, "the shell must stay grey"
        assert "svg(stdlib)" in completed.stdout

    def test_the_matplotlib_path_writes_a_png_when_it_is_there(self, staged,
                                                               placement,
                                                               tmp_path):
        pytest.importorskip("matplotlib")
        out = tmp_path / "die.png"
        completed = self._render(staged, placement, out, "png")
        assert completed.returncode == 0, completed.stderr
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert "png(matplotlib)" in completed.stdout

    def test_auto_falls_back_to_svg_rather_than_drawing_nothing(self, staged,
                                                                placement,
                                                                tmp_path):
        out = tmp_path / "auto.png"
        completed = subprocess.run(
            [sys.executable, str(staged / "host" / "render_die_map.py"),
             "--csv", str(placement), "--out", str(out), "--format", "auto"],
            capture_output=True, text=True,
            env={"PYTHONPATH": str(tmp_path / "no-matplotlib"),
                 "PATH": "/usr/bin:/bin"})
        assert completed.returncode == 0, completed.stderr
        assert out.exists() or out.with_suffix(".svg").exists()

    def test_a_csv_without_coordinates_refuses(self, staged, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("name,class\na,shell\n", encoding="utf-8")
        completed = self._render(staged, path, tmp_path / "x.svg", "svg")
        assert completed.returncode == 2
        assert "column(s)" in completed.stderr

    def test_a_missing_checkpoint_csv_names_where_it_comes_from(self, staged,
                                                                tmp_path):
        completed = self._render(
            staged, tmp_path / "absent.csv", tmp_path / "x.svg", "svg")
        assert completed.returncode == 2
        assert "mine_checkpoint.sh" in completed.stderr
