"""[ODIN P8] the DEPLOYMENT package: one upload that carries the network too.

``make_package.py --deployment BUNDLE`` ships the v4 bring-up package plus an
exported bundle, its replay, and the index phase 8 reads to learn WHICH network
it should run. These gates pin that contract without paying for a cosimulation:
the staging rules, the refusals, and the one field ``run_all.sh`` parses out of
the index with ``sed`` on a node that may have no JSON tool at all.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from integration.odin_hacc_harness import (
    WIDE_PLATFORM_RESOLVED,
    prepare_step,
    wide_config_overrides,
)

from mimarsinan.chip_simulation.odin_deployment_bundle import seal
from mimarsinan.chip_simulation.odin_fpga.chip_configs import STOCK_CHIP, WIDE_CHIP
from mimarsinan.chip_simulation.odin_hacc.artifact import render_bundle
from mimarsinan.pipelining.pipeline_steps.verification.odin_hacc_deployment_step import (
    OdinHaccDeploymentStep,
)

REPO = Path(__file__).resolve().parents[3]
RUN_ALL = REPO / "scripts" / "hacc" / "package" / "run_all.sh"


@pytest.fixture(scope="module")
def packager():
    """``scripts/hacc/make_package.py`` as a module (it is the only author)."""
    path = REPO / "scripts" / "hacc" / "make_package.py"
    spec = importlib.util.spec_from_file_location("odin_make_package", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bundle_pair(tmp_path_factory):
    """One exported bundle + its replay, from the REAL export step."""
    monkeypatch = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("odin_hacc_pkg_export")
    try:
        pipeline, step = prepare_step(
            monkeypatch, OdinHaccDeploymentStep, working_directory=str(root))
        step.process()
    finally:
        monkeypatch.undo()
    stats = pipeline.cache["OdinHaccDeploymentStep.odin_hacc_deployment_bundle"]
    source = Path(stats["paths"]["bundle"])
    named = source.with_name("micro_deploy.json")
    named.write_text(render_bundle(step.document), encoding="utf-8")
    Path(stats["paths"]["capture"]).rename(
        named.with_name("micro_deploy_capture.json"))
    return named, step.document


@pytest.fixture(scope="module")
def wide_bundle_pair(tmp_path_factory):
    """The same network exported against the WIDE fabric, ready to package."""
    monkeypatch = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("odin_hacc_pkg_export_wide")
    try:
        pipeline, step = prepare_step(
            monkeypatch, OdinHaccDeploymentStep, working_directory=str(root),
            config_overrides=wide_config_overrides(),
            platform_resolved=WIDE_PLATFORM_RESOLVED)
        step.process()
    finally:
        monkeypatch.undo()
    stats = pipeline.cache["OdinHaccDeploymentStep.odin_hacc_deployment_bundle"]
    source = Path(stats["paths"]["bundle"])
    named = source.with_name("wide_deploy.json")
    named.write_text(render_bundle(step.document), encoding="utf-8")
    Path(stats["paths"]["capture"]).rename(
        named.with_name("wide_deploy_capture.json"))
    return named, step.document


def _stage(packager, bundles, root: Path):
    written = {}

    def copy(source: Path, relative: str) -> None:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source).read_bytes())
        written[relative] = target

    def write_text(relative: str, text: str) -> None:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        written[relative] = target

    packager.stage_deployment(bundles, copy, write_text)
    return written


class TestTheDeploymentPackageDeclaresWhatItDeploys:
    def test_the_bundle_its_replay_and_an_index_are_staged(
        self, packager, bundle_pair, tmp_path,
    ):
        path, document = bundle_pair
        written = _stage(packager, [path], tmp_path)
        assert set(written) == {
            "deployment/micro_deploy.json",
            "deployment/micro_deploy_capture.json",
            "deployment/DEPLOYMENT.json",
        }
        index = json.loads(written["deployment/DEPLOYMENT.json"].read_text())
        assert index["default"] == "deployment/micro_deploy.json"
        assert index["default_replay"] == "deployment/micro_deploy_capture.json"
        entry = index["bundles"][0]
        assert entry["self_hash"] == document["self_hash"]
        assert entry["samples"] == len(document["samples"])
        assert entry["cores"] == len(document["cores"])
        assert entry["provenance"] == document["provenance"]

    def test_run_all_reads_the_default_out_of_the_index(
        self, packager, bundle_pair, tmp_path,
    ):
        """The node parses this file with ``sed``; a JSON tool is not promised."""
        path, _document = bundle_pair
        written = _stage(packager, [path], tmp_path)
        index = written["deployment/DEPLOYMENT.json"]
        script = (
            f'DEPLOYMENT_INDEX="{index}"\n'
            + _function_source("deployment_default")
            + "\ndeployment_default\n"
        )
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "deployment/micro_deploy.json"


class TestThePackageBuildsTheFabricItsBundleWasMappedOn:
    """A bundle names no chip; its CLAIMS do, and a package must build one."""

    def test_the_index_names_the_fabric_of_every_bundle(
        self, packager, bundle_pair, tmp_path,
    ):
        path, _document = bundle_pair
        written = _stage(packager, [path], tmp_path)
        index = json.loads(written["deployment/DEPLOYMENT.json"].read_text())
        assert index["default_chip"] == STOCK_CHIP
        assert index["bundles"][0]["chip"] == STOCK_CHIP

    def test_a_wide_bundle_makes_the_package_build_the_wide_fabric(
        self, packager, wide_bundle_pair, tmp_path,
    ):
        path, _document = wide_bundle_pair
        written = _stage(packager, [path], tmp_path)
        index = json.loads(written["deployment/DEPLOYMENT.json"].read_text())
        assert index["default_chip"] == WIDE_CHIP

    @pytest.mark.parametrize("chip", [STOCK_CHIP, WIDE_CHIP])
    def test_run_all_reads_the_chip_the_same_way_the_node_will(
        self, packager, bundle_pair, wide_bundle_pair, tmp_path, chip,
    ):
        """``sed`` on the index — a board node is promised no JSON tool."""
        path = (bundle_pair if chip == STOCK_CHIP else wide_bundle_pair)[0]
        written = _stage(packager, [path], tmp_path / chip)
        index = written["deployment/DEPLOYMENT.json"]
        script = (
            f'DEPLOYMENT_INDEX="{index}"\n'
            + _function_source("deployment_chip")
            + "\ndeployment_chip\n"
        )
        out = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == chip

    def test_a_bundle_whose_envelope_no_fabric_can_be_refuses(
        self, packager, bundle_pair, tmp_path,
    ):
        path, document = bundle_pair
        narrowed = json.loads(path.read_text())
        narrowed["chip_config"] = dict(narrowed["chip_config"])
        narrowed["chip_config"]["effective_max_axons"] = 7
        resealed = tmp_path / "narrow.json"
        resealed.write_text(
            render_bundle(seal(narrowed)), encoding="utf-8")
        (tmp_path / "narrow_capture.json").write_text("{}", encoding="utf-8")
        with pytest.raises(packager.PackagingRefusal, match="name no fabric"):
            _stage(packager, [resealed], tmp_path / "stage")


class TestPackagingRefusesAnUnrunnableDeployment:
    def test_a_bundle_without_its_replay_refuses(
        self, packager, bundle_pair, tmp_path,
    ):
        path, _document = bundle_pair
        orphan = tmp_path / "orphan.json"
        orphan.write_text(path.read_text(), encoding="utf-8")
        with pytest.raises(packager.PackagingRefusal, match="no orphan_capture"):
            _stage(packager, [orphan], tmp_path / "stage")

    def test_a_replay_handed_in_as_a_bundle_refuses(
        self, packager, bundle_pair, tmp_path,
    ):
        path, _document = bundle_pair
        replay = path.with_name("micro_deploy_capture.json")
        with pytest.raises(packager.PackagingRefusal, match="is not"):
            packager.require_bundle_document(replay)

    def test_an_edited_bundle_refuses_before_it_is_staged(
        self, packager, bundle_pair, tmp_path,
    ):
        path, _document = bundle_pair
        tampered = tmp_path / "tampered.json"
        document = json.loads(path.read_text())
        document["cycles_per_sample"] = int(document["cycles_per_sample"]) + 1
        tampered.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(packager.PackagingRefusal, match="self-hash"):
            packager.require_bundle_document(tampered)

    def test_a_missing_bundle_refuses_by_path(self, packager, tmp_path):
        with pytest.raises(packager.PackagingRefusal, match="no such file"):
            packager.require_bundle_document(tmp_path / "absent.json")


def _function_source(name: str) -> str:
    """Lift one shell function out of ``run_all.sh`` so bash can run it alone."""
    lines = RUN_ALL.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{name}()"))
    end = next(i for i in range(start, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start:end + 1])
