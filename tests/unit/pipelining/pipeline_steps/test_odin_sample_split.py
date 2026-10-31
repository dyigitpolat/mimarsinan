"""[U-HV] The ODIN steps read the declared sample split: the development split by default, the test split by name only.

The HACC export (``OdinHaccDeploymentStep``) ships a card campaign and certifies a subset of it; the device run
(``OdinFpgaDeploymentStep``) runs samples through a device. Both draw their samples from ONE declared split,
``odin_sample_split``: absent, the development split (the provider's validation carve), so neither step builds or
reads the test split; ``"test"`` reads the test split, which only a run under the test lock declares; any other
value is refused by name before any sample is read. The split and its per-split sample indices are written where the
samples go: the bundle's provenance (and its capture's), each step's cache entry (``sample_read``), and from there
the sealed deployment record (``test_device_samples_record.py``).

Before U-HV the export read ``load_test_pairs_by_index`` and the device run ``load_test_samples_by_index``: the test
split, always. The fixture provider here counts every test-split construction and read on the provider itself, and
the read spy sees the split every verified read was asked for; neither touches a real dataset's test file.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

import integration.odin_fpga_harness as F
import integration.odin_hacc_harness as H
from conftest import MockPipeline

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.data_handling import sample_loader
from mimarsinan.pipelining.pipeline_steps.verification.odin import odin_fpga_simulation_step as fpga_module
from mimarsinan.pipelining.pipeline_steps.verification.odin.odin_fpga_simulation_step import OdinFpgaDeploymentStep
from mimarsinan.pipelining.pipeline_steps.verification.odin.odin_hacc_deployment_step import OdinHaccDeploymentStep

from unit.data_handling.split_fixtures import SplitCountingFactory

KEY = "odin_sample_split"
SHIPPED = 6  # the HACC harness's odin_hacc_bundle_samples
REAL_PAIRS_BY_INDEX = sample_loader._pairs_by_index


def _same(pairs_a, pairs_b) -> bool:
    return len(pairs_a) == len(pairs_b) and all(
        torch.equal(xa, xb) and torch.equal(ya, yb) for (xa, ya), (xb, yb) in zip(pairs_a, pairs_b))


def _one_hot_split(shift: int, count: int = 8):
    xs = torch.tensor(np.stack([H.one_hot(k + shift)[0] for k in range(count)]), dtype=torch.float32)
    ys = torch.tensor([(k + shift) % H.CLASSES for k in range(count)], dtype=torch.long)
    return xs, ys


def _hacc_factory() -> SplitCountingFactory:
    """Validation sample k is class k; test sample k is class k + 1: the shipped labels tell the splits apart."""
    return SplitCountingFactory(_one_hot_split(0), _one_hot_split(1), num_classes=H.CLASSES)


def _fpga_factory() -> SplitCountingFactory:
    """The device fixture relays a saturated raster, so both splits are all-ones inputs; their labels differ."""
    ones = torch.ones(4, F.INPUT_LINES, dtype=torch.float32)
    return SplitCountingFactory((ones, torch.zeros(4, dtype=torch.long)), (ones.clone(), torch.ones(4, dtype=torch.long)),
                                num_classes=F.NEURONS)


def _spy(monkeypatch):
    """Every verified read the run makes: the split it was asked for, its indices and the pairs it returned."""
    reads = []

    def spy(data_provider_factory, indices, *, num_workers, split):
        pairs = REAL_PAIRS_BY_INDEX(data_provider_factory, indices, num_workers=num_workers, split=split)
        reads.append({"split": split, "indices": [int(i) for i in indices], "pairs": pairs})
        return pairs

    monkeypatch.setattr(sample_loader, "_pairs_by_index", spy)
    return reads


def _hacc(tmp_path, **overrides):
    factory = _hacc_factory()
    pipeline = MockPipeline(config=H.pipeline_config(**overrides), working_directory=str(tmp_path),
                            data_provider_factory=factory)
    pipeline.set_target_metric(0.5)
    pipeline.seed("model", object(), step_name="Model Configuration")
    pipeline.seed("hard_core_mapping", H.hybrid_program(), step_name="Hard Core Mapping")
    pipeline.seed("platform_constraints_resolved", dict(H.PLATFORM_RESOLVED), step_name="Model Configuration")
    step = OdinHaccDeploymentStep(pipeline)
    pipeline.prepare_step(step)
    return pipeline, step, factory


def _fpga(tmp_path, monkeypatch, **overrides):
    from unit.pipelining.pipeline_steps.test_odin_fpga_step import TwinTransport

    monkeypatch.setattr(fpga_module, "build_transport", lambda _config: TwinTransport(F.two_core_mapping()))
    factory = _fpga_factory()
    pipeline = MockPipeline(config=F.pipeline_config(**overrides), working_directory=str(tmp_path),
                            data_provider_factory=factory)
    pipeline.set_target_metric(0.5)
    pipeline.seed("model", object(), step_name="Model Configuration")
    pipeline.seed("hard_core_mapping", F.hybrid_program(), step_name="Hard Core Mapping")
    pipeline.seed("platform_constraints_resolved", dict(F.PLATFORM_RESOLVED), step_name="Model Configuration")
    step = OdinFpgaDeploymentStep(pipeline)
    pipeline.prepare_step(step)
    return pipeline, step, factory


def _expected_read(split: str, indices, backend: str) -> dict:
    return {"backend": backend, "declared_by": KEY, "split": split, "indices": list(indices)}


class TestTheHaccCampaignReadsTheDeclaredSplit:
    def test_the_default_builds_and_reads_no_test_split(self, tmp_path):
        _pipeline, step, factory = _hacc(tmp_path)
        step.process()
        assert factory.test_events() == {"constructions": 0, "reads": 0}

    def test_the_default_ships_the_validation_pairs(self, tmp_path, monkeypatch):
        reads = _spy(monkeypatch)
        _pipeline, step, _factory = _hacc(tmp_path)
        step.process()
        expected = REAL_PAIRS_BY_INDEX(_hacc_factory(), list(range(SHIPPED)), num_workers=0, split="validation")
        assert [(read["split"], read["indices"]) for read in reads] == [("validation", list(range(SHIPPED)))]
        assert _same(reads[0]["pairs"], expected)
        assert [row["label"] for row in step.document["samples"]] == [int(y) for _x, y in expected]

    def test_the_bundle_its_capture_and_the_entry_record_the_split_and_the_ids(self, tmp_path):
        pipeline, step, _factory = _hacc(tmp_path)
        step.process()
        read = _expected_read("validation", range(SHIPPED), "odin_hacc")
        provenance = {k: v for k, v in read.items() if k != "backend"}
        assert step.document["provenance"]["samples"] == provenance
        entry = pipeline.cache["OdinHaccDeploymentStep.odin_hacc_deployment_bundle"]
        assert entry["sample_read"] == read
        with open(entry["paths"]["capture"], encoding="utf-8") as handle:
            assert json.load(handle)["provenance"]["samples"] == provenance
        with open(entry["paths"]["bundle"], encoding="utf-8") as handle:
            assert json.load(handle)["provenance"]["samples"] == provenance

    def test_a_declared_test_split_reads_the_test_pairs(self, tmp_path, monkeypatch):
        reads = _spy(monkeypatch)
        _pipeline, step, factory = _hacc(tmp_path, **{KEY: "test"})
        step.process()
        expected = REAL_PAIRS_BY_INDEX(_hacc_factory(), list(range(SHIPPED)), num_workers=0, split="test")
        assert [read["split"] for read in reads] == ["test"]
        assert _same(reads[0]["pairs"], expected)
        assert [row["label"] for row in step.document["samples"]] == [int(y) for _x, y in expected]
        assert step.document["provenance"]["samples"]["split"] == "test"
        assert factory.test_events()["reads"] > 0

    def test_a_declared_development_split_is_the_default(self, tmp_path, monkeypatch):
        reads = _spy(monkeypatch)
        _pipeline, step, factory = _hacc(tmp_path, **{KEY: "validation"})
        step.process()
        assert [read["split"] for read in reads] == ["validation"]
        assert factory.test_events() == {"constructions": 0, "reads": 0}

    @pytest.mark.parametrize("value", ["development", "valid", "Test"])
    def test_an_unknown_split_is_refused_by_name_before_any_read(self, tmp_path, value):
        _pipeline, step, factory = _hacc(tmp_path, **{KEY: value})
        with pytest.raises(ToolchainRefusal, match=f"{KEY}={value!r}"):
            step.process()
        assert factory.built == 0
        assert not os.path.exists(os.path.join(str(tmp_path), "odin_hacc"))


class TestTheDeviceRunReadsTheDeclaredSplit:
    def test_the_default_builds_and_reads_no_test_split(self, tmp_path, monkeypatch):
        _pipeline, step, factory = _fpga(tmp_path, monkeypatch)
        step.process()
        assert factory.test_events() == {"constructions": 0, "reads": 0}

    def test_the_default_runs_the_validation_samples_and_records_them(self, tmp_path, monkeypatch):
        reads = _spy(monkeypatch)
        pipeline, step, _factory = _fpga(tmp_path, monkeypatch, odin_fpga_sample_count=2)
        step.process()
        assert [(read["split"], read["indices"]) for read in reads] == [("validation", [0, 1])]
        report = pipeline.cache["OdinFpgaDeploymentStep.odin_fpga_deployment_results"]
        assert report["sample_read"] == _expected_read("validation", [0, 1], "odin_fpga")
        assert step.step_verdict()["detail"]["sample_split"] == "validation"

    def test_a_declared_test_split_runs_the_test_samples(self, tmp_path, monkeypatch):
        reads = _spy(monkeypatch)
        pipeline, step, factory = _fpga(tmp_path, monkeypatch, **{KEY: "test"})
        step.process()
        assert [read["split"] for read in reads] == ["test"]
        report = pipeline.cache["OdinFpgaDeploymentStep.odin_fpga_deployment_results"]
        assert report["sample_read"] == _expected_read("test", [0], "odin_fpga")
        assert factory.test_events()["reads"] > 0

    def test_an_unknown_split_is_refused_by_name_before_any_read(self, tmp_path, monkeypatch):
        _pipeline, step, factory = _fpga(tmp_path, monkeypatch, **{KEY: "holdout"})
        with pytest.raises(ToolchainRefusal, match=f"{KEY}='holdout'"):
            step.process()
        assert factory.built == 0


class TestTheDeclaration:
    def test_the_key_has_no_schema_default_and_absence_reads_the_development_split(self):
        from mimarsinan.config_schema.defaults import CONFIG_KEYS_SET, get_default_deployment_parameters
        from mimarsinan.config_schema.registry import NO_DEFAULT, REGISTRY, effective_value

        entry = REGISTRY[KEY]
        assert entry.default is NO_DEFAULT and entry.provenance == "consumer frozen default"
        assert effective_value({}, KEY) == "validation" == sample_loader.DEVELOPMENT_SPLIT
        assert KEY in CONFIG_KEYS_SET and KEY not in get_default_deployment_parameters()

    def test_it_is_an_enum_of_the_reader_s_splits(self):
        from mimarsinan.config_schema.registry import REGISTRY, FieldType

        entry = REGISTRY[KEY]
        assert entry.type is FieldType.ENUM
        assert tuple(entry.resolved_options()) == sample_loader.SAMPLE_SPLITS == ("validation", "test")

    def test_it_bears_on_admissibility(self):
        from mimarsinan.config_schema.registry import REGISTRY
        from mimarsinan.config_schema.registry.decision_classes import ADMISSIBILITY_BEARING

        assert REGISTRY[KEY].decision_class == ADMISSIBILITY_BEARING

    def test_it_is_relevant_where_an_odin_step_runs(self):
        from mimarsinan.config_schema.registry import REGISTRY

        relevant = REGISTRY[KEY].relevant.evaluate
        spiking = {"core_semantics": "spiking"}  # the registry adds the event domain's own condition
        assert relevant({**spiking, "enable_odin_hacc_export": True})
        assert relevant({**spiking, "enable_odin_fpga_simulation": True})
        assert not relevant({**spiking, "enable_odin_hacc_export": False, "enable_odin_fpga_simulation": False})

    def test_both_doors_refuse_an_unknown_split_by_name(self):
        import copy

        from mimarsinan.config_schema.validation import validate_deployment_config
        from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
        from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

        path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "fixtures", "deployment_configs",
                            "odin_wide_simplemlp.json")
        with open(path, encoding="utf-8") as handle:
            base = json.load(handle)
        for value in ("validation", "test"):
            document = copy.deepcopy(base)
            document["deployment_parameters"][KEY] = value
            assert validate_deployment_config(document) == []
        document = copy.deepcopy(base)
        document["deployment_parameters"][KEY] = "dev"
        assert any(KEY in error for error in validate_deployment_config(document))
        with pytest.raises(ToolchainRefusal, match=f"{KEY}='dev'"):  # the run door: merge, then resolve
            build_platform_constraints_resolved(merge_pipeline_config(
                copy.deepcopy(document["deployment_parameters"]), copy.deepcopy(document["platform_constraints"])))
