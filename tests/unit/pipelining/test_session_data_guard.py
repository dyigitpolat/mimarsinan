"""The session's data guard: a declared seal holds before any provider is built, and a handed-in factory serves the
provider the document declares, so the deployment record names the provider that ran."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import MockDataProviderFactory

from mimarsinan.config_schema.registry.value_schema import DeclarationValueError
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory, DataProviderFactory
from mimarsinan.data_handling.sealed_paths import SealedPathRefusal
from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import _workload_key
from mimarsinan.pipelining.session import PipelineSession, parse_deployment_config


def _config(tmp_path, **overrides):
    cfg = {
        "experiment_name": "guard_test",
        "data_provider_name": "MNIST_DataProvider",
        "generated_files_path": str(tmp_path),
        "deployment_parameters": {
            "model_type": "mlp_mixer",
            "spiking_mode": "lif",
            "model_config_mode": "user",
            "hw_config_mode": "fixed",
            "model_config": {},
        },
        "platform_constraints": {},
    }
    cfg.update(overrides)
    return cfg


def _held(tmp_path) -> Path:
    held = tmp_path / "held_out" / "test"
    held.parent.mkdir(parents=True, exist_ok=True)
    held.write_bytes(b"held out")
    return held


class _ServingFactory(DataProviderFactory):
    """A handed-in factory that declares the provider it serves."""

    def __init__(self, served: str):
        self._served = served
        self._inner = MockDataProviderFactory()

    @property
    def provider_name(self) -> str:
        return self._served

    def create(self):
        return self._inner.create()


class TestDeclaredSeal:
    def test_the_seal_holds_and_the_run_directory_records_it(self, tmp_path):
        held = _held(tmp_path)
        parsed = parse_deployment_config(
            _config(tmp_path), data_provider_factory=MockDataProviderFactory(), sealed_paths=[str(held)]
        )
        assert parsed.sealed_paths == (str(held.resolve()),)
        with pytest.raises(SealedPathRefusal):
            open(held, "rb")
        saved = json.loads((Path(parsed.working_directory) / "_RUN_CONFIG" / "sealed_paths.json").read_text())
        assert saved == {"sealed_paths": [str(held.resolve())]}

    def test_no_seal_records_nothing(self, tmp_path):
        parsed = parse_deployment_config(_config(tmp_path), data_provider_factory=MockDataProviderFactory())
        assert parsed.sealed_paths == ()
        assert not (Path(parsed.working_directory) / "_RUN_CONFIG" / "sealed_paths.json").exists()

    def test_the_seal_holds_before_the_provider_is_built(self, tmp_path):
        held = _held(tmp_path)

        class _OpeningFactory(MockDataProviderFactory):
            def create(self):
                open(held, "rb").close()
                return super().create()

        with pytest.raises(SealedPathRefusal):
            PipelineSession.from_config(_config(tmp_path), data_provider_factory=_OpeningFactory(),
                                        sealed_paths=[str(held)])


class TestHandedInFactory:
    def test_a_factory_serving_another_provider_is_refused_by_name(self, tmp_path):
        with pytest.raises(DeclarationValueError) as caught:
            parse_deployment_config(_config(tmp_path), data_provider_factory=_ServingFactory("Other_DataProvider"))
        assert "Other_DataProvider" in str(caught.value) and "MNIST_DataProvider" in str(caught.value)

    def test_a_factory_serving_the_declared_provider_is_taken(self, tmp_path):
        factory = _ServingFactory("MNIST_DataProvider")
        parsed = parse_deployment_config(_config(tmp_path), data_provider_factory=factory)
        assert parsed.data_provider_factory is factory

    def test_a_factory_that_declares_nothing_keeps_its_reading(self, tmp_path):
        factory = MockDataProviderFactory()
        parsed = parse_deployment_config(_config(tmp_path), data_provider_factory=factory)
        assert parsed.data_provider_factory is factory

    def test_the_registry_factory_serves_its_registered_name(self):
        assert BasicDataProviderFactory("MNIST_DataProvider", "./datasets").provider_name == "MNIST_DataProvider"
        assert DataProviderFactory().provider_name is None


class TestRecordWorkload:
    @staticmethod
    def _pipeline(factory):
        class _Pipeline:
            data_provider_factory = factory

        return _Pipeline()

    def test_the_record_names_the_provider_a_handed_in_factory_serves(self):
        assert _workload_key(self._pipeline(_ServingFactory("E01_DataProvider"))) == "E01_DataProvider"

    def test_the_registry_factory_and_an_undeclaring_factory_read_as_before(self):
        assert _workload_key(self._pipeline(BasicDataProviderFactory("MNIST_DataProvider", "./d"))) == "MNIST_DataProvider"
        assert _workload_key(self._pipeline(MockDataProviderFactory())) == "MockDataProviderFactory"
