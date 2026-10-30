"""[U-H seam] A deployment bundle names the run that froze it, read from the session, never from the merged config.

The lineage injected ``experiment_name`` into the deployment parameters (d9703808) so the HACC export step could read
it off ``pipeline.config``. The release head's run config carries no run name (``parse_deployment_config`` lays out
the sections as declared, and its tests pin that), so the port hands the run's name to the pipeline instead: the
session sets ``pipeline.deployment_name`` from the parsed document, and the export step's provenance and its default
bundle name read it there, as the session reads it (verbatim, as it names the working directory).
"""

from __future__ import annotations

from types import SimpleNamespace

from conftest import MockDataProviderFactory

from mimarsinan.pipelining.pipeline_steps.verification.odin import odin_hacc_deployment_step as S
from mimarsinan.pipelining.session import PipelineSession


def _config(tmp_path):
    return {
        "experiment_name": "odin_run_name_test",
        "data_provider_name": "MNIST_DataProvider",
        "generated_files_path": str(tmp_path),
        "deployment_parameters": {"model_type": "mlp_mixer", "spiking_mode": "lif", "model_config_mode": "user",
                                  "hw_config_mode": "fixed", "model_config": {}},
        "platform_constraints": {},
    }


def test_the_session_hands_its_run_name_to_the_pipeline(tmp_path):
    session = PipelineSession.from_config(_config(tmp_path), data_provider_factory=MockDataProviderFactory())
    assert session.pipeline.deployment_name == "odin_run_name_test"
    assert "experiment_name" not in session.pipeline.config   # the release head's run config, unchanged


def test_the_default_bundle_name_is_the_run_name():
    assert S._default_name(SimpleNamespace(deployment_name="mnist_run", config={})) == "mnist_run"


def test_a_pipeline_without_a_run_name_takes_the_generic_name():
    assert S._default_name(SimpleNamespace(config={})) == "odin_hacc_deployment"
    assert S._default_name(SimpleNamespace(deployment_name="", config={})) == "odin_hacc_deployment"


def test_the_merged_config_is_not_where_the_name_is_read():
    assert S._default_name(SimpleNamespace(config={"experiment_name": "x"})) == "odin_hacc_deployment"


def test_the_provenance_names_the_run():
    step = S.OdinHaccDeploymentStep.__new__(S.OdinHaccDeploymentStep)
    step.pipeline = SimpleNamespace(deployment_name="frozen_by", config={"experiment_name": "not_this"})
    assert step._provenance(["hcm"])["experiment"] == "frozen_by"
