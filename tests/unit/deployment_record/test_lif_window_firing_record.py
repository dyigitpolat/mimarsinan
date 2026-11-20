"""[U-LIF2 W7] The sealed record states what a synchronized LIF hop ran: the identity carries the run's window firing
and every pass its dependency depth D, under a new format version; a format-1 record is migrated explicitly on read
(it ran firing while integrating, one level a pass), and a window firing other than fire_while_integrating enters the
cell key's point tag, so every historical key is unchanged. Red on the set export: format 1, no such fields."""

from __future__ import annotations

import dataclasses
import importlib
import json
from types import SimpleNamespace

import pytest

from mimarsinan.code_generation.cpp_chip_model_types import SpikeSource
from mimarsinan.config_schema.runtime import derive_run_config
from mimarsinan.deployment_record.schema import DeploymentRecord
from mimarsinan.deployment_record.schema.record import DEPLOYMENT_RECORD_FORMAT_VERSION

from unit.deployment_record.record_fixtures import make_full_record

FWI, ITE = "fire_while_integrating", "integrate_then_emit"
_CORES = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 8}]}


@pytest.fixture
def declared(window_firing_declared):
    """Set the entry's data (the declared set and the default) for one test (the shared fixture)."""
    return window_firing_declared


def _run(**dp):
    return derive_run_config({"model_type": "simple_mlp", "spiking_family": "lif", "spiking_variant": "synchronized",
                              **dp}, dict(_CORES))


def test_t20_the_format_is_2_and_the_identity_states_the_window_firing():
    assert DEPLOYMENT_RECORD_FORMAT_VERSION == 3  # [composed head, rule 9]
    data = make_full_record().to_dict()
    assert data["format_version"] == data["identity"]["format_version"] == 3
    assert "window_firing" in data["identity"]


def test_t20_a_format_1_record_is_migrated_explicitly():
    data = json.loads(json.dumps(make_full_record().to_dict()))
    data["format_version"] = data["identity"]["format_version"] = 1
    data["identity"].pop("window_firing")
    data["timing"]["latency"]["host_ops_s"] = data["timing"]["latency"].pop("host_ops_s_raw_total")  # rule 9
    loaded = DeploymentRecord.from_dict(data)
    assert loaded.format_version == loaded.identity.format_version == 3
    assert loaded.identity.window_firing is None
    assert all(segment.dependency_depth is None for segment in loaded.schedule.segments())


def test_t20_the_identity_carries_the_run_window_firing():
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
    from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import identity_from_pipeline

    for config, expected in ((_run(), ITE), (_run(lif_window_firing=FWI), FWI), (_run(spiking_variant="streamed"), None)):
        pipeline = SimpleNamespace(working_directory="run", config=config,
                                   data_provider_factory=SimpleNamespace(_name="MNIST_DataProvider"))
        identity = identity_from_pipeline(pipeline, DeploymentPlan.resolve(config), {"cores": []})
        assert identity.window_firing == expected
        record = dataclasses.replace(make_full_record(), identity=identity)
        assert DeploymentRecord.from_dict(json.loads(json.dumps(record.to_dict()))).identity.window_firing == expected


def _segment(reads):
    """One pass whose core i reads the cores ``reads[i]`` (and a network input when it reads none)."""
    cores = []
    for producers in reads:
        sources = [SpikeSource(p, 0) for p in producers] or [SpikeSource(-2, 0, is_input=True)]
        cores.append(SimpleNamespace(axon_sources=sources + [SpikeSource(-3, 0, is_always_on=True)], latency=0))
    return SimpleNamespace(cores=cores)


@pytest.mark.parametrize("reads,depth", [
    ([], 0), ([()], 1), ([(), (0,), (1,)], 3), ([(), (0,), (0, 1)], 3), ([(), (), (0, 1)], 2),
    ([(), (0,), (1,), (0, 2)], 4),
])
def test_t20_each_pass_states_its_dependency_depth_from_the_wiring(reads, depth):
    depth_of = getattr(importlib.import_module("mimarsinan.deployment_record.build.program_census"),
                       "pass_dependency_depth")
    assert depth_of(_segment(reads)) == depth


def test_t20_a_window_firing_other_than_fire_while_integrating_enters_the_cell_key(declared):
    from mimarsinan.chip_simulation.certification import CertificationCell
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

    declared((FWI, ITE), FWI)
    historical = CertificationCell.from_mode_policy(DeploymentPlan.resolve(_run()).mode_policy(), backend="hcm")
    assert historical.variant is None and historical.cell_key == "lif@hcm"
    ite = CertificationCell.from_mode_policy(
        DeploymentPlan.resolve(_run(lif_window_firing=ITE)).mode_policy(), backend="hcm")
    assert ite.variant == ITE and ite.cell_key != historical.cell_key
