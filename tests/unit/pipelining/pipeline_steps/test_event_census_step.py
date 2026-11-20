"""[U-S S-2] The opt-in event census step: declared by ``event_census_samples``, absent otherwise.

A run that declares ``event_census_samples = N`` gains one step, "Event Census", after the simulation tail and before
the Deployment Record: it runs the first N VALIDATION samples (never the test split) through the deployed program's
Hard Core Mapping executor one sample at a time, takes each sample's census (``census_from_run_record``) and persists
the per-sample counts and their mean per inference for the record step, which then requires the entry. A run that
does not declare the key resolves exactly as before: no step, no required entry, no default in the resolved config
(so every record stays byte for byte). A semantics whose executor records no per-cycle spikes refuses by name at
assembly."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline, TinyDataProvider, default_config
from mimarsinan.chip_simulation.synaptic_events import EventCensusRefusal, census_from_run_record
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.defaults import CONFIG_KEYS_SET, get_default_deployment_parameters
from mimarsinan.config_schema.registry import NO_DEFAULT, REGISTRY
from mimarsinan.config_schema.registry.decision_classes import COST_ONLY
from mimarsinan.data_handling.sample_loader import load_validation_samples_by_index
from mimarsinan.mapping.crossbar_utilization import CrossbarUtilizationReport
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import get_pipeline_step_specs
from mimarsinan.pipelining.core.simulation_factory import build_spiking_hybrid_flow
from mimarsinan.pipelining.pipeline_steps import DeploymentRecordStep, EventCensusStep

from unit.mapping.residency.executed import vehicles as V

KEY = "event_census_samples"


class _SplitProvider(TinyDataProvider):
    """Validation and test splits that differ; the test split refuses every read."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._val = torch.utils.data.TensorDataset(
            torch.rand(6, V.IN, generator=torch.Generator().manual_seed(3)), torch.zeros(6, dtype=torch.long))

    def _get_validation_dataset(self):
        return self._val

    def _get_test_dataset(self):
        raise AssertionError("the event census read the test split")

    def get_validation_batch_size(self):
        return 4


class _SplitFactory:
    def __init__(self):
        self._provider = None

    def create(self):
        if self._provider is None:
            self._provider = _SplitProvider(input_shape=(V.IN,), num_classes=4, size=6)
        return self._provider


def _config(**overrides):
    config = default_config()
    config.update(input_shape=(V.IN,), num_classes=4, spiking_mode="lif", cycle_accurate_lif_forward=True,
                  thresholding_mode="<=", simulation_steps=V.T, spike_generation_mode="Uniform")
    config.update(overrides)
    return config


def _census_pipeline(tmp_path, samples: int = 3):
    pipeline = MockPipeline(config=_config(**{KEY: samples}), working_directory=str(tmp_path / "run"),
                            data_provider_factory=_SplitFactory())
    os.makedirs(pipeline.working_directory, exist_ok=True)
    mapping = V.build(V.ir_of(V.chain_repr(seed=0)), 2)
    pipeline.seed("model", nn.Module())
    pipeline.seed("hard_core_mapping", mapping)
    return pipeline, mapping


class TestTheDeclaration:
    def test_the_key_is_registered_without_a_default_and_costs_only(self):
        entry = REGISTRY[KEY]
        assert entry.default is NO_DEFAULT
        assert entry.type.value == "int" and entry.bounds == (0, None)
        assert entry.decision_class == COST_ONLY
        assert KEY in CONFIG_KEYS_SET and KEY not in get_default_deployment_parameters()

    def test_the_plan_reads_the_declaration_and_absence_is_zero(self):
        fwi = {"lif_window_firing": "fire_while_integrating"}  # [U-LIF2 W13] an external backend executes it
        assert DeploymentPlan.resolve(_config(**fwi)).event_census_samples == 0
        assert DeploymentPlan.resolve(_config(**{KEY: 64})).event_census_samples == 64
        # integrate then emit, which no external backend executes: the executor of record's census (SANA-FE's count)
        assert DeploymentPlan.resolve(_config()).event_census_samples == int(_config().get("sanafe_sample_count") or 1)


class TestTheStepJoinsOnlyWhenDeclared:
    def test_absent_declaration_resolves_the_steps_unchanged(self):
        assert EventCensusStep not in {cls for _, cls in get_pipeline_step_specs({})}

    def test_a_declared_census_runs_after_the_simulation_tail_before_the_record(self):
        names = [name for name, _ in get_pipeline_step_specs({KEY: 8})]
        assert names.index("Event Census") == names.index("Deployment Record") - 1

    def test_the_record_step_requires_the_census_only_when_declared(self, tmp_path):
        pipeline, _ = _census_pipeline(tmp_path)
        assert "event_census" in DeploymentRecordStep(pipeline).requires
        pipeline.config.pop(KEY)
        assert "event_census" in DeploymentRecordStep(pipeline).requires  # [U-LIF2 W13] integrate then emit: derived
        pipeline.config["lif_window_firing"] = "fire_while_integrating"
        assert "event_census" not in DeploymentRecordStep(pipeline).requires


class TestValidationInputsOnly:
    def test_the_loader_reads_the_validation_split_by_index(self):
        provider = _SplitFactory()
        samples = load_validation_samples_by_index(provider, [0, 2], num_workers=0)
        expected = provider.create()._get_validation_dataset().tensors[0]
        assert torch.equal(samples[0], expected[0:1]) and torch.equal(samples[1], expected[2:3])

    def test_an_index_past_the_validation_set_refuses(self):
        with pytest.raises(IndexError, match="validation"):
            load_validation_samples_by_index(_SplitFactory(), [9], num_workers=0)


class TestTheCensusStep:
    def test_it_persists_each_validation_samples_census_and_their_mean(self, tmp_path):
        pipeline, mapping = _census_pipeline(tmp_path, samples=3)
        step = EventCensusStep(pipeline)
        pipeline.prepare_step(step)
        step.run()
        census = pipeline.cache[step.name + ".event_census"]
        xs = pipeline.data_provider_factory.create()._get_validation_dataset().tensors[0][:3]
        flow = build_spiking_hybrid_flow(pipeline, mapping).eval()
        with torch.no_grad():
            _, records = flow.forward_with_recordings(xs)
        expected = [census_from_run_record(r, mapping) for r in records]
        assert census["per_sample"] == expected and min(expected) > 0
        assert census["mean_per_inference"] == pytest.approx(float(np.mean(expected)))
        assert census["samples"] == 3 and census["split"] == "validation"
        assert census["simulation_steps"] == V.T
        cells = CrossbarUtilizationReport.from_hybrid_mapping(mapping).cells_used
        assert census["cells_used"] == cells
        assert census["activity"] == pytest.approx(float(np.mean(expected)) / (cells * V.T))

    @pytest.mark.parametrize("overrides", [
        {"spiking_mode": "ttfs", "firing_mode": "TTFS", "spike_generation_mode": "TTFS"},
        {"core_semantics": "mvm"},
    ], ids=["analytical_ttfs", "value_domain"])
    def test_a_semantics_without_a_per_cycle_record_refuses_by_name_at_assembly(self, tmp_path, overrides):
        pipeline = MockPipeline(config=_config(**{KEY: 2}, **overrides), working_directory=str(tmp_path))
        with pytest.raises(EventCensusRefusal, match="event census") as refused:
            EventCensusStep(pipeline)
        assert isinstance(refused.value, ToolchainRefusal)
