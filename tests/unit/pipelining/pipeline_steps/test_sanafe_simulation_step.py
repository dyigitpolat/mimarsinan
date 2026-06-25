"""SanafeSimulationStep — pipeline contract + parity gate + rich-stats publishing.

The step:
  * loads ``sanafe_sample_count`` deterministic samples,
  * runs the SANA-FE backend on each,
  * for each sample, builds an HCM reference ``RunRecord`` and feeds
    ``compare_records(ref, actual_subset)``; fails the pipeline with
    ``format_first_diff`` on any divergence,
  * persists the per-sample ``SanafeRunRecord``s as a ``SanafeStepReport``
    under cache key ``sanafe_simulation_results``,
  * reports headline metrics (parity, total energy, max sim time,
    spike & packet totals).

All SANA-FE machinery is mocked here.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import MockPipeline, MockDataProviderFactory, make_tiny_supermodel
from mimarsinan.chip_simulation.sanafe.records import (
    SanafeEnergyBreakdown,
    SanafeRunRecord,
)
from mimarsinan.pipelining.pipeline_steps.verification.sanafe_simulation_step import (
    SanafeSimulationStep,
)


# ---------------------------------------------------------------------------
# Reporter + DataLoaderFactory stubs (same shape as the Loihi step test)
# ---------------------------------------------------------------------------


class _RecordingReporter:
    def __init__(self):
        self.events: list[tuple[str, object]] = []

    def report(self, name, value):
        self.events.append((name, value))


class _FakeDataLoaderFactory:
    def __init__(self, data_provider_factory, num_workers=4):
        self.data_provider_factory = data_provider_factory
        self.num_workers = num_workers

    def create_data_provider(self):
        return self.data_provider_factory.create()

    def create_test_loader(self, batch_size, data_provider):
        dataset = data_provider._get_test_dataset()
        xs = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)
        ys = torch.tensor([int(dataset[i][1]) for i in range(len(dataset))])
        return [(xs, ys)]


# ---------------------------------------------------------------------------
# Fake records
# ---------------------------------------------------------------------------


def _fake_hcm_record(sample_index=0):
    return SimpleNamespace(
        sample_index=sample_index,
        T=4,
        segments={0: SimpleNamespace(cores=[SimpleNamespace()])},
    )


def _fake_sanafe_record(sample_index=0, energy_total=2.0,
                        sim_time_s=1.0e-6, total_spikes=10, total_packets=4):
    """Object that quacks like a SanafeRunRecord for the step's aggregation."""
    return SanafeRunRecord(
        arch_preset="loihi",
        arch_name="t",
        sample_index=sample_index,
        T=4,
        segments={},
        compute_outputs={},
        aggregate_energy=SanafeEnergyBreakdown(
            synapse_j=0.4 * energy_total, dendrite_j=0.1 * energy_total,
            soma_j=0.3 * energy_total, network_j=0.2 * energy_total,
            total_j=energy_total,
        ),
        aggregate_sim_time_s=sim_time_s,
        total_spikes=total_spikes,
        total_packets=total_packets,
    )


# ---------------------------------------------------------------------------
# Step setup
# ---------------------------------------------------------------------------


def _prepare_step(monkeypatch, *,
                  diffs=None, sample_count=1, arch_preset="loihi"):
    import mimarsinan.pipelining.pipeline_steps.verification.sanafe_simulation_step as step_mod

    calls = {
        "hcm_built": 0,
        "runner_inits": [],
        "runner_run_samples": [],
    }

    class FakeHCM:
        def __init__(self, *args, **kwargs):
            calls["hcm_built"] += 1
            self.args = args; self.kwargs = kwargs

        def to(self, device):  return self
        def eval(self):        return self

        def forward_with_recording(self, sample, sample_index=0):
            return torch.zeros((1, 1)), _fake_hcm_record(sample_index=sample_index)

    class FakeRunner:
        instance_count = 0

        def __init__(self, mapping, simulation_length, **kw):
            FakeRunner.instance_count += 1
            calls["runner_inits"].append(
                {"mapping": mapping, "simulation_length": simulation_length, **kw}
            )

        def run(self, sample, sample_index):
            calls["runner_run_samples"].append(int(sample_index))
            return _fake_sanafe_record(sample_index=sample_index)

    def _fake_load_samples(factory, indices, num_workers=4):
        loader = _FakeDataLoaderFactory(factory, num_workers=num_workers)
        provider = loader.create_data_provider()
        dataset = provider._get_test_dataset()
        return [dataset[int(i)][0].unsqueeze(0) for i in indices]

    def _fake_record_hcm(pipeline, mapping, sample, sample_index=0, device=None):
        calls["hcm_built"] += 1
        return FakeHCM(), _fake_hcm_record(sample_index=sample_index)

    monkeypatch.setattr(step_mod, "load_test_samples_by_index", _fake_load_samples)
    monkeypatch.setattr(step_mod, "record_hcm_reference", _fake_record_hcm)
    monkeypatch.setattr(step_mod, "SanafeRunner", FakeRunner)

    def _parity_or_raise(ref, actual):
        if diffs:
            raise AssertionError("formatted sanafe diff")

    monkeypatch.setattr(step_mod, "assert_spike_parity_or_raise", _parity_or_raise)

    pipeline = MockPipeline(
        data_provider_factory=MockDataProviderFactory(input_shape=(1, 8, 8), size=4),
    )
    pipeline.config["spiking_mode"] = "lif"
    pipeline.config["simulation_steps"] = 4
    pipeline.config["thresholding_mode"] = "<"
    pipeline.config["sanafe_sample_count"] = sample_count
    pipeline.config["sanafe_arch_preset"] = arch_preset
    pipeline.config["sanafe_custom_arch_path"] = None
    pipeline.reporter = _RecordingReporter()
    pipeline.set_target_metric(0.875)
    pipeline.seed("model", make_tiny_supermodel(), step_name="Model Configuration")
    pipeline.seed("hard_core_mapping", object(), step_name="Hard Core Mapping")

    step = SanafeSimulationStep(pipeline)
    step.name = "SANA-FE Simulation"
    pipeline.prepare_step(step)
    return step, pipeline, calls


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_step_declares_requires_promises_updates_clears(monkeypatch):
    step, _, _ = _prepare_step(monkeypatch)
    assert sorted(step.requires) == ["hard_core_mapping", "model"]
    assert step.promises == ["sanafe_simulation_results"]
    assert step.updates == []
    assert step.clears == []


def test_step_validate_returns_target_metric_when_no_local_metric(monkeypatch):
    step, _, _ = _prepare_step(monkeypatch)
    step.run()
    assert step.validate() == 0.875


def test_step_ttfs_uses_ttfs_reference_path(monkeypatch):
    import mimarsinan.pipelining.pipeline_steps.verification.sanafe_simulation_step as step_mod

    step, pipeline, calls = _prepare_step(monkeypatch)
    ttfs_calls = []

    def _fake_ttfs_ref(pipeline, mapping, sample, sample_index=0):
        ttfs_calls.append(sample_index)
        from mimarsinan.chip_simulation.ttfs.ttfs_recorder import TtfsRunRecord

        return None, TtfsRunRecord(
            sample_index=sample_index, simulation_length=4, spiking_mode="ttfs",
        )

    def _fake_ttfs_subset(self, spiking_mode="ttfs"):
        from mimarsinan.chip_simulation.ttfs.ttfs_recorder import TtfsRunRecord

        return TtfsRunRecord(
            sample_index=self.sample_index,
            simulation_length=self.T,
            spiking_mode=spiking_mode,
        )

    monkeypatch.setattr(step_mod, "record_ttfs_hcm_reference", _fake_ttfs_ref)
    monkeypatch.setattr(
        SanafeRunRecord, "to_ttfs_contract_subset", _fake_ttfs_subset, raising=False,
    )
    monkeypatch.setattr(
        SanafeRunRecord, "to_ttfs_hardware_subset", _fake_ttfs_subset, raising=False,
    )
    pipeline.config["spiking_mode"] = "ttfs_quantized"
    pipeline.config["firing_mode"] = "TTFS"
    pipeline.config["spike_generation_mode"] = "TTFS"
    step.run()
    assert ttfs_calls == [0]


# ---------------------------------------------------------------------------
# Single-sample happy path
# ---------------------------------------------------------------------------


def test_step_runs_one_sample_and_publishes_cache_entry(monkeypatch):
    step, pipeline, calls = _prepare_step(monkeypatch)
    step.run()

    assert calls["runner_run_samples"] == [0]
    key = "SANA-FE Simulation.sanafe_simulation_results"
    assert key in pipeline.cache
    report = pipeline.cache[key]
    assert report.arch_preset == "loihi"
    assert report.sample_indices == [0]
    assert len(report.per_sample) == 1


def test_step_runner_init_threads_thresholding_and_preset(monkeypatch):
    step, _, calls = _prepare_step(monkeypatch, arch_preset="truenorth")
    step.run()
    assert len(calls["runner_inits"]) == 1
    init = calls["runner_inits"][0]
    assert init["contract"].thresholding_mode == "<"
    assert init["arch_preset"] == "truenorth"
    assert init["simulation_length"] == 4


def test_step_reports_rich_metrics_on_success(monkeypatch):
    step, pipeline, _ = _prepare_step(monkeypatch)
    step.run()

    names = [e[0] for e in pipeline.reporter.events]
    assert "SANA-FE Parity" in names
    assert "SANA-FE Total Energy (mJ)" in names
    assert "SANA-FE Max Sim Time (s)" in names
    assert "SANA-FE Total Spikes" in names
    assert "SANA-FE Total NoC Packets" in names

    metrics = dict(pipeline.reporter.events)
    assert metrics["SANA-FE Parity"] == 1.0
    assert metrics["SANA-FE Total Energy (mJ)"] == pytest.approx(2.0 * 1000.0)
    assert metrics["SANA-FE Total Spikes"] == 10
    assert metrics["SANA-FE Total NoC Packets"] == 4


# ---------------------------------------------------------------------------
# Parity gate
# ---------------------------------------------------------------------------


def test_step_fails_with_formatted_record_diff_when_parity_breaks(monkeypatch):
    step, _, _ = _prepare_step(monkeypatch, diffs=["someDiff"])
    with pytest.raises(AssertionError, match="formatted sanafe diff"):
        step.run()


# ---------------------------------------------------------------------------
# Multi-sample
# ---------------------------------------------------------------------------


def test_step_loads_sanafe_sample_count_samples(monkeypatch):
    step, _, calls = _prepare_step(monkeypatch, sample_count=3)
    step.run()
    assert calls["runner_run_samples"] == [0, 1, 2]


def test_step_aggregates_across_samples(monkeypatch):
    step, pipeline, _ = _prepare_step(monkeypatch, sample_count=2)
    step.run()
    report = pipeline.cache["SANA-FE Simulation.sanafe_simulation_results"]
    assert report.sample_indices == [0, 1]
    # 2 samples × 2.0 J each = 4.0 J = 4000 mJ
    assert report.aggregate["total_energy_mj"] == pytest.approx(4000.0)


def test_step_rebuilds_runner_per_sample(monkeypatch):
    """A fresh ``SanafeRunner`` is constructed per sample so per-sample chip
    state doesn't bleed across samples."""
    step, _, calls = _prepare_step(monkeypatch, sample_count=3)
    step.run()
    assert len(calls["runner_inits"]) == 3


# ---------------------------------------------------------------------------
# Measured cost-record emission (cost-emit): the deployment NOW drops a
# measured ``cost_record.json`` alongside the run as a pure additive side
# effect — never crashing the run, never altering the deployment result.
# ---------------------------------------------------------------------------


def _emitted_cost_record(pipeline):
    import os

    from mimarsinan.chip_simulation.cost_extraction import (
        COST_RECORD_FILENAME,
        load_cost_record,
    )

    path = os.path.join(pipeline.working_directory, COST_RECORD_FILENAME)
    assert os.path.exists(path), f"no cost record at {path}"
    return load_cost_record(path)


def test_step_emits_measured_cost_record(monkeypatch):
    step, pipeline, _ = _prepare_step(monkeypatch)
    step.run()

    record = _emitted_cost_record(pipeline)
    # The cell is the (firing × sync) coordinate of the run, measured on SANA-FE.
    assert record.backend == "sanafe"
    assert record.cell_key == "lif@sanafe"
    # The deployed accuracy of record is the pipeline's target metric (R6 / E5).
    assert record.acc_deploy == pytest.approx(0.875)
    # The energy axis mirrors the SANA-FE aggregate (1 sample × 2.0 J = 2000 mJ).
    assert record.mj_per_sample == pytest.approx(2000.0)
    assert record.spikes == 10


def test_cost_emission_does_not_alter_step_result(monkeypatch):
    """Cost emission is a pure additive side-effect: the step's return (the
    target metric) and the published report are byte-identical to a run with
    cost emission disabled."""
    step, pipeline, _ = _prepare_step(monkeypatch)
    step.run()

    assert step.validate() == 0.875
    report = pipeline.cache["SANA-FE Simulation.sanafe_simulation_results"]
    assert report.aggregate["total_energy_mj"] == pytest.approx(2000.0)
    assert report.sample_indices == [0]


def test_cost_extraction_raise_is_swallowed_and_run_succeeds(monkeypatch):
    """A cost-extraction failure is logged and SWALLOWED — it never crashes the
    deployment nor changes its result; no cost file is required to exist."""
    import os

    import mimarsinan.pipelining.pipeline_steps.verification.sanafe_simulation_step as step_mod

    step, pipeline, _ = _prepare_step(monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("cost extraction blew up")

    # Detonate the in-step extractor; the step must still complete normally.
    monkeypatch.setattr(step_mod, "extract_cost_record", _boom)

    step.run()  # must NOT raise

    # The deployment result is unchanged.
    assert step.validate() == 0.875
    key = "SANA-FE Simulation.sanafe_simulation_results"
    assert key in pipeline.cache
    # No cost file was written (extraction raised), and that is tolerated.
    from mimarsinan.chip_simulation.cost_extraction import COST_RECORD_FILENAME

    assert not os.path.exists(
        os.path.join(pipeline.working_directory, COST_RECORD_FILENAME)
    )


def test_cost_save_raise_is_swallowed_and_run_succeeds(monkeypatch):
    """A failure in the WRITE path (save_cost_record) is also swallowed."""
    import mimarsinan.pipelining.pipeline_steps.verification.sanafe_simulation_step as step_mod

    step, pipeline, _ = _prepare_step(monkeypatch)

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(step_mod, "save_cost_record", _boom)

    step.run()  # must NOT raise
    assert step.validate() == 0.875
