"""[§17] rung-2 derivation: when the twin certificate is armed, the SCM step's
identity accuracy is DERIVED (identity ≡ packed counts ⇒ identical decisions),
not re-measured — the deployed read lands at Hard Core Mapping."""

from __future__ import annotations

import pytest

from conftest import MockPipeline
from mimarsinan.pipelining.core.spike_count_gate import certificate_gate_armed
from mimarsinan.pipelining.core.steps.pipeline_step import METRIC_CARRIED
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
    SoftCoreMappingStep,
)


def test_gate_armed_predicate_lif_counts_on():
    p = MockPipeline(config={
        "spiking_mode": "lif", "spike_count_parity_samples": 2,
    })
    assert certificate_gate_armed(p) is True


def test_gate_armed_predicate_off_for_zero_samples_and_analytic_modes():
    assert certificate_gate_armed(MockPipeline(config={
        "spiking_mode": "lif", "spike_count_parity_samples": 0,
    })) is False
    assert certificate_gate_armed(MockPipeline(config={
        "spiking_mode": "ttfs", "spike_count_parity_samples": 2,
    })) is False


def test_derived_metric_is_carried_and_loud():
    pipeline = MockPipeline(config={
        "spiking_mode": "lif", "spike_count_parity_samples": 2,
    })
    pipeline.get_target_metric = lambda: 0.86
    step = SoftCoreMappingStep(pipeline)
    step._identity_metric_derived = True
    step._soft_core_spiking_metric = None

    assert step.validate() == pytest.approx(0.86)
    assert step.pipeline_metric() == pytest.approx(0.86)
    assert step.validate_metric_kind() == METRIC_CARRIED


def test_legacy_path_unchanged_when_metric_ran():
    step = SoftCoreMappingStep(MockPipeline())
    step._soft_core_spiking_metric = 0.8125
    assert step.validate() == pytest.approx(0.8125)
    assert step.validate_metric_kind() != METRIC_CARRIED
