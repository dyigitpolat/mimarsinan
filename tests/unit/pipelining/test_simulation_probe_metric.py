"""Sim-role respec reaches the pipeline gate: nevresim is a decision-parity
PROBE (N=25 subsample), so its small-N accuracy must be reported, never set as
the pipeline metric — the accuracy verdict is the SCM identity read.

The measured failure: two healthy runs (SCM 0.9576 / 0.94, every parity
contract 1.0000) were killed by the retention gate firing on the probe's
25-sample binomial read (0.76 = 19/25; the column scatters 0.76-1.00 around
the SCM verdict across the softwall wave).
"""

from __future__ import annotations

from mimarsinan.pipelining.pipeline_steps.verification.loihi_simulation_step import (
    LoihiSimulationStep,
)
from mimarsinan.pipelining.pipeline_steps.verification.sanafe_simulation_step import (
    SanafeSimulationStep,
)
from mimarsinan.pipelining.pipeline_steps.verification.simulation_step import (
    SimulationStep,
)


class _Reporter:
    def __init__(self):
        self.reports = []

    def report(self, name, value):
        self.reports.append((name, value))


class _Pipeline:
    def __init__(self):
        self.config = {}
        self.reporter = _Reporter()

    def get_target_metric(self):
        return 0.9576


def test_probe_accuracy_never_becomes_the_pipeline_metric():
    step = SimulationStep(_Pipeline())
    step.probe_accuracy = 0.76  # 19/25 on the fixed seed-0 subsample
    assert step.validate() == 0.9576, (
        "the nevresim probe read must not drive the retention gate"
    )


def test_validate_preserves_the_metric_before_any_probe_ran():
    step = SimulationStep(_Pipeline())
    assert step.validate() == 0.9576


def test_probe_accuracy_is_reported_separately():
    pipeline = _Pipeline()
    step = SimulationStep(pipeline)
    step._report_probe(0.76)
    assert ("nevresim_probe_accuracy", 0.76) in pipeline.reporter.reports


def test_all_simulator_steps_share_the_metric_neutral_contract():
    # Loihi and SANA-FE already preserve the incoming metric (their verdicts
    # are parity assertions); nevresim now follows the same contract.
    for step_cls in (SimulationStep, LoihiSimulationStep, SanafeSimulationStep):
        step = step_cls.__new__(step_cls)
        step.pipeline = _Pipeline()
        step.metric = None
        step.probe_accuracy = None
        assert step.validate() == 0.9576, step_cls.__name__
