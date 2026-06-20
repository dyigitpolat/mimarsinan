"""Optional Loihi-target spike-parity step.

Added to the pipeline when ``enable_loihi_simulation`` is true in
``deployment_parameters``.  Runs only for LIF mode (see
``LavaLoihiRunner`` for the rationale — TTFS does not map onto Loihi's
LIF dynamics).

Runs *after* :class:`SimulationStep` so users still get the nevresim
cycle-accurate reference; this step then checks one deterministic sample
with HCM-vs-Lava spike-count parity for every neural segment.
"""

from mimarsinan.chip_simulation.lava_loihi import LavaLoihiRunner
from mimarsinan.data_handling.test_sample_loader import load_test_sample_by_index
from mimarsinan.pipelining.core.engine.pipeline_helpers import require_spiking_mode_supported
from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.pipelining.core.simulation_factory import (
    assert_spike_parity_or_raise,
    build_neural_behavior_config,
    record_hcm_reference,
)


class LoihiSimulationStep(PipelineStep):
    REQUIRES = ("model", "hard_core_mapping")

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

        self.metric = None

    def validate(self):
        if self.metric is not None:
            return self.metric
        return self.pipeline.get_target_metric()

    def process(self):
        self.get_entry("model")
        hard_core_mapping = self.get_entry("hard_core_mapping")
        simulation_length = int(self.pipeline.config["simulation_steps"])
        require_spiking_mode_supported(
            self.pipeline, "LoihiSimulationStep", backend="loihi",
        )

        sample_index = int(self.pipeline.config.get("loihi_parity_sample_index", 0))
        sample = load_test_sample_by_index(
            self.pipeline.data_provider_factory,
            sample_index,
            num_workers=int(self.pipeline.config.get("num_workers", 4)),
        )
        device = self.pipeline.config["device"]

        _flow, ref = record_hcm_reference(
            self.pipeline,
            hard_core_mapping,
            sample,
            sample_index=sample_index,
            device=device,
        )

        behavior = build_neural_behavior_config(self.pipeline)
        runner = LavaLoihiRunner(
            mapping=hard_core_mapping,
            simulation_length=simulation_length,
            behavior=behavior,
        )
        actual = runner.run_segments_from_reference(ref)
        assert_spike_parity_or_raise(ref, actual)

        checked_segments = len(ref.segments)
        checked_cores = sum(len(seg.cores) for seg in ref.segments.values())
        self.metric = self.pipeline.get_target_metric()
        self.pipeline.reporter.report("Loihi Spike Parity", 1.0)
        print(
            "Loihi spike parity: PASS "
            f"({checked_segments} neural segment(s), {checked_cores} core(s), "
            f"sample_index={ref.sample_index})"
        )
