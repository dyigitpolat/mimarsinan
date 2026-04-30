"""Optional Loihi-target spike-parity step.

Added to the pipeline when ``enable_loihi_simulation`` is true in
``deployment_parameters``.  Runs only for LIF mode (see
``LavaLoihiRunner`` for the rationale — TTFS does not map onto Loihi's
LIF dynamics).

Runs *after* :class:`SimulationStep` so users still get the nevresim
cycle-accurate reference; this step then checks one deterministic sample
with HCM-vs-Lava spike-count parity for every neural segment.
"""

from mimarsinan.chip_simulation.lava_loihi_runner import LavaLoihiRunner
from mimarsinan.chip_simulation.spike_recorder import compare_records, format_first_diff
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory, shutdown_data_loader
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow
from mimarsinan.pipelining.pipeline_step import PipelineStep


class LoihiSimulationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "hard_core_mapping"]
        promises = []
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.metric = None

    def validate(self):
        if self.metric is not None:
            return self.metric
        return self.pipeline.get_target_metric()

    def _load_parity_sample(self):
        sample_index = int(self.pipeline.config.get("loihi_parity_sample_index", 0))
        if sample_index < 0:
            raise ValueError("loihi_parity_sample_index must be >= 0")

        factory = DataLoaderFactory(
            self.pipeline.data_provider_factory,
            num_workers=int(self.pipeline.config.get("num_workers", 4)),
        )
        provider = factory.create_data_provider()
        loader = factory.create_test_loader(provider.get_test_batch_size(), provider)
        seen = 0
        try:
            for xs, _ys in loader:
                batch_size = int(xs.shape[0])
                if seen + batch_size <= sample_index:
                    seen += batch_size
                    continue
                local_idx = sample_index - seen
                return xs[local_idx:local_idx + 1]
        finally:
            shutdown_data_loader(loader)

        raise IndexError(
            f"loihi_parity_sample_index={sample_index} is outside the test set"
        )

    def process(self):
        # Access ``model`` explicitly to satisfy the step contract and to make
        # the dependency clear: Loihi parity is a post-HCM check of the mapped
        # model, not an independent accuracy evaluation.
        self.get_entry("model")
        hard_core_mapping = self.get_entry("hard_core_mapping")
        simulation_length = int(self.pipeline.config["simulation_steps"])
        spiking_mode = self.pipeline.config.get("spiking_mode", "lif")
        if spiking_mode != "lif":
            raise ValueError(
                f"LoihiSimulationStep requires spiking_mode='lif'; got {spiking_mode!r}"
            )

        sample = self._load_parity_sample()
        device = self.pipeline.config["device"]

        hcm = SpikingHybridCoreFlow(
            self.pipeline.config["input_shape"],
            hard_core_mapping,
            simulation_length,
            None,
            self.pipeline.config["firing_mode"],
            self.pipeline.config["spike_generation_mode"],
            self.pipeline.config["thresholding_mode"],
            spiking_mode=spiking_mode,
        ).to(device).eval()

        import torch
        with torch.no_grad():
            _, ref = hcm.forward_with_recording(
                sample.to(device), sample_index=int(
                    self.pipeline.config.get("loihi_parity_sample_index", 0)
                ),
            )

        runner = LavaLoihiRunner(
            pipeline=None,
            mapping=hard_core_mapping,
            simulation_length=simulation_length,
            thresholding_mode=self.pipeline.config.get("thresholding_mode", "<"),
        )
        actual = runner.run_segments_from_reference(ref)
        diffs = compare_records(ref, actual)
        if diffs:
            raise AssertionError(format_first_diff(diffs))

        checked_segments = len(ref.segments)
        checked_cores = sum(len(seg.cores) for seg in ref.segments.values())
        self.metric = self.pipeline.get_target_metric()
        self.pipeline.reporter.report("Loihi Spike Parity", 1.0)
        print(
            "Loihi spike parity: PASS "
            f"({checked_segments} neural segment(s), {checked_cores} core(s), "
            f"sample_index={ref.sample_index})"
        )
