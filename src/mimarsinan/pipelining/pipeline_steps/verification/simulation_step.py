from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep

from mimarsinan.chip_simulation.simulation_runner import SimulationRunner


class SimulationStep(PipelineStep):
    REQUIRES = ("hard_core_mapping",)

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

        self.probe_accuracy = None

    def validate(self):
        # nevresim is a decision-parity PROBE on a small subsample; its
        # binomial-noise accuracy is reported, never the pipeline metric —
        # the accuracy verdict is the SCM identity read (retention-gated
        # there). Loihi and SANA-FE follow the same metric-neutral contract.
        return self.pipeline.get_target_metric()

    def _report_probe(self, accuracy) -> None:
        print("Simulation accuracy: ", accuracy)
        self.pipeline.reporter.report("nevresim_probe_accuracy", float(accuracy))

    def process(self):
        runner = SimulationRunner(
            self.pipeline,
            self.get_entry('hard_core_mapping'),
            int(self.pipeline.config["simulation_steps"]),
        )

        self.probe_accuracy = runner.run()
        self._report_probe(self.probe_accuracy)
