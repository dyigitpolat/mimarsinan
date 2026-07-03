"""Nevresim SimulationRunner orchestration."""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from mimarsinan.chip_simulation.test_subsample import compute_test_subsample_indices
from mimarsinan.chip_simulation.nevresim.connectivity import resolve_nevresim_connectivity_mode
from mimarsinan.chip_simulation.spiking_semantics import requires_ttfs_firing
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory, shutdown_data_loader
from mimarsinan.chip_simulation.simulation_runner.flat import SimulationFlatMixin
from mimarsinan.chip_simulation.simulation_runner.hybrid import SimulationHybridMixin
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan


class SimulationRunner(SimulationFlatMixin, SimulationHybridMixin):
    def __init__(self, pipeline, mapping, simulation_length, preprocessor=None):
        self._preprocessor = preprocessor if preprocessor is not None else nn.Identity()
        plan = DeploymentPlan.of(pipeline)
        self.spike_generation_mode = pipeline.config["spike_generation_mode"]
        self.firing_mode = pipeline.config["firing_mode"]
        self.thresholding_mode = pipeline.config.get("thresholding_mode", "<=")
        self.spiking_mode = plan.spiking_mode
        self.nevresim_connectivity_mode = resolve_nevresim_connectivity_mode(pipeline.config)

        # Deliberately not routed through DeploymentPlan: this runner keeps the legacy weight_quantization default of True, which diverges from the plan's False default for a config that omits the key.
        wt_q = pipeline.config.get("weight_quantization", True)
        self.weight_type = int if wt_q else float

        is_ttfs = requires_ttfs_firing(self.spiking_mode)
        self.threshold_type = float if is_ttfs else self.weight_type

        self.input_size = pipeline.config["input_size"]
        self.num_classes = pipeline.config["num_classes"]
        self.working_directory = pipeline.working_directory
        self.mapping = mapping
        self.simulation_length = simulation_length

        self.test_input = []
        self.test_targets = []

        data_loader_factory = DataLoaderFactory.for_pipeline(pipeline)
        data_provider = data_loader_factory.create_data_provider()
        test_loader = data_loader_factory.create_test_loader(
            data_provider.get_test_batch_size(), data_provider,
        )

        try:
            for xs, ys in test_loader:
                self.test_input.extend(self._preprocessor(xs).detach().cpu())
                self.test_targets.extend(ys.detach().cpu() if hasattr(ys, "detach") else ys)
            self.test_data = [*zip(np.stack(self.test_input), np.stack(self.test_targets))]
        finally:
            shutdown_data_loader(test_loader)

        max_samples = int(pipeline.config.get("max_simulation_samples", 0) or 0)
        total_samples = len(self.test_data)
        if max_samples and 0 < max_samples < total_samples:
            indices = compute_test_subsample_indices(
                total_samples=total_samples,
                seed=int(pipeline.config.get("seed", 0)),
                max_samples=max_samples,
            )
            self.test_data = [self.test_data[i] for i in indices]
            self.test_input = [self.test_input[i] for i in indices]
            self.test_targets = [self.test_targets[i] for i in indices]
            print(f"  [SimulationRunner] Subsampled {max_samples} / {total_samples} test samples")

    def _evaluate_chip_output(self, predictions):
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for y, p in zip(self.test_targets, predictions):
            confusion_matrix[y.item()][p] += 1
        print("Confusion matrix:")
        print(confusion_matrix)

        total = 0
        correct = 0
        for y, p in zip(self.test_targets, predictions):
            correct += int(y.item() == p)
            total += 1
        return float(correct) / total

    def run(self) -> float:
        if isinstance(self.mapping, HybridHardCoreMapping):
            segments = self.mapping.get_neural_segments()
            compute_ops = self.mapping.get_compute_ops()
            if len(segments) == 1 and len(compute_ops) == 0:
                return self._run_flat_mapping(segments[0])
            print(
                f"  Hybrid mapping: {len(segments)} neural segments, "
                f"{len(compute_ops)} compute ops"
            )
            return self._run_hybrid(self.mapping)
        return self._run_flat_mapping(self.mapping)
