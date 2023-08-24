from mimarsinan.mapping.chip_latency import ChipLatency
from mimarsinan.chip_simulation.nevresim_driver import NevresimDriver
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

import numpy as np

class SimulationRunner:
    def __init__(self, pipeline, mapping, simulation_length):
        self.input_size = pipeline.config["input_size"]
        self.num_classes = pipeline.config["num_classes"]

        self.working_directory = pipeline.working_directory

        self.test_input = []
        self.test_targets = []

        data_loader_factory = DataLoaderFactory(pipeline.data_provider_factory)
        data_provider = data_loader_factory.create_data_provider()

        test_loader = data_loader_factory.create_test_loader(
            data_provider.get_test_batch_size(), data_provider)
        
        for xs, ys in test_loader:
            self.test_input.extend(xs)
            self.test_targets.extend(ys)
            
        self.test_data = \
            [*zip(np.stack(self.test_input), np.stack(self.test_targets))]

        self.mapping = mapping
        self.simulation_length = simulation_length

    def _evaluate_chip_output(self, predictions):
        confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=int)
        
        for (y, p) in zip(self.test_targets, predictions):
            confusion_matrix[y.item()][p] += 1
        print("Confusion matrix:")
        print(confusion_matrix)
        
        total = 0
        correct = 0
        for (y, p) in zip(self.test_targets, predictions):
            correct += int(y.item() == p)
            total += 1

        return float(correct) / total

    def run(self):
        delay = ChipLatency(self.mapping).calculate()
        print(f"  delay: {delay}")

        simulation_driver = NevresimDriver(
            self.input_size,
            self.mapping,
            self.working_directory,
            int
        )

        simulation_steps = delay + int(self.simulation_length)
        
        predictions = simulation_driver.predict_spiking(
            self.test_data,
            simulation_steps)

        print("Evaluating simulator output...")
        accuracy = self._evaluate_chip_output(predictions)
        return accuracy