from mimarsinan.mapping.chip_latency import ChipLatency
from mimarsinan.chip_simulation.nevresim_driver import NevresimDriver

import numpy as np

class SimulationRunner:
    def __init__(self, pipeline, mapping, threshold_scale, simulation_length):
        self.model = pipeline.model
        self.input_size = pipeline.input_size
        self.num_classes = pipeline.num_classes
        self.working_directory = pipeline.working_directory

        self.test_input = []
        self.test_targets = []
        for xs, ys in pipeline.test_dataloader:
            self.test_input.extend(xs)
            self.test_targets.extend(ys)
        self.test_data = zip(np.array(self.test_input), np.array(self.test_targets))

        self.mapping = mapping
        self.threshold_scale = threshold_scale
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

        simulation_steps = delay + int(
            self.threshold_scale * self.simulation_length)
        
        predictions = simulation_driver.predict_spiking(
            self.test_data,
            simulation_steps)

        print("Evaluating simulator output...")
        accuracy = self._evaluate_chip_output(predictions)
        print("SNN accuracy on MNIST is:", accuracy*100, "%")
        return accuracy