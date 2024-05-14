from mimarsinan.mapping.chip_latency import *

import torch.nn as nn
import torch

class StableSpikingCoreFlow(nn.Module):
    def __init__(self, input_shape, core_mapping, simulation_length, preprocessor):
        super(StableSpikingCoreFlow, self).__init__()
        self.input_shape = input_shape

        self.preprocessor = preprocessor

        self.core_mapping = core_mapping
        self.cores = core_mapping.cores
        self.output_sources = core_mapping.output_sources
        self.core_params = nn.ParameterList(
            [nn.Parameter(torch.tensor(
                self.cores[core].core_matrix.transpose(), dtype=torch.float32, requires_grad=False))\
                    for core in range(len(self.cores))]
        )

        self.thresholds = nn.ParameterList(
            [nn.Parameter(torch.tensor(
                self.cores[core].threshold, dtype=torch.float32), requires_grad=False)\
                    for core in range(len(self.cores))])

        self.simulation_length = simulation_length
        self.cycles = ChipLatency(core_mapping).calculate() + simulation_length

        # Stats
        self.core_sums = [None] * len(self.cores)

        self.spike_rates = []
        self.total_spikes = 0
    
    def get_signal(
            self, spike_train_cache, batch_size, device, core, neuron, is_input, is_off, is_always_on, cycle):

        if is_input:
            return spike_train_cache[-1][cycle][:, neuron]
        
        if is_off:
            return torch.zeros(batch_size, device=device)
        
        if is_always_on:
            return torch.ones(batch_size, device=device)
        
        return spike_train_cache[core][cycle][:, neuron]
    
    def get_signal_tensor(self, spike_train_cache, batch_size, device, sources, cycle):
        signal_tensor = torch.empty(batch_size, len(sources), device=device)

        for idx, spike_source in enumerate(sources):
            signal_tensor[:, idx] = self.get_signal(
                spike_train_cache, 
                batch_size,
                device,
                spike_source.core_, 
                spike_source.neuron_, 
                spike_source.is_input_, 
                spike_source.is_off_, 
                spike_source.is_always_on_,
                cycle)
        
        return signal_tensor
      
    def to_spikes(self, tensor):
        return (torch.rand(tensor.shape, device=tensor.device) < tensor).float()
    
    def generate_input_spike_train(self, x):
        shape = [self.simulation_length] + [*x.shape]
        print(shape)
        spikes = torch.zeros(shape, device=x.device)
        for t in range(self.simulation_length):
            # spikes[t] = self.to_spikes(x)
            spikes[t] = x
        
        return spikes
    
    def _process_core(self, batch_size, spike_train_cache, core, core_idx, device):
        
        membrane_potentials = \
            torch.zeros(batch_size, core.get_output_count(), device=device)
        
        # out_spikes = \
        #     torch.zeros(self.simulation_length, batch_size, core.get_output_count(), device=device)
            
        core_input_signals = []
        for cycle in range(self.simulation_length):
            core_input_signals.append(
                self.get_signal_tensor(
                    spike_train_cache, 
                    batch_size,
                    device,
                    core.axon_sources, 
                    cycle))
        
        avg_values = torch.zeros(batch_size, core.get_output_count(), device=device)
        for cycle in range(self.simulation_length):
            val = torch.matmul(
                    self.core_params[core_idx], 
                    core_input_signals[cycle].T).T
            #integrate
            membrane_potentials += val
                
            # average
            avg_values += val / self.simulation_length

            # novena reset
            # membrane_potentials[membrane_potentials > self.thresholds[core_idx]] = 0.0

            # # normal reset
            membrane_potentials[membrane_potentials > self.thresholds[core_idx]] -= self.thresholds[core_idx]

        ideal_out_spikes = \
            torch.zeros(self.simulation_length, batch_size, core.get_output_count(), device=device)
        
        # calculate idealized spikes for core
        ideal_membrane_potentials = \
            torch.zeros(batch_size, core.get_output_count(), device=device)
        
        for cycle in range(self.simulation_length):
            ideal_membrane_potentials += avg_values
        
            # fire
            ideal_out_spikes[cycle] = (ideal_membrane_potentials > self.thresholds[core_idx]).float()

            # # normal reset
            ideal_membrane_potentials[ideal_membrane_potentials > self.thresholds[core_idx]] -= self.thresholds[core_idx]

            # novena reset
            # ideal_membrane_potentials[ideal_membrane_potentials > self.thresholds[core_idx]] = 0

        spike_train_cache[core_idx] = ideal_out_spikes

    def get_core_spike_rates(self):
        return self.spike_rates
    
    def get_total_spikes(self):
        return self.total_spikes

    def forward(self, x):
        x = self.preprocessor(x)
        x = x.view(x.shape[0], -1)
        batch_size = x.shape[0]

        spike_train_cache = {}

        # init input spike train cache item
        spike_train_cache[-1] = self.generate_input_spike_train(x)

        chip_latency = ChipLatency(self.core_mapping).calculate()

        # traverse cores in latency order so input is always available.
        for t in range(chip_latency):
            for core_idx, core in enumerate(self.cores):
                if core.latency == t:
                    self._process_core(batch_size, spike_train_cache, core, core_idx, x.device)


        output_signals = torch.zeros(x.shape[0], len(self.output_sources), device=x.device)
        for cycle in range(self.simulation_length):
            output_signals += self.get_signal_tensor(spike_train_cache, batch_size, x.device, self.output_sources, cycle)
        

        self.spike_rates = []
        for core_idx in range(len(self.cores)):
            rate = torch.sum(
                spike_train_cache[core_idx]).item() / (self.simulation_length * batch_size * self.cores[core_idx].get_output_count())
            self.spike_rates.append(rate)

        self.total_spikes = torch.sum(output_signals).item()
        print("total spikes: ", self.total_spikes)

        return output_signals