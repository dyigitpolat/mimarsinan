from mimarsinan.mapping.chip_latency import *

import torch.nn as nn
import torch

class SpikingCoreFlow(nn.Module):
    def __init__(self, input_shape, core_mapping, simulation_length, preprocessor, firing_mode):
        super(SpikingCoreFlow, self).__init__()
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
        
        self.firing_mode = firing_mode
        assert firing_mode in ["Default", "Novena"]

        self.latency = ChipLatency(core_mapping).calculate()
        self.cycles = self.latency + simulation_length
        self.simulation_length = simulation_length

        # Stats
        self.core_avgs = None
        self.total_spikes = 0

    def set_simulation_length(self, simulation_length):
        self.cycles = self.latency + simulation_length
        self.simulation_length = simulation_length
        
    def update_cores(self):
        for idx, core in enumerate(self.cores):
            core.core_matrix[:,:] = \
                self.core_params[idx].detach().numpy().transpose()
            
    def refresh_thresholds(self):
        self.thresholds = nn.ParameterList(
            [nn.Parameter(torch.tensor(
                self.cores[core].threshold, dtype=torch.float32), requires_grad=False)\
                    for core in range(len(self.cores))])
    
    def get_signal(
            self, x, buffers, core, neuron, is_input, is_off, is_always_on, cycle):

        if is_input:
            return x[:, neuron]
        
        if is_off:
            return torch.zeros_like(x[:, 0])
        
        if is_always_on:
            return torch.ones_like(x[:, 0])
        
        return buffers[core][:, neuron]
    
    def get_signal_tensor(self, x, buffers, sources, cycle):
        signal_tensor = torch.empty(x.shape[0], len(sources), device=x.device)
        for idx, spike_source in enumerate(sources):
            signal_tensor[:, idx] = self.get_signal(
                x, 
                buffers,
                spike_source.core_, 
                spike_source.neuron_, 
                spike_source.is_input_, 
                spike_source.is_off_, 
                spike_source.is_always_on_,
                cycle)
        
        return signal_tensor

    def update_stats(self, buffers, batch_size):
        for i in range(len(self.cores)):
            self.core_avgs[i] += torch.sum(buffers[i]).item() / (self.simulation_length * batch_size * self.cores[i].get_output_count())

    def get_core_spike_rates(self):
        return self.core_avgs
    
    def get_total_spikes(self):
        return self.total_spikes
    
    def to_lif_spikes(self, tensor, membrane_potentials, threshold):
        membrane_potentials += tensor
        spikes = (membrane_potentials > threshold).float()
        membrane_potentials[spikes == 1] -= threshold
        return spikes

    def to_spikes(self, tensor):
        #return tensor
        return (torch.rand(tensor.shape, device=tensor.device) < tensor).float()
    
    def forward(self, x):
        x = self.preprocessor(x)
        
        x = x.view(x.shape[0], -1)

        self.core_avgs = [0.0] * len(self.cores)

        buffers = []
        input_signals = []
        membrane_potentials = []
        input_membrane_potentials = torch.zeros(
            x.shape[0], x.shape[1], device=x.device)
        for core in self.cores:
            buffers.append(torch.zeros(x.shape[0], core.get_output_count(), device=x.device))
            input_signals.append(
                torch.zeros(x.shape[0], core.get_input_count(), device=x.device))
            membrane_potentials.append(
                torch.zeros(x.shape[0], core.get_output_count(), device=x.device))
        
        output_signals = torch.zeros(x.shape[0], len(self.output_sources), device=x.device)
        for cycle in range(self.cycles):
            #input_spikes = self.to_lif_spikes(x, input_membrane_potentials, 1.0)
            input_spikes = self.to_spikes(x)
            for core_idx in range(len(self.cores)):
                input_signals[core_idx] = self.get_signal_tensor(
                    input_spikes, 
                    buffers, self.cores[core_idx].axon_sources,
                    cycle)

            for core_idx in range(len(self.cores)):
                if (self.cores[core_idx].latency is not None) and cycle >= self.cores[core_idx].latency:
                    memb = membrane_potentials[core_idx]

                    memb += \
                        torch.matmul(
                            self.core_params[core_idx], 
                            input_signals[core_idx].T).T
                    
                    buffers[core_idx] = (memb > self.thresholds[core_idx]).float()

                    # novena reset
                    if self.firing_mode == "Novena":
                        memb[memb > self.thresholds[core_idx]] = 0.0
                    
                    # subtract reset
                    if self.firing_mode == "Default":
                        memb[memb > self.thresholds[core_idx]] -= self.thresholds[core_idx]
        
            self.update_stats(buffers, x.shape[0])
            output_signals += self.get_signal_tensor(input_spikes, buffers, self.output_sources, cycle)
            
        self.total_spikes = torch.sum(output_signals).item()
        print("total spikes: ", self.total_spikes)
        return output_signals