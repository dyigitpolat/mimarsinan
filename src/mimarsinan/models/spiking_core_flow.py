from mimarsinan.mapping.chip_latency import *

import torch.nn as nn
import torch

class SpikingCoreFlow(nn.Module):
    def __init__(self, input_shape, core_mapping, simulation_length):
        super(SpikingCoreFlow, self).__init__()
        self.input_shape = input_shape

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

        self.cycles = ChipLatency(core_mapping).calculate() + simulation_length

        # Stats
        self.core_sums = [None] * len(self.cores)
    def update_cores(self):
        for idx, core in enumerate(self.cores):
            core.core_matrix[:,:] = \
                self.core_params[idx].detach().numpy().transpose()
    
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

    def update_stats(self, buffers):
        for i in range(len(self.cores)):
            if self.core_sums[i] is None:
                self.core_sums[i] = torch.sum(buffers[i]).item()
            else:
                self.core_sums[i] += torch.sum(buffers[i]).item()
            
    def to_spikes(self, tensor):
        return (torch.rand(tensor.shape, device=tensor.device) < tensor).float()
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)

        buffers = []
        input_signals = []
        membrane_potentials = []
        for core in self.cores:
            buffers.append(torch.zeros(x.shape[0], core.get_output_count(), device=x.device))
            input_signals.append(
                torch.zeros(x.shape[0], core.get_input_count(), device=x.device))
            membrane_potentials.append(
                torch.zeros(x.shape[0], core.get_output_count(), device=x.device))
        
        output_signals = torch.zeros(x.shape[0], len(self.output_sources), device=x.device)
        for cycle in range(self.cycles):
            for core_idx in range(len(self.cores)):
                input_signals[core_idx] = self.get_signal_tensor(
                    self.to_spikes(x), 
                    buffers, self.cores[core_idx].axon_sources,
                    cycle)

            for core_idx in range(len(self.cores)):
                if cycle >= self.cores[core_idx].latency:
                    memb = membrane_potentials[core_idx]

                    memb += \
                        torch.matmul(
                            self.core_params[core_idx], 
                            input_signals[core_idx].T).T
                    
                    buffers[core_idx] = (memb > self.thresholds[core_idx]).float()

                    # novena reset
                    memb[memb > self.thresholds[core_idx]] = 0.0
        
            self.update_stats(buffers)
            output_signals += self.get_signal_tensor(self.to_spikes(x), buffers, self.output_sources, cycle)
            
        print("total spikes: ", torch.sum(output_signals).item())
        return output_signals