import torch
import torch.nn as nn
import numpy as np
import pytest
import sys
import os

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.mapping.mapping_utils import InputMapper, PerceptronMapper, ModelRepresentation, EinopsRearrangeMapper
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.ir import NeuralCore, ir_graph_to_soft_core_mapping, IRSource
from mimarsinan.mapping.hybrid_hardcore_mapping import HardCore, HardCoreMapping
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow
from mimarsinan.mapping.per_source_scales import compute_per_source_scales
from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.models.layers import TransformedActivation, SavedTensorDecorator

class SimpleMLPModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # Mirroring SimpleMLP: Rearrange -> Perceptron(norm=LazyBatchNorm1d) -> Perceptron(norm=Identity)
        self.p1 = Perceptron(32, 28*28, normalization=nn.LazyBatchNorm1d())
        self.p2 = Perceptron(10, 32, normalization=nn.Identity())
        
        inp = InputMapper((1, 28, 28)) 
        rearrange = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
        m1 = PerceptronMapper(rearrange, self.p1)
        m2 = PerceptronMapper(m1, self.p2)
        self._mapper_repr = ModelRepresentation(m2)

    def get_perceptrons(self):
        return [self.p1, self.p2]

    def get_mapper_repr(self):
        return self._mapper_repr
    
    def forward(self, x):
        return self._mapper_repr(x)

def run_hard_analytical(hard_mapping, inputs, device):
    state = {(-2, i): inputs[i].item() for i in range(len(inputs))}
    core_outputs = {}
    
    # Process cores in a way that respects dependencies (one-pass is enough if topographical)
    for i in range(len(hard_mapping.cores)):
        core = hard_mapping.cores[i]
        axon_vals = []
        for src in core.axon_sources:
            if src.is_off_: axon_vals.append(0.0)
            elif src.is_input_: axon_vals.append(state[(-2, src.neuron_)])
            elif src.is_always_on_: axon_vals.append(1.0)
            else: 
                if (src.core_, src.neuron_) not in core_outputs:
                    axon_vals.append(0.0)
                else:
                    axon_vals.append(core_outputs[(src.core_, src.neuron_)])
        
        axon_vals = torch.tensor(axon_vals, device=device).to(torch.float32)
        weights = torch.tensor(core.core_matrix, device=device).to(torch.float32)
        
        raw = weights.T @ axon_vals
        acts = torch.clamp(raw, min=0) / float(core.threshold)
        
        for n_idx in range(len(acts)):
            core_outputs[(i, n_idx)] = acts[n_idx].item()
    
    final_out = []
    for src in hard_mapping.output_sources:
        if src.is_off_: final_out.append(0.0)
        elif src.is_input_: final_out.append(state[(-2, src.neuron_)])
        elif src.is_always_on_: final_out.append(1.0)
        else:
            final_out.append(core_outputs[(src.core_, src.neuron_)])
    return torch.tensor(final_out, device=device)

def test_coalescing_accuracy_with_rearrange():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt = PerceptronTransformer()
    # Use a fixed input for comparison
    torch.manual_seed(42)
    x = torch.rand(1, 1, 28, 28).to(device)

    def run_sim_case(q_max, use_quantization=False, allow_coalescing=True):
        # Fresh model per case to avoid side effects
        model = SimpleMLPModel(28*28).to(device)
        model.eval()

        # Capture and set activation scales
        for p in model.get_perceptrons():
            p.input_activation = TransformedActivation(p.input_activation, [])
            p.input_activation.decorate(SavedTensorDecorator())
        model(torch.randn(10, 1, 28, 28).to(device))
        for p in model.get_perceptrons():
            dec = p.input_activation.pop_decorator()
            p.set_input_activation_scale(dec.latest_input.max().item())
        model.p1.set_activation_scale(model.p2.input_activation_scale.item())
        model.p2.set_activation_scale(1.0)

        # Apply quantization if requested
        if use_quantization:
            for p in model.get_perceptrons():
                p.set_parameter_scale(float(q_max))
                w = pt.get_effective_weight(p)
                w_q = torch.round(w * q_max) / q_max
                pt.apply_effective_weight_transform(p, lambda _, wq=w_q: wq)
                b = pt.get_effective_bias(p)
                b_q = torch.round(b * q_max) / q_max
                pt.apply_effective_bias_transform(p, lambda _, bq=b_q: bq)
        else:
            for p in model.get_perceptrons():
                p.set_parameter_scale(1.0)

        with torch.no_grad():
            torch_out = model(x)

        ir_mapping = IRMapping(
            q_max=q_max, 
            firing_mode="TTFS",
            max_axons=128, # Trigger decomposition
            max_neurons=1024,
            allow_core_coalescing=allow_coalescing,
        )
        
        repr = model.get_mapper_repr()
        compute_per_source_scales(repr)
        ir_graph = ir_mapping.map(repr)

        # Soft Core Simulation
        flow = SpikingUnifiedCoreFlow(
            (1, 28, 28), ir_graph, 32, nn.Identity(), "TTFS", "TTFS", "<=", spiking_mode="ttfs"
        ).to(device)
        with torch.no_grad():
            sim_soft = flow(x)
        
        # Hard Core Simulation
        soft_map = ir_graph_to_soft_core_mapping(ir_graph)
        hardware_cores = [HardCore(2048, 32) for _ in range(100)]
        hard_map = HardCoreMapping(hardware_cores)
        hard_map.map(soft_map)
        sim_hard = run_hard_analytical(hard_map, x.flatten(), device)
        
        mse_soft = torch.mean((torch_out - sim_soft)**2).item()
        mse_hard = torch.mean((torch_out - sim_hard)**2).item()
        
        mode_str = f"Quantized(q={q_max})" if use_quantization else "Unquantized"
        print(f"  {mode_str}: SoftCore MSE: {mse_soft:.2e}, HardCore MSE: {mse_hard:.2e}")
        
        assert mse_soft < 1e-10, f"{mode_str} SoftCore failed: MSE={mse_soft}"
        assert mse_hard < 1e-10, f"{mode_str} HardCore failed: MSE={mse_hard}"

    # Execution phases
    def sweep_all(coalescing):
        print(f"\n--- Testing Mode: Coalescing={'ON' if coalescing else 'OFF'} ---")
        run_sim_case(q_max=1, use_quantization=False, allow_coalescing=coalescing)
        run_sim_case(q_max=127, use_quantization=True, allow_coalescing=coalescing)

    print("Verifying Mapping Accuracy...")
    sweep_all(coalescing=True)
    sweep_all(coalescing=False)
    print("\nOK: All modes verified (MSE < 1e-10)")

if __name__ == "__main__":
    test_coalescing_accuracy_with_rearrange()
