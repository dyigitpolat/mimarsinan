#!/usr/bin/env python3

"""
Hybrid runtime sanity check.

Validates that we can:
1) Map a model containing ComputeOps (e.g., MaxPool2d) into a unified IRGraph
2) Compile that IRGraph into a HybridHardCoreMapping:
     neural segment (HardCoreMapping) -> ComputeOp barrier -> neural segment
3) Execute the hybrid program with SpikingHybridCoreFlow end-to-end (shape check)
"""

import sys

import torch
import torch.nn as nn


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def main():
    if "./src" not in sys.path:
        sys.path.append("./src")

    from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
    from mimarsinan.mapping.ir_mapping import IRMapping
    from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow
    from mimarsinan.models.simple_conv import SimpleConvMapper

    device = torch.device("cpu")

    input_shape = (1, 28, 28)
    num_classes = 10

    model = SimpleConvMapper(
        device,
        input_shape,
        num_classes,
        conv_out_channels=3,
        conv_kernel_size=3,
        conv_stride=4,
        conv_padding=1,
        use_pool=True,
        pool_kernel_size=2,
        pool_stride=2,
        pool_padding=0,
        max_axons=256,
        max_neurons=256,
        name="simple_conv_pool",
    )

    # Initialize any LazyModules (e.g., LazyBatchNorm*) before PerceptronTransformer touches params.
    with torch.no_grad():
        _ = model(torch.randn(2, *input_shape))

    ir_mapping = IRMapping(
        q_max=127,
        firing_mode="Default",
        max_axons=256,
        max_neurons=256,
        allow_axon_tiling=False,
    )
    ir_graph = ir_mapping.map(model.get_mapper_repr())

    compute_ops = ir_graph.get_compute_ops()
    _assert(len(compute_ops) == 1, f"Expected 1 ComputeOp, got {len(compute_ops)}")

    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir_graph,
        cores_config=[{"max_axons": 256, "max_neurons": 256, "count": 50}],
    )
    _assert(len(hybrid.get_compute_ops()) == 1, "Hybrid mapping should contain 1 ComputeOp stage")
    _assert(len(hybrid.get_neural_segments()) == 2, "Expected 2 neural segments around the pool barrier")

    # End-to-end execution shape check.
    T = 8
    flow = SpikingHybridCoreFlow(
        input_shape=torch.Size(input_shape),
        hybrid_mapping=hybrid,
        simulation_length=T,
        preprocessor=nn.Identity(),  # We feed preprocessed flat rates directly for this sanity test.
        firing_mode="Default",
        spike_mode="Uniform",
        thresholding_mode="<=",
    )

    # Feed flat input rates in [0,1]; shape matches InputMapper flattening.
    x = torch.rand(2, 28 * 28)
    with torch.no_grad():
        y = flow(x)
    _assert(tuple(y.shape) == (2, num_classes), f"Output shape mismatch: {tuple(y.shape)}")

    print("OK")


if __name__ == "__main__":
    main()


