"""TTFS input layout and output gathering tests (rung-2 identity executor).

See plan section 5.5: flattened input and output spans match IR expectations.
"""

import pytest
import torch
import torch.nn as nn

from conftest import make_tiny_ir_graph
from mimarsinan.models.spiking.hybrid.identity_flow import build_identity_spiking_flow


def test_ttfs_flatten_order_matches_ir_input_sources():
    """Input dimension and assignment to first layer match IR input_sources."""
    in_dim = 8
    hidden_dim = 4
    out_dim = 4
    ir_graph = make_tiny_ir_graph(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    flow = build_identity_spiking_flow(
        input_shape=(in_dim,),
        ir_graph=ir_graph,
        simulation_length=32,
        preprocessor=nn.Identity(),
        firing_mode="TTFS",
        spike_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode="ttfs",
    )
    # Batch of ones: (B, in_dim)
    x = torch.ones(2, in_dim)
    with torch.no_grad():
        out = flow(x)
    assert out.shape == (2, out_dim)
    # First layer core consumes from input (node_id=-2); flatten is just x here.
    # Input span for node 0 should cover in_dim inputs + 1 bias
    total_input_sources = len(ir_graph.nodes[0].input_sources.flatten())
    assert total_input_sources == in_dim + 1


def test_ttfs_output_spans_indices():
    """Output span count equals num_classes / last layer output dimension."""
    in_dim = 8
    hidden_dim = 4
    num_classes = 4
    ir_graph = make_tiny_ir_graph(
        in_dim=in_dim, hidden_dim=hidden_dim, out_dim=num_classes
    )
    flow = build_identity_spiking_flow(
        input_shape=(in_dim,),
        ir_graph=ir_graph,
        simulation_length=32,
        preprocessor=nn.Identity(),
        firing_mode="TTFS",
        spike_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode="ttfs",
    )
    output_sources = list(flow.hybrid_mapping.output_sources.flatten())
    assert len(output_sources) == num_classes
    # Output signals have one value per output source
    x = torch.ones(1, in_dim)
    with torch.no_grad():
        out = flow(x)
    assert out.shape == (1, num_classes)
