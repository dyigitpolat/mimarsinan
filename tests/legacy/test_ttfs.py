"""
Quick test for the TTFS (Time-to-First-Spike) spiking simulation.

Builds a small 2-layer MLP by hand, converts it to an IRGraph,
and runs both UnifiedCoreFlow and HybridCoreFlow in TTFS mode.

Key property: TTFS output ordering (argmax) should match the ReLU
model's prediction, demonstrating the ReLU↔TTFS equivalence.

Tests include both unquantized and quantized (pipeline-matching) versions.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import numpy as np

from mimarsinan.mapping.ir import NeuralCore, IRGraph, IRSource
from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def make_simple_relu_model(in_dim, hidden_dim, out_dim):
    """Create a 2-layer ReLU network with positive weights (easier for spiking)."""
    model = nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=True),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim, bias=True),
    )
    with torch.no_grad():
        model[0].weight.data = torch.abs(model[0].weight.data) * 0.5
        model[0].bias.data = torch.abs(model[0].bias.data) * 0.1
        model[2].weight.data = torch.abs(model[2].weight.data) * 0.5
        model[2].bias.data = torch.abs(model[2].bias.data) * 0.1
    return model


def relu_model_to_ir_graph(model, in_dim, *, quantize=False, weight_bits=8):
    """
    Convert Sequential(Linear, ReLU, Linear) → IRGraph with 2 NeuralCores.

    Bias is encoded as an extra "always-on" axon row (IRSource node_id=-3).

    If quantize=True, replicates the pipeline's SoftCoreMappingStep behavior:
      core_matrix = round(core_matrix * parameter_scale)
      threshold = parameter_scale
    """
    w1 = model[0].weight.data.numpy()  # (hidden, in)
    b1 = model[0].bias.data.numpy()
    w2 = model[2].weight.data.numpy()  # (out, hidden)
    b2 = model[2].bias.data.numpy()

    hidden_dim = w1.shape[0]
    out_dim = w2.shape[0]

    q_max = (2 ** (weight_bits - 1)) - 1  # 127 for 8-bit

    # -- Core 1 (hidden layer) --
    core1_matrix = np.vstack([w1.T, b1.reshape(1, -1)])
    # Compute parameter_scale: scale to fill quantization range
    abs_max_1 = max(np.abs(core1_matrix).max(), 1e-12)
    ps1 = q_max / abs_max_1

    input_sources_1 = np.array(
        [IRSource(node_id=-2, index=i) for i in range(in_dim)]
        + [IRSource(node_id=-3, index=0)],
        dtype=object,
    )

    if quantize:
        core1_matrix_q = np.round(core1_matrix * ps1)
        threshold_1 = ps1
    else:
        core1_matrix_q = core1_matrix
        threshold_1 = 1.0

    core1 = NeuralCore(
        id=0, name="hidden",
        input_sources=input_sources_1,
        core_matrix=core1_matrix_q,
        threshold=threshold_1,
        parameter_scale=torch.tensor(1.0) if quantize else torch.tensor(ps1),
        latency=0,
    )

    # -- Core 2 (output layer) --
    core2_matrix = np.vstack([w2.T, b2.reshape(1, -1)])
    abs_max_2 = max(np.abs(core2_matrix).max(), 1e-12)
    ps2 = q_max / abs_max_2

    input_sources_2 = np.array(
        [IRSource(node_id=0, index=i) for i in range(hidden_dim)]
        + [IRSource(node_id=-3, index=0)],
        dtype=object,
    )

    if quantize:
        core2_matrix_q = np.round(core2_matrix * ps2)
        threshold_2 = ps2
    else:
        core2_matrix_q = core2_matrix
        threshold_2 = 1.0

    core2 = NeuralCore(
        id=1, name="output",
        input_sources=input_sources_2,
        core_matrix=core2_matrix_q,
        threshold=threshold_2,
        parameter_scale=torch.tensor(1.0) if quantize else torch.tensor(ps2),
        latency=1,
    )

    output_sources = np.array(
        [IRSource(node_id=1, index=i) for i in range(out_dim)],
        dtype=object,
    )
    return IRGraph(nodes=[core1, core2], output_sources=output_sources)


# ----------------------------------------------------------------
# Test 1: TTFS input encoding
# ----------------------------------------------------------------
def test_ttfs_encoding():
    print("=" * 60)
    print("TEST 1: TTFS Input Encoding")
    print("=" * 60)

    T = 16
    ir_graph = IRGraph(nodes=[], output_sources=np.array([], dtype=object))
    flow = SpikingUnifiedCoreFlow(
        input_shape=(5,),
        ir_graph=ir_graph,
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="TTFS",
        spike_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode="ttfs",
    )

    activations = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
    spike_train = flow._ttfs_encode_input(activations)  # (T, 1, 5)

    print(f"  Activations: {activations[0].tolist()}")
    all_ok = True
    for i in range(5):
        spike_cycles = (spike_train[:, 0, i] > 0).nonzero(as_tuple=True)[0].tolist()
        expected_time = round(T * (1.0 - activations[0, i].item()))
        print(f"    act={activations[0,i]:.2f}: spike at cycle {spike_cycles}, expected ~{expected_time}")
        if expected_time < T:
            if len(spike_cycles) != 1:
                print(f"    FAIL: expected 1 spike, got {len(spike_cycles)}")
                all_ok = False
            elif spike_cycles[0] != expected_time:
                print(f"    FAIL: expected spike at {expected_time}, got {spike_cycles[0]}")
                all_ok = False
        else:
            if len(spike_cycles) != 0:
                print(f"    FAIL: expected no spike (time={expected_time}), got {spike_cycles}")
                all_ok = False

    if all_ok:
        print("  PASSED\n")
    else:
        print("  FAILED\n")
    return all_ok


# ----------------------------------------------------------------
# Test 2: TTFS via UnifiedCoreFlow (unquantized)
# ----------------------------------------------------------------
def test_ttfs_unified_core_flow():
    print("=" * 60)
    print("TEST 2: TTFS UnifiedCoreFlow (unquantized, threshold=1.0)")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    in_dim, hidden_dim, out_dim = 8, 16, 4
    T = 64

    model = make_simple_relu_model(in_dim, hidden_dim, out_dim)
    x = torch.rand(8, in_dim)

    with torch.no_grad():
        relu_out = model(x)
        relu_preds = relu_out.argmax(dim=1)

    ir_graph = relu_model_to_ir_graph(model, in_dim, quantize=False)

    flow = SpikingUnifiedCoreFlow(
        input_shape=(in_dim,),
        ir_graph=ir_graph,
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="TTFS",
        spike_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode="ttfs",
    )

    with torch.no_grad():
        ttfs_out = flow(x)
        ttfs_preds = ttfs_out.argmax(dim=1)

    print(f"  ReLU predictions: {relu_preds.tolist()}")
    print(f"  TTFS predictions: {ttfs_preds.tolist()}")
    print(f"  ReLU outputs[0]:  {[f'{v:.4f}' for v in relu_out[0].tolist()]}")
    print(f"  TTFS outputs[0]:  {[f'{v:.4f}' for v in ttfs_out[0].tolist()]}")

    agreement = (relu_preds == ttfs_preds).float().mean().item()
    print(f"  Agreement: {agreement*100:.0f}% ({(relu_preds == ttfs_preds).sum()}/{len(relu_preds)})")

    passed = agreement == 1.0
    print(f"  {'PASSED' if passed else 'FAILED'}\n")
    return passed


# ----------------------------------------------------------------
# Test 3: TTFS via HybridCoreFlow (unquantized)
# ----------------------------------------------------------------
def test_ttfs_hybrid_core_flow():
    print("=" * 60)
    print("TEST 3: TTFS HybridCoreFlow (unquantized, threshold=1.0)")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    in_dim, hidden_dim, out_dim = 8, 16, 4
    T = 64

    model = make_simple_relu_model(in_dim, hidden_dim, out_dim)
    x = torch.rand(8, in_dim)

    with torch.no_grad():
        relu_out = model(x)
        relu_preds = relu_out.argmax(dim=1)

    ir_graph = relu_model_to_ir_graph(model, in_dim, quantize=False)

    cores_config = [{"max_axons": 256, "max_neurons": 256, "count": 10}]
    hybrid_mapping = build_hybrid_hard_core_mapping(
        ir_graph=ir_graph,
        cores_config=cores_config,
    )

    flow = SpikingHybridCoreFlow(
        input_shape=(in_dim,),
        hybrid_mapping=hybrid_mapping,
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode="ttfs",
    )

    with torch.no_grad():
        ttfs_out = flow(x)
        ttfs_preds = ttfs_out.argmax(dim=1)

    print(f"  ReLU predictions: {relu_preds.tolist()}")
    print(f"  TTFS predictions: {ttfs_preds.tolist()}")
    print(f"  ReLU outputs[0]:  {[f'{v:.4f}' for v in relu_out[0].tolist()]}")
    print(f"  TTFS outputs[0]:  {[f'{v:.4f}' for v in ttfs_out[0].tolist()]}")

    agreement = (relu_preds == ttfs_preds).float().mean().item()
    print(f"  Agreement: {agreement*100:.0f}% ({(relu_preds == ttfs_preds).sum()}/{len(relu_preds)})")

    passed = agreement == 1.0
    print(f"  {'PASSED' if passed else 'FAILED'}\n")
    return passed


# ----------------------------------------------------------------
# Test 4: TTFS with QUANTIZED weights (pipeline-matching)
# ----------------------------------------------------------------
def test_ttfs_quantized():
    """
    This test replicates what SoftCoreMappingStep does:
      core_matrix = round(core_matrix * parameter_scale)
      threshold = parameter_scale
      parameter_scale = 1.0

    The analytical TTFS forward computes:
      out = relu(W_quantized @ input) / threshold

    Which should approximately recover relu(W_float @ input).
    """
    print("=" * 60)
    print("TEST 4: TTFS with QUANTIZED weights (pipeline-matching)")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    in_dim, hidden_dim, out_dim = 16, 32, 10
    T = 32

    model = make_simple_relu_model(in_dim, hidden_dim, out_dim)
    x = torch.rand(32, in_dim)

    with torch.no_grad():
        relu_out = model(x)
        relu_preds = relu_out.argmax(dim=1)

    # Quantized IR graph (matching pipeline behavior)
    ir_graph = relu_model_to_ir_graph(model, in_dim, quantize=True, weight_bits=8)

    # Print quantization info
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            print(f"  Core '{node.name}': matrix range [{node.core_matrix.min():.0f}, {node.core_matrix.max():.0f}], "
                  f"threshold={node.threshold:.1f}")

    # Test UnifiedCoreFlow
    unified = SpikingUnifiedCoreFlow(
        input_shape=(in_dim,),
        ir_graph=ir_graph,
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="TTFS",
        spike_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode="ttfs",
    )

    with torch.no_grad():
        ttfs_out = unified(x)
        ttfs_preds = ttfs_out.argmax(dim=1)

    print(f"\n  ReLU predictions[:8]:     {relu_preds[:8].tolist()}")
    print(f"  TTFS-Q predictions[:8]:   {ttfs_preds[:8].tolist()}")
    print(f"  ReLU outputs[0]:  {[f'{v:.3f}' for v in relu_out[0].tolist()]}")
    print(f"  TTFS-Q outputs[0]: {[f'{v:.3f}' for v in ttfs_out[0].tolist()]}")

    agreement = (relu_preds == ttfs_preds).float().mean().item()
    print(f"\n  Argmax agreement (Unified): {agreement*100:.0f}% ({(relu_preds == ttfs_preds).sum()}/{len(relu_preds)})")

    # Test HybridCoreFlow
    cores_config = [{"max_axons": 256, "max_neurons": 256, "count": 10}]
    hybrid_mapping = build_hybrid_hard_core_mapping(
        ir_graph=ir_graph,
        cores_config=cores_config,
    )
    hybrid = SpikingHybridCoreFlow(
        input_shape=(in_dim,),
        hybrid_mapping=hybrid_mapping,
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode="ttfs",
    )

    with torch.no_grad():
        h_out = hybrid(x)
        h_preds = h_out.argmax(dim=1)

    h_agreement = (relu_preds == h_preds).float().mean().item()
    print(f"  Argmax agreement (Hybrid):  {h_agreement*100:.0f}% ({(relu_preds == h_preds).sum()}/{len(relu_preds)})")

    uh_agreement = (ttfs_preds == h_preds).float().mean().item()
    print(f"  Unified-Hybrid agreement:   {uh_agreement*100:.0f}%")

    passed = agreement >= 0.8 and h_agreement >= 0.8
    print(f"  {'PASSED' if passed else 'FAILED'} (threshold: >= 80% agreement)\n")
    return passed


# ----------------------------------------------------------------
# Test 5: Fire-once semantics (output range check)
# ----------------------------------------------------------------
def test_ttfs_fire_once():
    print("=" * 60)
    print("TEST 5: Fire-Once Semantics (output range)")
    print("=" * 60)

    torch.manual_seed(0)
    in_dim, hidden_dim, out_dim = 4, 8, 2
    T = 32

    model = make_simple_relu_model(in_dim, hidden_dim, out_dim)
    ir_graph = relu_model_to_ir_graph(model, in_dim)

    flow = SpikingUnifiedCoreFlow(
        input_shape=(in_dim,),
        ir_graph=ir_graph,
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="TTFS",
        spike_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode="ttfs",
    )

    x = torch.rand(4, in_dim)
    with torch.no_grad():
        out = flow(x)

    # Analytical TTFS output should be >= 0 (ReLU)
    ok = (out >= -1e-6).all().item()
    print(f"  Output range: [{out.min().item():.4f}, {out.max().item():.4f}] (expected >= 0)")
    print(f"  {'PASSED' if ok else 'FAILED'}\n")
    return ok


# ----------------------------------------------------------------
# Test 6: Unified vs Hybrid agreement in TTFS mode
# ----------------------------------------------------------------
def test_unified_vs_hybrid_ttfs():
    print("=" * 60)
    print("TEST 6: Unified vs Hybrid TTFS Agreement")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    in_dim, hidden_dim, out_dim = 8, 16, 4
    T = 64

    model = make_simple_relu_model(in_dim, hidden_dim, out_dim)
    x = torch.rand(4, in_dim)

    ir_graph = relu_model_to_ir_graph(model, in_dim)

    unified = SpikingUnifiedCoreFlow(
        input_shape=(in_dim,),
        ir_graph=ir_graph,
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="TTFS",
        spike_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode="ttfs",
    )

    cores_config = [{"max_axons": 256, "max_neurons": 256, "count": 10}]
    hybrid_mapping = build_hybrid_hard_core_mapping(
        ir_graph=ir_graph,
        cores_config=cores_config,
    )
    hybrid = SpikingHybridCoreFlow(
        input_shape=(in_dim,),
        hybrid_mapping=hybrid_mapping,
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode="ttfs",
    )

    with torch.no_grad():
        u_out = unified(x)
        h_out = hybrid(x)

    u_preds = u_out.argmax(dim=1)
    h_preds = h_out.argmax(dim=1)

    print(f"  Unified preds: {u_preds.tolist()}")
    print(f"  Hybrid  preds: {h_preds.tolist()}")
    print(f"  Unified out[0]: {[f'{v:.4f}' for v in u_out[0].tolist()]}")
    print(f"  Hybrid  out[0]: {[f'{v:.4f}' for v in h_out[0].tolist()]}")

    # Check numerical closeness, not just argmax
    diff = (u_out - h_out).abs().max().item()
    print(f"  Max absolute difference: {diff:.6f}")

    agreement = (u_preds == h_preds).float().mean().item()
    print(f"  Agreement: {agreement*100:.0f}%")

    passed = agreement >= 0.75 and diff < 0.1
    print(f"  {'PASSED' if passed else 'WARNING: low agreement or large diff'}\n")
    return passed


# ----------------------------------------------------------------
# Test 7: TTFS with realistic mixed-sign weights + quantization
# ----------------------------------------------------------------
def test_ttfs_realistic_mixed_weights():
    """
    Replicate pipeline conditions more closely:
    - Mixed positive/negative weights (like a real trained model)
    - Multiple layers
    - Quantized weights
    - Various input patterns
    """
    print("=" * 60)
    print("TEST 7: TTFS with realistic mixed-sign weights (quantized)")
    print("=" * 60)

    torch.manual_seed(123)
    np.random.seed(123)

    in_dim, hidden_dim, out_dim = 16, 32, 10

    # Create model with realistic mixed weights (as from training)
    model = nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=True),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim, bias=True),
    )
    # Don't force positive — use default init (mixed sign)
    x = torch.rand(64, in_dim)

    with torch.no_grad():
        relu_out = model(x)
        relu_preds = relu_out.argmax(dim=1)

    # Build quantized IR graph
    ir_graph = relu_model_to_ir_graph(model, in_dim, quantize=True, weight_bits=8)

    # Print core info
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            print(f"  Core '{node.name}': "
                  f"matrix [{node.core_matrix.min():.0f}, {node.core_matrix.max():.0f}], "
                  f"threshold={node.threshold:.1f}, "
                  f"nonzero={np.count_nonzero(node.core_matrix)}/{node.core_matrix.size}")

    # UnifiedCoreFlow TTFS
    flow = SpikingUnifiedCoreFlow(
        input_shape=(in_dim,),
        ir_graph=ir_graph,
        simulation_length=32,
        preprocessor=nn.Identity(),
        firing_mode="TTFS",
        spike_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode="ttfs",
    )

    with torch.no_grad():
        ttfs_out = flow(x)
        ttfs_preds = ttfs_out.argmax(dim=1)

    # Print detailed comparison
    print(f"\n  ReLU outputs[0]:   {[f'{v:.4f}' for v in relu_out[0].tolist()]}")
    print(f"  TTFS-Q outputs[0]: {[f'{v:.4f}' for v in ttfs_out[0].tolist()]}")
    print(f"  ReLU outputs[1]:   {[f'{v:.4f}' for v in relu_out[1].tolist()]}")
    print(f"  TTFS-Q outputs[1]: {[f'{v:.4f}' for v in ttfs_out[1].tolist()]}")

    # Check if outputs are all the same (degenerate case)
    unique_preds = len(set(ttfs_preds.tolist()))
    print(f"\n  TTFS unique predictions: {unique_preds} (out of {out_dim} classes)")
    if unique_preds == 1:
        print("  WARNING: All TTFS predictions are the same class!")

    agreement = (relu_preds == ttfs_preds).float().mean().item()
    print(f"  Argmax agreement: {agreement*100:.0f}% ({(relu_preds == ttfs_preds).sum()}/{len(relu_preds)})")

    passed = agreement >= 0.7
    print(f"  {'PASSED' if passed else 'FAILED'} (threshold: >= 70%)\n")
    return passed


# ----------------------------------------------------------------
# Test 8: Load pipeline artifacts and test (if available)
# ----------------------------------------------------------------
def test_ttfs_pipeline_artifacts():
    """
    If a pipeline run has been completed, load its IR graph and test
    the analytical TTFS directly.
    """
    print("=" * 60)
    print("TEST 8: Pipeline artifact test (loads from generated/)")
    print("=" * 60)

    import pickle

    cache_dir = './generated/mnist_ttfs_ttfs_deployment_run'
    ir_pickle = os.path.join(cache_dir, 'Soft Core Mapping.ir_graph.pickle')
    model_pt = os.path.join(cache_dir, 'Normalization Fusion.model.pt')

    if not os.path.exists(ir_pickle) or not os.path.exists(model_pt):
        print("  SKIPPED: Pipeline artifacts not found.")
        print(f"  Expected: {ir_pickle}")
        print(f"           {model_pt}\n")
        return True  # Not a failure, just skipped

    with open(ir_pickle, 'rb') as f:
        ir_graph = pickle.load(f)

    loaded = torch.load(model_pt, map_location='cpu', weights_only=False)
    # Pipeline saves (model, device) tuple
    model = loaded[0] if isinstance(loaded, tuple) else loaded

    # Build preprocessor like the pipeline does
    preprocessor = nn.Sequential(model.get_preprocessor(), model.in_act)

    cores = ir_graph.get_neural_cores()
    print(f"  IR Graph: {len(cores)} neural cores")
    for i, core in enumerate(cores[:5]):
        print(f"    Core {i} ({core.name}): "
              f"shape={core.core_matrix.shape}, "
              f"threshold={core.threshold:.2f}, "
              f"range=[{core.core_matrix.min():.0f}, {core.core_matrix.max():.0f}]")
    if len(cores) > 5:
        print(f"    ... ({len(cores) - 5} more)")

    flow = SpikingUnifiedCoreFlow(
        input_shape=(1, 28, 28),  # MNIST
        ir_graph=ir_graph,
        simulation_length=32,
        preprocessor=preprocessor,
        firing_mode="TTFS",
        spike_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode="ttfs",
    )

    # Create a small batch of test inputs
    from mimarsinan.data_handling.data_providers.mnist_data_provider import MNIST_DataProvider
    try:
        dp = MNIST_DataProvider()
        test_loader = torch.utils.data.DataLoader(
            dp.get_test_dataset(), batch_size=100, shuffle=False
        )
        x_batch, y_batch = next(iter(test_loader))
    except Exception as e:
        print(f"  Could not load MNIST data: {e}")
        # Create random input
        x_batch = torch.rand(100, 1, 28, 28)
        y_batch = torch.randint(0, 10, (100,))

    with torch.no_grad():
        ttfs_out = flow(x_batch)
        ttfs_preds = ttfs_out.argmax(dim=1)

    # Check outputs
    print(f"\n  TTFS output shape: {ttfs_out.shape}")
    print(f"  TTFS output range: [{ttfs_out.min().item():.4f}, {ttfs_out.max().item():.4f}]")
    print(f"  TTFS output mean: {ttfs_out.mean().item():.4f}")

    # Check if all outputs are the same
    unique_per_sample = [(row != row[0]).any().item() for row in ttfs_out[:10]]
    all_same = not any(unique_per_sample)
    if all_same:
        print("  WARNING: All output values are identical across classes!")
        print(f"  Sample output[0]: {[f'{v:.4f}' for v in ttfs_out[0].tolist()[:10]]}")

    # Check intermediate activations
    # Re-run with debug to see what's happening inside
    print("\n  Debugging intermediate activations...")
    x_debug = x_batch[:4]
    x_proc = preprocessor(x_debug)
    x_proc = x_proc.view(x_proc.shape[0], -1)
    print(f"  Preprocessed input: shape={x_proc.shape}, "
          f"range=[{x_proc.min().item():.4f}, {x_proc.max().item():.4f}], "
          f"mean={x_proc.mean().item():.4f}")

    # Check each core's output
    from mimarsinan.mapping.ir_source_spans import compress_ir_sources
    activation_cache = {}
    for node_idx, node in enumerate(ir_graph.nodes):
        if isinstance(node, NeuralCore):
            weight = torch.tensor(node.core_matrix.T, dtype=torch.float32)
            threshold = torch.tensor(node.threshold, dtype=torch.float32)

            spans = compress_ir_sources(list(node.input_sources.flatten()))
            in_dim = len(node.input_sources.flatten())

            inp = torch.zeros(4, in_dim)
            # Fill from spans
            for sp in spans:
                d0, d1 = int(sp.dst_start), int(sp.dst_end)
                if sp.kind == "off":
                    continue
                if sp.kind == "on":
                    inp[:, d0:d1] = 1.0
                    continue
                if sp.kind == "input":
                    inp[:, d0:d1] = x_proc[:, int(sp.src_start):int(sp.src_end)]
                    continue
                inp[:, d0:d1] = activation_cache[int(sp.src_node_id)][:, int(sp.src_start):int(sp.src_end)]

            raw = torch.matmul(weight, inp.T).T
            out = torch.nn.functional.relu(raw) / threshold

            activation_cache[node.id] = out

            if node_idx < 5 or node_idx >= len(ir_graph.nodes) - 3:
                print(f"    Core {node_idx} ({node.name}): "
                      f"inp range=[{inp.min():.3f}, {inp.max():.3f}], "
                      f"raw range=[{raw.min():.1f}, {raw.max():.1f}], "
                      f"out range=[{out.min():.4f}, {out.max():.4f}]")

    accuracy = (ttfs_preds == y_batch).float().mean().item()
    print(f"\n  TTFS accuracy on 100 samples: {accuracy*100:.1f}%")

    passed = accuracy > 0.2  # Better than random chance
    print(f"  {'PASSED' if passed else 'FAILED'} (threshold: > 20%)\n")
    return passed


# ----------------------------------------------------------------
if __name__ == "__main__":
    results = {}
    results["encoding"]       = test_ttfs_encoding()
    results["fire_once"]      = test_ttfs_fire_once()
    results["unified"]        = test_ttfs_unified_core_flow()
    results["hybrid"]         = test_ttfs_hybrid_core_flow()
    results["quantized"]      = test_ttfs_quantized()
    results["unified_hybrid"] = test_unified_vs_hybrid_ttfs()
    results["mixed_weights"]  = test_ttfs_realistic_mixed_weights()
    results["pipeline"]       = test_ttfs_pipeline_artifacts()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name:20s}: {'PASS' if ok else 'FAIL/WARN'}")
    
    all_pass = all(results.values())
    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED/WARNED'}")
    print("=" * 60)
