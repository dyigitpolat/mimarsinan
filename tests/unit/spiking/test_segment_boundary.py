"""SSOT segment-boundary contract: decode + descriptor + shim identity.

The encode paths (``encode_segment_input`` / ``encode_compute_boundary``) are
exercised end-to-end by ``test_segment_boundary_encode.py``; here we pin the decode
contract and the declarative ``SegmentBoundary`` descriptor's inert Round-2 defaults.
"""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.spiking import segment_boundary as sb


def test_decode_segment_output_numpy_is_counts_over_T():
    counts = np.array([0, 2, 4, 8], dtype=np.float64)
    out = sb.decode_segment_output(counts, 4)
    assert out.shape == (1, 4)
    np.testing.assert_allclose(out, np.array([[0.0, 0.5, 1.0, 2.0]]))


def test_decode_clamps_T_to_at_least_one():
    counts = np.array([3.0])
    np.testing.assert_allclose(sb.decode_segment_output(counts, 0), np.array([[3.0]]))


def test_decode_segment_output_torch_preserves_batch():
    counts = torch.tensor([[2.0, 4.0], [0.0, 8.0]])  # (B=2, N=2)
    out = sb.decode_segment_output_torch(counts, 4)
    assert out.shape == (2, 2)
    torch.testing.assert_close(out, torch.tensor([[0.5, 1.0], [0.0, 2.0]]))


def test_numpy_and_torch_decode_agree():
    counts = np.array([1, 3, 4, 7], dtype=np.float64)
    n = sb.decode_segment_output(counts, 4).reshape(-1)
    t = sb.decode_segment_output_torch(torch.tensor(counts), 4).reshape(-1).numpy()
    np.testing.assert_allclose(n, t)


def test_segment_boundary_descriptor_round2_fields_inert_by_default():
    """Round-2 seams default to the identity so Round-1 behavior is exact."""
    bnd = sb.SegmentBoundary(
        producer_kind="neural", consumer_node_id=3, encode_mode="uniform",
    )
    assert bnd.shift == 0.0
    assert bnd.placement == "subsume"
    assert bnd.spike_generation_mode == "Uniform"
    assert bnd.decode == "count_over_T"


def test_boundary_config_defaults():
    cfg = sb.BoundaryConfig(simulation_length=4, spiking_mode="lif", cycle_accurate=True)
    # Inherited behavior toggle.
    assert cfg.use_cycle_accurate_trains is True
    # Declared-but-inert Round-2 toggles.
    assert cfg.negative_shift is False
    assert cfg.spike_generation_mode == "Uniform"
