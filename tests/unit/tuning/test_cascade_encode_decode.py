"""Spike encode/decode behaviour of the genuine TTFS cascade across segment boundaries.

Host-side value-domain compute ops between neural segments force a spike
*decode -> host compute -> re-encode* boundary. These tests pin how the deployed
single-spike cascade behaves at those boundaries (and how it diverges from the
value-domain staircase proxy with depth, §2.2):

1. Host compute ops cut the cascade into >1 neural segment (real decode/re-encode
   boundaries); a single encoding entry still matches the analytical TTFS kernel
   exactly — the boundary encode/decode is exact, divergence is a depth effect.
2. Gradient contract across a boundary: a *subsume* entry charges the decoded
   value directly (differentiable — gradient crosses upstream); an *offload* /
   host-ComputeOp entry reads a round-based re-encoded spike train
   (non-differentiable — upstream gradient severed). The latter is the structural
   mechanism behind §2.2's deep-layer attenuation; making the re-encode a
   surrogate so the genuine backward trains every segment is the D2/D3 work, and
   this pins the current contract so such a change is deliberate and visible.
"""

from __future__ import annotations

import torch

from cascade_fixtures import (
    build_cascade_flow,
    cascade_forward,
    install_ttfs_nodes,
    segment_count,
)


class TestSegmentStructure:
    def test_single_segment_has_one_neural_segment(self):
        flow, _ = build_cascade_flow(host_ops=False, depth=3)
        assert segment_count(flow) == 1

    def test_host_ops_split_into_multiple_segments(self):
        # depth=3 stages with a host compute op after stages 1 and 2 -> 3 segments.
        flow, _ = build_cascade_flow(host_ops=True, depth=3)
        assert segment_count(flow) >= 2, (
            "host compute ops must cut the cascade into >1 neural segment so the "
            "spike decode/re-encode boundary is exercised"
        )


class TestSingleEntryMatchesAnalyticalKernel:
    """At a single encoding entry the genuine cascade equals the analytical TTFS
    kernel — the boundary encode/decode is exact; divergence is a depth effect."""

    def test_single_perceptron_cascade_matches_kernel(self):
        from mimarsinan.models.nn.activations import TTFSCycleActivation

        S = 16
        flow, x = build_cascade_flow(host_ops=False, depth=1, in_dim=5, out_dim=3, S=S)
        out = cascade_forward(flow, x, S)
        p = flow.get_perceptrons()[0]
        kernel = TTFSCycleActivation(T=S, activation_scale=p.activation_scale,
                                     thresholding_mode="<=")
        with torch.no_grad():
            expected = kernel(p.layer(x.double()))
        torch.testing.assert_close(out, expected, atol=1e-9, rtol=0)


class TestEncodeDecodeGradientContract:
    """The genuine cascade's gradient contract across segment boundaries.

    A *subsume* segment entry charges the decoded value directly (differentiable),
    so gradient crosses the boundary. An *offload* / host-ComputeOp entry reads a
    re-encoded single-spike train (``round`` TTFS encode), which is the deployed
    HCM behaviour and is NON-differentiable — upstream gradient is severed.
    Making the boundary re-encode a surrogate (so the genuine backward trains all
    segments) is research direction D2/D3; this test pins the CURRENT contract so
    such a change is a deliberate, visible update."""

    def test_within_single_segment_all_layers_train(self):
        # No host-op cut: the whole cascade is one segment and every layer gets
        # gradient through the within-segment surrogate path.
        flow, x = build_cascade_flow(host_ops=False, depth=3)
        for p in flow.get_perceptrons():
            p.layer.weight.grad = None
        out = cascade_forward(flow, x.clone().requires_grad_(True), 4, grad=True)
        out.sum().backward()
        for p in flow.get_perceptrons():
            g = p.layer.weight.grad
            assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0

    def _build_two_segment(self, *, host_op, surrogate_temp=None, return_out=False):
        import torch.nn as nn
        from mimarsinan.torch_mapping.converter import convert_torch_model
        from cascade_fixtures import _calibrate_scales

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1, self.a1 = nn.Linear(6, 5), nn.ReLU()
                self.fc2, self.a2 = nn.Linear(5, 4), nn.ReLU()

            def forward(self, x):
                h = self.a1(self.fc1(x))
                if host_op:
                    h = h * 0.7 + 0.05  # host ComputeOp -> offload-style boundary
                return self.a2(self.fc2(h))

        torch.manual_seed(0)
        m = _M()
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                nn.init.uniform_(mod.weight, -0.4, 0.4)
                nn.init.uniform_(mod.bias, -0.05, 0.05)
        flow = convert_torch_model(m, (6,), 4, device="cpu")
        if not host_op:
            flow.get_perceptrons()[1].is_encoding_layer = True  # force a subsume cut
        x = torch.rand(16, 6, dtype=torch.float64)
        _calibrate_scales(flow, x)
        install_ttfs_nodes(flow, 32)
        for p in flow.get_perceptrons():
            p.layer.weight.grad = None
        out = cascade_forward(
            flow, x.requires_grad_(True), 32, grad=True, surrogate_temp=surrogate_temp,
        )
        out.sum().backward()
        return (flow, out) if return_out else flow

    def test_subsume_encode_cut_propagates_gradient_upstream(self):
        flow = self._build_two_segment(host_op=False)
        grads = [p.layer.weight.grad for p in flow.get_perceptrons()]
        assert all(g is not None and g.abs().sum() > 0 for g in grads), (
            "a subsume (value-charge) boundary must propagate gradient upstream"
        )

    def test_offload_computeop_boundary_severs_upstream_gradient(self):
        # Default contract (no surrogate): the offload boundary re-encode is the
        # non-differentiable round-based encode, so upstream gradient is severed.
        flow = self._build_two_segment(host_op=True, surrogate_temp=None)
        grads = [p.layer.weight.grad for p in flow.get_perceptrons()]
        assert grads[-1] is not None and grads[-1].abs().sum() > 0
        assert grads[0] is None or grads[0].abs().sum() == 0, (
            "offload/ComputeOp boundary severs upstream gradient by default "
            "(round-based re-encode); the STE surrogate flips this"
        )

    def test_offload_boundary_surrogate_propagates_gradient_upstream(self):
        # With the STE surrogate the genuine backward flows through the offload
        # boundary, so the UPSTREAM segment trains too (D2/D3 — the real lever).
        flow = self._build_two_segment(host_op=True, surrogate_temp=1.0)
        grads = [p.layer.weight.grad for p in flow.get_perceptrons()]
        assert all(g is not None and g.abs().sum() > 0 for g in grads), (
            "the boundary STE must propagate gradient to every segment "
            "(including the upstream segment past the offload boundary)"
        )

    def test_surrogate_leaves_forward_bit_exact(self):
        # The STE is backward-only: the forward (hence NF↔SCM parity and deployed
        # accuracy) must be byte-identical with and without the surrogate.
        _, out_severed = self._build_two_segment(
            host_op=True, surrogate_temp=None, return_out=True,
        )
        _, out_ste = self._build_two_segment(
            host_op=True, surrogate_temp=1.0, return_out=True,
        )
        torch.testing.assert_close(out_ste, out_severed, atol=0, rtol=0)
