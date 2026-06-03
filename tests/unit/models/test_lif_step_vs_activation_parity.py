"""Per-cycle LIF dynamics parity: the chip step vs the SpikingJelly node.

The chip-aligned NF reference and HCM must fire identically cycle-by-cycle. This
pins the linchpin: ``LIFCyclePolicy.step`` (HCM crossbar: ``memb += W@s + b``,
signed integrate-and-fire) produces the *same* per-cycle spikes as the SpikingJelly
node ``LIFActivation._forward_single_step`` when fed the equivalent real-valued
pre-activation. Equivalence holds only because the relu was removed from the node
(signed IF); a relu on either side breaks it whenever a cycle's weighted input is
negative.
"""

from __future__ import annotations

import pytest
import torch
from spikingjelly.activation_based import functional

from mimarsinan.models.spiking.cycle_policy import LIFCyclePolicy
from mimarsinan.models.nn.activations import LIFActivation


def _hcm_counts(W, b, inp, T, n_out, *, thresholding_mode, firing_mode):
    pol = LIFCyclePolicy(firing_mode)
    st = pol.make_state(inp.shape[1], n_out, "cpu", torch.float64)
    thr = torch.tensor(1.0, dtype=torch.float64)
    spikes = [
        pol.step(
            st, W.double(), inp[t], thr,
            hw_bias=b.double(), thresholding_mode=thresholding_mode,
            output_dtype=torch.float64,
        ).clone()
        for t in range(T)
    ]
    return torch.stack(spikes)


def _node_counts(W, b, inp, T, *, thresholding_mode):
    lif = LIFActivation(
        T=T, activation_scale=torch.tensor(1.0), thresholding_mode=thresholding_mode,
    )
    lif.set_cycle_accurate(True)
    functional.reset_net(lif.if_node)
    spikes = [
        lif._forward_single_step(inp[t].float() @ W.T + b).clone()
        for t in range(T)
    ]
    return torch.stack(spikes).double()


@pytest.mark.parametrize("thresholding_mode", ["<", "<="])
@pytest.mark.parametrize("seed", list(range(40)))
def test_lif_step_matches_node_per_cycle_signed(thresholding_mode, seed):
    """Default-reset LIF: signed crossbar step == SpikingJelly node, every cycle."""
    torch.manual_seed(seed)
    n_in, n_out, T, B = 5, 4, 8, 1
    W = torch.randn(n_out, n_in) * 1.5  # mixed-sign weights → negative cycle inputs
    b = torch.randn(n_out)
    inp = (torch.rand(T, B, n_in) > 0.5).double()

    hcm = _hcm_counts(
        W, b, inp, T, n_out,
        thresholding_mode=thresholding_mode, firing_mode="Default",
    )
    node = _node_counts(W, b, inp, T, thresholding_mode=thresholding_mode)
    assert torch.equal(hcm, node), (
        f"per-cycle mismatch (seed={seed}, mode={thresholding_mode}): "
        f"HCM={hcm[:, 0].sum(0).tolist()} node={node[:, 0].sum(0).tolist()}"
    )


def test_parity_holds_in_negative_input_regime():
    """Explicit negative-pre-activation case: this is the regime where the old
    relu-on-membrane diverged from HCM. Signed-IF must agree exactly."""
    T, n_out = 8, 3
    # All-negative bias and weights so most cycle inputs are negative.
    W = torch.tensor([[-1.0, -0.5, 0.2], [0.3, -1.2, -0.4], [-0.7, 0.1, -0.9]])
    b = torch.tensor([-0.5, -0.2, -0.3])
    inp = (torch.rand(T, 1, 3) > 0.4).double()
    pre = inp[:, 0].float() @ W.T + b
    assert pre.min() < 0  # deterministic-enough: weights/bias are mostly negative

    for mode in ("<", "<="):
        hcm = _hcm_counts(W, b, inp, T, n_out, thresholding_mode=mode, firing_mode="Default")
        node = _node_counts(W, b, inp, T, thresholding_mode=mode)
        assert torch.equal(hcm, node), f"mode={mode}: {hcm[:,0].sum(0)} vs {node[:,0].sum(0)}"


def test_scaled_node_matches_scaled_step():
    """Non-unit activation_scale: node normalizes by scale, HCM bakes 1/scale into
    the effective weight/bias — both reduce to the same signed-IF firing."""
    torch.manual_seed(0)
    n_in, n_out, T = 6, 5, 8
    scale = 2.5
    W = torch.randn(n_out, n_in) * 1.5
    b = torch.randn(n_out)
    inp = (torch.rand(T, 1, n_in) > 0.5).double()

    # Node side: real pre-activation, scale handled inside the node.
    lif = LIFActivation(T=T, activation_scale=torch.tensor(scale), thresholding_mode="<=")
    lif.set_cycle_accurate(True)
    functional.reset_net(lif.if_node)
    node = torch.stack(
        [(lif._forward_single_step(inp[t].float() @ W.T + b) / scale).clone() for t in range(T)]
    ).double()

    # HCM side: effective weight/bias fold 1/scale; threshold 1.
    hcm = _hcm_counts(
        W / scale, b / scale, inp, T, n_out,
        thresholding_mode="<=", firing_mode="Default",
    )
    assert torch.equal(hcm, node)
