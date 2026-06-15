"""D2: stochastic axes own their decision RNG via ``set_decision_seed``.

The stochastic-mask path (``ActQuantAxis`` → ``RandomMaskAdjustmentStrategy``,
``NoiseAxis`` → ``NoisyDropout``) used unseeded global ``torch.rand``, so a
re-evaluation of the same committed state drew different masks and the I6
determinism invariant held only at the mock level. ``set_decision_seed`` gives
each stochastic axis its own seeded ``torch.Generator``: a re-seed reproduces the
exact masks regardless of intervening global RNG. The default (never seeded)
path is bit-exact with the legacy global-RNG behaviour.
"""

import torch

from conftest import make_tiny_supermodel, default_config
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)
from mimarsinan.tuning.axes import ActQuantAxis


def _act_quant_axis(rate=0.5):
    cfg = default_config()
    cfg["activation_quantization"] = True
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    axis = ActQuantAxis()
    axis.attach(model, manager, cfg)
    axis.set_rate(rate)  # intermediate rate → the random mask is live
    return axis, model


def _fwd(model, x):
    model.eval()
    with torch.no_grad():
        return model(x).clone()


def test_decision_seed_reproduces_masks_without_global_seed():
    axis, model = _act_quant_axis()
    x = torch.randn(2, 1, 8, 8)

    axis.set_decision_seed(123)
    a = _fwd(model, x)
    torch.rand(17)               # perturb GLOBAL rng between the two passes
    axis.set_decision_seed(123)  # reset the axis's OWN generator
    b = _fwd(model, x)
    assert torch.equal(a, b), "set_decision_seed must own determinism, not global RNG"


def test_different_decision_seed_changes_masks():
    axis, model = _act_quant_axis()
    x = torch.randn(2, 1, 8, 8)
    axis.set_decision_seed(1)
    a = _fwd(model, x)
    axis.set_decision_seed(2)
    b = _fwd(model, x)
    assert not torch.equal(a, b), "a different decision seed should change the masks"


def test_unseeded_axis_is_bit_exact_with_global_rng():
    # No set_decision_seed → the masks come from the GLOBAL rng exactly as before,
    # so a global manual_seed fully controls the forward (legacy bit-exact path).
    axis, model = _act_quant_axis()
    x = torch.randn(2, 1, 8, 8)
    torch.manual_seed(7)
    a = _fwd(model, x)
    torch.manual_seed(7)
    b = _fwd(model, x)
    assert torch.equal(a, b)


def test_decision_seed_survives_rate_changes():
    # The generator is re-wired after each set_rate rebuild, so reproducibility
    # holds even after the rate (and decorator stack) is updated.
    axis, model = _act_quant_axis(rate=0.3)
    x = torch.randn(2, 1, 8, 8)

    axis.set_decision_seed(55)
    axis.set_rate(0.7)
    a = _fwd(model, x)
    axis.set_decision_seed(55)
    axis.set_rate(0.7)
    b = _fwd(model, x)
    assert torch.equal(a, b)
