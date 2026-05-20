"""``ChipAlignedForward``: model.forward → SpikingHybridCoreFlow.forward.

Used by the post-LIF chip-aligned finetune to train host-side encoding
Perceptron weights *through* the actual chip simulator (per-core latencies,
integer firing, surrogate gradient). The on-chip core weights live in the
hybrid mapping as numpy ndarrays and remain frozen — gradients flow only
to the host-side ``nn.Parameter``s (encoding Perceptrons, etc.).

Unlike the earlier broken draft this wrapper has **no blend-rate bypass**:
it is only installed *after* ``LIFAdaptationTuner`` reaches ``rate==1.0``.
While ``LIFBlendActivation.rate < 1.0`` the per-cycle membrane evolution of
the blend has no analogue inside the chip simulator, so blending and
chip-alignment are kept as separate sequential phases (blend first, then
chip-aligned finetune).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ChipAlignedForward:
    """Picklable forward wrapper that routes through a SpikingHybridCoreFlow.

    The hybrid mapping is built lazily on first call from the current model
    weights. Re-build is currently *manual* (call ``refresh(hybrid_mapping)``)
    rather than auto-detected — the finetune tuner rebuilds at most once per
    epoch, far cheaper than re-running the soft-core mapping per step.
    """

    def __init__(self, model: nn.Module, pipeline_config: dict[str, Any]):
        self.model = model
        self.pipeline_config = pipeline_config
        self._flow = None
        self._hybrid_mapping = None

    def _call_unpatched_forward(self, x: torch.Tensor) -> torch.Tensor:
        return type(self.model).forward(self.model, x)

    def refresh(self, hybrid_mapping) -> None:
        """Install/refresh the chip-aligned flow with the given hybrid mapping."""
        from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow
        from mimarsinan.spiking.lif_utils import apply_cycle_accurate_trains_to_model

        cfg = self.pipeline_config
        device = cfg.get("device", "cpu")
        preprocessor = getattr(self.model, "preprocessor", None)
        self._hybrid_mapping = hybrid_mapping
        self._flow = SpikingHybridCoreFlow(
            cfg["input_shape"],
            hybrid_mapping,
            int(cfg["simulation_steps"]),
            preprocessor,
            cfg.get("firing_mode", "Default"),
            cfg.get("spike_generation_mode", "Uniform"),
            cfg.get("thresholding_mode", "<="),
            spiking_mode=cfg.get("spiking_mode", "lif"),
            cycle_accurate_lif_forward=True,
        ).to(device)
        apply_cycle_accurate_trains_to_model(self.model, True)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._flow is None:
            raise RuntimeError(
                "ChipAlignedForward: call .refresh(hybrid_mapping) before "
                "invoking the wrapper. The chip-aligned flow needs a built "
                "HybridHardCoreMapping (typically from Soft Core Mapping)."
            )
        # Mirror the model's train/eval flag so the diff kernel is only used
        # in training; eval runs the bit-exact integer path.
        self._flow._chip_aligned_training = bool(self.model.training)
        if self.model.training:
            self._flow.train()
        else:
            self._flow.eval()
        return self._flow(x)


def install_chip_aligned_forward(
    model: nn.Module,
    pipeline_config: dict[str, Any],
    hybrid_mapping,
) -> ChipAlignedForward:
    """Patch ``model.forward`` to route through a chip-aligned flow.

    Asserts no prior instance-level patch (symmetric with ``LIFAdaptationTuner``);
    callers must call :func:`uninstall_chip_aligned_forward` (typically in a
    ``try/finally``) to restore the class forward.
    """
    assert "forward" not in model.__dict__, (
        "ChipAlignedForward: model.forward is already patched; uninstall "
        "the previous wrapper first."
    )
    wrapper = ChipAlignedForward(model, pipeline_config)
    wrapper.refresh(hybrid_mapping)
    model.forward = wrapper
    return wrapper


def uninstall_chip_aligned_forward(model: nn.Module) -> None:
    """Remove the chip-aligned wrapper installed by :func:`install_chip_aligned_forward`."""
    try:
        del model.forward
    except AttributeError:
        pass
