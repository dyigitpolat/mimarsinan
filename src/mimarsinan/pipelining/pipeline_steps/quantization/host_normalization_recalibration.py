"""[U-POST G1] The host normalizations re-estimated at the end of Weight Quantization.

The conversions change what every host op receives: the activation staircase, the LIF swap and the weight projection
each move the values a host op's normalization sees, and Weight Quantization freezes only the projected perceptrons'
normalizations. Every other normalization (a host op's BatchNorm: a residual branch's, a shortcut's) kept the statistics
it was trained with, which no longer describe the values it normalizes (on residual networks whose host normalizations
follow converted cores, a loss of several points that no training step recovers).

At the end of the step, on its final weights, each such normalization's running statistics are reset and re-estimated as
the plain average over a declared number of training-split batches, through the model's own forward with only those
normalizations in train mode: no optimizer, no parameter moves. The result is kept only if the calibration read does not
fall; otherwise every statistic is restored. The integer program is untouched: only float host-op statistics move, and
the twins and the deployed program read the same host modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.membrane_lattice import integer_program_read

HOST_NORMALIZATION_BATCHES_KEY = "wq_host_normalization_batches"
#: The documented default: 60 training batches (the images drawn scale with the run's training batch size).
DEFAULT_HOST_NORMALIZATION_BATCHES = 60

__all__ = [
    "DEFAULT_HOST_NORMALIZATION_BATCHES",
    "HOST_NORMALIZATION_BATCHES_KEY",
    "HostNormalizationReport",
    "declared_host_normalization_batches",
    "host_normalizations",
    "reestimate_at_step_end",
    "reestimate_host_normalizations",
]


@dataclass(frozen=True)
class HostNormalizationReport:
    """One re-estimation: the normalizations it covered, the batches it drew, the calibration read before and after,
    and whether the result was kept."""

    modules: int
    batches: int
    before: float
    after: float
    kept: bool

    def line(self) -> str:
        verdict = "kept" if self.kept else "restored (the calibration read fell)"
        return (f"[WQ HOST NORMALIZATION] {self.modules} host normalization(s) re-estimated over {self.batches} "
                f"training batch(es): calibration read {self.before:.6f} -> {self.after:.6f}, {verdict}")

    def as_dict(self) -> dict:
        return asdict(self)


def declared_host_normalization_batches(config) -> int:
    """The declared batch count (``wq_host_normalization_batches``), else the documented default; 0 runs none."""
    return max(0, int(config.get(HOST_NORMALIZATION_BATCHES_KEY, DEFAULT_HOST_NORMALIZATION_BATCHES)))


def host_normalizations(model) -> list:
    """``(name, module)`` for every normalization with running statistics outside the model's perceptrons."""
    owned = {id(m) for p in model.get_perceptrons() for m in p.modules()}
    return [(name, m) for name, m in model.named_modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats
            and m.running_mean is not None and id(m) not in owned]


def reestimate_host_normalizations(model, batches: Iterable[torch.Tensor], *, calibration_read: Callable[[], float],
                                   device) -> HostNormalizationReport:
    """Re-estimate every host normalization of ``model`` over ``batches`` (inputs), kept only if
    ``calibration_read()`` does not fall. The CPU random state is preserved (the batches drawn advance their
    iterator)."""
    norms = [m for _, m in host_normalizations(model)]
    if not norms:
        return HostNormalizationReport(0, 0, float("nan"), float("nan"), False)
    before = float(calibration_read())
    saved = [(m, {k: v.detach().clone() for k, v in m.state_dict().items()}, m.momentum) for m in norms]
    modes = [(m, m.training) for m in model.modules()]
    drawn = 0
    model.eval()
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # the plain average over the drawn batches
        m.train()
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            for x in batches:
                model(x.to(device))
                drawn += 1
    finally:
        for m, _, momentum in saved:
            m.momentum = momentum
            m.eval()
    after = float(calibration_read()) if drawn else before
    kept = drawn > 0 and after >= before
    if not kept:
        for m, state, _ in saved:
            m.load_state_dict(state)
    for m, mode in modes:
        m.training = mode
    return HostNormalizationReport(len(norms), drawn, before, after, kept)


def reestimate_at_step_end(model, tuner, pipeline, *, calibration_read: Callable[[], float]):
    """Weight Quantization's end: re-estimate over the declared batches the tuner's trainer draws from the training
    split, in the plane the step's read runs in (an integer program's measurement plane), printed and reported. None
    when nothing is declared, no trainer serves batches or the model has no host normalization."""
    count, trainer = declared_host_normalization_batches(pipeline.config), getattr(tuner, "trainer", None)
    if not count or trainer is None or not host_normalizations(model):
        return None
    with integer_program_read(model):
        report = reestimate_host_normalizations(model, (trainer.next_training_batch()[0] for _ in range(count)),
                                                calibration_read=calibration_read, device=pipeline.config["device"])
    print(report.line(), flush=True)
    pipeline.reporter.report("wq_host_normalization", report.as_dict())
    return report
