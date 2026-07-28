"""GradualWarmupScheduler passes an epoch arg modern SequentialLR rejects."""

import pytest
import torch

from mimarsinan.model_training.training_utilities import EpochArgTolerantScheduler


def _sequential(lr=0.1, total=6):
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=lr)
    warm = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-3, end_factor=1.0, total_iters=3
    )
    main = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    return torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warm, main], milestones=[3]
    )


class TestEpochArgTolerance:
    def test_raw_sequential_rejects_the_epoch_arg(self):
        # The defect this shim exists for (guards against a torch change
        # silently making the shim unnecessary but still correct).
        with pytest.raises(TypeError):
            _sequential().step(None)

    def test_shim_accepts_the_none_epoch_arg(self):
        shim = EpochArgTolerantScheduler(_sequential())
        shim.step(None)  # must not raise
        shim.step()

    def test_trajectory_matches_the_unwrapped_scheduler(self):
        raw, wrapped = _sequential(), _sequential()
        shim = EpochArgTolerantScheduler(wrapped)
        raw_lrs, shim_lrs = [], []
        for _ in range(8):
            raw.step()
            shim.step(None)
            raw_lrs.append(raw.get_last_lr()[0])
            shim_lrs.append(shim.get_last_lr()[0])
        assert raw_lrs == shim_lrs

    def test_explicit_epoch_fails_loud(self):
        # Epoch-indexed stepping has no modern SequentialLR equivalent; a
        # silent reinterpretation would move the LR curve.
        shim = EpochArgTolerantScheduler(_sequential())
        with pytest.raises(ValueError, match="epoch-indexed"):
            shim.step(3)

    def test_attributes_proxy_to_the_wrapped_scheduler(self):
        inner = _sequential()
        shim = EpochArgTolerantScheduler(inner)
        shim.step(None)
        assert shim.last_epoch == inner.last_epoch
        assert shim.get_last_lr() == inner.get_last_lr()
        assert shim.optimizer is inner.optimizer


class TestEveryConstructionSiteIsWrapped:
    """The shim only helps where it is APPLIED: a partial edit left two of
    four GradualWarmupScheduler sites bare and t1_10 failed identically."""

    def test_no_bare_after_scheduler_argument(self):
        import re
        from pathlib import Path

        import mimarsinan.model_training as pkg

        offenders = []
        for path in Path(pkg.__file__).parent.glob("*.py"):
            for match in re.finditer(r"after_scheduler=([A-Za-z_][\w.]*)",
                                     path.read_text()):
                if match.group(1) != "EpochArgTolerantScheduler":
                    offenders.append(f"{path.name}: after_scheduler={match.group(1)}")
        assert not offenders, (
            f"unwrapped warmup after_scheduler sites (modern SequentialLR "
            f"rejects the epoch arg): {offenders}"
        )

