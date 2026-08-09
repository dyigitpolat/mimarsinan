"""ChipAlignedNFForward install machinery: pickle seam and stage handoff."""

from __future__ import annotations

import pytest
import torch.nn as nn


def test_pickle_alias_survives_the_move():
    from mimarsinan.tuning.forward_install import ChipAlignedNFForward
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import (
        _ChipAlignedNFForward,
    )

    assert _ChipAlignedNFForward is ChipAlignedNFForward


def test_stage_handoff_replaces_inherited_patch_but_not_own():
    from mimarsinan.tuning.forward_install import CascadeForwardInstall

    class _Tuner(CascadeForwardInstall):
        def __init__(self, model):
            self.model = model

    model = nn.Linear(2, 2)
    model.forward = "inherited-stage-patch"
    t = _Tuner(model)
    t._install_forward("own-patch")           # handoff: replaces inherited
    assert model.__dict__["forward"] == "own-patch"
    with pytest.raises(AssertionError):
        t._install_forward("double-patch")    # within-owner still loud
