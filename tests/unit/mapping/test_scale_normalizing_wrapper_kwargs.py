"""[calculus §16.10] the wrapper accepts call-site module kwargs (IR executor
pattern): constructor-owned kwargs win on collision; extras pass through."""

import torch
import torch.nn as nn

from mimarsinan.mapping.support.compute_modules import ScaleNormalizingWrapper


class _KwargSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x, *, need_weights=True, scale_hint=None):
        self.seen = {"need_weights": need_weights, "scale_hint": scale_hint}
        return x * 2.0


def test_call_site_kwargs_win_and_owned_fill_the_gaps():
    spy = _KwargSpy()
    w = ScaleNormalizingWrapper(
        spy, [torch.tensor([2.0])], torch.tensor([1.0]),
        module_kwargs={"need_weights": False},
    )
    x = torch.ones(2, 3)
    out = w(x, need_weights=True, scale_hint="ir")
    assert spy.seen == {"need_weights": True, "scale_hint": "ir"}
    w(x)
    assert spy.seen == {"need_weights": False, "scale_hint": None}
    torch.testing.assert_close(out, torch.full((2, 3), 4.0))
