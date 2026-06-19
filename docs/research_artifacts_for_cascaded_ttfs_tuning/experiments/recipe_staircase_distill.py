"""Self-distill the genuine cascade toward its OWN complete-sum staircase output.

The structural residual is the greedy partial-sum firing: the genuine output drifts
from the (~lossless) staircase output of the SAME weights. Directly penalise that
drift — target = staircase_logits(flow,x).detach() — so FT is explicitly told "make
your greedy-firing output equal your complete-sum output", on top of CE + teacher KD.
A near-lossless-targeted objective the plain output KD lacks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from recipe_harness import (
    batches, genuine_logits, kd_ce_loss, staircase_logits, teacher_logits,
    trainable_params,
)

NAME = "staircase_distill"


def train(flow, xtr, ytr, xva, yva, S, base, teacher, *, steps, seed,
          lr=2e-3, bs=256, beta=1.0, T=3.0):
    opt = torch.optim.Adam(trainable_params(flow), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for x, y in batches(xtr, ytr, bs, steps, seed):
        stair = staircase_logits(flow, x).detach()      # complete-sum target (this network's lossless mode)
        gen = genuine_logits(flow, x, S)
        distill = F.kl_div(F.log_softmax(gen / T, -1), F.softmax(stair / T, -1),
                           reduction="batchmean") * T * T
        loss = kd_ce_loss(gen, y, teacher_logits(base, x), alpha=0.3) + beta * distill
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return flow
