"""Shared harness for cascaded-TTFS calibrate+FT recipes (stable+fast research).

GOAL: convert an already-trained ANN to the GENUINE cascaded single-spike TTFS
deployment quickly and stably, with minimal accuracy loss. Cold genuine cascade is
chance (greedy partial-sum firing death cascade); the analytical staircase forward
is ~lossless (==continuous); the gap is closed by FT through the genuine cascade,
which is UNSTABLE/slow at depth from a cold start. A recipe is a `train(...)` fn.

This module gives recipes clean, correct primitives so they don't re-derive the
(subtle) cascade dynamics:

  genuine_logits(flow, x, S)   -> differentiable GENUINE single-spike cascade logits
  staircase_logits(flow, x)    -> differentiable analytical staircase logits (~cont)
  teacher_logits(base, x)      -> frozen continuous-ANN logits (for KD)
  kd_ce_loss(...)              -> KD + CE blend
  genuine_acc(flow, x, y, S)   -> deployed-metric accuracy (no grad)
  set_S(flow, S) / per-node helpers

Recipe contract (each recipe file `recipe_<name>.py` exports):
    NAME = "<name>"
    def train(flow, xtr, ytr, xva, yva, S, base, teacher, *, steps, seed) -> flow
        # fine-tune `flow` in place (or return a new one); weights start = the
        # cold converted ANN. `base` is the continuous nn.Module (KD teacher),
        # `teacher` is the per-perceptron activation-mean dict. Return the flow
        # whose genuine_acc will be measured.

All on cuda/float (float64 only where the cascade requires .double()). See
[[cascade-lab-use-gpu]].
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_HERE, os.path.join(_HERE, ".."), os.path.join(_REPO, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ft_budget import DEV, build, genuine_acc  # noqa: E402,F401  (re-export)
from cascade_lab import _accuracy  # noqa: E402,F401  (re-export)
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation  # noqa: E402
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward  # noqa: E402


def genuine_logits(flow, x, S):
    """Differentiable genuine single-spike ramp cascade logits (the deployed path)."""
    return TTFSSegmentForward(flow.get_mapper_repr(), S)(x.double())


def staircase_logits(flow, x):
    """Differentiable analytical staircase logits (cycle_accurate OFF) — ~continuous,
    the 'easy' forward that keeps a good basin during a curriculum handoff."""
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(False)
    return flow.double()(x.double())


def teacher_logits(base, x):
    base.eval()
    with torch.no_grad():
        return base(x.float())


def kd_ce_loss(student, y, teacher=None, *, T=3.0, alpha=0.3, label_smooth=0.0):
    """alpha*CE + (1-alpha)*KD(teacher). If teacher is None -> plain CE."""
    ce = F.cross_entropy(student, y, label_smoothing=label_smooth)
    if teacher is None:
        return ce
    kd = F.kl_div(F.log_softmax(student / T, -1), F.softmax(teacher / T, -1),
                  reduction="batchmean") * T * T
    return alpha * ce + (1 - alpha) * kd


def trainable_params(flow):
    return [p for p in flow.parameters() if p.requires_grad]


def batches(xtr, ytr, bs, steps, seed):
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, xtr.shape[0], (bs,), generator=g).to(xtr.device)
        yield xtr[idx].double(), ytr[idx]


def promote_theta_per_channel(flow, requires_grad=False):
    """Per-output-channel activation_scale (encoding layer kept fixed). Returns the
    list of scale params. Use requires_grad=True to co-train theta with weights.

    NOTE: perceptron.set_activation_scale only copies ``.data`` into the existing
    parameter, so it cannot install a NEW trainable param the forward reads. We
    rebind ``activation_scale`` on BOTH the perceptron and its TTFSActivation node
    to the same new Parameter so the optimiser's gradient reaches the forward."""
    scales = []
    for p in flow.get_perceptrons():
        if getattr(p, "is_encoding_layer", False):
            continue
        s = p.activation_scale.detach()
        out_dim = p.layer.weight.shape[0]
        vec = (s * torch.ones(out_dim, dtype=s.dtype, device=s.device)
               if s.dim() == 0 else s.clone())
        param = nn.Parameter(vec, requires_grad=requires_grad)
        p.activation_scale = param
        node = getattr(p, "activation", None)
        if isinstance(node, TTFSActivation):
            node.activation_scale = param
        scales.append(param)
    return scales
