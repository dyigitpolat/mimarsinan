"""WS-W probe 2: readout recalibration candidates on the post-WQ trained artifact."""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path[:0] = ["./src", "./spikingjelly"]

import torch
import torch.nn as nn

RUN = sys.argv[1] if len(sys.argv) > 1 else "generated/wsw_t0_21_base_phased_deployment_run"
DEV = "cuda:0"

from mimarsinan.pipelining.session import apply_determinism

apply_determinism(0)

model, _dev = torch.load(f"{RUN}/Weight Quantization.model.pt", map_location="cpu", weights_only=False)
model = model.to(DEV).eval()

import mimarsinan.data_handling.data_providers  # noqa: F401
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

factory = BasicDataProviderFactory("MNIST_DataProvider", "./datasets", seed=0, batch_size=128)
provider = factory.create()


def _tensors(ds):
    loader = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


val_x, val_y = _tensors(provider._get_validation_dataset())
test_x, test_y = _tensors(provider._get_test_dataset())

classifier = None
for node in model.get_mapper_repr().execution_order():
    if getattr(node, "name", "") == "classifier":
        classifier = node.module
while hasattr(classifier, "module"):
    classifier = classifier.module
assert isinstance(classifier, nn.Linear)

cls_w0 = classifier.weight.detach().clone()
cls_b0 = classifier.bias.detach().clone()


def restore():
    with torch.no_grad():
        classifier.weight.data.copy_(cls_w0)
        classifier.bias.data.copy_(cls_b0)


@torch.no_grad()
def forward_logits(x):
    outs = []
    for i in range(0, x.shape[0], 1024):
        outs.append(model(x[i : i + 1024].to(DEV)).float().cpu())
    return torch.cat(outs)


def acc(logits, y):
    return float((logits.argmax(-1) == y).float().mean())


FIT_N = 2000

# cache features+logits once (readout-input features via hook)
feats = []
h = classifier.register_forward_hook(lambda _m, inp, _o: feats.append(inp[0].detach().float().cpu()))
try:
    val_logits0 = forward_logits(val_x)
    n_val_feats = len(feats)
    test_logits0 = forward_logits(test_x)
finally:
    h.remove()
val_F = torch.cat(feats[:n_val_feats])
test_F = torch.cat(feats[n_val_feats:])
assert val_F.shape[0] == val_x.shape[0] and test_F.shape[0] == test_x.shape[0]


def reads(w=None, b=None):
    """(val, heldout, test) reads from cached features under readout (w, b)."""
    w = cls_w0.cpu() if w is None else w.cpu()
    b = cls_b0.cpu() if b is None else b.cpu()
    lv = val_F @ w.T + b
    lt = test_F @ w.T + b
    return acc(lv, val_y), acc(lv[FIT_N:], val_y[FIT_N:]), acc(lt, test_y)


e_v, e_h, e_t = reads()
print(f"entry               val {e_v:.4f} heldout {e_h:.4f} TEST {e_t:.4f}")
sv = torch.equal((val_F @ cls_w0.cpu().T + cls_b0.cpu()).argmax(-1), val_logits0.argmax(-1))
print(f"feature-cache consistency with live forward: {sv}")


def prior_match_bias(logits, rates, iters=300, eta=5.0):
    b = torch.zeros(10)
    for _ in range(iters):
        pred = torch.bincount((logits + b).argmax(-1), minlength=10).float() / logits.shape[0]
        gap = rates - pred
        if float(gap.abs().max()) < 5e-4:
            break
        b += eta * gap
    return b


def ridge(X, y, lam):
    Y = torch.nn.functional.one_hot(y, 10).double()
    Xa = torch.cat([X.double(), torch.ones(X.shape[0], 1, dtype=torch.float64)], dim=1)
    P = torch.linalg.solve(Xa.T @ Xa + lam * torch.eye(Xa.shape[1], dtype=torch.float64), Xa.T @ Y)
    return P[:-1].T.float(), P[-1].float()


# C4 prior matching on fit slice / full val
for tag, sl in (("fit2000", slice(None, FIT_N)), ("full3000", slice(None))):
    rates = torch.bincount(val_y[sl], minlength=10).float() / val_y[sl].shape[0]
    b_adj = prior_match_bias(val_F[sl] @ cls_w0.cpu().T + cls_b0.cpu(), rates)
    v, hh, t = reads(b=cls_b0.cpu() + b_adj)
    print(f"C4 prior-match {tag:9} val {v:.4f} heldout {hh:.4f} TEST {t:.4f} (|b| {float(b_adj.abs().mean()):.3f})")

# C5 ridge on fit slice / full val
for tag, sl in (("fit2000", slice(None, FIT_N)), ("full3000", slice(None))):
    for lam in (1.0, 10.0):
        W, B = ridge(val_F[sl], val_y[sl], lam)
        v, hh, t = reads(w=W, b=B)
        print(f"C5 ridge {tag:9} lam={lam:<4} val {v:.4f} heldout {hh:.4f} TEST {t:.4f}")

# C6 ridge then prior-match (composed), fit slice
W, B = ridge(val_F[:FIT_N], val_y[:FIT_N], 1.0)
rates = torch.bincount(val_y[:FIT_N], minlength=10).float() / FIT_N
b_adj = prior_match_bias(val_F[:FIT_N] @ W.T + B, rates)
v, hh, t = reads(w=W, b=B + b_adj)
print(f"C6 ridge+prior fit2000     val {v:.4f} heldout {hh:.4f} TEST {t:.4f}")

# C7 greedy per-class bias coordinate ascent on fit-slice accuracy
logits_fit = val_F[:FIT_N] @ cls_w0.cpu().T + cls_b0.cpu()
b = torch.zeros(10)
best_fit = acc(logits_fit, val_y[:FIT_N])
step_sizes = (0.4, 0.2, 0.1, 0.05)
for step in step_sizes:
    improved = True
    while improved:
        improved = False
        for c in range(10):
            for sgn in (+1.0, -1.0):
                cand = b.clone()
                cand[c] += sgn * step
                a = acc(logits_fit + cand, val_y[:FIT_N])
                if a > best_fit:
                    b = cand
                    best_fit = a
                    improved = True
v, hh, t = reads(b=cls_b0.cpu() + b)
print(f"C7 coord-ascent bias fit2000 val {v:.4f} heldout {hh:.4f} TEST {t:.4f} (fit {best_fit:.4f}, |b| {float(b.abs().mean()):.3f})")

# C8: C4 on full val computed from TRUE prior assumption (uniform 0.1)
b_adj = prior_match_bias(val_F @ cls_w0.cpu().T + cls_b0.cpu(), torch.full((10,), 0.1))
v, hh, t = reads(b=cls_b0.cpu() + b_adj)
print(f"C8 prior-match uniform     val {v:.4f} heldout {hh:.4f} TEST {t:.4f} (|b| {float(b_adj.abs().mean()):.3f})")
