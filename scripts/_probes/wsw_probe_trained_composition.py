"""WS-W probe: per-hop composed error of the TRAINED sync composition (post-WQ artifact).

Measures, on the trained artifact only (validation/calibration data, never test labels):
  1. deployed val/test reads vs the artifact's OWN float-envelope twin
  2. prefix sweep (staircase hops 0..j, float beyond) -> composed-error localization
  3. per-hop first-moment gaps (deployed vs twin preact channel means), in theta/S units
  4. readout per-class logit bias (deployed - twin)
  5. candidate folds applied in-memory + val reads (evidence for the landing decision)
"""
import os
import sys
import copy

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path[:0] = ["./src", "./spikingjelly"]

import torch
import torch.nn as nn

RUN = sys.argv[1] if len(sys.argv) > 1 else "generated/wsw_t0_21_base_phased_deployment_run"
ENTRY = sys.argv[2] if len(sys.argv) > 2 else "Weight Quantization.model"
DEV = "cuda:0"
S = 8

from mimarsinan.pipelining.session import apply_determinism

apply_determinism(0)

model, _dev = torch.load(f"{RUN}/{ENTRY}.pt", map_location="cpu", weights_only=False)
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
print(f"val {tuple(val_x.shape)} test {tuple(test_x.shape)}", flush=True)

perceptrons = list(model.get_perceptrons())
from mimarsinan.spiking.sync_first_moment import perceptron_forward_order

order = perceptron_forward_order(model)
print(f"perceptrons: {len(perceptrons)}, forward order {order}")
for k in order:
    p = perceptrons[k]
    theta = torch.as_tensor(p.activation_scale).reshape(-1)
    print(
        f"  hop k={k} name={getattr(p, 'name', '?')} "
        f"w{tuple(p.layer.weight.shape)} theta={theta[:3].tolist()}(n={theta.numel()}) "
        f"enc={getattr(p, 'is_encoding_layer', False)} "
        f"act={type(p.activation).__name__} "
        f"in_act={type(p.input_activation).__name__} "
        f"pscale={float(p.parameter_scale):.4f} bscale={float(p.bias_scale):.4f}"
    )


def _theta(p, like):
    th = torch.as_tensor(p.activation_scale, device=like.device, dtype=like.dtype)
    return th.clamp(min=1e-12)


def stair_bypass_handles(float_hops):
    """Bypass the staircase (clamp envelope) + input grid snap on the given hop indices."""
    handles = []
    for k in float_hops:
        p = perceptrons[k]

        def bypass(_m, inp, _out, p=p):
            z = inp[0]
            th = _theta(p, z)
            if th.numel() > 1:
                shape = [1] * z.dim()
                axis = getattr(p, "output_channel_axis", -1)
                shape[axis % z.dim()] = th.numel()
                th = th.reshape(shape)
            return torch.minimum(z.clamp(min=0.0), th)

        handles.append(p.activation.register_forward_hook(bypass))
        if not isinstance(p.input_activation, nn.Identity):
            handles.append(
                p.input_activation.register_forward_hook(lambda _m, inp, _o: inp[0])
            )
    return handles


@torch.no_grad()
def forward_logits(x, float_hops=()):
    handles = stair_bypass_handles(float_hops)
    try:
        outs = []
        for i in range(0, x.shape[0], 1024):
            outs.append(model(x[i : i + 1024].to(DEV)).float().cpu())
        return torch.cat(outs)
    finally:
        for h in handles:
            h.remove()


def acc(logits, y):
    return float((logits.argmax(-1) == y).float().mean())


all_hops = list(order)
dep_val = forward_logits(val_x)
twin_val = forward_logits(val_x, float_hops=all_hops)
print(f"\n[reads] deployed val {acc(dep_val, val_y):.4f} | float-twin val {acc(twin_val, val_y):.4f}")
dep_test = forward_logits(test_x)
twin_test = forward_logits(test_x, float_hops=all_hops)
print(f"[reads] deployed TEST {acc(dep_test, test_y):.4f} | float-twin TEST {acc(twin_test, test_y):.4f} (report-only)")

print("\n[prefix sweep] staircase on hops <= j (forward order), float beyond:")
for j in range(len(all_hops) + 1):
    float_hops = all_hops[j:]
    a = acc(forward_logits(val_x, float_hops=float_hops), val_y)
    print(f"  stair prefix {j:2d}/{len(all_hops)} -> val {a:.4f}")

print("\n[single-hop] ONLY hop j staircased (others float):")
for j, k in enumerate(all_hops):
    float_hops = [h for h in all_hops if h != k]
    a = acc(forward_logits(val_x, float_hops=float_hops), val_y)
    print(f"  only hop {j:2d} (k={k}) stair -> val {a:.4f}")

print("\n[single-hop-off] ONLY hop j float (others staircased):")
for j, k in enumerate(all_hops):
    a = acc(forward_logits(val_x, float_hops=[k]), val_y)
    print(f"  only hop {j:2d} (k={k}) float -> val {a:.4f}")

# per-hop first-moment gaps: deployed vs twin preacts
from mimarsinan.spiking.dfq_bias_correction import (
    _effective_bias_shift,
    perceptron_channel_mean,
    perceptron_preactivation_samples,
)

CAL_N = 3000
cal_x = val_x[:CAL_N].to(DEV)

with torch.no_grad():
    dep_pre = perceptron_preactivation_samples(model, cal_x)
handles = stair_bypass_handles(all_hops)
try:
    with torch.no_grad():
        twin_pre = perceptron_preactivation_samples(model, cal_x)
finally:
    for h in handles:
        h.remove()

print("\n[per-hop first moments] mean over channels of (dep - twin) preact, theta/S units:")
print(f"{'hop':>4} {'k':>3} {'mean(d)':>10} {'mean|d|':>10} {'in grid-steps':>14} {'starved%':>9}")
for j, k in enumerate(all_hops):
    p = perceptrons[k]
    dm = perceptron_channel_mean(p, dep_pre[k])
    tm = perceptron_channel_mean(p, twin_pre[k])
    th = float(torch.as_tensor(p.activation_scale).reshape(-1).mean())
    d = (dm - tm)
    step = th / S
    # starved: channel deployed preact mean <= 0 while twin mean > 0.01*theta
    starved = float(((dm <= 0) & (tm > 0.01 * th)).float().mean()) * 100
    print(
        f"{j:>4} {k:>3} {float(d.mean()):>10.5f} {float(d.abs().mean()):>10.5f} "
        f"{float(d.mean()) / step:>14.3f} {starved:>8.1f}%"
    )

# the TRUE readout is the host-side classifier Linear (ComputeOpMapper), float and un-gridded
classifier = None
for node in model.get_mapper_repr().execution_order():
    if getattr(node, "name", "") == "classifier":
        classifier = node.module
while hasattr(classifier, "module"):
    classifier = classifier.module
assert isinstance(classifier, nn.Linear), type(classifier)
print(f"\n[readout] host classifier {classifier} (float, un-gridded)")

print("[readout] per-class mean logit delta (deployed - twin) on val:")
delta = (dep_val - twin_val).mean(0)
print("  delta:", [f"{v:.4f}" for v in delta.tolist()])
print("  |delta| mean:", float(delta.abs().mean()))

# per-class prediction marginals vs the true prior (label-free systematics)
pred_rate = torch.bincount(dep_val.argmax(-1), minlength=10).float() / dep_val.shape[0]
true_rate = torch.bincount(val_y, minlength=10).float() / val_y.shape[0]
twin_rate = torch.bincount(twin_val.argmax(-1), minlength=10).float() / twin_val.shape[0]
print("  deployed pred rates:", [f"{v:.3f}" for v in pred_rate.tolist()])
print("  twin     pred rates:", [f"{v:.3f}" for v in twin_rate.tolist()])
print("  true          rates:", [f"{v:.3f}" for v in true_rate.tolist()])
per_class_recall = torch.zeros(10)
for c in range(10):
    m = val_y == c
    per_class_recall[c] = float((dep_val[m].argmax(-1) == c).float().mean())
print("  deployed per-class recall:", [f"{v:.3f}" for v in per_class_recall.tolist()])

# ---- candidate folds (in-memory, keep-best style evidence) ----

state0 = copy.deepcopy(model.state_dict())
cls_w0 = classifier.weight.detach().clone()
cls_b0 = classifier.bias.detach().clone()


def restore():
    model.load_state_dict(state0)
    with torch.no_grad():
        classifier.weight.data.copy_(cls_w0)
        classifier.bias.data.copy_(cls_b0)


FIT_N = 2000  # fit on val[:2000], guard/report on the held-out val[2000:]


def val_read():
    return acc(forward_logits(val_x), val_y)


def held_read():
    return acc(forward_logits(val_x[FIT_N:]), val_y[FIT_N:])


def test_read():
    return acc(forward_logits(test_x), test_y)


print(f"\n[candidates] entry val {val_read():.4f} heldout {held_read():.4f} TEST {test_read():.4f}")

# C1: readout per-class bias fold toward the twin (the named identity-fold candidate)
d = delta.to(classifier.bias.device, classifier.bias.dtype)
with torch.no_grad():
    classifier.bias.data -= d
print(f"  C1 readout bias fold toward twin    -> val {val_read():.4f} heldout {held_read():.4f}")
restore()

# C4: label-free prior-matching bias (deployed prediction marginal -> class prior),
# fixed-point iteration on the FIT slice of the calibration inputs only
with torch.no_grad():
    logits = dep_val[:FIT_N].clone()
    fit_rate = torch.bincount(val_y[:FIT_N], minlength=10).float() / FIT_N
    b_adj = torch.zeros(10)
    for _ in range(200):
        rates = torch.bincount((logits + b_adj).argmax(-1), minlength=10).float() / logits.shape[0]
        gap = fit_rate - rates
        if float(gap.abs().max()) < 5e-4:
            break
        b_adj += 0.5 * gap * 10.0
    classifier.bias.data += b_adj.to(classifier.bias.device, classifier.bias.dtype)
print(f"  C4 prior-matching bias (label-free) -> val {val_read():.4f} heldout {held_read():.4f} (|b| mean {float(b_adj.abs().mean()):.4f})")
restore()

# C5: closed-form ridge refit of the host classifier on DEPLOYED features
# (fit slice of validation only; labels are calibration labels, never test)
feats = []


def _cap(_m, inp, _o):
    feats.append(inp[0].detach().float().cpu())


h = classifier.register_forward_hook(_cap)
try:
    _ = forward_logits(val_x)
finally:
    h.remove()
X = torch.cat(feats)[:FIT_N].double()
Y = torch.nn.functional.one_hot(val_y[:FIT_N], 10).double()
Xa = torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64)], dim=1)
for lam in (1e-3, 1e-1, 1.0, 10.0):
    P = torch.linalg.solve(
        Xa.T @ Xa + lam * torch.eye(Xa.shape[1], dtype=torch.float64), Xa.T @ Y
    )
    with torch.no_grad():
        classifier.weight.data.copy_(P[:-1].T.to(classifier.weight.dtype))
        classifier.bias.data.copy_(P[-1].to(classifier.bias.dtype))
    print(f"  C5 ridge readout refit lam={lam:<5} -> val {val_read():.4f} heldout {held_read():.4f}")
    restore()

# C5b: ridge refit BLENDED with the trained readout (residual-limited refit)
lam = 1.0
P = torch.linalg.solve(
    Xa.T @ Xa + lam * torch.eye(Xa.shape[1], dtype=torch.float64), Xa.T @ Y
)
# scale-match: trained logits ~ theta-scale, one-hot fit ~ [0,1] -> compare argmax only;
# blend in the trained readout's logit scale via least-squares alignment
with torch.no_grad():
    trained_logits = dep_val[:FIT_N].double()
    refit_logits = Xa @ P
    s = float((trained_logits * refit_logits).sum() / (refit_logits * refit_logits).sum())
P_dev = P.to(cls_w0.device)
for mix in (0.25, 0.5):
    with torch.no_grad():
        w_new = (1 - mix) * cls_w0.double() + mix * s * P_dev[:-1].T
        b_new = (1 - mix) * cls_b0.double() + mix * s * P_dev[-1]
        classifier.weight.data.copy_(w_new.to(classifier.weight.dtype))
        classifier.bias.data.copy_(b_new.to(classifier.bias.dtype))
    print(f"  C5b blend mix={mix:<4} (s={s:.3f})    -> val {val_read():.4f} heldout {held_read():.4f}")
    restore()

# C2: trained-composition sequential first-moment fold toward the artifact's own twin
# (twin reference includes the baked half-steps; fold measured through the already-folded prefix)
print("  C2 sequential per-hop first-moment fold (float), greedy keep-best per hop:")
best = val_read()
kept = []
for j, k in enumerate(all_hops):
    p = perceptrons[k]
    with torch.no_grad():
        cur = perceptron_preactivation_samples(model, cal_x, indices=(k,))[k]
    cm = perceptron_channel_mean(p, cur)
    tm = perceptron_channel_mean(p, twin_pre[k]).to(cm.device)
    gap = (cm - tm)
    snap = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
    bias = p.layer.bias
    n = min(gap.numel(), bias.numel())
    full = torch.zeros(bias.numel(), dtype=gap.dtype, device=gap.device)
    full[:n] = gap[:n]
    _effective_bias_shift(p, full)
    r = val_read()
    tag = "KEEP" if r >= best else "ROLLBACK"
    print(f"    hop {j:2d} (k={k}): fold mean|gap|={float(gap.abs().mean()):.5f} -> val {r:.4f} [{tag}]")
    if r >= best:
        best = r
        kept.append(j)
    else:
        model.load_state_dict(snap)
print(f"    kept hops {kept}, final val {best:.4f}")
restore()

# C3: C2 without guard (fold all hops sequentially)
for j, k in enumerate(all_hops):
    p = perceptrons[k]
    with torch.no_grad():
        cur = perceptron_preactivation_samples(model, cal_x, indices=(k,))[k]
    cm = perceptron_channel_mean(p, cur)
    tm = perceptron_channel_mean(p, twin_pre[k]).to(cm.device)
    gap = (cm - tm)
    bias = p.layer.bias
    n = min(gap.numel(), bias.numel())
    full = torch.zeros(bias.numel(), dtype=gap.dtype, device=gap.device)
    full[:n] = gap[:n]
    _effective_bias_shift(p, full)
print(f"  C3 fold-all-hops (no guard)       -> val {val_read():.4f}")
restore()

print(f"\n[exit] restored entry val read {val_read():.4f}")
