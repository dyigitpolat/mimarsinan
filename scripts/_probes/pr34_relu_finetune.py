"""PR34 (calculus §15.16): is the origin->analytic (GELU->clamped-ReLU) gap
BUDGET-limited or a family floor? Proper multi-epoch fine-tune of the origin
backbone with the deployable LIF value-twin target, KD-to-origin, full
trainset, keep-best on analytic. >=~0.86 => budget (recipe fix); plateau => floor.
"""
import copy
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["MIMARSINAN_DISABLE_FFCV"] = "1"
sys.path[:0] = ["./src"]

import torch
import torch.nn.functional as F

torch.backends.cudnn.enabled = False
DEV = os.environ.get("PR34_DEV", "cuda:0")
T = 32
EPOCHS = float(os.environ.get("PR34_EPOCHS", "6"))
LR = float(os.environ.get("PR34_LR", "1e-4"))
BS = int(os.environ.get("PR34_BS", "32"))
ALPHA = float(os.environ.get("PR34_ALPHA", "0.5"))
TEMP = float(os.environ.get("PR34_TEMP", "4.0"))
RUN = "generated/t2_04_aaorigin_ab6_phased_deployment_run"
SAVE = os.environ.get("PR34_SAVE", "")

from mimarsinan.pipelining.session import apply_determinism

apply_determinism(0)

model, _ = torch.load(
    f"{RUN}/Reference Teacher Snapshot.reference_teacher_model.pt",
    map_location="cpu", weights_only=False,
)
model = model.to(DEV)
for p in model.parameters():
    p.requires_grad_(True)
teacher = copy.deepcopy(model).eval()
for p in teacher.parameters():
    p.requires_grad_(False)

import mimarsinan.data_handling.data_providers  # noqa: F401
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

provider = BasicDataProviderFactory(
    "CIFAR100_DataProvider", "./datasets", seed=0, batch_size=BS,
    preprocessing={"interpolation": "bicubic", "resize_to": 224, "normalize": "imagenet"},
).create()
import torch.utils.data as D

val = list(D.DataLoader(provider._get_validation_dataset(), batch_size=64))
train_ds = provider._get_training_dataset()
steps_per_epoch = len(train_ds) // BS
STEPS = int(EPOCHS * steps_per_epoch)

import torch.nn as nn


class ClampReLU(nn.Module):
    """The deployable value transfer: clamp(z, 0, theta) — the T->inf LIF rate
    limit. Fast (no 32-step IF), isolates the GELU->clamped-ReLU artifact axis
    from the separate grid axis."""

    def __init__(self, theta: float):
        super().__init__()
        self.theta = float(theta)

    def forward(self, x):
        return x.clamp(min=0.0, max=self.theta)


perceptrons = list(model.get_perceptrons())
# theta = per-perceptron max over a calibration pass (activation_scale_quantile=1.0).
cal = list(D.DataLoader(train_ds, batch_size=64, shuffle=False))[:8]
pre = {k: 0.0 for k in range(len(perceptrons))}
hs = []
for k, p in enumerate(perceptrons):
    def _h(_m, inp, _o, _k=k):
        pre[_k] = max(pre[_k], float(inp[0].detach().abs().max()))
    hs.append(p.activation.register_forward_hook(_h))
with torch.no_grad():
    for x, _ in cal:
        model(x.to(DEV))
for h in hs:
    h.remove()
for k, p in enumerate(perceptrons):
    p.activation = ClampReLU(max(pre[k], 1e-6)).to(DEV)


def analytic(n):
    c = t = 0
    with torch.no_grad():
        for x, y in val[:n]:
            x, y = x.to(DEV), y.to(DEV)
            c += int((model(x).argmax(1) == y).sum())
            t += int(y.numel())
    return c / t


entry = analytic(8)
print(f"[PR34] steps={STEPS} ({EPOCHS} ep) lr={LR} | entry analytic={entry:.4f} "
      f"(origin GELU 0.8678)", flush=True)

opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                        lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR * 0.05)
best = [entry, copy.deepcopy(model.state_dict())]
loader = D.DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=4,
                      drop_last=True, persistent_workers=True)
model.train()
t0 = time.time()
step = 0
done = False
import math
for _epoch in range(math.ceil(EPOCHS) + 1):
    if done:
        break
    for x, y in loader:
        if step >= STEPS:
            done = True
            break
        x, y = x.to(DEV), y.to(DEV)
        with torch.no_grad():
            tl = teacher(x)
        sl = model(x)
        ce = F.cross_entropy(sl, y)
        kd = F.kl_div(F.log_softmax(sl / TEMP, -1), F.softmax(tl / TEMP, -1),
                      reduction="batchmean") * (TEMP * TEMP)
        loss = (1 - ALPHA) * ce + ALPHA * kd
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 300 == 0:
            model.eval()
            a = analytic(8)
            model.train()
            if a > best[0]:
                best = [a, copy.deepcopy(model.state_dict())]
            print(f"[PR34] step {step}/{STEPS}: analytic(512)={a:.4f} "
                  f"best={best[0]:.4f} loss={float(loss.detach()):.3f} "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)

model.load_state_dict(best[1])
model.eval()
census = analytic(len(val))
print(f"[PR34] BEST analytic census (n={64*len(val)}): {census:.4f} "
      f"(entry {entry:.4f}, origin 0.8678)", flush=True)
if SAVE:
    torch.save(model.state_dict(), SAVE)
    print(f"[PR34] saved -> {SAVE}", flush=True)
print("PR34-DONE", flush=True)
