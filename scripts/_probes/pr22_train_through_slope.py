"""PR22 (calculus §14.6): the train-THROUGH slope — bounded recovery of the
genuine composition with KD-to-origin, gradients through the per-cycle
surrogate. Probe-grade (no session/cache overhead) so it fits one window.
"""
import copy
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path[:0] = ["./src"]

import torch
import torch.nn.functional as F

RUN = sys.argv[1] if len(sys.argv) > 1 else (
    "generated/t2_04_origingate_ab2_phased_deployment_run"
)
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 180
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4
DEV = os.environ.get("PR22_DEV", "cuda:0")
T = 32
BS_TRAIN = int(os.environ.get("PR22_BS", "16"))
ACCUM = int(os.environ.get("PR22_ACCUM", "1"))
KD_A = float(os.environ.get("PR22_ALPHA", "0.5"))
KD_T = float(os.environ.get("PR22_TEMP", "4.0"))
DITHER = bool(int(os.environ.get("PR22_DITHER", "0")))
V0 = float(os.environ.get("PR22_V0", "0.0"))

from mimarsinan.pipelining.session import apply_determinism

apply_determinism(0)

model, _ = torch.load(
    f"{RUN}/Activation Quantization.model.pt", map_location="cpu", weights_only=False
)
model = model.to(DEV)
teacher, _ = torch.load(
    f"{RUN}/Reference Teacher Snapshot.reference_teacher_model.pt",
    map_location="cpu", weights_only=False,
)
teacher = teacher.to(DEV).eval()
for p in teacher.parameters():
    p.requires_grad_(False)

import mimarsinan.data_handling.data_providers  # noqa: F401
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

provider = BasicDataProviderFactory(
    "CIFAR100_DataProvider", "./datasets", seed=0, batch_size=64,
    preprocessing={"interpolation": "bicubic", "resize_to": 224, "normalize": "imagenet"},
).create()
import torch.utils.data as D

val_loader = list(D.DataLoader(provider._get_validation_dataset(), batch_size=64))[:8]
train_ds = provider._get_training_dataset()
train_loader = D.DataLoader(train_ds, batch_size=BS_TRAIN, shuffle=True,
                            num_workers=2, drop_last=True)

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

for p in model.get_perceptrons():
    p.activation = LIFActivation(
        T=T, activation_scale=p.activation_scale,
        thresholding_mode="<", firing_mode="Default", bias_mode="on_chip",
        membrane_init=V0,
    ).to(DEV)
print(f"[PR22] composition: phase_dither={DITHER} membrane_init={V0}", flush=True)


def genuine_acc(n_batches=4):
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader[:n_batches]:
            x, y = x.to(DEV), y.to(DEV)
            logits = chip_aligned_segment_forward(
                model, x, T, retime=True, phase_dither=DITHER,
            )
            correct += int((logits.argmax(1) == y).sum())
            total += int(y.numel())
    return correct / total


def analytic_acc(n_batches=4):
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader[:n_batches]:
            x, y = x.to(DEV), y.to(DEV)
            logits = model(x)
            correct += int((logits.argmax(1) == y).sum())
            total += int(y.numel())
    return correct / total


t0 = time.time()
entry = genuine_acc(8)
print(f"[PR22] entry genuine (n=512): {entry:.4f}  ({time.time()-t0:.0f}s)", flush=True)

opt = torch.optim.AdamW(
    [q for q in model.parameters() if q.requires_grad], lr=LR, weight_decay=0.01,
)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR * 0.1)
_best = [0.0, None]
model.train()
alpha, temp = KD_A, KD_T
t0 = time.time()
step = 0
_micro = 0
import itertools
for x, y in itertools.chain(train_loader, train_loader):
    if step >= STEPS:
        break
    x, y = x.to(DEV), y.to(DEV)
    with torch.no_grad():
        t_logits = teacher(x)
    s_logits = chip_aligned_segment_forward(
        model, x, T, retime=True, phase_dither=DITHER,
    )
    ce = F.cross_entropy(s_logits, y)
    kd = F.kl_div(
        F.log_softmax(s_logits / temp, -1), F.softmax(t_logits / temp, -1),
        reduction="batchmean",
    ) * (temp * temp)
    loss = (1 - alpha) * ce + alpha * kd
    (loss / ACCUM).backward()
    if (_micro + 1) % ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
    _micro += 1
    if step > 0 and step % 60 == 0 and _micro % ACCUM == 0:
        model.eval()
        acc, a_acc = genuine_acc(4), analytic_acc(4)
        model.train()
        print(f"[PR22] step {step}: genuine(n=256)={acc:.4f} "
              f"analytic(n=256)={a_acc:.4f} loss={float(loss):.3f} "
              f"({(time.time()-t0)/step:.1f}s/step)", flush=True)
        if acc > _best[0]:
            _best[0] = acc
            _best[1] = copy.deepcopy(model.state_dict())

model.eval()
final = genuine_acc(8)
if _best[1] is not None and _best[0] > final:
    model.load_state_dict(_best[1])
    final = genuine_acc(8)
    print(f"[PR22] restored keep-best state ({_best[0]:.4f} @256)", flush=True)
print(f"[PR22] FINAL genuine (n=512): {final:.4f} analytic(n=512)={analytic_acc(8):.4f} "
      f"(entry {entry:.4f}, slope {final-entry:+.4f} over {step} steps)", flush=True)
out = os.environ.get("PR22_SAVE", "")
if out:
    torch.save(model.state_dict(), out)
    print(f"[PR22] saved trained state -> {out}", flush=True)
