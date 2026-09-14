"""PR32 (calculus §15.15): analytic swap-target screen — which MAPPABLE target
activation, installed on ORIGIN weights with NO training, retains the most
accuracy. Lowest raw swap loss = best fine-tune basin for the artifact axis.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["MIMARSINAN_DISABLE_FFCV"] = "1"
sys.path[:0] = ["./src"]

import torch

torch.backends.cudnn.enabled = False
DEV = os.environ.get("PR32_DEV", "cuda:0")
T = 32
RUN = "generated/t2_04_stemheal_ab3_phased_deployment_run"

from mimarsinan.pipelining.session import apply_determinism

apply_determinism(0)

teacher, _ = torch.load(
    f"{RUN}/Reference Teacher Snapshot.reference_teacher_model.pt",
    map_location="cpu", weights_only=False,
)
teacher = teacher.to(DEV).eval()

import mimarsinan.data_handling.data_providers  # noqa: F401
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

provider = BasicDataProviderFactory(
    "CIFAR100_DataProvider", "./datasets", seed=0, batch_size=64,
    preprocessing={"interpolation": "bicubic", "resize_to": 224, "normalize": "imagenet"},
).create()
import torch.utils.data as D

val = list(D.DataLoader(provider._get_validation_dataset(), batch_size=64))[:8]
cal = list(D.DataLoader(provider._get_training_dataset(), batch_size=64))[:4]

from mimarsinan.models.nn.activations import LIFActivation

perceptrons = list(teacher.get_perceptrons())
originals = [p.activation for p in perceptrons]


def restore():
    for p, a in zip(perceptrons, originals):
        p.activation = a


def acc(n=8):
    c = t = 0
    with torch.no_grad():
        for x, y in val[:n]:
            x, y = x.to(DEV), y.to(DEV)
            c += int((teacher(x).argmax(1) == y).sum())
            t += int(y.numel())
    return c / t


# Per-perceptron pre-activation stats (input to the activation) from calib.
pre = {k: [] for k in range(len(perceptrons))}
hs = []
for k, p in enumerate(perceptrons):
    def _h(_m, inp, _o, _k=k):
        pre[_k].append(inp[0].detach().float())
    hs.append(p.activation.register_forward_hook(_h))
with torch.no_grad():
    for x, _ in cal:
        teacher(x.to(DEV))
for h in hs:
    h.remove()


def theta_scalar(k, q=1.0):
    z = torch.cat([t.reshape(-1) for t in pre[k]]).abs().clamp(min=1e-6)
    if q >= 1.0:
        return z.max().item()
    return z[torch.randint(0, z.numel(), (200000,))].quantile(q).item()


def theta_channel(k, q=1.0):
    z = torch.cat([t.reshape(-1, t.shape[-1]) for t in pre[k]], 0).abs().clamp(min=1e-6)
    if q >= 1.0:
        return z.amax(dim=0)
    idx = torch.randint(0, z.shape[0], (min(z.shape[0], 200000),))
    return z[idx].quantile(q, dim=0)


print(f"[PR32] origin GELU baseline: {acc():.4f}", flush=True)

for label, mk in (
    ("scalar-theta LIF", lambda k: LIFActivation(
        T=T, activation_scale=torch.tensor(theta_scalar(k)),
        thresholding_mode="<", firing_mode="Default", bias_mode="on_chip")),
    ("per-channel-theta LIF", lambda k: LIFActivation(
        T=T, activation_scale=theta_channel(k),
        thresholding_mode="<", firing_mode="Default", bias_mode="on_chip")),
    ("1.5x-theta LIF", lambda k: LIFActivation(
        T=T, activation_scale=torch.tensor(1.5 * theta_scalar(k)),
        thresholding_mode="<", firing_mode="Default", bias_mode="on_chip")),
    ("per-channel + guard", lambda k: LIFActivation(
        T=T, activation_scale=theta_channel(k),
        thresholding_mode="<", firing_mode="Default", bias_mode="on_chip",
        membrane_init=-0.25)),
):
    for k, p in enumerate(perceptrons):
        p.activation = mk(k).to(DEV)
    print(f"[PR32] {label:>24}: analytic = {acc():.4f}", flush=True)
    restore()
print("PR32-DONE", flush=True)
