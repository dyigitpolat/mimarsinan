"""PR33 (calculus §15.14): is the hop-1 magnitude term GRID-limited?

Capture each early hop's real pre-activation from a genuine pass, then for a
range of T compute the LIF value-twin output RMS vs the float clamp target.
RMS ∝ 1/T ⇒ grid-limited ⇒ per-hop temporal allocation (s_allocation) recovers
it; RMS flat in T ⇒ timing/retime-limited ⇒ a different lever.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["MIMARSINAN_DISABLE_FFCV"] = "1"
sys.path[:0] = ["./src"]

import torch

torch.backends.cudnn.enabled = False
DEV = os.environ.get("PR33_DEV", "cuda:0")
RUN = "generated/t2_04_stemheal_ab3_phased_deployment_run"
HOPS = (0, 1, 2, 6)
TS = (32, 64, 128, 256)

from mimarsinan.pipelining.session import apply_determinism

apply_determinism(0)

model, _ = torch.load(
    f"{RUN}/Activation Quantization.model.pt", map_location="cpu", weights_only=False
)
model = model.to(DEV)

from mimarsinan.models.nn.activations import LIFActivation

for p in model.get_perceptrons():
    p.activation = LIFActivation(
        T=32, activation_scale=p.activation_scale,
        thresholding_mode="<", firing_mode="Default", bias_mode="on_chip",
        membrane_init=-0.25,
    ).to(DEV)
model.eval()
perceptrons = list(model.get_perceptrons())

import mimarsinan.data_handling.data_providers  # noqa: F401
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

provider = BasicDataProviderFactory(
    "CIFAR100_DataProvider", "./datasets", seed=0, batch_size=64,
    preprocessing={"interpolation": "bicubic", "resize_to": 224, "normalize": "imagenet"},
).create()
import torch.utils.data as D

val = list(D.DataLoader(provider._get_validation_dataset(), batch_size=64))[:4]

pre = {k: [] for k in HOPS}
hs = []
for k in HOPS:
    def _h(_m, inp, _o, _k=k):
        pre[_k].append(inp[0].detach().float())
    hs.append(perceptrons[k].activation.register_forward_hook(_h))
with torch.no_grad():
    for x, _ in val:
        model(x.to(DEV))
for h in hs:
    h.remove()


def lif_twin(z, theta, T):
    """Deployable LIF value transfer at T on constant input z (signed IF)."""
    from spikingjelly.activation_based import neuron, functional
    from mimarsinan.models.nn.activations.lif import StrictATanSurrogate

    node = neuron.IFNode(
        v_threshold=1.0, v_reset=None,
        surrogate_function=StrictATanSurrogate(), step_mode="m", backend="torch",
    )
    node.set_reset_value("v", -0.25)
    functional.reset_net(node)
    zt = (z / theta).unsqueeze(0).expand(T, *z.shape).contiguous()
    spikes = node(zt)
    return spikes.mean(0) * theta


print(f"{'hop':>4} {'theta':>7} " + " ".join(f"T={t:<7}" for t in TS)
      + "   slope", flush=True)
for k in HOPS:
    z = torch.cat([t.reshape(-1) for t in pre[k]])[:200000].to(DEV)
    theta = float(torch.as_tensor(perceptrons[k].activation_scale).float().mean())
    target = z.clamp(min=0.0, max=theta)  # float clamp = the T→∞ transfer
    rms = []
    for T in TS:
        out = lif_twin(z, max(theta, 1e-6), T)
        rms.append(float((out - target).pow(2).mean().sqrt()) / theta)
    ratio = rms[0] / max(rms[-1], 1e-9)
    print(f"[PR33] {k:>2} {theta:>7.3f} "
          + " ".join(f"{r:<9.5f}" for r in rms)
          + f"   x{ratio:.1f} (T x{TS[-1] // TS[0]})", flush=True)
print("[PR33] grid-limited iff RMS ~ 1/T (ratio ~ T-ratio); flat ratio ~1 = timing",
      flush=True)
print("PR33-DONE", flush=True)
