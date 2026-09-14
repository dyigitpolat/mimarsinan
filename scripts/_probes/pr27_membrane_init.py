"""PR27 (calculus §15.10): membrane-init pairing — V0=0 (floor, arrival-fragile)
vs V0=theta/2 (round, arrival-robust). Part 1: isolated soft-reset IF assay over
arrival patterns. Part 2: zero-training model A/B on W_a and W_g (injection =
IFNode registered reset value, so every reset restores V0).
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path[:0] = ["./src"]

import torch

DEV = "cuda:0"
T = 32


def soft_reset_if(charges: torch.Tensor, v0: float) -> torch.Tensor:
    """Strict soft-reset IF over a (T, N) charge sequence; returns counts/T."""
    v = torch.full_like(charges[0], v0)
    count = torch.zeros_like(charges[0])
    for t in range(charges.shape[0]):
        v = v + charges[t]
        fire = (v > 1.0).to(v)
        v = v - fire
        count = count + fire
    return count / charges.shape[0]


def patterns(c: torch.Tensor, T: int) -> dict[str, torch.Tensor]:
    """Arrival patterns delivering total charge c*T over T cycles."""
    N = c.shape[0]
    uni = (c.unsqueeze(0)).expand(T, N).clone()
    back = torch.zeros(T, N)
    back[3 * T // 4:] = (c * T / (T - 3 * T // 4)).unsqueeze(0)
    front = torch.zeros(T, N)
    front[: T // 4] = (c * T / (T // 4)).unsqueeze(0)
    signed = uni.clone()
    signed[0::2] = 2.5 * c.unsqueeze(0)
    signed[1::2] = -0.5 * c.unsqueeze(0)
    from mimarsinan.spiking.spike_trains import uniform_spike_train

    train = uniform_spike_train(c.clamp(0, 1), T)
    return {"uniform": uni, "back_loaded": back, "front_loaded": front,
            "signed_alt": signed, "spike_train": train}


print("=== PART 1: isolated soft-reset IF assay (strict '>', n=20000) ===", flush=True)
torch.manual_seed(0)
c = torch.rand(20000) * 1.2 - 0.1
target = torch.round((c * T).clamp(0, T)) / T
for name, ch in patterns(c, T).items():
    row = [name]
    for v0 in (0.0, 0.5):
        err = (soft_reset_if(ch, v0) - target).abs().mean()
        row.append(f"V0={v0}: {float(err):.4f}")
    print(f"[PR27-1] {row[0]:>12}  " + "  ".join(row[1:]), flush=True)

print("=== PART 2: zero-training model A/B ===", flush=True)
from mimarsinan.pipelining.session import apply_determinism

apply_determinism(0)
RUN = "generated/t2_04_stemheal_ab3_phased_deployment_run"
SCRATCH = ("/tmp/claude-1005/-home-yigit-repos-research-stuff-mimarsinan/"
           "c73daf12-2a91-4dc4-9e8f-3b7f618308d5/scratchpad")

import mimarsinan.data_handling.data_providers  # noqa: F401
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

provider = BasicDataProviderFactory(
    "CIFAR100_DataProvider", "./datasets", seed=0, batch_size=64,
    preprocessing={"interpolation": "bicubic", "resize_to": 224, "normalize": "imagenet"},
).create()
import torch.utils.data as D

val_loader = list(D.DataLoader(provider._get_validation_dataset(), batch_size=64))[:8]

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward


def build(state_path: str | None):
    model, _ = torch.load(
        f"{RUN}/Activation Quantization.model.pt", map_location="cpu",
        weights_only=False,
    )
    model = model.to(DEV)
    for p in model.get_perceptrons():
        p.activation = LIFActivation(
            T=T, activation_scale=p.activation_scale,
            thresholding_mode="<", firing_mode="Default", bias_mode="on_chip",
        ).to(DEV)
    if state_path:
        model.load_state_dict(torch.load(state_path, map_location=DEV))
    return model.eval()


def set_v0(model, v0: float) -> None:
    for m in model.modules():
        if isinstance(m, LIFActivation):
            m.if_node._memories_rv["v"] = v0
            m.if_node.v = v0


def census(model, genuine: bool) -> float:
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEV), y.to(DEV)
            logits = (chip_aligned_segment_forward(model, x, T, retime=True)
                      if genuine else model(x))
            correct += int((logits.argmax(1) == y).sum())
            total += int(y.numel())
    return correct / total


for label, state in (("W_a(artifact)", None),
                     ("W_g(trained)", f"{SCRATCH}/pr22c_best_state.pt")):
    model = build(state)
    for v0 in (0.0, 0.5):
        set_v0(model, v0)
        a = census(model, genuine=False)
        g = census(model, genuine=True)
        print(f"[PR27-2] {label} V0={v0}: analytic={a:.4f} genuine={g:.4f} "
              f"tau={a - g:+.4f}", flush=True)
    del model
    torch.cuda.empty_cache()
print("PR27-DONE", flush=True)
