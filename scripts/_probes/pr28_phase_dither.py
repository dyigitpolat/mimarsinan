"""PR28 (calculus §15.11): per-channel phase-dithered uniform encode.

to_uniform_spikes fires every active channel at cycle 0 (synchronized burst)
then phase-locked combs; with signed charge + irreversible fires this is the
transient-overfire source. Rotating each channel's comb by a deterministic
offset mod T preserves counts EXACTLY (decode-invariant) and only spreads
arrival — a pure temporal-shape A/B, zero training.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path[:0] = ["./src"]

import torch

DEV, T = "cuda:0", 32
RUN = "generated/t2_04_stemheal_ab3_phased_deployment_run"
SCRATCH = ("/tmp/claude-1005/-home-yigit-repos-research-stuff-mimarsinan/"
           "c73daf12-2a91-4dc4-9e8f-3b7f618308d5/scratchpad")

from mimarsinan.pipelining.session import apply_determinism

apply_determinism(0)

from mimarsinan.chip_simulation.recording import spike_modes as sm

_orig = sm.to_uniform_spikes


def dithered_to_uniform_spikes(tensor, cycle, simulation_length):
    T = simulation_length
    n = torch.round(tensor * T).to(torch.long)
    mask = (n != 0) & (n != T) & (cycle < T)
    n_safe = torch.clamp(n, min=1)
    spacing = T / n_safe.float()
    C = tensor.shape[-1]
    off = torch.floor(
        (torch.arange(C, device=tensor.device, dtype=torch.float32)
         * 0.6180339887).frac() * T
    )
    e = (float(cycle) + off) % T
    result = mask & (torch.floor(e / spacing) < n_safe) & (torch.floor(e % spacing) == 0)
    result = result.float()
    result[n == T] = 1.0
    return result


def _install(fn):
    import mimarsinan.chip_simulation as cs

    sm.to_uniform_spikes = fn
    if hasattr(cs, "spike_modes"):
        cs.spike_modes.to_uniform_spikes = fn


# sanity: exact count preservation on random rates
torch.manual_seed(0)
r = torch.rand(4096)
base = torch.stack([_orig(r, c, T) for c in range(T)]).sum(0)
_install(dithered_to_uniform_spikes)
dith = torch.stack([sm.to_uniform_spikes(r, c, T) for c in range(T)]).sum(0)
_install(_orig)
print(f"[PR28] count preservation: max|diff|={float((base - dith).abs().max()):.1f} "
      f"(must be 0)", flush=True)
assert float((base - dith).abs().max()) == 0.0

import mimarsinan.data_handling.data_providers  # noqa: F401
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

provider = BasicDataProviderFactory(
    "CIFAR100_DataProvider", "./datasets", seed=0, batch_size=64,
    preprocessing={"interpolation": "bicubic", "resize_to": 224, "normalize": "imagenet"},
).create()
import torch.utils.data as D

val = list(D.DataLoader(provider._get_validation_dataset(), batch_size=64))[:8]

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward


def build(state):
    m, _ = torch.load(
        f"{RUN}/Activation Quantization.model.pt", map_location="cpu",
        weights_only=False,
    )
    m = m.to(DEV)
    for p in m.get_perceptrons():
        p.activation = LIFActivation(
            T=T, activation_scale=p.activation_scale,
            thresholding_mode="<", firing_mode="Default", bias_mode="on_chip",
        ).to(DEV)
    if state:
        m.load_state_dict(torch.load(state, map_location=DEV))
    return m.eval()


def gcensus(m):
    c = t = 0
    with torch.no_grad():
        for x, y in val:
            x, y = x.to(DEV), y.to(DEV)
            lg = chip_aligned_segment_forward(m, x, T, retime=True)
            c += int((lg.argmax(1) == y).sum())
            t += int(y.numel())
    return c / t


def set_v0(m, v0):
    for mod in m.modules():
        if isinstance(mod, LIFActivation):
            mod.if_node._memories_rv["v"] = v0
            mod.if_node.v = v0


m = build(None)
for encode, fn in (("locked", _orig), ("dither", dithered_to_uniform_spikes)):
    _install(fn)
    for v0 in (0.0, -0.25, -0.5, -0.75):
        set_v0(m, v0)
        print(f"[PR28] W_a encode={encode} V0={v0:+.2f}: genuine={gcensus(m):.4f}",
              flush=True)
_install(_orig)
del m
torch.cuda.empty_cache()

m = build(f"{SCRATCH}/pr22c_best_state.pt")
for encode, fn in (("locked", _orig), ("dither", dithered_to_uniform_spikes)):
    _install(fn)
    set_v0(m, 0.0)
    print(f"[PR28] W_g encode={encode} V0=+0.00: genuine={gcensus(m):.4f}", flush=True)
_install(_orig)
print("PR28-DONE", flush=True)
