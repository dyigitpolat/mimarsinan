"""H2/PR23 (calculus §15.5): the decisive Phase-Deploy run — DeployedRiskFinetune
on the 0.8377 artifact, checkpoint-chained across process windows.
Prints PR23-DONE + the census verdict only when the full step budget completes.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path[:0] = ["./src"]

import torch

RUN = os.environ.get(
    "PR23_RUN", "generated/t2_04_origingate_ab2_phased_deployment_run"
)
CKPT = os.environ.get(
    "PR23_CKPT",
    "/tmp/claude-1005/-home-yigit-repos-research-stuff-mimarsinan/"
    "c73daf12-2a91-4dc4-9e8f-3b7f618308d5/scratchpad/pr23_stage.ckpt",
)
STEPS = int(os.environ.get("PR23_STEPS", "1200"))
LR = float(os.environ.get("PR23_LR", "2e-5"))
DEV = os.environ.get("PR23_DEV", "cuda:0")
T = 32
BS = int(os.environ.get("PR23_BS", "16"))
DITHER = bool(int(os.environ.get("PR23_DITHER", "0")))
V0 = float(os.environ.get("PR23_V0", "0.0"))

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

val_loader = list(D.DataLoader(provider._get_validation_dataset(), batch_size=64))
train_loader = D.DataLoader(
    provider._get_training_dataset(), batch_size=BS, shuffle=True,
    num_workers=2, drop_last=True,
)

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.tuning.orchestration.deployed_risk_finetune import (
    run_deployed_risk_finetune,
)

for p in model.get_perceptrons():
    p.activation = LIFActivation(
        T=T, activation_scale=p.activation_scale,
        thresholding_mode="<", firing_mode="Default", bias_mode="on_chip",
        membrane_init=V0,
    ).to(DEV)
print(f"[PR23] composition: phase_dither={DITHER} membrane_init={V0} bs={BS}", flush=True)


def genuine_acc(n_batches: int) -> float:
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


def device_batches():
    for x, y in train_loader:
        yield x.to(DEV), y.to(DEV)


result = run_deployed_risk_finetune(
    model, teacher,
    lambda x: chip_aligned_segment_forward(model, x, T, retime=True, phase_dither=DITHER),
    train_batches=device_batches(),
    eval_genuine=lambda: genuine_acc(4),
    steps=STEPS, lr=LR, eval_every=100, warmup_frac=0.1,
    kd_alpha=0.5, kd_temperature=4.0,
    checkpoint_path=CKPT, checkpoint_every=150,
)
print(f"[PR23] steps_run={result.steps_run}/{STEPS} resumed_from={result.resumed_from} "
      f"entry={result.entry_genuine:.4f} best={result.best_genuine:.4f} "
      f"final={result.final_genuine:.4f}", flush=True)

if result.steps_run >= STEPS:
    census = genuine_acc(39)
    print(f"[PR23] CENSUS genuine (n=2496): {census:.4f}  "
          f"(PR23 gate >= 0.82: {'PASS' if census >= 0.82 else 'MISS'})", flush=True)
    save = CKPT.replace("stage.ckpt", "final_state.pt")
    torch.save(model.state_dict(), save)
    print(f"[PR23] saved -> {save}", flush=True)
    print("PR23-DONE", flush=True)
