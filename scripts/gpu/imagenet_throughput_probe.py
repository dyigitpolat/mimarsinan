"""Measure ResNet-50 ImageNet train throughput on this hardware (torchvision loader).

Answers the Group-2 question: is the torchvision fallback fast enough to hit the
~67%/<1hr target on the RTX PRO 6000 Blackwells, or is the (risky) FFCV install
required? Prints imgs/sec + the extrapolated 16-epoch wall time. Read-only on the
data; trains nothing to convergence. Pick the GPU via CUDA_VISIBLE_DEVICES.
"""
import os
import time

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models import resnet50

ROOT = os.environ.get("IMAGENET_ROOT", "/data/ImageNet")
BATCH = int(os.environ.get("PROBE_BATCH", "256"))
WORKERS = int(os.environ.get("PROBE_WORKERS", "32"))
SIZE = int(os.environ.get("PROBE_SIZE", "176"))  # progressive-resize start size
WARMUP, MEASURE = 5, 40


def main() -> None:
    assert torch.cuda.is_available(), "no CUDA visible"
    dev = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)} batch={BATCH} workers={WORKERS} size={SIZE}", flush=True)

    t0 = time.time()
    tfm = T.Compose([
        T.RandomResizedCrop(SIZE), T.RandomHorizontalFlip(), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = torchvision.datasets.ImageNet(ROOT, split="train", transform=tfm)
    print(f"dataset constructed: {len(ds)} imgs in {time.time()-t0:.1f}s", flush=True)

    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=WORKERS,
                    pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)

    model = resnet50(num_classes=1000).to(dev, memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-5)
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    model.train()

    it = iter(dl)
    seen, t_start = 0, None
    for step in range(WARMUP + MEASURE):
        x, y = next(it)
        x = x.to(dev, memory_format=torch.channels_last, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss = lossf(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step == WARMUP:
            torch.cuda.synchronize()
            t_start = time.time()
        elif step >= WARMUP:
            seen += x.size(0)
    torch.cuda.synchronize()
    dt = time.time() - t_start
    ips = seen / dt
    epoch_s = 1_281_167 / ips
    print(f"throughput: {ips:.0f} imgs/s ({dt/MEASURE*1000:.0f} ms/step) on 1 GPU", flush=True)
    print(f"1 epoch ~= {epoch_s/60:.1f} min/GPU  ->  16 epochs on 4 GPUs (DDP, ideal) ~= {16*epoch_s/4/60:.1f} min", flush=True)
    print(f"VERDICT: torchvision fallback {'SUFFICES (<60min)' if 16*epoch_s/4/60 < 60 else 'is TOO SLOW -> FFCV needed'} for the <1hr target", flush=True)


if __name__ == "__main__":
    main()
