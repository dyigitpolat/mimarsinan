"""
ANN-to-SNN pipeline on CIFAR-10 (VGG9 + SpikingJelly).
  Step 1: Train ANN VGG9                          -> checkpoints/ann.pt
  Step 2: Quantize (ReLU -> ClampFloor) + finetune -> checkpoints/quant.pt
  Step 3: Distill to SNN VGG9 (LIF, T=8)          -> checkpoints/snn.pt

Usage:
    python train.py              # all steps
    python train.py --step 1     # ANN only
    python train.py --step 2     # quantize only (needs step 1)
    python train.py --step 3     # SNN distill only (needs step 2)
"""
import os, sys, logging, argparse, numpy as np
from collections import OrderedDict
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torchvision import datasets, transforms

from models import ANN_VGG9, SNN_VGG9, ClampFloor, transfer_weights

# ── args ──────────────────────────────────────────────────────────
P = argparse.ArgumentParser()
P.add_argument('--step', type=int, default=0, help='0=all 1=ANN 2=quant 3=SNN')
P.add_argument('--data', default=os.path.expanduser('~/data'))
P.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
A = P.parse_args()

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ── logger ────────────────────────────────────────────────────────
def get_logger(name, path):
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fh = logging.FileHandler(path, mode='w'); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(message)s')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(ch)
    return lg

# ── data ──────────────────────────────────────────────────────────
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
tr_tf = transforms.Compose([
    transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
tr_aug_tf = transforms.Compose([
    transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
te_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def loader(ds, bs, shuffle=True):
    return torch.utils.data.DataLoader(ds, bs, shuffle, num_workers=4,
                                       pin_memory=True, drop_last=shuffle)

class RepeatT(torch.utils.data.Dataset):
    """Direct coding wrapper: repeat image T times."""
    def __init__(self, ds, T): self.ds, self.T = ds, T
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        x, y = self.ds[i]; return x.unsqueeze(0).repeat(self.T, 1, 1, 1), y

# ── utils ─────────────────────────────────────────────────────────
@torch.no_grad()
def accuracy(model, dl, snn=False):
    model.eval(); c = t = 0
    for x, y in dl:
        x, y = x.to(A.device), y.to(A.device)
        o = model(x.transpose(0, 1)) if snn else model(x)
        c += o.argmax(1).eq(y).sum().item(); t += y.size(0)
    return 100. * c / t

def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    H, W = x.shape[-2:]
    r = np.sqrt(1 - lam)
    cw, ch = int(W * r), int(H * r)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, y1 = max(cx - cw // 2, 0), max(cy - ch // 2, 0)
    x2, y2 = min(cx + cw // 2, W), min(cy + ch // 2, H)
    x[..., y1:y2, x1:x2] = x[idx][..., y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return x, y, y[idx], lam

# ══════════════════════════════════════════════════════════════════
# STEP 1: Train ANN VGG9
# ══════════════════════════════════════════════════════════════════
def step1():
    log = get_logger('step1', 'logs/step1_ann.log')
    log.info('=' * 50)
    log.info('STEP 1: Train ANN VGG9 on CIFAR-10')
    log.info('=' * 50)

    model = ANN_VGG9().to(A.device)
    tr_dl = loader(datasets.CIFAR10(A.data, True, download=True, transform=tr_tf), 128)
    te_dl = loader(datasets.CIFAR10(A.data, False, download=True, transform=te_tf), 256, False)

    opt = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, 60)
    crit = nn.CrossEntropyLoss()
    best = 0

    for ep in range(1, 61):
        model.train(); tc = tt = ls = 0
        for x, y in tr_dl:
            x, y = x.to(A.device), y.to(A.device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            tc += model(x).detach().argmax(1).eq(y).sum().item()
            tt += y.size(0); ls += loss.item() * y.size(0)
        sch.step()
        acc = accuracy(model, te_dl)
        s = ''
        if acc > best: best = acc; torch.save(model.state_dict(), 'checkpoints/ann.pt'); s = ' *'
        log.info(f'[{ep:2d}/60] lr={opt.param_groups[0]["lr"]:.4f} '
                 f'train={100*tc/tt:.2f}% loss={ls/tt:.4f} test={acc:.2f}%{s}')

    log.info(f'Best ANN: {best:.2f}%')
    return best

# ══════════════════════════════════════════════════════════════════
# STEP 2: Quantize ANN (ReLU -> ClampFloor T=4)
# ══════════════════════════════════════════════════════════════════
def step2():
    log = get_logger('step2', 'logs/step2_quantize.log')
    log.info('=' * 50)
    log.info('STEP 2: Quantize ANN -> ClampFloor(T=4)')
    log.info('=' * 50)

    T_q = 4
    model = ANN_VGG9().to(A.device)
    model.load_state_dict(torch.load('checkpoints/ann.pt', map_location=A.device, weights_only=True))
    model.eval()

    # ── calibrate max activation per ReLU ──
    calib = loader(torch.utils.data.Subset(
        datasets.CIFAR10(A.data, True, download=True, transform=te_tf), range(2048)), 256, False)
    maxv = OrderedDict()
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.ReLU):
            def mk(n):
                def fn(_, __, out):
                    v = out[out > 0]
                    maxv[n] = max(maxv.get(n, 0.), torch.quantile(v.float(), 0.999).item() if v.numel() else 1.)
                return fn
            hooks.append(m.register_forward_hook(mk(name)))
    with torch.no_grad():
        for x, _ in calib: model(x.to(A.device))
    for h in hooks: h.remove()
    log.info(f'Calibrated {len(maxv)} layers')

    # ── weight-threshold balancing ──
    # Collect (relu_name, bn, next_weight) triples
    relu_names = [n for n, m in model.named_modules() if isinstance(m, nn.ReLU)]
    bn_list = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    conv_list = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    # BN[i] feeds ReLU[i], which feeds Conv/Linear[i+1]
    with torch.no_grad():
        for i, rn in enumerate(relu_names):
            if rn not in maxv: continue
            s = max(maxv[rn], 1e-6)
            bn_list[i].weight.data.div_(s)
            bn_list[i].bias.data.div_(s)
            if i + 1 < len(conv_list):
                conv_list[i + 1].weight.data.mul_(s)
    log.info('Weight-threshold balancing done')

    # ── replace ReLU -> ClampFloor ──
    def replace(m):
        for n, mod in m.named_children():
            if isinstance(mod, nn.ReLU): setattr(m, n, ClampFloor(T_q))
            else: replace(mod)
    replace(model)

    te_dl = loader(datasets.CIFAR10(A.data, False, download=True, transform=te_tf), 256, False)
    log.info(f'After quantize (before finetune): {accuracy(model, te_dl):.2f}%')

    # ── finetune ──
    tr_dl = loader(datasets.CIFAR10(A.data, True, download=True, transform=tr_tf), 128)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, 30)
    crit = nn.CrossEntropyLoss()
    best = 0

    for ep in range(1, 31):
        model.train()
        for x, y in tr_dl:
            x, y = x.to(A.device), y.to(A.device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sch.step()
        acc = accuracy(model, te_dl)
        s = ''
        if acc > best: best = acc; torch.save(model.state_dict(), 'checkpoints/quant.pt'); s = ' *'
        log.info(f'[{ep:2d}/30] test={acc:.2f}%{s}')

    log.info(f'Best Quantized ANN: {best:.2f}%')
    return best

# ══════════════════════════════════════════════════════════════════
# STEP 3: Distill to SNN (KD + T=8 + CutMix + AutoAugment)
# ══════════════════════════════════════════════════════════════════
def step3():
    log = get_logger('step3', 'logs/step3_snn.log')
    log.info('=' * 50)
    log.info('STEP 3: Knowledge Distillation -> SNN VGG9')
    log.info('=' * 50)

    T = 8; EPOCHS = 100; LR = 2e-3; WARMUP = 5
    KD_ALPHA = 0.7; KD_TEMP = 3.0; CUTMIX_P = 0.5

    # ── teacher ──
    teacher = ANN_VGG9(act=lambda: ClampFloor(4)).to(A.device)
    teacher.load_state_dict(
        torch.load('checkpoints/quant.pt', map_location=A.device, weights_only=True), strict=False)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    te_dl_ann = loader(datasets.CIFAR10(A.data, False, download=True, transform=te_tf), 256, False)
    log.info(f'Teacher accuracy: {accuracy(teacher, te_dl_ann):.2f}%')

    # ── student ──
    student = SNN_VGG9(T=T, tau=2.0, v_threshold=1.0).to(A.device)
    quant_sd = torch.load('checkpoints/quant.pt', map_location='cpu', weights_only=True)
    n = transfer_weights(student, quant_sd)
    log.info(f'Transferred {n} weights from quantized ANN (conv only, fresh BN)')

    # ── data ──
    tr_dl = loader(RepeatT(
        datasets.CIFAR10(A.data, True, download=True, transform=tr_aug_tf), T), 64)
    te_dl = loader(RepeatT(
        datasets.CIFAR10(A.data, False, download=True, transform=te_tf), T), 64, False)

    init_acc = accuracy(student, te_dl, snn=True)
    log.info(f'Student init: {init_acc:.2f}%')

    # ── optimizer ──
    opt = optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS - WARMUP)
    crit = nn.CrossEntropyLoss()
    best = 0

    log.info(f'Config: T={T}, epochs={EPOCHS}, lr={LR}, warmup={WARMUP}, '
             f'KD(alpha={KD_ALPHA}, temp={KD_TEMP}), CutMix={CUTMIX_P}')
    log.info('-' * 50)

    for ep in range(1, EPOCHS + 1):
        # warmup
        if ep <= WARMUP:
            for pg in opt.param_groups: pg['lr'] = LR * ep / WARMUP
        else:
            sch.step()

        student.train(); tc = tt = ls = 0
        for x, y in tr_dl:
            x, y = x.to(A.device), y.to(A.device)

            # cutmix
            use_cm = np.random.random() < CUTMIX_P
            if use_cm: x, ya, yb, lam = cutmix(x, y)

            # teacher logits (single frame)
            with torch.no_grad(): t_logits = teacher(x[:, 0])

            opt.zero_grad()
            s_logits = student(x.transpose(0, 1))

            # KD loss
            kl = F.kl_div(F.log_softmax(s_logits / KD_TEMP, 1),
                          F.softmax(t_logits / KD_TEMP, 1),
                          reduction='batchmean') * KD_TEMP ** 2
            if use_cm:
                ce = lam * crit(s_logits, ya) + (1 - lam) * crit(s_logits, yb)
            else:
                ce = crit(s_logits, y)
            loss = KD_ALPHA * kl + (1 - KD_ALPHA) * ce

            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

            tc += s_logits.detach().argmax(1).eq(y).sum().item()
            tt += y.size(0); ls += loss.item() * y.size(0)

        acc = accuracy(student, te_dl, snn=True)
        s = ''
        if acc > best: best = acc; torch.save(student.state_dict(), 'checkpoints/snn.pt'); s = ' *'
        log.info(f'[{ep:3d}/{EPOCHS}] lr={opt.param_groups[0]["lr"]:.6f} '
                 f'train={100*tc/tt:.2f}% loss={ls/tt:.4f} test={acc:.2f}%{s}')

    log.info('=' * 50)
    log.info(f'Best SNN: {best:.2f}%')
    return best

# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    s = A.step
    if s in (0, 1): step1()
    if s in (0, 2): step2()
    if s in (0, 3): step3()
