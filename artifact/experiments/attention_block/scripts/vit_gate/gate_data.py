""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

from gate_common import DATA_DIR, SPLIT_JSON, TRAIN_MD5, md5_file

NUM_CLASSES = 100
MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)
IMG_MEAN_8BIT = tuple(min(255, round(255 * x)) for x in MEAN)
_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)
_LEVEL_DENOM = 10.0


def load_train_batch(data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    ""
    path = Path(data_dir) / "train"
    digest = md5_file(path)
    if digest != TRAIN_MD5:
        raise SystemExit(f"REFUSED: {path} md5 {digest} != {TRAIN_MD5}")
    with open(path, "rb") as f:
        entry = pickle.load(f, encoding="latin1")
    images = np.asarray(entry["data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    labels = np.asarray(entry["fine_labels"], dtype=np.int64)
    return images, labels


def stratified_split(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    ""
    rng = np.random.default_rng(seed)
    picks = [rng.permutation(np.flatnonzero(labels == c))[:per_class] for c in range(NUM_CLASSES)]
    return np.sort(np.concatenate(picks))


def indices_sha256(indices) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in indices).encode()).hexdigest()


def load_split(labels: np.ndarray, split_json: Path = SPLIT_JSON) -> tuple[np.ndarray, np.ndarray]:
    ""
    split = json.loads(Path(split_json).read_text())
    val = np.asarray(split["val_indices"], dtype=np.int64)
    if indices_sha256(val) != split["val_indices_sha256"]:
        raise SystemExit("REFUSED: split file indices do not match their recorded sha256")
    rederived = stratified_split(labels, split["per_class"], split["seed"])
    if not np.array_equal(rederived, val):
        raise SystemExit("REFUSED: split file differs from the declared stratified rule")
    mask = np.ones(len(labels), dtype=bool)
    mask[val] = False
    return np.flatnonzero(mask), val


def calibration_indices(train_idx: np.ndarray, n: int, seed: int) -> np.ndarray:
    ""
    rng = np.random.default_rng(int(seed) + 1000)
    return np.asarray(train_idx)[rng.permutation(len(train_idx))[:n]]


def derived_seed(*parts) -> int:
    ""
    return int(hashlib.sha256("/".join(str(p) for p in parts).encode()).hexdigest()[:15], 16)


def _negate(rng: random.Random, v: float) -> float:
    return -v if rng.random() > 0.5 else v


def _resample(rng: random.Random):
    return rng.choice(_RANDOM_INTERPOLATION)


def _affine(img, matrix, rng):
    return img.transform(img.size, Image.AFFINE, matrix, resample=_resample(rng), fillcolor=IMG_MEAN_8BIT)


def _solarize_add(img, add, thresh=128):
    lut = [min(255, i + add) if i < thresh else i for i in range(256)]
    return img.point(lut * 3) if img.mode == "RGB" else img.point(lut)


def _posterize(img, bits):
    return img if bits >= 8 else ImageOps.posterize(img, int(bits))


def _enhance(kind):
    def f(img, m, rng):
        factor = max(0.1, 1.0 + _negate(rng, (m / _LEVEL_DENOM) * 0.9))
        return getattr(ImageEnhance, kind)(img).enhance(factor)
    return f


RAND_INCREASING_OPS = {
    "AutoContrast": lambda img, m, rng: ImageOps.autocontrast(img),
    "Equalize": lambda img, m, rng: ImageOps.equalize(img),
    "Invert": lambda img, m, rng: ImageOps.invert(img),
    "Rotate": lambda img, m, rng: img.rotate(_negate(rng, (m / _LEVEL_DENOM) * 30.0), resample=_resample(rng),
                                             fillcolor=IMG_MEAN_8BIT),
    "PosterizeIncreasing": lambda img, m, rng: _posterize(img, 4 - int((m / _LEVEL_DENOM) * 4)),
    "SolarizeIncreasing": lambda img, m, rng: ImageOps.solarize(img, 256 - int((m / _LEVEL_DENOM) * 256)),
    "SolarizeAdd": lambda img, m, rng: _solarize_add(img, int((m / _LEVEL_DENOM) * 110)),
    "ColorIncreasing": _enhance("Color"),
    "ContrastIncreasing": _enhance("Contrast"),
    "BrightnessIncreasing": _enhance("Brightness"),
    "SharpnessIncreasing": _enhance("Sharpness"),
    "ShearX": lambda img, m, rng: _affine(img, (1, _negate(rng, (m / _LEVEL_DENOM) * 0.3), 0, 0, 1, 0), rng),
    "ShearY": lambda img, m, rng: _affine(img, (1, 0, 0, _negate(rng, (m / _LEVEL_DENOM) * 0.3), 1, 0), rng),
    "TranslateXRel": lambda img, m, rng: _affine(
        img, (1, 0, _negate(rng, (m / _LEVEL_DENOM) * 0.45) * img.size[0], 0, 1, 0), rng),
    "TranslateYRel": lambda img, m, rng: _affine(
        img, (1, 0, 0, 0, 1, _negate(rng, (m / _LEVEL_DENOM) * 0.45) * img.size[1]), rng),
}
RAND_OP_NAMES = tuple(RAND_INCREASING_OPS)


def rand_augment(img: Image.Image, rng: random.Random, *, magnitude: float = 9.0, magnitude_std: float = 0.5,
                 num_layers: int = 2, prob: float = 0.5) -> Image.Image:
    ""
    for _ in range(num_layers):
        name = RAND_OP_NAMES[rng.randrange(len(RAND_OP_NAMES))]
        if prob < 1.0 and rng.random() > prob:
            continue
        m = rng.gauss(magnitude, magnitude_std) if magnitude_std > 0 else magnitude
        m = max(0.0, min(m, _LEVEL_DENOM))
        img = RAND_INCREASING_OPS[name](img, m, rng)
    return img


def random_resized_crop(img: Image.Image, rng: random.Random, size: int = 32, scale=(0.8, 1.0),
                        ratio=(3.0 / 4.0, 4.0 / 3.0)) -> Image.Image:
    ""
    W, H = img.size
    area = W * H
    box = None
    for _ in range(10):
        target = rng.uniform(*scale) * area
        aspect = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        w = int(round(math.sqrt(target * aspect)))
        h = int(round(math.sqrt(target / aspect)))
        if w <= W and h <= H:
            i = rng.randint(0, H - h)
            j = rng.randint(0, W - w)
            box = (i, j, h, w)
            break
    if box is None:
        in_ratio = W / H
        if in_ratio < min(ratio):
            w, h = W, int(round(W / min(ratio)))
        elif in_ratio > max(ratio):
            h, w = H, int(round(H * max(ratio)))
        else:
            w, h = W, H
        box = ((H - h) // 2, (W - w) // 2, h, w)
    i, j, h, w = box
    return img.crop((j, i, j + w, i + h)).resize((size, size), _resample(rng))


def train_image(img: Image.Image, rng: random.Random) -> Image.Image:
    img = random_resized_crop(img, rng)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return rand_augment(img, rng)


def to_uint8_chw(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).copy())


class CifarAugmented(torch.utils.data.Dataset):
    ""

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray, *, train: bool, seed: int,
                 phase: str):
        self.images, self.labels, self.indices = images, labels, np.asarray(indices)
        self.train, self.seed, self.phase = train, int(seed), phase
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        img = Image.fromarray(self.images[j])
        if self.train:
            img = train_image(img, random.Random(derived_seed(self.seed, self.phase, self.epoch, j)))
        return to_uint8_chw(img), int(self.labels[j])


def _worker_init(_):
    torch.set_num_threads(1)


def epoch_order(n: int, seed: int, phase: str, epoch: int) -> list[int]:
    g = torch.Generator()
    g.manual_seed(derived_seed(seed, phase, "order", epoch))
    return torch.randperm(n, generator=g).tolist()


def make_loader(dataset, batch_size: int, *, order: list[int] | None, workers: int, drop_last: bool):
    sampler = order if order is not None else list(range(len(dataset)))
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=True, drop_last=drop_last,
        persistent_workers=False, prefetch_factor=4 if workers > 0 else None,
        worker_init_fn=_worker_init if workers > 0 else None)


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=x_uint8.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x_uint8.device).view(1, 3, 1, 1)
    return (x_uint8.float() / 255.0 - mean) / std


def random_erasing(x: torch.Tensor, rng: np.random.Generator, gen: torch.Generator, *, prob: float = 0.25,
                   min_area: float = 0.02, max_area: float = 1 / 3, min_aspect: float = 0.3) -> torch.Tensor:
    ""
    b, c, h, w = x.shape
    log_ratio = (math.log(min_aspect), math.log(1.0 / min_aspect))
    boxes = np.zeros((b, 4), dtype=np.int64)
    for k in range(b):
        if rng.random() > prob:
            continue
        for _ in range(10):
            target = rng.uniform(min_area, max_area) * h * w
            aspect = math.exp(rng.uniform(*log_ratio))
            hh = int(round(math.sqrt(target * aspect)))
            ww = int(round(math.sqrt(target / aspect)))
            if ww < w and hh < h:
                boxes[k] = (rng.integers(0, h - hh + 1), rng.integers(0, w - ww + 1), hh, ww)
                break
    if not boxes[:, 2].any():
        return x
    t = torch.from_numpy(boxes).to(x.device)
    ys = torch.arange(h, device=x.device).view(1, h, 1)
    xs = torch.arange(w, device=x.device).view(1, 1, w)
    top, left, hh, ww = (t[:, i].view(b, 1, 1) for i in range(4))
    mask = (ys >= top) & (ys < top + hh) & (xs >= left) & (xs < left + ww)
    noise = torch.empty(x.shape, device=x.device, dtype=x.dtype).normal_(generator=gen)
    return torch.where(mask.unsqueeze(1), noise, x)


def mixup_target(target: torch.Tensor, num_classes: int, lam: float, smoothing: float) -> torch.Tensor:
    ""
    off = smoothing / num_classes
    on = 1.0 - smoothing + off
    y1 = torch.full((target.size(0), num_classes), off, device=target.device).scatter_(1, target.view(-1, 1), on)
    y2 = torch.full((target.size(0), num_classes), off, device=target.device).scatter_(1, target.flip(0).view(-1, 1), on)
    return y1 * lam + y2 * (1.0 - lam)


def mixup_cutmix(x: torch.Tensor, target: torch.Tensor, rng: np.random.Generator, *, mixup_alpha: float = 0.8,
                 cutmix_alpha: float = 1.0, prob: float = 1.0, switch_prob: float = 0.5, smoothing: float = 0.1,
                 num_classes: int = NUM_CLASSES) -> tuple[torch.Tensor, torch.Tensor, dict]:
    ""
    lam, use_cutmix = 1.0, False
    if rng.random() < prob:
        use_cutmix = rng.random() < switch_prob
        lam = float(rng.beta(cutmix_alpha, cutmix_alpha) if use_cutmix else rng.beta(mixup_alpha, mixup_alpha))
    draw = {"lam": lam, "cutmix": use_cutmix}
    if lam != 1.0:
        if use_cutmix:
            h, w = x.shape[-2:]
            ratio = math.sqrt(1.0 - lam)
            cut_h, cut_w = int(h * ratio), int(w * ratio)
            cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
            yl, yh = int(np.clip(cy - cut_h // 2, 0, h)), int(np.clip(cy + cut_h // 2, 0, h))
            xl, xh = int(np.clip(cx - cut_w // 2, 0, w)), int(np.clip(cx + cut_w // 2, 0, w))
            x = x.clone()
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
            lam = 1.0 - (yh - yl) * (xh - xl) / float(h * w)
            draw.update(lam=lam, box=(yl, yh, xl, xh))
        else:
            x = x * lam + x.flip(0) * (1.0 - lam)
    return x, mixup_target(target, num_classes, lam, smoothing), draw


def batch_ops(x_uint8: torch.Tensor, target: torch.Tensor, *, seed: int, phase: str, epoch: int, step: int,
              recipe: dict) -> tuple[torch.Tensor, torch.Tensor]:
    ""
    rng = np.random.default_rng(derived_seed(seed, phase, "batch", epoch, step))
    gen = torch.Generator(device=x_uint8.device)
    gen.manual_seed(derived_seed(seed, phase, "noise", epoch, step) % (1 << 62))
    x = normalize(x_uint8)
    x = random_erasing(x, rng, gen, prob=float(recipe["reprob"]))
    x, y, _ = mixup_cutmix(x, target, rng, mixup_alpha=float(recipe["mixup"]), cutmix_alpha=float(recipe["cutmix"]),
                           prob=float(recipe["mixup_prob"]), switch_prob=float(recipe["mixup_switch_prob"]),
                           smoothing=float(recipe["smoothing"]))
    return x, y
