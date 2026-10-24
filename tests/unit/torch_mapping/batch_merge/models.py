"""Witness models of the batch-merge push-forward (coverage W7): window attention as torchvision and timm write it
(the batch merged into the window axis, ``(B * nW, N, C)``, and split back), and the forms that must stay refused by
name (an op mixing the merged axis, a window-major merge or split, a batch-leading operand)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

H = W = 8
WS = 4
NW = (H // WS) * (W // WS)
N = WS * WS
C = 8
NH = 2
SHAPE = (H, W, C)


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_w = nn.Parameter(torch.randn(3 * C, C) * 0.3)
        self.qkv_b = nn.Parameter(torch.randn(3 * C) * 0.1)
        self.table = nn.Parameter(torch.randn(1, NH, N, N) * 0.5)
        self.proj = nn.Linear(C, C)
        self.head = nn.Linear(H * W * C, 5)

    def partition(self, x):
        B = x.size(0)
        x = x.view(B, H // WS, WS, W // WS, WS, C).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(B * NW, N, C)

    def reverse(self, y, B):
        y = y.view(B, H // WS, W // WS, WS, WS, C).permute(0, 1, 3, 2, 4, 5)
        return y.reshape(B, H, W, C)

    def finish(self, x):
        return self.head(torch.flatten(x, 1))


class WinAttn(Base):
    """Swin's window attention as torchvision writes it (functional linear, permute to heads, getitem)."""

    def forward(self, x):
        B = x.size(0)
        w = self.partition(x)
        qkv = F.linear(w, self.qkv_w, self.qkv_b).reshape(w.size(0), w.size(1), 3, NH, C // NH)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)[0], qkv.permute(2, 0, 3, 1, 4)[1], qkv.permute(2, 0, 3, 1, 4)[2]
        attn = (q * (C // NH) ** -0.5).matmul(k.transpose(-2, -1)) + self.table
        attn = F.dropout(F.softmax(attn, dim=-1), p=0.0, training=False)
        y = attn.matmul(v).transpose(1, 2).reshape(w.size(0), w.size(1), C)
        return self.finish(self.reverse(self.proj(y), B))


class WinAttnModules(Base):
    """The same with module calls (timm style): nn.Linear, nn.Softmax(dim=-1), nn.Dropout, nn.GELU."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(C, 3 * C)
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(0.1)
        self.act = nn.GELU()

    def forward(self, x):
        B = x.size(0)
        w = self.partition(x)
        qkv = self.qkv(w).reshape(-1, N, 3, NH, C // NH).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.drop(self.softmax(q @ k.transpose(-2, -1) + self.table))
        y = self.act((attn @ v).transpose(1, 2).reshape(-1, N, C))
        return self.finish(self.reverse(self.proj(y), B))


class WinMask(Base):
    """A shifted-window form: the attention split to (B, nW, ...) for a batch-free mask, then merged again."""

    def __init__(self):
        super().__init__()
        self.mask = nn.Parameter(torch.randn(NW, N, N))

    def forward(self, x):
        B = x.size(0)
        w = self.partition(torch.roll(x, shifts=(-2, -2), dims=(1, 2)))
        qkv = F.linear(w, self.qkv_w, self.qkv_b).reshape(w.size(0), w.size(1), 3, NH, C // NH).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q.matmul(k.transpose(-2, -1)) + self.table
        attn = attn.view(w.size(0) // NW, NW, NH, N, N) + self.mask.unsqueeze(1).unsqueeze(0)
        attn = F.softmax(attn.view(-1, NH, N, N), dim=-1)
        y = attn.matmul(v).transpose(1, 2).reshape(w.size(0), w.size(1), C)
        return self.finish(torch.roll(self.reverse(self.proj(y), B), shifts=(2, 2), dims=(1, 2)))


class Multiplicity(Base):
    """A merged value reshaped to another multiplicity, (16B, 4, C) to (8B, 8, C), then split."""

    def forward(self, x):
        B = x.size(0)
        y = x.reshape(B * 16, 4, C)
        y = self.proj(y.reshape(B * 8, 8, C)) * 2.0
        return self.finish(y.reshape(B, H, W, C))


class FlattenMerge(Base):
    """``flatten(0, 1)`` merges the batch with the rows; a Linear; ``view(B, ...)`` splits."""

    def forward(self, x):
        B = x.size(0)
        y = self.proj(torch.flatten(x, 0, 1))
        return self.finish(y.view(B, H, W, C))


class TokenSlice(Base):
    """Per-window slices of the merged value (``y[:, :1]``, ``y[:, 1:]``) joined along a per-window axis."""

    def forward(self, x):
        B = x.size(0)
        y = self.proj(self.partition(x))
        z = torch.cat([y[:, :1], y[:, 1:] * 2.0], 1)
        return self.finish(self.reverse(z, B))


class ExpandToken(Base):
    """A batch-free token expanded to the merged extent (``token.expand(w.size(0), -1, -1)``), joined, sliced off."""

    def __init__(self):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, C))

    def forward(self, x):
        B = x.size(0)
        w = self.partition(x)
        t = torch.cat([self.token.expand(w.size(0), -1, -1), w], 1)
        return self.finish(self.reverse(self.proj(t)[:, 1:], B))


class Cosine(Base):
    """Swin V2's cosine attention: F.normalize(dim=-1), a clamped logit scale, a batch-free bias."""

    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((NH, 1, 1))))

    def forward(self, x):
        B = x.size(0)
        w = self.partition(x)
        qkv = F.linear(w, self.qkv_w, self.qkv_b).reshape(w.size(0), w.size(1), 3, NH, C // NH).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = attn * torch.clamp(self.logit_scale, max=4.6).exp() + self.table
        y = F.softmax(attn, dim=-1).matmul(v).transpose(1, 2).reshape(w.size(0), w.size(1), C)
        return self.finish(self.reverse(self.proj(y), B))


class MatrixLeft(Base):
    """A batch-free matrix times the merged value, ``P @ y``: a contraction over a per-window axis."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.randn(N, N) * 0.3)

    def forward(self, x):
        B = x.size(0)
        return self.finish(self.reverse(self.p @ self.partition(x), B))


# ------------------------------------------------------------------ the merged axis mixed: must stay refused
def _mixing(name, op):
    def forward(self, x):
        B = x.size(0)
        return self.finish(self.reverse(op(self, self.partition(x)), B))
    return type(name, (Base,), {"forward": forward})


Sum0 = _mixing("Sum0", lambda s, y: y - y.mean(0, keepdim=True))
Softmax0 = _mixing("Softmax0", lambda s, y: F.softmax(y, dim=0))
Flip0 = _mixing("Flip0", lambda s, y: y.flip(0))
Cumsum0 = _mixing("Cumsum0", lambda s, y: y.cumsum(0))
Roll0 = _mixing("Roll0", lambda s, y: torch.roll(y, 1, 0))
Roll0Method = _mixing("Roll0Method", lambda s, y: y.roll(3, dims=0))
Normalize0 = _mixing("Normalize0", lambda s, y: F.normalize(y, dim=0))
Cat0 = _mixing("Cat0", lambda s, y: torch.cat([y[1:], y[:1]], 0))
Slice0 = _mixing("Slice0", lambda s, y: torch.cat([y[: y.size(0) // 2] * 2.0, y[y.size(0) // 2:]], 0))
SizeScale = _mixing("SizeScale", lambda s, y: y * y.size(0))
Contract0 = _mixing("Contract0", lambda s, y: (y.reshape(y.size(0), N * C).t() @ y.reshape(y.size(0), N * C)
                                               ).sum() + y)
TransposeMatmul = _mixing("TransposeMatmul", lambda s, y: 1e-3 * (y.transpose(0, 1) @ y.transpose(0, 1).transpose(
    1, 2)).sum() + y)
DropoutTrain = _mixing("DropoutTrain", lambda s, y: F.dropout(y, 0.5, training=True))


class WindowMajor(Base):
    """Windows merged window-major, ``(B, nW, N, C).transpose(0, 1).reshape(nW * B, N, C)``, split back the same way."""

    def forward(self, x):
        B = x.size(0)
        w = self.partition(x).reshape(B, NW, N, C).transpose(0, 1).reshape(NW * B, N, C)
        y = self.proj(w).reshape(NW, B, N, C).transpose(0, 1).reshape(B * NW, N, C)
        return self.finish(self.reverse(y, B))


class SplitWindowMajor(Base):
    """Sample-major windows split window-major, ``view(nW, B, N, C)``: natively a mix of the samples."""

    def forward(self, x):
        B = x.size(0)
        y = self.proj(self.partition(x)).view(NW, -1, N, C).transpose(0, 1).reshape(B * NW, N, C)
        return self.finish(self.reverse(y, B))


class BatchOperand(Base):
    """The merged value plus a batch-leading value repeated window-major (``h.repeat(nW, 1, 1)``)."""

    def forward(self, x):
        B = x.size(0)
        w = self.partition(x)
        h = x.reshape(B, H * W, C)[:, :1]
        return self.finish(self.reverse(self.proj(w + h.repeat(NW, 1, 1)), B))


class MergedOutput(Base):
    """The merged value is the output."""

    def forward(self, x):
        return self.proj(self.partition(x))


class LeftBroadcast(Base):
    """A batch-free operand of higher rank reaching the merged axis from the left (exact natively)."""

    def __init__(self):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, N, C))

    def forward(self, x):
        B = x.size(0)
        y = (self.partition(x) + self.q)[0]
        return self.finish(self.reverse(y, B))


class SizeSplit(Base):
    """A reshape to multiplicity 2 whose leading extent the program computes from the merged extent,
    ``y.reshape(y.size(0) // 2, 2, ...)``, then merged to multiplicity 4 again."""

    def forward(self, x):
        B = x.size(0)
        y = self.proj(self.partition(x))
        z = y.reshape(y.size(0) // 2, 2, N, C).reshape(B * NW, N, C)
        return self.finish(self.reverse(z, B))

class DtypeView(Base):
    """A ``view(dtype)`` that doubles the leading extent: ``(B,)`` float64 viewed as ``(2B,)`` float32, scaled by 2
    (exact on the bits' float32 halves only by accident), viewed back: a reinterpretation, never a reshape."""

    def __init__(self):
        super().__init__()
        self.out = nn.Linear(1, 3)

    def forward(self, x):
        s = self.proj(x.reshape(x.size(0), H * W, C)).sum((1, 2))
        z = s.view(torch.float32) * 1.0
        return self.out(z.view(torch.float64).unsqueeze(-1))


class TensorStartNarrow(Base):
    """A ``narrow`` along a per-window axis whose start is a tensor buffer: an operand the axis rules do not read (the
    rule along a named axis takes no tensor operand but the merged one)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("start", torch.tensor(1))

    def forward(self, x):
        B = x.size(0)
        y = self.proj(self.partition(x))
        z = torch.cat([y[:, :1], torch.narrow(y, 1, self.start, N - 1) * 2.0], 1)
        return self.finish(self.reverse(z, B))


# Second resume: torch operators whose aten name has a pointwise overload beside a reduction (``max``, ``min``), other
# reductions and selections of the merged axis, a module's own ``dim`` on it, and leaf helpers named like the
# operators the rules take whose code mixes it.
MaxDim0 = _mixing("MaxDim0", lambda s, y: y - torch.max(y, 0, keepdim=True)[0])
MinDim0 = _mixing("MinDim0", lambda s, y: y - y.min(dim=0, keepdim=True).values)
Unbind0 = _mixing("Unbind0", lambda s, y: y * y.unbind(0)[0])
Select0 = _mixing("Select0", lambda s, y: y * y.select(0, 0))
Narrow0 = _mixing("Narrow0", lambda s, y: torch.cat([torch.narrow(y, 0, 1, y.size(0) - 1), torch.narrow(y, 0, 0, 1)],
                                                    0))
Sort0 = _mixing("Sort0", lambda s, y: torch.sort(y, 0)[0])
LogSumExp0 = _mixing("LogSumExp0", lambda s, y: y - torch.logsumexp(y, 0, keepdim=True))


class SoftmaxModule0(Base):
    """``nn.Softmax(dim=0)`` on the merged value."""

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        B = x.size(0)
        return self.finish(self.reverse(self.softmax(self.proj(self.partition(x))), B))


@torch.fx.wrap
def relu(y):
    """A leaf helper named like a pointwise operator whose code subtracts the mean over axis 0."""
    return y - y.mean(0, keepdim=True)


@torch.fx.wrap
def dropout(y, p, training):
    """A leaf helper named and called like ``F.dropout(y, p, False)`` whose code flips axis 0."""
    return y.flip(0) * (1.0 - p) if not training else y


class NamedPointwiseHelper(Base):
    def forward(self, x):
        B = x.size(0)
        return self.finish(self.reverse(relu(self.proj(self.partition(x))), B))


class NamedDropoutHelper(Base):
    def forward(self, x):
        B = x.size(0)
        return self.finish(self.reverse(dropout(self.proj(self.partition(x)), 0.5, False), B))
