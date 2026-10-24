"""Witness models of batch-free values: the relative-position biases of Swin, MaxViT and Swin V2 at toy size, and the
forms the batch-free rule must keep on their existing path (a random draw, a write in place, an opaque call)."""

from __future__ import annotations

import torch
import torch.fx
import torch.nn as nn

from operand_completeness.harness import keep_opaque

N = 3  # window side: N * N tokens per window
T = N * N
H = 2  # heads
SHAPE = (H, T, T)  # per-sample attention logits


def relative_index(n: int = N) -> torch.Tensor:
    """Swin's relative-position index of an n x n window, flattened (n**4,), into a (2n - 1)**2 table."""
    coords = torch.stack(torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")).flatten(1)
    rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
    rel[:, :, 0] += n - 1
    rel[:, :, 1] += n - 1
    rel[:, :, 0] *= 2 * n - 1
    return rel.sum(-1).flatten()


def relative_bias(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """The gather, view, permute, copy and unsqueeze of torchvision's ``_get_relative_position_bias``."""
    bias = table[index].view(T, T, -1)
    return bias.permute(2, 0, 1).contiguous().unsqueeze(0)


def _wrapped_bias(table: torch.Tensor, index: torch.Tensor, window: list) -> torch.Tensor:
    """An FX leaf function of the bias, as torchvision registers Swin's (leaf-call inlining splices its body)."""
    side = window[0] * window[1]
    bias = table[index].view(side, side, -1)
    return bias.permute(2, 0, 1).contiguous().unsqueeze(0)


torch.fx.wrap("_wrapped_bias")


def _opaque_attention(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """An FX leaf function whose body branches on a value (it stays opaque, so the program is impure)."""
    keep_opaque(logits)
    return torch.softmax(logits + bias, dim=-1)


torch.fx.wrap("_opaque_attention")


def _opaque_scale(table: torch.Tensor) -> torch.Tensor:
    keep_opaque(table)
    return table * 2.0


torch.fx.wrap("_opaque_scale")


def _double_in_place(value: torch.Tensor) -> torch.Tensor:
    keep_opaque(value)
    value.mul_(2.0)
    return value


torch.fx.wrap("_double_in_place")


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.randn((2 * N - 1) ** 2, H))
        self.register_buffer("index", relative_index())
        self.head = nn.Linear(H * T * T, 5)

    def out(self, logits: torch.Tensor) -> torch.Tensor:
        return self.head(logits.flatten(1))


class RelativeBias(_Head):
    """Swin's and MaxViT's relative-position bias added to the attention logits."""

    def forward(self, x):
        return self.out(torch.softmax(x + relative_bias(self.table, self.index), dim=-1))


class WrappedBias(_Head):
    """The bias computed in an FX leaf function, as Swin's ``_get_relative_position_bias``."""

    def forward(self, x):
        return self.out(torch.softmax(x + _wrapped_bias(self.table, self.index, [N, N]), dim=-1))


class CoordinateBias(nn.Module):
    """Swin V2's continuous position bias: an MLP on a coordinate buffer, gathered, permuted and squashed."""

    def __init__(self):
        super().__init__()
        side = torch.arange(-(N - 1), N, dtype=torch.float32)
        table = torch.stack(torch.meshgrid([side, side], indexing="ij")).permute(1, 2, 0).contiguous().unsqueeze(0)
        self.register_buffer("coords", table / (N - 1))
        self.register_buffer("index", relative_index())
        self.cpb = nn.Sequential(nn.Linear(2, 8), nn.ReLU(inplace=True), nn.Linear(8, H, bias=False))
        self.head = nn.Linear(H * T * T, 5)

    def forward(self, x):
        bias = 16 * torch.sigmoid(relative_bias(self.cpb(self.coords).view(-1, H), self.index))
        return self.head(torch.softmax(x + bias, dim=-1).flatten(1))


class OpaqueConsumer(_Head):
    """The bias read by a call that stays opaque (Swin's shifted-window attention): an impure program."""

    def forward(self, x):
        return self.out(_opaque_attention(x, relative_bias(self.table, self.index)))


class TwoConsumers(_Head):
    """One bias read by two operators."""

    def forward(self, x):
        bias = relative_bias(self.table, self.index)
        return self.out(torch.softmax(x + bias, dim=-1) * bias)


class ClassToken(nn.Module):
    """A computed class token, expanded to the batch and concatenated before the tokens."""

    def __init__(self):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, T))
        self.scale = nn.Parameter(torch.randn(T))
        self.head = nn.Linear((T + 1) * T, 5)

    def forward(self, x):
        tokens = x.flatten(1, 2)  # (B, H * T, T)
        token = self.cls * self.scale
        joined = torch.cat([token.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        return self.head(joined[:, : T + 1].flatten(1))


class AttentionMask(nn.Module):
    """A float attention mask gathered from a table, given to ``nn.MultiheadAttention``."""

    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.randn((2 * N - 1) ** 2))
        self.register_buffer("index", relative_index())
        self.mha = nn.MultiheadAttention(4, 2, batch_first=True)
        self.head = nn.Linear(T * 4, 5)

    def forward(self, x):  # x: (B, T, 4)
        mask = self.table[self.index].view(T, T)
        return self.head(self.mha(x, x, x, attn_mask=mask, need_weights=False)[0].flatten(1))


class ConstantOutput(nn.Module):
    """An output computed from the state alone: it has no batch axis."""

    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.randn(4, 5))

    def forward(self, x):
        return (self.table * 2.0)[0:1]


class PlainParameter(nn.Module):
    """A parameter read as it is: the bound-tensor path of the base, unchanged."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(H, T, T))
        self.head = nn.Linear(H * T * T, 5)

    def forward(self, x):
        return self.head((x + self.bias).flatten(1))


class RandomCone(_Head):
    """A random draw in the bias's computation: drawn once per call natively, so never a batch-free value."""

    def forward(self, x):
        noisy = self.table + torch.randn(self.table.shape, dtype=self.table.dtype)
        return self.out(x + relative_bias(noisy, self.index))


class WrittenState(_Head):
    """A parameter the forward writes in place, read by the bias's computation."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()), requires_grad=False)

    def forward(self, x):
        scale = self.scale.mul_(1.5)
        return self.out(x + relative_bias(self.table * scale, self.index))


class OpaqueCone(_Head):
    """The bias computed through a call that stays opaque."""

    def forward(self, x):
        return self.out(x + relative_bias(_opaque_scale(self.table), self.index))


class WrittenByOpaqueCall(_Head):
    """An opaque call writes the bias in place, and a later operator reads it: it reads the written value."""

    def forward(self, x):
        bias = relative_bias(self.table, self.index)
        doubled = _double_in_place(bias)
        return self.out(x + bias + doubled)


class InputDtypeFactory(nn.Module):
    """A factory reading an activation's dtype and device (``x.new_ones``): a metadata read, batch-free (coverage
    W8; in W6 it read the input)."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(H * T * T, 5)

    def forward(self, x):
        return self.head((x + x.new_ones(T)).flatten(1))


class StateShape(nn.Module):
    """A shape read from a batch-free value: a static shape value, folded at conversion as before."""

    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.randn(T, 2))
        self.head = nn.Linear(H * T * T, 5)

    def forward(self, x):
        width = (self.table * 2.0).shape[0]
        return self.head(x.reshape(x.shape[0], -1, width).flatten(1))


class StateViewToOpaqueCall(_Head):
    """A view of the table given to a call that stays opaque (which may write it), the bias gathered after it: the
    table is not batch-free state (the tracer keeps the view, ``KEPT_OPERANDS``), so the bias is not batch-free."""

    def forward(self, x):
        scaled = _opaque_scale(self.table.view(-1))
        return self.out(x + relative_bias(self.table, self.index) + scaled.sum())


class StateViewToLaterOpaqueCall(_Head):
    """The same with the bias gathered before the opaque call (the tracer orders the call after its readers,
    ``RUN_AFTER``)."""

    def forward(self, x):
        bias = relative_bias(self.table, self.index)
        scaled = _opaque_scale(self.table.view(-1))
        return self.out(x + bias + scaled.sum())


class StateCopyToOpaqueCall(_Head):
    """A copy of the table (``clone``) given to a call that stays opaque: the table stays batch-free state."""

    def forward(self, x):
        scaled = _opaque_scale(self.table.clone())
        return self.out(x + relative_bias(self.table, self.index) + scaled.sum())


class RandomMask(nn.Module):
    """A float mask gathered from a random draw (not batch-free), given to ``nn.MultiheadAttention``."""

    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.randn((2 * N - 1) ** 2))
        self.register_buffer("index", relative_index())
        self.mha = nn.MultiheadAttention(4, 2, batch_first=True)
        self.head = nn.Linear(T * 4, 5)

    def forward(self, x):  # x: (B, T, 4)
        mask = torch.rand_like(self.table)[self.index].view(T, T)
        return self.head(self.mha(x, x, x, attn_mask=mask, need_weights=False)[0].flatten(1))
