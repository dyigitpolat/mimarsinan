""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gate_quant import OperandContraction, QuantLinear, QuantPatchEmbed, StaticGrid

ARMS = ("float", "twin", "operand")
ATTENTION_FORMS = ("softmax", "relu")


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    ""
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = (keep + torch.rand(shape, dtype=x.dtype, device=x.device)).floor_()
    return x.div(keep) * mask


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, *, qkv_bias: bool, attn_drop: float, proj_drop: float, act_bits: int,
                 weight_bits: int, payload_bits: int, name: str, attention: str = "softmax"):
        super().__init__()
        if attention not in ATTENTION_FORMS:
            raise ValueError(f"unknown attention form {attention!r}; declared {ATTENTION_FORMS}")
        self.attention = attention
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = QuantLinear(dim, 3 * dim, bias=qkv_bias, act_bits=act_bits, weight_bits=weight_bits, name=f"{name}.qkv")
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = QuantLinear(dim, dim, act_bits=act_bits, weight_bits=weight_bits, name=f"{name}.proj")
        self.proj_drop = nn.Dropout(proj_drop)

        self.qk = OperandContraction(heads, act_bits=act_bits, payload_bits=payload_bits, name=f"{name}.qk")
        self.pv = OperandContraction(heads, act_bits=act_bits, payload_bits=payload_bits, name=f"{name}.pv")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.qk(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) if self.attention == "softmax" else F.relu(attn) / n
        attn = self.attn_drop(attn)
        out = self.pv(attn, v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, *, block: str, qkv_bias: bool, attn_drop: float, drop: float,
                 drop_path_rate: float, act_bits: int, weight_bits: int, payload_bits: int, name: str,
                 attention: str = "softmax"):
        super().__init__()
        if block not in ("cct", "preln"):
            raise ValueError(f"unknown block {block!r}")
        self.block = block
        self.pre_norm = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                   act_bits=act_bits, weight_bits=weight_bits, payload_bits=payload_bits, name=name,
                                   attention=attention)
        self.linear1 = QuantLinear(dim, mlp_dim, act_bits=act_bits, weight_bits=weight_bits, name=f"{name}.linear1")
        self.dropout1 = nn.Dropout(drop)
        self.norm1 = nn.LayerNorm(dim)
        self.linear2 = QuantLinear(mlp_dim, dim, act_bits=act_bits, weight_bits=weight_bits, name=f"{name}.linear2")
        self.dropout2 = nn.Dropout(drop)
        self.drop_path_rate = float(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + drop_path(self.self_attn(self.pre_norm(x)), self.drop_path_rate, self.training)
        if self.block == "cct":
            x = self.norm1(x)
            mlp_in = x
        else:
            mlp_in = self.norm1(x)
        y = self.linear2(self.dropout1(F.gelu(self.linear1(mlp_in))))
        return x + drop_path(self.dropout2(y), self.drop_path_rate, self.training)


class ViTLite(nn.Module):
    def __init__(self, *, num_classes: int = 100, img_size: int = 32, in_ch: int = 3, patch: int = 4, dim: int = 256,
                 depth: int = 7, heads: int = 4, mlp_ratio: float = 2.0, block: str = "cct", qkv_bias: bool = False,
                 attn_drop: float = 0.1, drop: float = 0.0, stochastic_depth: float = 0.1, act_bits: int = 8,
                 weight_bits: int = 8, payload_bits: int = 8, attention: str = "softmax"):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.tokens = (img_size // patch) ** 2
        self.tokenizer = QuantPatchEmbed(in_ch, dim, patch, act_bits=act_bits, weight_bits=weight_bits)
        self.class_emb = nn.Parameter(torch.zeros(1, 1, dim))
        self.positional_emb = nn.Parameter(torch.zeros(1, self.tokens + 1, dim))
        nn.init.trunc_normal_(self.positional_emb, std=0.2)
        self.dropout = nn.Dropout(drop)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, depth)]
        self.blocks = nn.ModuleList(
            EncoderLayer(dim, heads, int(dim * mlp_ratio), block=block, qkv_bias=qkv_bias, attn_drop=attn_drop,
                         drop=drop, drop_path_rate=dpr[i], act_bits=act_bits, weight_bits=weight_bits,
                         payload_bits=payload_bits, name=f"blocks.{i}", attention=attention)
            for i in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.fc = QuantLinear(dim, num_classes, act_bits=act_bits, weight_bits=weight_bits, name="fc")
        self.apply(self._init_weight)
        self.attention = attention
        self.arm = "float"

    @staticmethod
    def _init_weight(m):
        if isinstance(m, QuantLinear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        cls = self.class_emb.expand(x.shape[0], -1, -1)
        x = torch.cat((cls.to(x.dtype), x), dim=1) + self.positional_emb.to(x.dtype)
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        return self.fc(self.norm(x)[:, 0])


    def crossbars(self) -> list:
        return [m for m in self.modules() if isinstance(m, (QuantLinear, QuantPatchEmbed))]

    def contractions(self) -> list[OperandContraction]:
        return [m for m in self.modules() if isinstance(m, OperandContraction)]

    def grids(self) -> dict[str, StaticGrid]:
        return {n: m for n, m in self.named_modules() if isinstance(m, StaticGrid)}

    def payload_grids(self) -> list[StaticGrid]:
        return [c.b_grid for c in self.contractions()]

    def observe(self, on: bool) -> None:
        ""
        self.set_arm("float")
        mode = "observe" if on else "off"
        for m in self.crossbars():
            m.in_grid.mode = mode
        for c in self.contractions():
            c.a_grid.mode = c.b_grid.mode = mode

    def drop_stats(self) -> None:
        for g in self.grids().values():
            g.reset_stats()

    def set_arm(self, arm: str) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; declared {ARMS}")
        quant = arm in ("twin", "operand")
        for m in self.crossbars():
            m.quantized = quant
            m.in_grid.mode = "on" if quant else "off"
        for c in self.contractions():
            c.on_chip = arm == "operand"
            c.a_on = c.b_on = True
            c.a_grid.mode = c.b_grid.mode = "on" if arm == "operand" else "off"
        self.arm = arm

    def configure(self, arm: str, *, payload_bits: int | None = None, rule: str = "toolchain", payload_on: bool = True,
                  streamed_on: bool = True, p_signed: bool = True, qp_pooled: bool = False) -> None:
        ""
        self.set_arm(arm)
        if arm == "float":
            return
        for m in self.crossbars():
            m.in_grid.apply_rule(rule)
        if arm != "operand":
            return
        for blk in self.blocks:
            a = blk.self_attn
            for c in (a.qk, a.pv):
                c.b_grid.apply_rule(rule, bits=payload_bits if payload_bits is not None else c.b_grid.bits,
                                    signed=True, pooled=False)
                c.a_grid.apply_rule(rule, signed=True if c is a.qk else bool(p_signed), pooled=bool(qp_pooled))
                c.a_on, c.b_on = bool(streamed_on), bool(payload_on)

    def model_parameters(self) -> list[tuple[str, nn.Parameter]]:
        ""
        return list(self.named_parameters())


def vit_lite_7_4(num_classes: int = 100, **kw) -> ViTLite:
    ""
    return ViTLite(num_classes=num_classes, img_size=32, patch=4, dim=256, depth=7, heads=4, mlp_ratio=2.0, **kw)


def network_parameter_count(model: ViTLite) -> int:
    return sum(p.numel() for _, p in model.model_parameters())


def leaf_state_to_gate(leaf_state: dict, depth: int) -> dict:
    ""
    out = {
        "tokenizer.weight": leaf_state["patch_embed.weight"], "tokenizer.bias": leaf_state["patch_embed.bias"],
        "class_emb": leaf_state["cls_token"], "positional_emb": leaf_state["pos_embed"],
        "norm.weight": leaf_state["norm.weight"], "norm.bias": leaf_state["norm.bias"],
        "fc.weight": leaf_state["head.weight"], "fc.bias": leaf_state["head.bias"],
    }
    for i in range(depth):
        p, g = f"blocks.{i}.", f"blocks.{i}."
        out.update({
            g + "pre_norm.weight": leaf_state[p + "norm1.weight"], g + "pre_norm.bias": leaf_state[p + "norm1.bias"],
            g + "self_attn.qkv.weight": leaf_state[p + "attn.in_proj_weight"],
            g + "self_attn.qkv.bias": leaf_state[p + "attn.in_proj_bias"],
            g + "self_attn.proj.weight": leaf_state[p + "attn.out_proj.weight"],
            g + "self_attn.proj.bias": leaf_state[p + "attn.out_proj.bias"],
            g + "norm1.weight": leaf_state[p + "norm2.weight"], g + "norm1.bias": leaf_state[p + "norm2.bias"],
            g + "linear1.weight": leaf_state[p + "fc1.weight"], g + "linear1.bias": leaf_state[p + "fc1.bias"],
            g + "linear2.weight": leaf_state[p + "fc2.weight"], g + "linear2.bias": leaf_state[p + "fc2.bias"],
        })
    return out
