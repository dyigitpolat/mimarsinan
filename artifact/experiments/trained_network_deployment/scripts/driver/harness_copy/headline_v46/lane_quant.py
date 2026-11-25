""

from __future__ import annotations

import operator
from dataclasses import dataclass, field

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F

import screen_quant
from mimarsinan.models.nn.activations.autograd import LIF_EXACT_QAT_THETA_FLOOR, _gated_lsq_backward
from screen_quant import BN_TYPES, CROSSBAR_TYPES, CrossbarInputQuantizer, QConv2d, QLinear, quantize_model

ACT_MODULES = (nn.ReLU, nn.ReLU6, nn.Hardtanh)
RELU_LIKE_MODULES = (nn.ReLU, nn.ReLU6)
REDUCE_MODULES = (nn.AvgPool1d, nn.AvgPool2d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d)
WIRING_MODULES = (nn.Flatten, nn.Identity)
MAX_MODULES = (nn.MaxPool1d, nn.MaxPool2d, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d)
ADD_FUNCTIONS = (operator.add, torch.add)
ACT_FUNCTIONS = (F.relu, torch.relu, F.relu6, F.hardtanh, torch.clamp)
REDUCE_FUNCTIONS = (F.avg_pool1d, F.avg_pool2d, F.adaptive_avg_pool1d, F.adaptive_avg_pool2d)
WIRING_FUNCTIONS = (torch.cat, torch.flatten, torch.reshape)
ADD_METHODS = ("add",)
ACT_METHODS = ("relu", "clamp")
WIRING_METHODS = ("view", "reshape", "flatten")
LANE_CLASS = {"input": "I", "mm": "L", "affine": "A", "add": "E", "reduce": "P", "act": "L0"}
CONTAINER = "lane_quant"


ROUNDINGS = ("nearest", "floor")
LANE_ROUNDING = "nearest"
INPUT_ROUNDING = "floor"


def nearest_forward(x, theta, shift, levels: int):
    ""
    dtype = torch.promote_types(x.dtype, torch.float32)
    safe = theta.to(dtype).clamp(min=LIF_EXACT_QAT_THETA_FLOOR)
    r = (x.to(dtype) + shift.to(dtype)) / safe
    q = torch.floor(levels * r + 0.5).clamp(0.0, float(levels)) / levels
    return (safe * q - shift.to(dtype)).to(x.dtype)


def nearest_backward(g, x, theta, shift, levels: int):
    ""
    dtype = torch.promote_types(x.dtype, torch.float32)
    safe = theta.to(dtype).clamp(min=LIF_EXACT_QAT_THETA_FLOOR)
    r = (x.to(dtype) + shift.to(dtype)) / safe
    q = torch.floor(levels * r + 0.5).clamp(0.0, float(levels)) / levels
    g = g.to(dtype)
    grad_z, grad_theta = _gated_lsq_backward(g, r, q, safe)
    return grad_z.to(x.dtype), grad_theta.to(theta.dtype), (grad_z.sum() - g.sum()).to(shift.dtype)


_FUSED_NEAREST: dict = {}


def _nearest_kernels(x: torch.Tensor):
    if not (screen_quant.FUSE_ON_CUDA and x.is_cuda):
        return nearest_forward, nearest_backward
    if not _FUSED_NEAREST:
        _FUSED_NEAREST["fwd"] = torch.compile(nearest_forward, dynamic=True)
        _FUSED_NEAREST["bwd"] = torch.compile(nearest_backward, dynamic=True)
    return _FUSED_NEAREST["fwd"], _FUSED_NEAREST["bwd"]


class NearestCountStaircase(torch.autograd.Function):
    ""

    @staticmethod
    def forward(ctx, x, theta, shift, levels):
        ctx.save_for_backward(x, theta, shift)
        ctx.levels = levels
        return _nearest_kernels(x)[0](x, theta, shift, levels)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (g,) = grad_outputs
        x, theta, shift = ctx.saved_tensors
        grad_x, grad_theta, grad_shift = _nearest_kernels(x)[1](g, x, theta, shift, ctx.levels)
        return grad_x, grad_theta, (grad_shift if ctx.needs_input_grad[2] else None), None


def fused_kernel_check(device, n: int = 1 << 22) -> dict:
    ""
    out = {f"floor_{k}": v for k, v in screen_quant.fused_kernel_check(device, n).items()}
    g = torch.Generator(device="cpu").manual_seed(0)
    for signed in (False, True):
        x = torch.randn(n, generator=g).to(device)
        x = (x if signed else torch.relu(x)).to(torch.bfloat16)
        theta = torch.tensor(2.3, device=device)
        shift = torch.tensor(1.7 if signed else 0.0, device=device)
        grad = torch.randn(n, generator=g).to(device).to(torch.bfloat16)
        fwd, bwd = _nearest_kernels(x)
        y_f, y_e = fwd(x, theta, shift, 15), nearest_forward(x, theta, shift, 15)
        b_f, b_e = bwd(grad, x, theta, shift, 15), nearest_backward(grad, x, theta, shift, 15)
        out["nearest_signed" if signed else "nearest_unsigned"] = {
            "forward_mismatches": int((y_f != y_e).sum()), "elements": n,
            "grad_x_mismatches": int((b_f[0] != b_e[0]).sum()),
            "grad_theta_rel_err": float((b_f[1] - b_e[1]).abs() / b_e[1].abs().clamp_min(1e-12)),
        }
    return out


class LaneOutputQuantizer(CrossbarInputQuantizer):
    ""

    def __init__(self, bits: int, rounding: str = LANE_ROUNDING):
        super().__init__(bits)
        if rounding not in ROUNDINGS:
            raise ValueError(rounding)
        self.rounding = rounding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode != "quantize" or self.rounding == "floor":
            return super().forward(x)
        if not self.calibrated:
            raise RuntimeError("LaneOutputQuantizer used before calibration")
        shift = self.act_shift if self.signed else torch.zeros((), device=x.device)
        return NearestCountStaircase.apply(x, self.act_theta, shift, self.levels)


class ProducerQuantized(nn.Module):
    ""

    def __init__(self):
        super().__init__()
        self.mode = "bypass"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _is_zero_pad(node: torch.fx.Node) -> bool:
    mode = node.kwargs.get("mode", node.args[2] if len(node.args) > 2 else "constant")
    value = node.kwargs.get("value", node.args[3] if len(node.args) > 3 else None)
    return mode == "constant" and value in (None, 0, 0.0)


def classify(node: torch.fx.Node, modules: dict) -> str:
    ""
    if node.op == "placeholder":
        return "input"
    if node.op == "output":
        return "output"
    if node.op == "call_module":
        m = modules[node.target]
        if isinstance(m, CROSSBAR_TYPES):
            return "mm"
        if isinstance(m, BN_TYPES):
            return "affine"
        if isinstance(m, ACT_MODULES):
            return "act"
        if isinstance(m, REDUCE_MODULES):
            return "reduce"
        if isinstance(m, WIRING_MODULES):
            return "wiring"
        if isinstance(m, MAX_MODULES):
            raise ValueError(f"{node.name}: max pooling needs class C, which is off in this sensitivity")
        raise ValueError(f"{node.name}: module {type(m).__name__} is outside the lane classes")
    if node.op == "call_function":
        if node.target in ADD_FUNCTIONS:
            return "add"
        if node.target in ACT_FUNCTIONS:
            return "act"
        if node.target in REDUCE_FUNCTIONS:
            return "reduce"
        if node.target in WIRING_FUNCTIONS:
            return "wiring"
        if node.target is F.pad:
            if not _is_zero_pad(node):
                raise ValueError(f"{node.name}: only zero constant padding is wiring")
            return "wiring"
        raise ValueError(f"{node.name}: function {getattr(node.target, '__name__', node.target)} is outside the lane classes")
    if node.op == "call_method":
        if node.target in ADD_METHODS:
            return "add"
        if node.target in ACT_METHODS:
            return "act"
        if node.target in WIRING_METHODS:
            return "wiring"
        raise ValueError(f"{node.name}: method {node.target} is outside the lane classes")
    raise ValueError(f"{node.name}: op {node.op} is outside the lane classes")


def tensor_inputs(node: torch.fx.Node) -> list[torch.fx.Node]:
    found: list[torch.fx.Node] = []
    torch.fx.node.map_arg((node.args, node.kwargs), lambda a: found.append(a))
    return found


@dataclass
class Lane:
    index: int
    nodes: list = field(default_factory=list)
    kinds: list = field(default_factory=list)
    quantized: bool = False
    logits: bool = False
    quantizer: str | None = None

    @property
    def head(self) -> torch.fx.Node:
        return self.nodes[0]

    @property
    def tail(self) -> torch.fx.Node:
        return self.nodes[-1]

    @property
    def lane_class(self) -> str:
        return LANE_CLASS[self.kinds[0]]

    def admits(self, kind: str) -> bool:
        if "act" in self.kinds or self.kinds[0] == "input":
            return False
        if kind == "add":
            return "add" not in self.kinds
        if kind == "reduce":
            return "reduce" not in self.kinds
        return kind in ("affine", "act")

    def describe(self) -> dict:
        return {"index": self.index, "class": self.lane_class, "head": self.head.name, "tail": self.tail.name,
                "kinds": list(self.kinds), "nodes": [n.name for n in self.nodes], "quantized": self.quantized,
                "logits": self.logits, "quantizer": self.quantizer}


@dataclass
class LanePlan:
    lanes: list
    wiring: list

    def census(self) -> dict:
        def count(pred):
            return sum(1 for lane in self.lanes if pred(lane))

        mm = [lane for lane in self.lanes if lane.lane_class == "L"]
        return {
            "lanes": len(self.lanes),
            "I": count(lambda l: l.lane_class == "I"),
            "L_with_add": sum(1 for l in mm if "add" in l.kinds),
            "L_without_add": sum(1 for l in mm if "add" not in l.kinds),
            "L_with_reduction": sum(1 for l in mm if "reduce" in l.kinds),
            "L_logits": sum(1 for l in mm if l.logits),
            "A": count(lambda l: l.lane_class == "A"),
            "E": count(lambda l: l.lane_class == "E"),
            "P": count(lambda l: l.lane_class == "P"),
            "L0": count(lambda l: l.lane_class == "L0"),
            "W": len(self.wiring),
            "quantizers": count(lambda l: l.quantized),
            "affine_in_lanes": sum(l.kinds.count("affine") for l in self.lanes),
        }


def plan_lanes(gm: torch.fx.GraphModule) -> LanePlan:
    ""
    modules = dict(gm.named_modules())
    lanes: list[Lane] = []
    open_tail: dict[torch.fx.Node, Lane] = {}
    closed: set[torch.fx.Node] = set()
    wiring: list[str] = []

    def new_lane(node, kind) -> Lane:
        lane = Lane(index=len(lanes), nodes=[node], kinds=[kind])
        lanes.append(lane)
        return lane

    def close(p: torch.fx.Node) -> None:
        if p in closed:
            return
        if p not in open_tail:
            raise ValueError(f"{p.name}: consumed before it was produced by a lane")
        lane = open_tail.pop(p)
        lane.quantized = True
        closed.add(p)

    for node in gm.graph.nodes:
        kind = classify(node, modules)
        ins = tensor_inputs(node)
        if len(set(ins)) != len(ins):
            raise ValueError(f"{node.name}: an op reading one tensor twice is outside the lane grammar")
        if kind == "input":
            lane = new_lane(node, "input")
            lane.quantized = True
            closed.add(node)
        elif kind == "wiring":
            for p in ins:
                close(p)
            closed.add(node)
            wiring.append(node.name)
        elif kind == "output":
            for p in ins:
                if p in open_tail:
                    open_tail.pop(p).logits = True
        elif kind == "mm":
            for p in ins:
                close(p)
            open_tail[node] = new_lane(node, "mm")
        else:
            if kind == "add" and len(ins) < 2:
                raise ValueError(f"{node.name}: an add with a constant operand is outside the lane grammar")
            joinable = [p for p in ins if p in open_tail and len(p.users) == 1 and open_tail[p].admits(kind)]
            if joinable:
                p = joinable[0]
                lane = open_tail.pop(p)
                for q in ins:
                    if q is not p:
                        close(q)
                lane.nodes.append(node)
                lane.kinds.append(kind)
                open_tail[node] = lane
            else:
                for q in ins:
                    close(q)
                open_tail[node] = new_lane(node, kind)
    if open_tail:
        raise ValueError(f"lanes without a consumer: {[n.name for n in open_tail]}")
    return LanePlan(lanes=lanes, wiring=wiring)


def _bn_after_mm(plan: LanePlan) -> dict[str, str]:
    ""
    out = {}
    for lane in plan.lanes:
        if lane.kinds[:2] == ["mm", "affine"] and lane.nodes[1].args[0] is lane.nodes[0]:
            out[lane.nodes[0].target] = lane.nodes[1].target
    return out


def insert_quantizers(gm: torch.fx.GraphModule, plan: LanePlan, bits: int) -> None:
    ""
    device = next((p.device for p in gm.parameters()), torch.device("cpu"))
    if not hasattr(gm, CONTAINER):
        gm.add_module(CONTAINER, nn.Module())
    container = getattr(gm, CONTAINER)
    for lane in plan.lanes:
        if not lane.quantized:
            continue
        tail = lane.tail
        name = tail.name
        rounding = INPUT_ROUNDING if lane.lane_class == "I" else LANE_ROUNDING
        container.add_module(name, LaneOutputQuantizer(bits, rounding).to(device))
        target = f"{CONTAINER}.{name}"
        with gm.graph.inserting_after(tail):
            qn = gm.graph.call_module(target, (tail,))
        tail.replace_all_uses_with(qn, delete_user_cb=lambda user, qn=qn: user is not qn)
        lane.quantizer = target
    gm.graph.lint()
    gm.recompile()


def lane_quantize(model: nn.Module, w_bits: int, a_bits: int) -> tuple[torch.fx.GraphModule, dict]:
    ""
    gm = torch.fx.symbolic_trace(model)
    plan = plan_lanes(gm)
    weights = quantize_model(gm, w_bits, a_bits)
    folded = {name: m._paired_bn[0] for name, m in gm.named_modules() if isinstance(m, (QConv2d, QLinear)) and m._paired_bn}
    named = dict(gm.named_modules())
    lane_pairs = _bn_after_mm(plan)
    if {k: named[v] for k, v in lane_pairs.items()} != folded:
        raise RuntimeError("lane partition and the screen's BN fold disagree on the MM-BN pairs")
    for m in gm.modules():
        if isinstance(m, (QConv2d, QLinear)):
            m.in_q = ProducerQuantized()
    insert_quantizers(gm, plan, a_bits)
    gm.lane_plan = plan
    census = {**plan.census(), "crossbars": weights["crossbars"], "folded_bn": weights["folded_bn"],
              "w_bits": int(w_bits), "a_bits": int(a_bits), **rounding_census(gm)}
    return gm, census


def lane_quantizers(model: nn.Module) -> list[tuple[str, LaneOutputQuantizer]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LaneOutputQuantizer)]


def rounding_census(model: nn.Module) -> dict:
    ""
    plan: LanePlan | None = getattr(model, "lane_plan", None)
    modules = dict(model.named_modules())
    counts = {r: 0 for r in ROUNDINGS}
    for _, q in lane_quantizers(model):
        counts[q.rounding] += 1
    declared = plan is not None and all(
        modules[lane.quantizer].rounding == (INPUT_ROUNDING if lane.lane_class == "I" else LANE_ROUNDING)
        for lane in plan.lanes if lane.quantized and lane.quantizer)
    return {"quantizers_nearest": counts["nearest"], "quantizers_floor": counts["floor"],
            "rounding_as_declared": bool(declared)}


def set_lane_rounding(model: nn.Module, rounding: str, include_input: bool = False) -> int:
    ""
    if rounding not in ROUNDINGS:
        raise ValueError(rounding)
    plan: LanePlan = model.lane_plan
    modules = dict(model.named_modules())
    n = 0
    for lane in plan.lanes:
        if lane.quantized and lane.quantizer and (include_input or lane.lane_class != "I"):
            modules[lane.quantizer].rounding = rounding
            n += 1
    return n


def set_lane_mode(model: nn.Module, mode: str) -> None:
    ""
    if mode not in ("quantize", "observe", "bypass"):
        raise ValueError(mode)
    for m in model.modules():
        if isinstance(m, (LaneOutputQuantizer, ProducerQuantized)):
            m.mode = mode


@torch.no_grad()
def calibrate_lanes(model: nn.Module, batches) -> dict:
    ""
    was_training = model.training
    model.eval()
    set_lane_mode(model, "observe")
    for x in batches:
        model(x)
    stats = {name: q.finalize_calibration() for name, q in lane_quantizers(model)}
    set_lane_mode(model, "quantize")
    model.train(was_training)
    plan: LanePlan | None = getattr(model, "lane_plan", None)
    relu_signed = []
    if plan is not None:
        modules = dict(model.named_modules())
        for lane in plan.lanes:
            last = lane.nodes[-1]
            if (lane.quantized and lane.kinds[-1] == "act" and last.op == "call_module"
                    and isinstance(modules.get(last.target), RELU_LIKE_MODULES) and stats[lane.quantizer]["signed"]):
                relu_signed.append(lane.quantizer)
    return {"quantizers": len(stats), "signed": sum(s["signed"] for s in stats.values()),
            "relu_lanes_signed": relu_signed, "per_lane": stats}


def lane_table(model: nn.Module) -> list[dict]:
    ""
    plan: LanePlan = model.lane_plan
    modules = dict(model.named_modules())
    rows = []
    for lane in plan.lanes:
        row = lane.describe()
        if lane.quantizer:
            q = modules[lane.quantizer]
            row.update(signed=bool(q.signed), theta=float(q.act_theta.detach()), shift=float(q.act_shift.detach()),
                       rounding=q.rounding)
        rows.append(row)
    return rows
