""

from __future__ import annotations

import json
import random
import sys
import time

TREE = sys.argv[1] if len(sys.argv) > 1 else ""
sys.path.insert(0, TREE + "/tests")

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
try:
    from mimarsinan.mapping.ir.deployment_dtype import VALUE_ARITHMETIC
except ImportError as exc:
    raise SystemExit(f"REFUSED: RECORD_ABSENT: the head carries no record of the deployed arithmetic "
                     f"(mapping/ir/deployment_dtype.py, U-DAR): {exc}")
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping

torch.set_num_threads(1)


RANGES = {"kinds": ["mlp", "conv", "residual"], "mlp_input": [4, 24], "mlp_layers": [2, 5], "mlp_width": [4, 32],
          "conv_channels_in": [1, 3], "conv_side": [6, 10], "conv_layers": [1, 3], "conv_channels": [2, 8],
          "residual_blocks": [1, 2], "residual_channels": [4, 8], "classes": [2, 6], "axons": [16, 32, 64, 128, 256],
          "neurons": [8, 16, 32], "coalescing_rate": 0.75, "max_passes": 64, "batch": 16, "input_spread": 3}


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout)
        self.conv2, self.bn2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout)
        self.relu1, self.relu2 = nn.ReLU(), nn.ReLU()
        self.shortcut = nn.Sequential() if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False),
                                                                          nn.BatchNorm2d(cout))

    def forward(self, x):
        return self.relu2(self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x))))) + self.shortcut(x))


class _Residual(nn.Module):
    def __init__(self, widths, classes: int) -> None:
        super().__init__()
        blocks, cin = [], 3
        for cout in widths:
            blocks.append(_Block(cin, cout))
            cin = cout
        self.blocks, self.pool, self.fc = nn.Sequential(*blocks), nn.AdaptiveAvgPool2d(1), nn.Linear(cin, classes)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.blocks(x)), 1))


def _randomize_bn(model: nn.Module, seed: int) -> nn.Module:
    generator = torch.Generator().manual_seed(seed + 1)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.running_mean.copy_(torch.randn(module.num_features, generator=generator) * 0.1)
            module.running_var.copy_(torch.rand(module.num_features, generator=generator) + 0.5)
    return model.eval()


def network(seed: int):
    ""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    kind = rng.choice(RANGES["kinds"])
    classes = rng.randint(*RANGES["classes"])
    if kind == "mlp":
        width_in = rng.randint(*RANGES["mlp_input"])
        widths = [rng.randint(*RANGES["mlp_width"]) for _ in range(rng.randint(*RANGES["mlp_layers"]) - 1)]
        layers, cin = [], width_in
        for k, w in enumerate(widths):
            layers += [nn.Linear(cin, w)] + ([nn.BatchNorm1d(w)] if k == 0 and rng.random() < 0.5 else []) + [nn.ReLU()]
            cin = w
        model, shape = nn.Sequential(*layers, nn.Linear(cin, classes)), (width_in,)
        facts = dict(kind=kind, layers=len(widths) + 1, widths=widths + [classes])
    elif kind == "conv":
        cin, side = rng.randint(*RANGES["conv_channels_in"]), rng.randint(*RANGES["conv_side"])
        chans = [rng.randint(*RANGES["conv_channels"]) for _ in range(rng.randint(*RANGES["conv_layers"]))]
        layers, c = [], cin
        for ch in chans:
            layers += [nn.Conv2d(c, ch, 3, padding=1), nn.ReLU()]
            c = ch
        model, shape = nn.Sequential(*layers, nn.Flatten(), nn.Linear(c * side * side, classes)), (cin, side, side)
        facts = dict(kind=kind, layers=len(chans) + 1, widths=chans + [classes], side=side)
    else:
        widths = [rng.randint(*RANGES["residual_channels"]) for _ in range(rng.randint(*RANGES["residual_blocks"]))]
        model, shape = _Residual(widths, classes), (3, 8, 8)
        facts = dict(kind=kind, layers=2 * len(widths) + 1, widths=widths + [classes], side=8)
    return _randomize_bn(model, seed), shape, classes, facts


def _equal(a, b) -> bool:
    return bool(torch.equal(torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)))


def main() -> int:
    from unit.chip_simulation.value_run.integer_programs import build_integer_program, integer_inputs

    out_path, s0, s1 = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    counts = [int(c) for c in sys.argv[5].split(",")]
    with open(out_path, "a") as out:
        for seed in range(s0, s1):
            started = time.time()
            try:
                model, shape, classes, facts = network(seed)
                program = build_integer_program(model, shape, num_classes=classes, seed=seed)
                x = integer_inputs(program, RANGES["batch"], seed=seed + 1, spread=RANGES["input_spread"]).double()
                identity_mapping = build_identity_hybrid_mapping(ir_graph=program.ir)
                with torch.no_grad():
                    identity = ValueHybridCoreFlow(identity_mapping, device=VALUE_ARITHMETIC.device, dtype=VALUE_ARITHMETIC.core_dtype)(x)
                codes, steps = run_golden(identity_mapping, x.numpy())
                golden_identity = _equal(codes * steps, identity)
            except Exception as exc:
                print(json.dumps(dict(seed=seed, family="value", vehicle_error=f"{type(exc).__name__}: {str(exc)[:200]}")),
                      file=out, flush=True)
                continue
            cores = program.ir.get_neural_cores()
            facts.update(ir_cores=len(cores), host_ops=len(program.ir.get_compute_ops()),
                         max_fan_in=max((int(c.get_input_count()) for c in cores), default=0),
                         max_width=max((int(c.get_output_count()) for c in cores), default=0))
            nonzero = int(np.count_nonzero(codes))
            rng = random.Random(10_000 + seed)
            for count in counts:
                axons, neurons = rng.choice(RANGES["axons"]), rng.choice(RANGES["neurons"])
                coalescing = rng.random() < RANGES["coalescing_rate"]
                row = dict(seed=seed, count=count, family="value", vehicle=facts, axons=axons, neurons=neurons,
                           coalescing=coalescing, golden_identity=golden_identity, nonzero_codes=nonzero)
                t0 = time.time()
                try:
                    mapping = program.packed_mapping(axons=axons, neurons=neurons, count=count, split=True,
                                                     coalescing=coalescing, passes=RANGES["max_passes"])
                except ToolchainRefusal as exc:
                    row.update(build="refused", refusal_class=type(exc).__name__, reason=str(exc)[:240])
                    print(json.dumps(row), file=out, flush=True)
                    continue
                except Exception as exc:
                    row.update(build="error", reason=f"{type(exc).__name__}: {str(exc)[:240]}")
                    print(json.dumps(row), file=out, flush=True)
                    continue
                passes = [len(s.hard_core_mapping.cores) for s in mapping.stages if s.kind == "neural"]
                identity_stages = sum(1 for s in identity_mapping.stages if s.kind == "neural")
                row.update(build="built", passes=passes, multipass=len(passes) > identity_stages)
                try:
                    with torch.no_grad():
                        packed = ValueHybridCoreFlow(mapping, device=VALUE_ARITHMETIC.device, dtype=VALUE_ARITHMETIC.core_dtype)(x)
                    packed_codes, packed_steps = run_golden(mapping, x.numpy())
                    row["value"] = "equal" if _equal(packed, identity) else "DIFFERS"
                    row["value_golden"] = "equal" if _equal(packed_codes * packed_steps, packed) else "DIFFERS"
                    row["value_disc"] = float(identity.std(0).max()) > 0 if identity.shape[0] > 1 else False
                except Exception as exc:
                    row["value"] = f"run_error {type(exc).__name__}: {str(exc)[:100]}"
                row["wall_s"] = round(time.time() - t0, 3)
                print(json.dumps(row), file=out, flush=True)
            del started
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
