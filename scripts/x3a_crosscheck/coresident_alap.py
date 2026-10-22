"""Probe (proof check, part 2): H1-latency is checked on the soft cores' IR latency (ASAP depth), while the
streamed executors run ChipLatency's latencies on the hard-core graph (ASAP, then every shiftable producer
raised to its deepest consumer minus one). Two soft cores of equal ASAP depth and unequal ALAP latency pass
the packing predicate (``canonical.canonical_is_mapping_possible``); sharing one hard core then raises the
shallow one, and its in-segment consumer reads it at delivery delay d = -1.

Instance (streamed LIF, every layer observed): A = n(input) -> B = n(A) -> Z = n(B, X); X = n(input);
Y = n(input) -> V = n(Y). ASAP: A, X, Y at 0; B, V at 1; Z at 2. ALAP: X at 1, Y at 0. X and Y have one
threshold, so under value-keyed residency classes (the screen's ``on_values_classes`` setting, every
hypothesis on) they may share a core; the declared budget (5 cores of 8 x 2) leaves the packer no other way.
Usage: ``python coresident_alap.py OUT.json``.
"""

from __future__ import annotations

import json
from typing import Any, Dict
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from mimarsinan.certification.randomized import executors as ex  # noqa: E402
from mimarsinan.certification.randomized.arms import ARMS, ON_VALUES  # noqa: E402
from mimarsinan.certification.randomized.cost_draws import platform_constraints  # noqa: E402
from mimarsinan.certification.randomized.instance import Instance, Layer, Network, STREAMED_LIF  # noqa: E402
from mimarsinan.certification.randomized.runner import _run, compare, deployment, reference_nodes, structure  # noqa: E402


def instance(T: int = 4) -> Instance:
    ones = lambda n, k=1: tuple(tuple(k for _ in range(n)) for _ in range(1))  # noqa: E731
    layers = (
        Layer("A", "neural", ("input",), 1, weights=ones(4), bias=(0,), theta=2),
        Layer("B", "neural", ("A",), 1, weights=ones(1), bias=(0,), theta=3),
        Layer("X", "neural", ("input",), 1, weights=ones(4), bias=(0,), theta=1),
        Layer("Z", "neural", ("B", "X"), 1, weights=ones(2), bias=(0,), theta=4),
        Layer("Y", "neural", ("input",), 1, weights=ones(4), bias=(0,), theta=1),
        Layer("V", "neural", ("Y",), 1, weights=ones(1), bias=(0,), theta=1),
    )
    net = Network(4, layers, tuple(layer.name for layer in layers))
    return Instance(seed=0, semantics=STREAMED_LIF, timesteps=T, hardware_bias=True, network=net,
                    inputs=((T, T, T, T), (1, 2, 3, 4)))


def main(argv) -> int:
    inst = instance()
    arm = ARMS[ON_VALUES]
    decl = {"cores": [{"max_axons": 8, "max_neurons": 2, "count": 5}], "allow_coalescing": False,
            "allow_neuron_splitting": False, "allow_scheduling": False, "max_schedule_passes": 1,
            "core_value_granularity": {}}
    out: Dict[str, Any] = {"declaration": decl}
    ref_ir, _ref_map, ref_nodes = reference_nodes(inst, arm)
    with arm.engaged():
        ir, mapping = deployment(inst, platform_constraints(decl, hardware_bias=True), 0, arm, relabel=False)
        out["structure"] = structure(mapping)
        nodes = _run(mapping, inst, arm)
        out["windows"], out["mismatches"], out["first"] = compare(ref_ir, ref_nodes, ir, nodes)
    cores = []
    for stage in mapping.stages:
        hcm = stage.hard_core_mapping
        if stage.kind == "neural" and hcm is not None:
            for i, core in enumerate(hcm.cores):
                placements = (getattr(hcm, "soft_core_placements_per_hard_core", None) or [[]] * len(hcm.cores))[i]
                cores.append({"core": i, "latency": core.latency,
                              "soft": [int(p.get("ir_node_id", -1)) for p in placements]})
    out["cores"] = cores
    want, got = ex.per_neuron(ref_ir, ref_nodes), ex.per_neuron(ir, nodes)
    out["per_layer"] = {str(k): {"identity": want[k].tolist(), "deployed": got[k].tolist()} for k in sorted(want)}
    Path(argv[0]).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: out[k] for k in ("structure", "mismatches", "first", "cores")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
