"""The model-level golden against the code-level runner on seeded code-level instances (proof check, part 2).

This bridge sits outside the golden package (``scripts/x3a_crosscheck/``, beside ``scripts/x3a_golden/``,
which stays toolchain-free): it imports both sides and patches the toolchain in-process for its ablations.
Its run records live in the research tree (``RA/10_wave1/g6_exactness_contract/x3a_crosscheck_runs/``).

Per seed and semantics it takes the code level's own instance (``certification.randomized.generator``),
runs the toolchain's identity-mapped single-pass program (``runner.reference_nodes``'s construction) and
translates the same integer network into a golden ``Program``:

- code ``weights`` are ``width x fan_in`` over the sources in listed order; the golden reads its sources in
  ascending node order, so rows are permuted to that order and transposed;
- host ops: ``identity`` is ``relay``; ``sum`` (the half sum of two equal sources) is ``add`` under a spiking
  semantics (the next entry re-encode clamps to ``[0, T]``) and the unclamped ``sum`` in the value domain;
  ``relu`` is ``relu``;
- the input and every segment entry use the LIF memo's Uniform comb (``encoder="comb"``); the value domain
  runs the base tree's affine core law (``value_law="affine"``, theta 1).

Four relations are reported per instance: code identity against the golden identity (``run_reference``),
code identity against N (``run_network``, where the golden claims and admits it), the golden identity
against N, and the code level's own N evaluator (``certification.randomized.network``) against the golden's
``run_network`` (two independent integer evaluators of one document). A code-identity mismatch is attributed to a stored-latency walk that is not a dependency order
(R6) when the toolchain's executed core order has a consumer before one of its producers.

Usage: ``python golden_vs_code.py [--motifs=a,b] OUT.jsonl SEED0 SEED1 [semantics ...]`` (through ``run_capped.sh``).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from x3a_golden import program as gp  # noqa: E402
from x3a_golden.hypotheses import admit_network  # noqa: E402
from x3a_golden.reference import network_claimed, run_network, run_reference  # noqa: E402

import mimarsinan  # noqa: E402
from mimarsinan.certification.randomized import executors as ex  # noqa: E402
from mimarsinan.certification.randomized import instance as ci  # noqa: E402
from mimarsinan.certification.randomized import network as cn  # noqa: E402
from mimarsinan.certification.randomized import program as cp  # noqa: E402
from mimarsinan.certification.randomized.arms import ARMS  # noqa: E402
from mimarsinan.certification.randomized.generator import generate  # noqa: E402
from mimarsinan.certification.randomized.runner import _run  # noqa: E402
from mimarsinan.chip_simulation.recording.spike_modes import to_uniform_spikes  # noqa: E402
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping  # noqa: E402

SEMANTICS = {ci.STREAMED_LIF: gp.STREAMED_LIF, ci.SYNCHRONIZED_LIF: gp.SYNC_LIF,
             ci.SYNCHRONIZED_TTFS: gp.SYNC_TTFS, ci.VALUE: gp.VALUE}


def comb_matches_toolchain(max_t: int = 8) -> bool:
    """The golden's ``comb`` encoder equals the toolchain's ``to_uniform_spikes`` for every count and window."""
    from x3a_golden.kernels import comb_trains
    for T in range(1, max_t + 1):
        n = np.arange(T + 1)
        want = comb_trains(n, T)
        got = np.stack([to_uniform_spikes(torch.tensor(n / T, dtype=torch.float64), c, T).numpy() for c in range(T)])
        if not np.array_equal(want, got.astype(np.int64)):
            return False
    return True


def to_golden(inst: ci.Instance):
    """``(Program, inputs, {layer name: nid})`` of the code-level instance's integer network."""
    sem = SEMANTICS[inst.semantics]
    decl = gp.Declaration(sem, T=inst.timesteps, comparator="<=", reset="subtract", granularity="per_cycle",
                          membrane_bits=None, act_bits=30, encoder="comb",
                          value_law="affine" if sem == gp.VALUE else "relu_shift")
    net = inst.network
    nid = {ci.INPUT: 0}
    nodes = [gp.Node(0, "input", net.input_width)]
    for layer in net.layers:
        srcs = [nid[s] for s in layer.sources]
        k = len(nodes)
        if layer.kind == ci.HOST:
            op = {"identity": "relay", "relu": "relu"}.get(layer.op) or ("sum" if sem == gp.VALUE else "add")
            nodes.append(gp.Node(k, "host", layer.width, tuple(srcs), op))
        else:
            # [fan_in, width] in listed order; a source listed twice reads one wire twice, so its rows add
            # (one charge per cycle either way).
            w = np.array(layer.weights, np.int64).reshape(layer.width, -1).T
            blocks, at = {}, 0
            for s in srcs:
                block = w[at:at + nodes[s].size]
                blocks[s] = blocks[s] + block if s in blocks else block
                at += nodes[s].size
            rows = np.concatenate([blocks[s] for s in sorted(blocks)], axis=0)
            srcs = sorted(blocks)
            nodes.append(gp.Node(k, "neural", layer.width, tuple(srcs), weights=rows,
                                 bias=np.array(layer.bias, np.int64), theta=int(layer.theta)))
        nid[layer.name] = k
    prog = gp.Program(decl, nodes)
    # The runner delivers the document's counts (the midpoint rate, ``executors.input_tensor``); round 0's
    # float32 rate k / T lost a spike at k = 5, T = 6 and is recorded in the round-0 runs only.
    inputs = {0: np.array(inst.inputs, np.int64)}
    return prog, inputs, nid


def code_identity(inst: ci.Instance):
    """``({layer name: (B, width) float64}, mapping)``: the toolchain's identity single-pass program per neuron."""
    mapped = cp.mapper_graph(inst)
    ir = cp.integer_ir(mapped, max_axons=None, max_neurons=None, allow_coalescing=False,
                       hardware_bias=inst.hardware_bias)
    mapping = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=ex.RUNS[inst.semantics].retimed)
    nodes = _run(mapping, inst, ARMS["on"], identity=True)
    per = ex.per_neuron(ir, nodes)
    out = {name: per[idx].numpy() for name, idx in mapped.perceptron_of.items()}
    if inst.semantics == ci.SYNCHRONIZED_TTFS:  # the contract reports decoded levels in [0, 1]
        out = {k: v * inst.timesteps for k, v in out.items()}
    return out, mapping


def walk_violations(mapping) -> int:
    """Edges inside a neural stage whose consumer the stored-latency walk runs before its producer (R6)."""
    bad = 0
    for stage in mapping.stages:
        if stage.kind != "neural":
            continue
        for sub in (getattr(stage, "retimed_level_stages", None) or [stage]):
            cores = sub.hard_core_mapping.cores
            order = sorted(range(len(cores)), key=lambda i: int(cores[i].latency or 0))
            pos = {c: p for p, c in enumerate(order)}
            for i, core in enumerate(cores):
                srcs = {int(s.core_) for s in core.axon_sources
                        if int(s.core_) >= 0 and not getattr(s, "is_off_", False)}
                bad += sum(1 for j in srcs if pos[j] > pos[i])
    return bad


def _diff(a: np.ndarray, b: np.ndarray) -> int:
    return int((np.abs(a - b) > 0).sum())


def asap_identity(inst: ci.Instance):
    """The code identity with ChipLatency's ALAP raise switched off in-process (no file edit): every
    stored-latency walk is then a dependency order (the R6 ablation; meaningful for windowed semantics)."""
    from mimarsinan.mapping.latency.chip import ChipLatency
    saved = ChipLatency._align_shiftable_cores
    ChipLatency._align_shiftable_cores = lambda self, current_max_latency: current_max_latency
    try:
        return code_identity(inst)[0]
    finally:
        ChipLatency._align_shiftable_cores = saved


class exact_staircases:
    """In-process ablation (no file edit): the synchronized LIF count staircase and the TTFS level staircase
    evaluated on the integer charge (``floor(Q / theta)``) instead of the float quotients
    ``floor(T * ((Q / T) / theta))`` (``wire_semantics.lif_count_staircase``, called by
    ``executors/sync_counts.py``) and ``ceil(S * (1 - V / theta))`` (``wire_semantics.ttfs_quantized_staircase_np``,
    via ``ttfs_segment``), both of which lose an exact tie ``Q = k theta``."""

    def __enter__(self):
        from mimarsinan.chip_simulation.ttfs import ttfs_segment
        from mimarsinan.models.spiking.hybrid.executors import sync_counts

        def lif(z, threshold, window, *, compare_mode="<="):
            t = int(window)
            q = torch.round(z * t)
            theta = torch.as_tensor(threshold, dtype=q.dtype, device=q.device).clamp(min=1e-12)
            c = torch.div(q, theta, rounding_mode="floor") if compare_mode == "<=" else \
                -torch.div(-q, theta, rounding_mode="floor") - 1
            return theta * c.clamp(0.0, float(t)) / t

        def ttfs(v, threshold, simulation_length, *, comparator_half_step=False):
            assert not comparator_half_step
            s_ = int(simulation_length)
            q = np.round(np.asarray(v, np.float64) * s_)
            level = np.clip(np.floor_divide(q, np.maximum(np.asarray(threshold, np.float64), 1e-12)), 0, s_)
            return level / s_

        self._saved = [(sync_counts, "lif_count_staircase", sync_counts.lif_count_staircase),
                       (ttfs_segment, "ttfs_quantized_activation_np", ttfs_segment.ttfs_quantized_activation_np)]
        sync_counts.lif_count_staircase = lif
        ttfs_segment.ttfs_quantized_activation_np = ttfs
        return self

    def __exit__(self, *exc):
        for owner, name, value in self._saved:
            setattr(owner, name, value)
        return False


def first_layer_ties(inst: ci.Instance, prog, ref, code, nid) -> dict:
    """At the first neural layer where the code identity departs from the golden one (its inputs agree),
    how many departing neurons sit at an exact threshold tie (Q a positive multiple of theta, code one
    level short): the staircase quotient evaluated in floating point."""
    for layer in inst.network.neural_layers():
        node = prog.nodes[nid[layer.name]]
        want, got = ref[node.nid], code[layer.name]
        bad = np.argwhere(np.abs(want - got) > 0)
        if not len(bad):
            continue
        x = np.concatenate([ref[s] for s in node.sources], axis=1)
        bias = node.bias if prog.decl.semantics == gp.VALUE else prog.decl.T * node.bias
        q = x @ node.weights + bias
        ties = sum(1 for b, j in bad if q[b, j] > 0 and q[b, j] % node.theta == 0 and got[b, j] == want[b, j] - 1
                   and q[b, j] // node.theta <= prog.decl.T)
        return {"layer": layer.name, "departing": int(len(bad)), "ties": int(ties),
                "witness": None if not ties else next(
                    {"Q": int(q[b, j]), "theta": int(node.theta), "T": int(prog.decl.T), "code": float(got[b, j]),
                     "golden": int(want[b, j])} for b, j in bad if q[b, j] > 0 and q[b, j] % node.theta == 0)}
    return {}


def check(seed: int, semantics: str, motifs: tuple = ()) -> dict:
    started = time.perf_counter()
    inst = generate(seed, semantics=semantics, motifs=motifs)
    row = {"seed": seed, "semantics": semantics, "T": inst.timesteps, "hardware_bias": inst.hardware_bias,
           "motifs": list(inst.motifs), "layers": len(inst.network.neural_layers())}
    try:
        code, mapping = code_identity(inst)
    except Exception as exc:  # recorded, never compared
        row.update(outcome="code_error", detail=f"{type(exc).__name__}: {str(exc)[:200]}")
        return row
    prog, inputs, nid = to_golden(inst)
    if semantics != ci.VALUE:
        delivered = ex.encoder_counts(mapping, inst).numpy()
        row["encoder_departures"] = int((delivered != np.array(inst.inputs, np.int64)).sum())
    ref, diag = run_reference(prog, inputs)
    net = run_network(prog, inputs)
    admitted, refused = admit_network(prog, diag, frozenset())
    names = [layer.name for layer in inst.network.neural_layers()]
    integral = all(np.allclose(code[n], np.round(code[n]), atol=0) for n in names)
    row["code_integral"] = bool(integral)
    row["neurons"] = int(sum(code[n].size for n in names))
    row["code_vs_golden_identity"] = sum(_diff(code[n], ref[nid[n]]) for n in names)
    row["first_identity_mismatch"] = next((n for n in names if _diff(code[n], ref[nid[n]])), None)
    row["network_claimed"] = bool(network_claimed(prog))
    row["network_admitted"] = bool(admitted)
    row["network_refused_by"] = refused
    if net is not None:
        row["code_vs_network"] = sum(_diff(code[n], net[nid[n]]) for n in names)
        row["golden_identity_vs_network"] = sum(_diff(ref[nid[n]], net[nid[n]]) for n in names)
        own = cn.evaluate(inst)
        row["code_network_vs_golden_network"] = sum(_diff(own[n], net[nid[n]]) for n in names)
    row["walk_violations"] = walk_violations(mapping)
    if row["code_vs_golden_identity"] and semantics != ci.STREAMED_LIF:
        row["first_layer"] = first_layer_ties(inst, prog, ref, code, nid)
        asap = asap_identity(inst)
        row["asap_vs_golden_identity"] = sum(_diff(asap[n], ref[nid[n]]) for n in names)
        with exact_staircases():
            exact = code_identity(inst)[0]
            both = asap_identity(inst)
        row["exact_staircase_vs_golden_identity"] = sum(_diff(exact[n], ref[nid[n]]) for n in names)
        row["asap_and_exact_vs_golden_identity"] = sum(_diff(both[n], ref[nid[n]]) for n in names)
    row["active"] = float(np.mean([np.mean(code[n] != 0) for n in names]))
    row["wall_s"] = round(time.perf_counter() - started, 4)
    return row


def main(argv) -> int:
    variant, motifs = "midpoint_inputs", ()
    if argv and argv[0].startswith("--motifs="):
        motifs = tuple(argv[0].split("=", 1)[1].split(","))
        argv = argv[1:]
    out, lo, hi = Path(argv[0]), int(argv[1]), int(argv[2])
    sems = argv[3:] or list(SEMANTICS)
    assert Path(mimarsinan.__file__).resolve().is_relative_to(ROOT / "src"), mimarsinan.__file__
    head = {"schema": "x3a-golden-crosscheck/2", "mimarsinan": str(Path(mimarsinan.__file__).resolve()),
            "comb_matches_toolchain": comb_matches_toolchain(), "seeds": [lo, hi], "semantics": sems,
            "variant": variant, "motifs": list(motifs)}
    with out.open("w") as fh:
        fh.write(json.dumps(head, sort_keys=True) + "\n")
        for seed in range(lo, hi):
            for sem in sems:
                fh.write(json.dumps(check(seed, sem, motifs), sort_keys=True) + "\n")
                fh.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
