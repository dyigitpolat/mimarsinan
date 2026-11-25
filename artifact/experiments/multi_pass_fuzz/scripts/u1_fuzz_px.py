""
import json
import os
import random
import sys
import time

TREE = sys.argv[1]
sys.path.insert(0, TREE + "/tests")
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_rule as W
import error_site as ES
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from unit.mapping.residency.executed import runs as R
from unit.mapping.residency.executed import vehicles as V


PINNED_LATENCIES, REPLAY_DELAYS = "pinned_latencies", "replay_delays"
RUNS = dict(zip(R.RUN_IDS, R.ALL_RUNS))
if os.environ.get("FUZZ_DITHER"):
    RUNS = {k: dict(v, phase_dither=True) for k, v in RUNS.items() if k in ("streamed_lif", "windowed_lif")}
FAMILY = os.environ.get("FUZZ_FAMILY") or "base"
W13 = bool(os.environ.get("FUZZ_W13"))
WINDOW = int(os.environ.get("FUZZ_T") or V.T)
if W13 and not os.environ.get("FUZZ_DITHER"):
    from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE
    RUNS["synchronized_ttfs"] = dict(spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized",
                                     firing_mode="TTFS", spike_mode="TTFS", pass_transfer=COLLAPSE)
if os.environ.get("FUZZ_RUNS"):
    RUNS = {k: v for k, v in RUNS.items() if k in os.environ["FUZZ_RUNS"].split(",")}
CAPS = dict(allow_coalescing=True, allow_neuron_splitting=True) if FAMILY == "coalesce" else {}
REFUSAL_CLASSES = (("streams under per-hop re-timing", "streaming_windowed_segment"),
                   ("one hard core starts at one latency", "co_packed_fused_latencies"),
                   ("cycles early", "positive_delay"), ("does not pack on", "planned_pass_packing"))


def refusal_class(exc) -> str:
    return next((name for key, name in REFUSAL_CLASSES if key in str(exc)), type(exc).__name__)


def vehicle_rep(seed: int):
    if FAMILY == "coalesce":
        return V.wide_rep(seed)
    if FAMILY == "instances":
        rng = random.Random(seed)
        tokens = rng.choice([2, 3, 4])
        deep_tokens = tuple(sorted(rng.sample(range(tokens), rng.randint(1, tokens - 1))))
        deep = tuple(rng.choice([4, 8]) for _ in range(rng.randint(1, 3)))
        return V.token_repr(deep=deep, tokens=tokens, deep_tokens=deep_tokens, seed=seed)
    return V.random_rep(seed)


def vehicle_ir(seed: int, rep=None, run=None):
    ""
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.latency.ir import IRLatency

    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=V.MAX_AXONS, max_neurons=V.MAX_NEURONS,
                   allow_coalescing=FAMILY == "coalesce").map(vehicle_rep(seed) if rep is None else rep)
    if run is not None:
        W.through_window(ir, run, thresholding_mode="<=", q_max=127)
    IRLatency(ir).calculate()
    return ir


def unbounded_ir(rep, run=None):
    ""
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.latency.ir import IRLatency

    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=None, max_neurons=None, allow_coalescing=False).map(rep)
    if run is not None:
        W.through_window(ir, run, thresholding_mode="<=", q_max=127)
    IRLatency(ir).calculate()
    return ir


def vehicle_facts(ir, unbounded) -> dict:
    ""
    cores = unbounded.get_neural_cores()
    return dict(perceptrons=len({int(c.perceptron_index) for c in cores if c.perceptron_index is not None}),
                host_ops=len(unbounded.get_compute_ops()), ir_cores=len(ir.get_neural_cores()),
                unbounded_ir_cores=len(cores),
                max_fan_in=max((int(c.get_input_count()) for c in cores), default=0),
                max_width=max((int(c.get_output_count()) for c in cores), default=0))


def flow(mapping, sem):
    from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

    return SpikingHybridCoreFlow((V.IN,), mapping, simulation_length=WINDOW, thresholding_mode="<=", **sem)


def stamps(mapping) -> dict:
    passes = V.neural_passes(mapping)
    return dict(pinned=sum(getattr(s.hard_core_mapping, PINNED_LATENCIES, None) is not None for s in passes),
                delayed=sum(len(getattr(s, REPLAY_DELAYS, None) or {}) for s in passes),
                max_delay=max((abs(d) for s in passes for d in (getattr(s, REPLAY_DELAYS, None) or {}).values()),
                              default=0),
                level_units=sum(len(getattr(s, "retimed_level_stages", None) or ()) for s in passes))


FORM = {name: W.cascade_of(sem) for name, sem in RUNS.items()}
FIRST = {}
for _name, _form in FORM.items():
    FIRST.setdefault(_form, _name)
if len({W.armed(RUNS[n]) for n, f in FORM.items() if f}) > 1:
    raise SystemExit("REFUSED: WINDOW_FORMS: the per-cycle runs are armed unlike")

LEAD = False if False in FIRST else True


def main():
    out_path, s0, s1 = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    counts = [int(c) for c in sys.argv[5].split(",")]
    with open(out_path, "a") as out:
        for seed in range(s0, s1):
            try:
                rep = vehicle_rep(seed)
                built = {}
                for form, name in FIRST.items():
                    run = RUNS[name] if form else None
                    ir_f = vehicle_ir(seed, rep, run)
                    built[form] = (ir_f, build_identity_hybrid_mapping(ir_graph=ir_f),
                                   unbounded_ir(rep, run) if W13 else None)
                ir, ident, unbounded = built[LEAD]
            except Exception as exc:
                print(json.dumps(dict(seed=seed, vehicle_error=f"{type(exc).__name__}: {str(exc)[:200]}",
                                      vehicle_error_site=ES.raise_site(exc))), file=out, flush=True)
                continue
            sizes = [len(s.hard_core_mapping.cores) for s in ident.stages if s.kind == "neural"]
            for count in counts:
                row = dict(seed=seed, count=count, family=FAMILY, identity_stage_cores=sizes,
                           identity_fits=max(sizes) <= count)
                if W13:
                    row.update(window=WINDOW, vehicle=vehicle_facts(ir, unbounded))
                if True in built:
                    row["relay_cores"] = sum(1 for n in built[True][0].get_neural_cores()
                                             if getattr(n, "_depth_relay", False))
                t0 = time.time()
                u1s = {}
                for form in sorted(built, key=lambda f: f != LEAD):
                    try:
                        u1s[form] = V.build(built[form][0], count, None, **CAPS)
                    except ToolchainRefusal as exc:
                        u1s[form] = exc
                    except Exception as exc:
                        u1s[form] = exc
                u1 = u1s[LEAD]
                if isinstance(u1, ToolchainRefusal):
                    row.update(u1="refused", refusal_class=refusal_class(u1),
                               reason=f"{type(u1).__name__}: {str(u1)[:240]}", u1_site=ES.raise_site(u1))
                    print(json.dumps(row), file=out, flush=True)
                    continue
                if isinstance(u1, Exception):
                    row.update(u1="error", reason=f"{type(u1).__name__}: {str(u1)[:240]}", u1_site=ES.raise_site(u1))
                    print(json.dumps(row), file=out, flush=True)
                    continue
                passes = [len(s.hard_core_mapping.cores) for s in V.neural_passes(u1)]
                row.update(u1="built", passes=passes, multipass=len(passes) > len(sizes), timing=stamps(u1))
                multi = {}
                for form, (_ir_f, ident_f, _unb_f) in built.items():
                    if isinstance(u1s[form], Exception):
                        multi[form] = True
                        continue
                    sizes_f = [len(s.hard_core_mapping.cores) for s in ident_f.stages if s.kind == "neural"]
                    passes_f = [len(s.hard_core_mapping.cores) for s in V.neural_passes(u1s[form])]
                    multi[form] = len(passes_f) > len(sizes_f)
                    if form != LEAD:
                        row["passes_relayed"] = passes_f
                if any(multi.values()):
                    for name, sem in RUNS.items():
                        if not multi[FORM[name]]:
                            continue
                        retimed = name == "windowed_lif"
                        ir_s, form_u1, unb_s = built[FORM[name]][0], u1s[FORM[name]], built[FORM[name]][2]
                        if isinstance(form_u1, Exception):
                            row[name] = (f"refused {refusal_class(form_u1)}" if isinstance(form_u1, ToolchainRefusal)
                                         else f"run_error build {type(form_u1).__name__}: {str(form_u1)[:100]}")
                            row[name + "_site"] = ES.raise_site(form_u1)
                            continue
                        identity = build_identity_hybrid_mapping(ir_graph=ir_s, retimed_level_stages=retimed)
                        try:
                            sched = form_u1 if not retimed else V.build(ir_s, count, None, retimed_level_stages=True,
                                                                        **CAPS)
                        except ToolchainRefusal as exc:
                            row[name] = f"refused {refusal_class(exc)}"
                            row[name + "_site"] = ES.raise_site(exc)
                            continue
                        except Exception as exc:
                            row[name] = f"run_error build {type(exc).__name__}: {str(exc)[:100]}"
                            row[name + "_site"] = ES.raise_site(exc)
                            continue
                        if retimed:
                            row["timing_windowed"] = stamps(sched)
                        xs = [torch.rand(6, V.IN, generator=torch.Generator().manual_seed(b)) for b in range(3)]
                        try:
                            with torch.no_grad():
                                wants = [flow(identity, sem)(x) for x in xs]
                        except Exception as exc:
                            row[name] = f"identity_error {type(exc).__name__}: {str(exc)[:100]}"
                            row[name + "_site"] = ES.raise_site(exc)
                            continue
                        try:
                            with torch.no_grad():
                                gots = [flow(sched, sem)(x) for x in xs]
                        except Exception as exc:
                            row[name] = f"run_error {type(exc).__name__}: {str(exc)[:100]}"
                            row[name + "_site"] = ES.raise_site(exc)
                            continue
                        row[name] = "equal" if all(bool(torch.equal(g, w)) for g, w in zip(gots, wants)) else "DIFFERS"
                        row[name + "_disc"] = max(float(w.std(0).max()) for w in wants) > 0
                        if W13:
                            try:
                                theorem = build_identity_hybrid_mapping(ir_graph=unb_s, retimed_level_stages=retimed)
                                with torch.no_grad():
                                    iotas = [flow(theorem, sem)(x) for x in xs]
                                row[name + "_unbounded"] = ("equal" if all(bool(torch.equal(g, w))
                                                                           for g, w in zip(gots, iotas)) else "DIFFERS")
                            except Exception as exc:
                                row[name + "_unbounded"] = f"identity_error {type(exc).__name__}: {str(exc)[:100]}"
                                row[name + "_unbounded_site"] = ES.raise_site(exc)
                row["wall_s"] = round(time.time() - t0, 2)
                print(json.dumps(row), file=out, flush=True)


if __name__ == "__main__":
    main()
