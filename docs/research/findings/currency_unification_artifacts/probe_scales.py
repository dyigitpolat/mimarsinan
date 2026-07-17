"""P0 probes for conversion_boundary_algebra.md — run from the project root.

P0-1  ViT LN scale table: input_activation_scale vs read_boundary_out_scales
      (NF side; the HCM-side identity-divisor hole is code-verified at
      segment_boundary.py:77-84 and re-measured here when a hybrid mapping is
      constructible).
P0-2  Seam mass: fraction of each LN seam's output saturated (>kappa) or
      negative under today's temporal encode.
P0-3  Tier-0 inertness: cached artifacts carry empty sigma tables and no
      value-op-after-neural; op-type inventory for the wire_transparent list.

Read-only; prints a report to stdout.
"""
import os
import pickle
import sys

sys.path.append("./src")
sys.path.append("./spikingjelly")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

VIT_WD = "generated/t2_04_lif_vit_wq_s32_sched_offload_pruned_phased_deployment_run"
T0_WD = "generated/t0_03_lif_deepcnn_d8_wq_s16_sched_phased_deployment_run"


def p0_1_scale_table():
    print("=" * 70)
    print("P0-1: ViT scale table (NF side) — kappa_fold vs entry scales")
    model, _ = torch.load(
        f"{VIT_WD}/LIF Adaptation.model.pt", map_location="cpu", weights_only=False,
    )
    from mimarsinan.spiking.scale_aware_boundaries import read_boundary_out_scales
    from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons

    repr_ = model.get_mapper_repr()
    out_scales = read_boundary_out_scales(repr_, input_data_scale=2.64)
    entries = segment_entry_perceptrons(repr_)
    print(f"segment entries: {len(entries)}")
    for i, p in enumerate(entries[:4]):
        theta_in = float(p.input_activation_scale)
        print(f"  entry[{i}] {getattr(p, 'name', '?')}: "
              f"input_activation_scale(kappa_fold)={theta_in:.4f}")
    # The producers feeding entries: what out-scale does the NF table assign?
    exec_order = repr_.execution_order()
    id2node = {id(n): n for n in exec_order}
    sample = list(out_scales.items())[:6]
    for node_id, scale in sample:
        node = id2node.get(node_id)
        name = type(node).__name__ if node is not None else "?"
        val = scale if isinstance(scale, float) else (
            f"tensor(mean={float(torch.as_tensor(scale).float().mean()):.4f})")
        print(f"  out_scale[{name}] = {val}")
    print("  (HCM-side boundary_normalization_scales identity-divisor hole is "
        "code-verified at segment_boundary.py:77-84: plain host producers "
        "carry no perceptron_wrapped_activation_scale => divisor 1)")


def p0_2_seam_mass():
    print("=" * 70)
    print("P0-2: seam mass — saturated / negative fractions at LN seams")
    model, _ = torch.load(
        f"{VIT_WD}/LIF Adaptation.model.pt", map_location="cpu", weights_only=False,
    )
    model.eval()
    from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons

    entries = segment_entry_perceptrons(model.get_mapper_repr())
    stats = {}
    hooks = []
    for i, p in enumerate(entries):
        def mk(i, p):
            def pre_hook(mod, inp):
                # the perceptron's raw input BEFORE any wire op — the true
                # host value at the seam.
                v = inp[0].detach().float()
                kappa = float(p.input_activation_scale)
                stats[i] = (
                    float((v < 0).float().mean()),
                    float((v > kappa).float().mean()),
                    float(v.min()), float(v.max()), kappa,
                )
            return pre_hook
        hooks.append(p.register_forward_pre_hook(mk(i, p)))
    torch.manual_seed(0)
    with torch.no_grad():
        model(torch.randn(8, 3, 224, 224) * 0.5)
    for h in hooks:
        h.remove()
    for i, (neg, sat, lo, hi, kappa) in sorted(stats.items()):
        print(f"  seam[{i}]: neg_mass={neg:.3f} sat_mass={sat:.3f} "
              f"range=[{lo:.2f},{hi:.2f}] kappa={kappa:.3f}")


def p0_3_tier0_inertness():
    print("=" * 70)
    print("P0-3: tier-0 inertness — sigma tables + host-op inventory (t0_03)")
    with open(f"{T0_WD}/Hard Core Mapping.hard_core_mapping.pickle", "rb") as f:
        hcm = pickle.load(f)
    shifts = getattr(hcm, "node_output_shifts", None)
    n_shifts = 0 if not shifts else sum(
        1 for v in (shifts.values() if hasattr(shifts, "values") else shifts)
        if v is not None
    )
    print(f"  node_output_shifts entries: {n_shifts} (inertness needs 0)")
    ops = getattr(hcm, "compute_ops", None) or getattr(hcm, "ops", None)
    if ops is None:
        print(f"  hcm attrs: {[a for a in dir(hcm) if not a.startswith('__')][:20]}")
    else:
        from collections import Counter
        kinds = Counter()
        for op in ops:
            mod = getattr(op, "module", None) or getattr(op, "payload", None)
            kinds[type(mod).__name__] += 1
        print(f"  host op module types: {dict(kinds)}")
    with open(f"{T0_WD}/Soft Core Mapping.ir_graph.pickle", "rb") as f:
        ir = pickle.load(f)
    from collections import Counter
    node_kinds = Counter(type(n).__name__ for n in getattr(ir, "nodes", []))
    print(f"  IR node kinds: {dict(node_kinds)}")


if __name__ == "__main__":
    p0_1_scale_table()
    p0_2_seam_mass()
    p0_3_tier0_inertness()
    print("=" * 70)
    print("DONE")
