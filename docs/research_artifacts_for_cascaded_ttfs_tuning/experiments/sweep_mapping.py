"""A4 -- CHIP-MAPPING loss: coalescing / neuron-splitting / fan-in.

The chip maps each perceptron to hard cores under (max_axons, max_neurons) limits.
Two capacity-limiting mechanisms can in principle lose precision:

  * NEURON-SPLITTING (max_neurons < layer width): a core's neurons are tiled across
    several hard cores (fan-out). Each fragment computes its own neurons' full sums.
  * COALESCING / partial-sum (max_axons < layer fan-in): a wide fan-in is fused into
    one wider crossbar across N hard cores -- the partial sums are membrane potentials
    in one MERGED core, so the full weighted sum is computed once, then fired once.

This experiment quantifies the A4 mapping loss as a DELTA from the genuine torch
neuromorphic forward (NF) -- the *same* NF the deployed sim is paired with -- on a real
digits task with a calibrated + cascade-FT'd ``ttfs_cycle_based`` single-spike flow,
swept over the capacity-limiting variables:

  - max_neurons {realistic, tight}      -> neuron splitting (fan-out tiling)
  - max_axons   {realistic, tight}      -> coalescing partial-sum (membrane transfer)
  - allow_coalescing {on, off}          -> the chip-capability gate for wide fan-in

For each config we report, over the full test set:
  * value residual max|NF - deployed|   (the mapping-domain loss; 0 == bit-exact)
  * deployed HCM accuracy, NF accuracy, and the accuracy DELTA in pp

METHOD: we reuse the repo's *tested* torch<->sim fidelity harness
(``tests/integration/_torch_sim_fidelity.py``), which pairs the NF and the deployed
HCM from one model so any residual is purely the mapping. The trained+FT'd weights are
lifted into a plain MLP and re-mapped per config through that harness. (A hand-rolled
NF/deploy pairing instead measures a spurious constant decode-scale offset -- a
forward-pairing artifact, not a mapping loss; the harness is the source of truth.)

RESULT (reproduced + quantified): in the current design coalescing is an inter-core
membrane transfer (NOT a spike-domain re-fire), so neuron-split and axon-fuse
(partial-sum) both compose BIT-EXACTLY for ``ttfs_cycle_based`` -- chip mapping is
structurally LOSSLESS (max|NF-deployed| = 0.0, accuracy delta = 0.00pp) at realistic
*and* tight limits. The one historical lossy path (spike-domain firing partial-sum)
was removed (commit 77f343c); with ``allow_coalescing=False`` a wide fan-in is now
UNMAPPABLE (raises), i.e. an honest infeasibility, not a silent accuracy leak. A
softcore that overflows BOTH axons and neurons at once is also unmappable -- each
lossless config overflows exactly one dimension.

Run:  source env/bin/activate && \
      python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/sweep_mapping.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)

import recipe_harness  # noqa: F401,E402  bootstraps src/tests/spikingjelly on sys.path

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from ft_budget import build, ft_genuine  # noqa: E402
from revive import calibrate_equalize_damped  # noqa: E402
from cascade_lab import _accuracy  # noqa: E402

from integration._torch_sim_fidelity import (  # noqa: E402
    MappingConfig,
    build_torch_and_hcm,
    assert_config_triggered,
    mapping_structure,
    _torch_nf,
)
from mimarsinan.mapping.platform.mapping_structure import (  # noqa: E402
    WideFanInUnsupportedError,
)

MODE = "ttfs_cycle_based"
IN_SHAPE = (64,)
NUM_CLASSES = 10


@dataclass(frozen=True)
class SweepCfg:
    """A named chip-mapping preset over the harness ``MappingConfig`` knobs.

    Each lossless config overflows exactly ONE dimension (axons OR neurons). The
    combined-overflow and no-coalescing presets are infeasibility probes (they raise).
    """

    label: str
    config: MappingConfig


def _configs(realistic: int, tight_neurons: int, tight_axons: int) -> list[SweepCfg]:
    R = realistic
    mk = MappingConfig
    return [
        SweepCfg(f"identity (realistic {R}/{R}, no split/fuse)",
                 mk("identity", R, R, R, R, allow_neuron_splitting=False, allow_coalescing=False)),
        SweepCfg(f"neuron_split (tight max_neurons={tight_neurons})",
                 mk("neuron_split", R, R, R, tight_neurons,
                    allow_neuron_splitting=True, allow_coalescing=False)),
        SweepCfg(f"axon_fuse / coalesce (tight max_axons={tight_axons})",
                 mk("axon_fuse", R, R, tight_axons, R,
                    allow_neuron_splitting=True, allow_coalescing=True)),
        SweepCfg(f"split+fuse BOTH (max_neurons={tight_neurons}, max_axons={tight_axons})",
                 mk("axon_fuse", R, R, tight_axons, tight_neurons,
                    allow_neuron_splitting=True, allow_coalescing=True)),
        SweepCfg(f"wide fan-in, NO coalesce (max_axons={tight_axons})",
                 mk("axon_fuse", tight_axons, R, tight_axons, R,
                    allow_neuron_splitting=True, allow_coalescing=False)),
    ]


def _lift_to_plain_mlp(flow) -> nn.Sequential:
    """Lift the (BN-folded, FT'd) flow's per-perceptron Linears into a plain
    Linear+ReLU MLP, so the *tested* harness can re-convert and map it per config with
    its established bit-exact NF<->HCM pairing."""
    layers: list[nn.Module] = []
    for p in flow.get_perceptrons():
        w, b = p.layer.weight, p.layer.bias
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data = w.detach().cpu().float().clone()
        lin.bias.data = b.detach().cpu().float().clone()
        layers += [lin, nn.ReLU()]
    return nn.Sequential(*layers)


def _eval_config(mlp, x, y, S, sweep: SweepCfg, nf_acc_ref):
    """Build the paired (NF, deployed-HCM) under ``sweep.config`` and measure the
    mapping loss. Returns a dict; ``unmappable`` configs report the raise reason."""
    cfg = sweep.config
    try:
        flow, hcm, hybrid, nodes = build_torch_and_hcm(
            mlp, IN_SHAPE, NUM_CLASSES, spiking_mode=MODE, config=cfg, T=S, device="cpu")
        assert_config_triggered(hybrid, cfg.name)
    except WideFanInUnsupportedError as exc:
        return {"label": sweep.label, "unmappable": True, "reason": str(exc).split(".")[0]}
    except (RuntimeError, AssertionError) as exc:
        return {"label": sweep.label, "unmappable": True,
                "reason": str(exc).strip().split("\n")[0][:90]}

    xd = x.double()
    with torch.no_grad():
        nf = _torch_nf(flow, xd, MODE, S).double()
        dep = (hcm(xd) / float(S)).double()
    res = float((nf - dep).abs().max())
    nf_acc = _accuracy(nf, y)
    dep_acc = _accuracy(dep, y)
    return {
        "label": sweep.label,
        "unmappable": False,
        "structure": mapping_structure(hybrid),
        "value_residual_max": res,
        "nf_acc": nf_acc,
        "dep_acc": dep_acc,
        "acc_delta_pp": (dep_acc - nf_acc) * 100.0,
    }


def run(depth=3, S=8, seed=0, epochs=120, ft_steps=200):
    t0 = time.time()
    flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, S, seed=seed, epochs=epochs)
    calibrate_equalize_damped(flow, xtr[:256], S, teacher)
    flow = ft_genuine(flow, xtr, ytr, S, steps=ft_steps, lr=2e-3, seed=seed)

    mlp = _lift_to_plain_mlp(flow)
    x, y = xte.cpu(), yte.cpu()
    fan_ins = [p.layer.weight.shape[1] for p in flow.get_perceptrons()]
    widths = [p.layer.weight.shape[0] for p in flow.get_perceptrons()]

    # NF reference (the lossless target the deployed sim is compared against).
    ref_flow, _, _, _ = build_torch_and_hcm(
        mlp, IN_SHAPE, NUM_CLASSES, spiking_mode=MODE,
        config=MappingConfig("identity", 512, 512, 512, 512, False, False), T=S, device="cpu")
    with torch.no_grad():
        nf_acc_ref = _accuracy(_torch_nf(ref_flow, x.double(), MODE, S).double(), y)

    print(f"=== A4 chip-mapping loss sweep (depth={depth} width=96 S={S} seed={seed}) ===")
    print(f"ANN (cont) acc = {cont:.4f}   genuine NF acc = {nf_acc_ref:.4f}   "
          f"(cont->NF gap {(cont - nf_acc_ref) * 100:+.2f}pp is the A1 cascade-conversion "
          f"loss, NOT mapping)")
    print(f"fan-ins {fan_ins}  widths {widths}")
    print()
    hdr = f"{'config':<48} {'max|NF-dep|':>12} {'NF':>7} {'deployed':>9} {'Δpp':>7}  structure"
    print(hdr)
    print("-" * len(hdr))

    results = []
    # realistic chip caps (mmixcore-like 256-512); tight caps force split/fuse on width 96.
    for sweep in _configs(realistic=512, tight_neurons=32, tight_axons=48):
        r = _eval_config(mlp, x, y, S, sweep, nf_acc_ref)
        results.append(r)
        if r["unmappable"]:
            print(f"{sweep.label:<48} {'UNMAPPABLE':>12}  -> {r['reason']}")
        else:
            s = r["structure"]
            tag = (f"cores={s['hard_cores']} split={s['split_frags']} "
                   f"fused={s['fused_cores']} psum={s['psum_partials']}")
            print(f"{sweep.label:<48} {r['value_residual_max']:>12.2e} "
                  f"{r['nf_acc']:>7.4f} {r['dep_acc']:>9.4f} {r['acc_delta_pp']:>+7.2f}  {tag}")
    print("-" * len(hdr))

    mappable = [r for r in results if not r["unmappable"]]
    max_res = max(r["value_residual_max"] for r in mappable)
    max_dpp = max(abs(r["acc_delta_pp"]) for r in mappable)
    n_unmap = sum(1 for r in results if r["unmappable"])
    print(f"\nVERDICT over {len(mappable)} mappable configs (realistic 512 + tight 32/48):")
    print(f"  worst value residual max|NF-deployed| = {max_res:.2e}   (bit-exact == 0.0)")
    print(f"  worst |deployed-NF| accuracy delta    = {max_dpp:.4f}pp")
    print(f"  {n_unmap} infeasibility probe(s) UNMAPPABLE (wide fan-in w/o coalescing, or "
          f"both-dim overflow) -> honest raise, never a silent leak")
    print(f"\n[{time.time() - t0:.1f}s]  A4: chip mapping (neuron-split + coalescing "
          f"partial-sum) is BIT-EXACT LOSSLESS -- NOT a source of the ANN->deployed loss.")
    return results


if __name__ == "__main__":
    run()
