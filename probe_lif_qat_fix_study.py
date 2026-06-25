"""LIF-specific FIX study on the STRONG CIFAR10 residual probe, by DECISION-FIDELITY.

Reuses the IDENTICAL trained ResNet checkpoint and the production deploy SSOT in
``probe_lif_resnet_decision_fidelity`` (P). Measures, for the collapsing depth at a
collapsing T, baseline-vs-with-fix LIF DECISION-FIDELITY = argmax-agreement with the
float ANN teacher on the SAME 2000 CIFAR10 test images.

Fixes (LIF-applicable; the diagnosis already settled boundary/T/DFQ, so this targets
the one durable lever it flagged un-measured for LIF: SPIKING-AWARE TRAINING / QAT /
KD through the genuine LIF cascade):
  baseline      : fold (bit-exact per-layer), no retrain
  resmerge      : fold + on-chip residual merge (boundary re-encode removed)
  dfq           : fold + LIF DFQ per-neuron bias correction
  highT2x       : fold, deploy at 2x T
  qat           : fold, then fine-tune flow weights through chip_aligned LIF NF with
                  KD (soft) toward the float ANN teacher, AT the deploy T
  qat_kd_hard   : qat with hard-label CE mixed in
  qat_resmerge  : qat + on-chip residual merge (compose)
  qat_highT     : qat trained at deploy T, deployed at 2x T (decoupled S)

Usage:
  PYTHONPATH=src:spikingjelly env/bin/python probe_lif_qat_fix_study.py \
      --ckpt probe_ckpt/resnet_res_d8_w32.pt --T 16 --gpu 1 \
      --fixes baseline,resmerge,dfq,qat,qat_resmerge --steps 300
"""
from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import probe_lif_resnet_decision_fidelity as P
from probe_sweep_fold import cifar_tensors

from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.spiking.lif_distribution_matching import match_lif_activation_distributions


def build_folded_flow(model, cal_x, T, device, *, onchip_residual_merge=False):
    """Fold per-layer scales into ``model``, convert to a flow, install LIF (scale=1.0,
    bit-exact NF==HCM), map. Returns (flow, hybrid, teacher, nseg)."""
    model = copy.deepcopy(model)
    P.fold_perlayer_scales(model, cal_x, quantile=0.99)

    flow = convert_torch_model(model, P.INPUT_SHAPE, P.NUM_CLASSES, device=device)
    flow.eval()
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    teacher = copy.deepcopy(flow).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    for p in flow.get_perceptrons():
        p.set_activation_scale(1.0)
        p.set_activation(LIFActivation(T=T, activation_scale=1.0, thresholding_mode="<="))
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)

    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=100000, max_neurons=100000,
        allow_coalescing=False, onchip_residual_merge=onchip_residual_merge,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 100000, "max_neurons": 100000, "count": 20000}],
        strategy=MappingStrategy.from_permissions(
            allow_neuron_splitting=False, allow_coalescing=False),
    )
    nseg = sum(1 for s in hybrid.stages if s.hard_core_mapping is not None)
    return flow, hybrid, teacher, nseg


def reinstall_T(flow, deploy_T):
    for p in flow.get_perceptrons():
        p.set_activation(LIFActivation(T=deploy_T, activation_scale=p.activation_scale,
                                       thresholding_mode="<="))


@torch.no_grad()
def teacher_logits_batched(teacher, X, device, batch=256):
    teacher = teacher.to(device)
    out = []
    for i in range(0, len(X), batch):
        out.append(teacher(X[i:i + batch].to(device)))
    return torch.cat(out)


def qat_finetune(flow, teacher, Xtr, ytr, T, device, *, steps, lr, batch,
                 kd_tau=4.0, kd_w=1.0, ce_w=1.0):
    """Fine-tune the flow's weights through the genuine chip-aligned LIF NF with a
    KD blend toward the frozen float ANN teacher (soft-target distillation), plus a
    hard-label CE term. Mutates the flow in place; returns it.

    KD/CE are computed on the RAW NF logits (the final-FC spike-count scale, ~[-4,6]
    here, which carries the discriminative dynamic range) -- NOT on ``nf/T`` (which
    crushes the logit range to ~[-0.25,0.4] and makes every softmax near-uniform, so
    there is no learnable signal). The teacher logits are temperature-matched by a
    per-batch logit-std rescale so the soft targets are commensurate."""
    flow = flow.to(device).train()
    # CRITICAL: the converted flow carries live BatchNorm1d per perceptron. In
    # .train() they use BATCH stats (and drift their running stats), making the
    # training forward a different function than the deployed .eval() forward
    # (measured train-vs-eval argmax agreement ~28%). Freeze every BN to eval so
    # QAT optimizes the EXACT deployed cycle-accurate LIF forward.
    for mod in flow.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.eval()
    teacher = teacher.to(device).eval()
    params = [p for p in flow.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    n = len(Xtr)
    g = torch.Generator(device="cpu").manual_seed(0)
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        xb = Xtr[idx].to(device)
        yb = ytr[idx].to(device)
        with torch.no_grad():
            t_logits = teacher(xb)
        nf = chip_aligned_segment_forward(flow, xb, T)  # raw spike-count logits
        opt.zero_grad()
        loss = torch.zeros((), device=device)
        if kd_w > 0.0:
            # match the teacher's logit spread to the NF logits' spread so the
            # softened distributions are comparable, then distill.
            nf_std = nf.std().clamp_min(1e-4)
            t_std = t_logits.std().clamp_min(1e-4)
            t_matched = (t_logits - t_logits.mean(1, keepdim=True)) * (nf_std / t_std)
            ls = F.log_softmax(nf / kd_tau, dim=1)
            ts = F.softmax(t_matched / kd_tau, dim=1)
            loss = loss + kd_w * (kd_tau ** 2) * F.kl_div(ls, ts, reduction="batchmean")
        if ce_w > 0.0:
            loss = loss + ce_w * F.cross_entropy(nf, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"    qat step {step:>4}/{steps} loss={loss.item():.4f}", flush=True)
    flow.eval()
    return flow


def measure(flow, teacher, eval_x, eval_y, T, device, *, label, ann_eval):
    df = P.decision_fidelity(flow, teacher, eval_x, T, device)
    nf = P.nf_logits(flow, eval_x, T, device)
    dep = (nf.argmax(1) == eval_y.cpu()).float().mean().item()
    print(f"  {label:24} T={T:>3} dec_fid={df:.4f} dep_top1={dep:.4f} "
          f"ret={100*dep/ann_eval:5.1f}%", flush=True)
    return df, dep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--data", default="/home/yigit/data")
    ap.add_argument("--n-cal", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--fixes", default="baseline,resmerge,dfq,qat,qat_resmerge")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = P.ResNet(ck["depth"], width=ck["width"], residual=ck["residual"])
    model.load_state_dict(ck["state_dict"])
    model = model.eval().to(device)

    cal_x, eval_x, eval_y = cifar_tensors(args.data, args.n_cal, args.n_eval, device)
    # training slice (separate from eval): take cal/train from the train split tensors
    import torchvision
    import torchvision.transforms as TT
    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    tf = TT.Compose([TT.ToTensor(), TT.Normalize(mean, std)])
    tr = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=tf)
    Xtr = torch.stack([tr[i][0] for i in range(args.n_train)]).cpu()
    ytr = torch.tensor([tr[i][1] for i in range(args.n_train)]).cpu()

    with torch.no_grad():
        ann_eval = (model(eval_x).argmax(1) == eval_y).float().mean().item()
    T = args.T
    print(f"CKPT {args.ckpt} depth={ck['depth']} residual={ck['residual']} "
          f"ANN(eval)={ann_eval:.4f} chance=0.1 n_eval={args.n_eval} T={T} "
          f"steps={args.steps} lr={args.lr}", flush=True)

    fixes = args.fixes.split(",")
    results = {}

    for fix in fixes:
        merge = fix in ("resmerge", "qat_resmerge")
        flow, hybrid, teacher, nseg = build_folded_flow(
            model, cal_x, T, device, onchip_residual_merge=merge)

        if fix == "baseline":
            pass
        elif fix == "resmerge":
            pass
        elif fix == "dfq":
            match_lif_activation_distributions(flow, teacher, cal_x, T)
        elif fix == "highT2x":
            reinstall_T(flow, 2 * T)
        elif fix in ("qat", "qat_resmerge"):
            qat_finetune(flow, teacher, Xtr, ytr, T, device,
                         steps=args.steps, lr=args.lr, batch=args.batch,
                         kd_tau=4.0, kd_w=1.0, ce_w=1.0)
        elif fix == "qat_ce":
            qat_finetune(flow, teacher, Xtr, ytr, T, device,
                         steps=args.steps, lr=args.lr, batch=args.batch,
                         kd_tau=4.0, kd_w=0.0, ce_w=1.0)
        elif fix == "qat_kd":
            qat_finetune(flow, teacher, Xtr, ytr, T, device,
                         steps=args.steps, lr=args.lr, batch=args.batch,
                         kd_tau=4.0, kd_w=1.0, ce_w=0.0)
        elif fix == "qat_highT":
            qat_finetune(flow, teacher, Xtr, ytr, T, device,
                         steps=args.steps, lr=args.lr, batch=args.batch,
                         kd_tau=4.0, kd_w=1.0, ce_w=1.0)
            reinstall_T(flow, 2 * T)
        elif fix == "qat_ce_highT":
            # train CE-only at T (the winning objective), deploy at 2T (the
            # dominant T lever) -- the headline composition.
            qat_finetune(flow, teacher, Xtr, ytr, T, device,
                         steps=args.steps, lr=args.lr, batch=args.batch,
                         kd_tau=4.0, kd_w=0.0, ce_w=1.0)
            reinstall_T(flow, 2 * T)
        elif fix == "qat_ce_train2T":
            # train CE-only directly AT 2T (full train==deploy at the higher T).
            reinstall_T(flow, 2 * T)
            qat_finetune(flow, teacher, Xtr, ytr, 2 * T, device,
                         steps=args.steps, lr=args.lr, batch=args.batch,
                         kd_tau=4.0, kd_w=0.0, ce_w=1.0)
        else:
            raise ValueError(f"unknown fix {fix}")

        deploy_T = 2 * T if fix in ("highT2x", "qat_highT", "qat_ce_highT",
                                    "qat_ce_train2T") else T
        df, dep = measure(flow, teacher, eval_x, eval_y, deploy_T, device,
                          label=fix, ann_eval=ann_eval)
        results[fix] = (df, dep, nseg)

    print("\n=== SUMMARY (dec_fid = argmax-agreement with ANN) ===", flush=True)
    base = results.get("baseline", (None,))[0]
    for fix, (df, dep, nseg) in results.items():
        gain = f"  (+{df-base:.4f})" if base is not None and fix != "baseline" else ""
        print(f"  {fix:24} dec_fid={df:.4f}  dep_top1={dep:.4f}  nseg={nseg}{gain}", flush=True)


if __name__ == "__main__":
    main()
