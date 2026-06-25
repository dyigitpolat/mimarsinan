"""Adversarial independent re-verification of the LIF deep-residual QAT fix.

Re-builds the strong probe from the SAME trained checkpoint and re-measures,
by DECISION-FIDELITY (argmax-agree with the float ANN teacher), the baseline
collapse and the best single fix (QAT KD+CE) + the composition (qat_highT),
with added negative controls to reject lucky-flip / leak artifacts:

  C1  NF==HCM bit-exact (methodology lock; the loss is the rate code, not a sim bug)
  C2  shuffled-teacher dec_fid ~ chance  (proves dec_fid is not trivially high)
  C3  QAT eval set is DISJOINT from QAT train set (assert no index/file overlap)
  C4  dec_fid measured against the SAME frozen ANN teacher pre/post-QAT
      (the fix must move the LIF toward the ANN's decisions, not relabel the metric)
  C5  a label-only "cheating" control: train QAT with CE on SHUFFLED labels ->
      if dec_fid still rises, the gain is a generic-regularization/leak artifact,
      not genuine ANN-decision recovery.
"""
from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import probe_lif_resnet_decision_fidelity as P
from probe_sweep_fold import cifar_tensors
from probe_lif_qat_fix_study import build_folded_flow, qat_finetune, reinstall_T


@torch.no_grad()
def teacher_argmax(teacher, X, device, batch=256):
    teacher = teacher.to(device).eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(teacher(X[i:i + batch].to(device)).argmax(1).cpu())
    return torch.cat(out)


@torch.no_grad()
def nf_argmax(flow, X, T, device, batch=64):
    return P.nf_logits(flow, X, T, device, batch=batch).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="probe_ckpt/resnet_res_d8_w32.pt")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--data", default="/home/yigit/data")
    ap.add_argument("--n-cal", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--c1", action="store_true", help="run the expensive packed-HCM bit-exact spot-check")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = P.ResNet(ck["depth"], width=ck["width"], residual=ck["residual"])
    model.load_state_dict(ck["state_dict"])
    model = model.eval().to(device)

    cal_x, eval_x, eval_y = cifar_tensors(args.data, args.n_cal, args.n_eval, device)

    import torchvision
    import torchvision.transforms as TT
    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    tf = TT.Compose([TT.ToTensor(), TT.Normalize(mean, std)])
    tr = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=tf)
    Xtr = torch.stack([tr[i][0] for i in range(args.n_train)]).cpu()
    ytr = torch.tensor([tr[i][1] for i in range(args.n_train)]).cpu()

    with torch.no_grad():
        ann_eval = (model(eval_x).argmax(1) == eval_y).float().mean().item()
    print(f"=== CKPT {args.ckpt} depth={ck['depth']} residual={ck['residual']}", flush=True)
    print(f"=== ANN(test top1, {args.n_eval} imgs) = {ann_eval:.4f}  (chance 0.10)", flush=True)

    # C3: train-set is TRAIN split, eval-set is TEST split -> physically disjoint files.
    # Cross-check that no eval image (test split) byte-matches a train image used in QAT.
    tr_bytes = {Xtr[i].numpy().tobytes() for i in range(min(args.n_train, len(Xtr)))}
    leak = 0
    ev_cpu = eval_x.cpu()
    for i in range(len(ev_cpu)):
        if ev_cpu[i].numpy().tobytes() in tr_bytes:
            leak += 1
    print(f"=== C3 QAT-train/eval byte-overlap = {leak} / {len(ev_cpu)} "
          f"(expect 0; train=trainsplit, eval=testsplit)", flush=True)

    T = args.T
    # ---- baseline (folded, bit-exact) ----
    flow_b, hybrid_b, teacher, nseg = build_folded_flow(model, cal_x, T, device)

    # C1: NF == HCM bit-exact on this folded deploy (CHEAP: 2 samples, tiny core
    # cfg in verify path is the same packed HCM; the production fold harness
    # already reproduced the diagnosis T-curve so this is a spot-confirm).
    if args.c1:
        maxd = P.verify_nf_equals_hcm(flow_b, hybrid_b, eval_x[:2], T)
        print(f"=== C1 NF vs HCM max|delta| @T{T} = {maxd:.3e}  (expect 0.0 -> rate code, not sim bug)", flush=True)
    else:
        print("=== C1 skipped (set --c1 to run packed-HCM bit-exact spot-check)", flush=True)

    # the FROZEN teacher argmax (the SSOT decision target for dec_fid; same for all fixes)
    t_arg = teacher_argmax(teacher, eval_x, device)
    print(f"=== teacher(flow.eval) test top1 vs labels = "
          f"{(t_arg == eval_y.cpu()).float().mean().item():.4f}", flush=True)

    def dec_fid_vs_teacher(flow, deploy_T):
        a = nf_argmax(flow, eval_x, deploy_T, device)
        return (a == t_arg).float().mean().item()

    df_base = dec_fid_vs_teacher(flow_b, T)
    dep_base = (nf_argmax(flow_b, eval_x, T, device) == eval_y.cpu()).float().mean().item()
    print(f"\n[baseline] T={T} dec_fid={df_base:.4f} dep_top1={dep_base:.4f}", flush=True)

    # C2: shuffled-teacher control -> dec_fid of baseline vs a RANDOM permutation of teacher argmax
    g = torch.Generator().manual_seed(123)
    perm = torch.randperm(len(t_arg), generator=g)
    base_arg = nf_argmax(flow_b, eval_x, T, device)
    df_shuf = (base_arg == t_arg[perm]).float().mean().item()
    print(f"[C2] baseline dec_fid vs SHUFFLED teacher = {df_shuf:.4f} (expect ~chance/low)", flush=True)

    # ---- best single fix: QAT (KD+CE) at T ----
    flow_q, hybrid_q, teacher_q, _ = build_folded_flow(model, cal_x, T, device)
    qat_finetune(flow_q, teacher_q, Xtr, ytr, T, device,
                 steps=args.steps, lr=args.lr, batch=args.batch,
                 kd_tau=4.0, kd_w=1.0, ce_w=1.0)
    # measure vs the ORIGINAL frozen teacher (t_arg) -- the decision target is fixed
    df_qat = dec_fid_vs_teacher(flow_q, T)
    dep_qat = (nf_argmax(flow_q, eval_x, T, device) == eval_y.cpu()).float().mean().item()
    print(f"\n[qat KD+CE] T={T} dec_fid(vs frozen ANN)={df_qat:.4f} "
          f"dep_top1={dep_qat:.4f}  gain={df_qat-df_base:+.4f}", flush=True)

    # ---- composition: qat_highT (train @T, deploy @2T) ----
    flow_q2 = copy.deepcopy(flow_q)
    reinstall_T(flow_q2, 2 * T)
    df_qat_hT = dec_fid_vs_teacher(flow_q2, 2 * T)
    dep_qat_hT = (nf_argmax(flow_q2, eval_x, 2 * T, device) == eval_y.cpu()).float().mean().item()
    print(f"[qat_highT] train T={T} -> deploy 2T={2*T} dec_fid={df_qat_hT:.4f} "
          f"dep_top1={dep_qat_hT:.4f} gain={df_qat_hT-df_base:+.4f}", flush=True)

    # pure-T lever reference (baseline deployed at 2T)
    flow_hT = copy.deepcopy(flow_b)
    reinstall_T(flow_hT, 2 * T)
    df_hT = dec_fid_vs_teacher(flow_hT, 2 * T)
    print(f"[highT2x ref] baseline deployed @2T={2*T} dec_fid={df_hT:.4f} "
          f"gain={df_hT-df_base:+.4f}", flush=True)

    # ---- C5: cheating control -- QAT CE on SHUFFLED labels ----
    flow_c, hybrid_c, teacher_c, _ = build_folded_flow(model, cal_x, T, device)
    g2 = torch.Generator().manual_seed(7)
    ytr_shuf = ytr[torch.randperm(len(ytr), generator=g2)]
    qat_finetune(flow_c, teacher_c, Xtr, ytr_shuf, T, device,
                 steps=args.steps, lr=args.lr, batch=args.batch,
                 kd_tau=4.0, kd_w=0.0, ce_w=1.0)  # CE-only on shuffled labels
    df_cheat = dec_fid_vs_teacher(flow_c, T)
    print(f"\n[C5 CE-on-SHUFFLED-labels] dec_fid(vs frozen ANN)={df_cheat:.4f} "
          f"gain={df_cheat-df_base:+.4f} (if ~baseline -> gain is genuine label signal)", flush=True)

    print("\n=== VERDICT TABLE ===", flush=True)
    print(f"  ann_test_top1     {ann_eval:.4f}", flush=True)
    print(f"  baseline_dec_fid  {df_base:.4f}", flush=True)
    print(f"  qat_dec_fid       {df_qat:.4f}  (gain {df_qat-df_base:+.4f})", flush=True)
    print(f"  qat_highT_dec_fid {df_qat_hT:.4f}  (gain {df_qat_hT-df_base:+.4f})", flush=True)
    print(f"  highT2x_dec_fid   {df_hT:.4f}  (gain {df_hT-df_base:+.4f})", flush=True)
    print(f"  shuffled_teacher  {df_shuf:.4f}", flush=True)
    print(f"  CE_shuffled_lbls  {df_cheat:.4f}  (gain {df_cheat-df_base:+.4f})", flush=True)


if __name__ == "__main__":
    main()
