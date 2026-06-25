"""Cheap independent NF==HCM bit-exact spot-check on the folded d8 residual deploy."""
from __future__ import annotations
import argparse, torch
import probe_lif_resnet_decision_fidelity as P
from probe_sweep_fold import cifar_tensors
from probe_lif_qat_fix_study import build_folded_flow

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=1)
ap.add_argument("--T", type=int, default=32)
ap.add_argument("--n", type=int, default=2)
args = ap.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
ck = torch.load("probe_ckpt/resnet_res_d8_w32.pt", map_location="cpu", weights_only=False)
model = P.ResNet(ck["depth"], width=ck["width"], residual=ck["residual"])
model.load_state_dict(ck["state_dict"]); model = model.eval().to(device)
cal_x, eval_x, eval_y = cifar_tensors("/home/yigit/data", 256, args.n, device)
flow, hybrid, teacher, nseg = build_folded_flow(model, cal_x, args.T, device)
maxd = P.verify_nf_equals_hcm(flow, hybrid, eval_x[:args.n], args.T)
print(f"C1 NF vs HCM max|delta| @T{args.T} n={args.n} nseg={nseg} = {maxd:.3e}", flush=True)
