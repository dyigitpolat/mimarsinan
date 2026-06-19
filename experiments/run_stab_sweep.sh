#!/usr/bin/env bash
# Deployed-cascade training (stabilization-steps) sweep against the fixed high-ANN
# cache generated/exp_hicache (ANN=0.9776). Tests whether training the deployed
# cycle-accurate LIF forward longer reliably closes the ~0.2-0.3pp deployed↔ANN
# gap (the L2 reconstruction lever) above the ~0.3pp run-to-run tuning noise.
# Each config resumes Activation Analysis -> Normalization Fusion.
set -u
cd /home/yigit/repos/research_stuff/mimarsinan
source env/bin/activate

read_nf() {
  python3 -c "import json;print(json.load(open('generated/exp_hicache/_GUI_STATE/steps.json'))['steps'].get('Normalization Fusion',{}).get('target_metric'))" 2>/dev/null
}

echo "===== STABILIZE-STEPS SWEEP (hicache ANN=0.9776) ====="
printf "%-18s %-8s %-12s %-10s\n" cfg exit deployed_NF wall_s
for cfg in exp_hi_base3 exp_hi_stab1500 exp_hi_stab3000 exp_hi_stab6000; do
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 python run.py --headless experiments/$cfg.json > /tmp/$cfg.log 2>&1
  ec=$?
  t1=$(date +%s)
  nf=$(read_nf)
  printf "%-18s %-8s %-12s %-10s\n" "$cfg" "$ec" "$nf" "$((t1-t0))"
  grep -E "Traceback|RuntimeError|AssertionError" /tmp/$cfg.log | tail -2
done
echo "===== STAB SWEEP DONE ====="
