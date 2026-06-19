#!/usr/bin/env bash
# Quantile + weight_bits conversion-bias sweep, resumed against the FIXED device-0
# cache generated/exp_qcache (ANN=0.9749). Each config re-runs Activation Analysis
# -> Normalization Fusion with lif_blend_fast (+stabilize=400), S=8, isolating the
# marginal effect of the activation_scale_quantile / weight_bits on the LIF
# ANN->SNN conversion. Sequential on GPU 0 (1-3 busy).
set -u
cd /home/yigit/repos/research_stuff/mimarsinan
source env/bin/activate

read_nf() {
  python3 -c "import json;print(json.load(open('generated/exp_qcache/_GUI_STATE/steps.json'))['steps'].get('Normalization Fusion',{}).get('target_metric'))" 2>/dev/null
}

ANN=$(python3 -c "import json;print(json.load(open('generated/exp_qcache/_GUI_STATE/steps.json'))['steps']['Pretraining']['target_metric'])" 2>/dev/null)
echo "===== Q/WB SWEEP (fixed cache, ANN=$ANN) ====="
printf "%-12s %-8s %-10s %-12s %-8s\n" cfg exit quantile deployed_NF wall_s
for cfg in exp_q_099 exp_q_0995 exp_q_0999 exp_q_100 exp_wb8; do
  t0=$(date +%s 2>/dev/null || echo 0)
  CUDA_VISIBLE_DEVICES=0 python run.py --headless experiments/$cfg.json > /tmp/$cfg.log 2>&1
  ec=$?
  t1=$(date +%s 2>/dev/null || echo 0)
  nf=$(read_nf)
  q=$(grep -ohE "q=[0-9.]+" /tmp/$cfg.log | head -1)
  printf "%-12s %-8s %-10s %-12s %-8s\n" "$cfg" "$ec" "$q" "$nf" "$((t1-t0))"
  grep -E "Traceback|RuntimeError|AssertionError" /tmp/$cfg.log | tail -2
done
echo "===== SWEEP DONE ====="
