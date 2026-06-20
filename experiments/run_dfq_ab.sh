#!/usr/bin/env bash
# DFQ-for-LIF A/B against the fixed high-ANN cache generated/exp_hicache
# (ANN=0.9776, baseline deployed_NF=0.9752). Each config resumes Activation
# Analysis -> Normalization Fusion; lif_distmatch runs the DFQ per-neuron bias
# correction in the LIF Adaptation post-stabilization hook. Reports the deployed
# full-test NF metric + the distmatch channel-mean gap before/after.
set -u
cd /home/yigit/repos/research_stuff/mimarsinan
source env/bin/activate

read_nf() {
  python3 -c "import json;print(json.load(open('generated/exp_hicache/_GUI_STATE/steps.json'))['steps'].get('Normalization Fusion',{}).get('target_metric'))" 2>/dev/null
}

echo "===== DFQ-for-LIF A/B (hicache ANN=0.9776, baseline deployed=0.9752) ====="
printf "%-16s %-8s %-12s %-26s\n" cfg exit deployed_NF distmatch_gap
for cfg in exp_hi_base2 exp_hi_dfq exp_hi_dfq_e03; do
  CUDA_VISIBLE_DEVICES=0 python run.py --headless experiments/$cfg.json > /tmp/$cfg.log 2>&1
  ec=$?
  nf=$(read_nf)
  gap=$(grep -oE "distmatch.*mean_gap_before[^}]*" /tmp/$cfg.log | tail -1 | grep -oE "mean_gap_(before|after)[^,}]*" | tr '\n' ' ')
  printf "%-16s %-8s %-12s %-26s\n" "$cfg" "$ec" "$nf" "$gap"
  grep -E "Traceback|RuntimeError|AssertionError" /tmp/$cfg.log | tail -2
done
echo "===== DFQ A/B DONE ====="
