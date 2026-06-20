#!/usr/bin/env bash
set -u
cd /home/yigit/repos/research_stuff/mimarsinan
source env/bin/activate
WD=generated/exp_offl_cache
read_step(){ python3 -c "import json;print(json.load(open('$WD/_GUI_STATE/steps.json'))['steps'].get('$1',{}).get('target_metric'))" 2>/dev/null; }
echo "===== OFFLOAD cache build (encoding_layer_placement=offload) ====="
CUDA_VISIBLE_DEVICES=0 python run.py --headless experiments/exp_offl_cache.json > /tmp/exp_offl_cache.log 2>&1
echo "  cache ANN=$(read_step Pretraining)  ActAnalysis=$(read_step 'Activation Analysis')"
grep -E "Traceback|RuntimeError|AssertionError" /tmp/exp_offl_cache.log | tail -3
echo "===== OFFLOAD STE A/B (severance maximal — every segment entry re-encoded) ====="
printf "%-16s %-8s %-10s %-12s %-8s\n" cfg exit ttfs_ft deployed_SCM wall_s
for cfg in exp_offl_off exp_offl_on; do
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 python run.py --headless experiments/$cfg.json > /tmp/$cfg.log 2>&1; ec=$?
  t1=$(date +%s)
  printf "%-16s %-8s %-10s %-12s %-8s\n" "$cfg" "$ec" "$(read_step 'TTFS Cycle Fine-Tuning')" "$(read_step 'Soft Core Mapping')" "$((t1-t0))"
  grep -E "Traceback|RuntimeError|AssertionError" /tmp/$cfg.log | tail -2
done
echo "===== OFFLOAD STE A/B DONE ====="
