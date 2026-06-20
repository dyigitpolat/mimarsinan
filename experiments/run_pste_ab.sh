#!/usr/bin/env bash
set -u
cd /home/yigit/repos/research_stuff/mimarsinan
source env/bin/activate
WD=generated/exp_g_s8_phased_deployment_run
read_step(){ python3 -c "import json;print(json.load(open('$WD/_GUI_STATE/steps.json'))['steps'].get('$1',{}).get('target_metric'))" 2>/dev/null; }
echo "===== PROXY-PATH STE A/B (exp_g_s8 cache, ANN=0.982, proxy baseline ~0.95) ====="
printf "%-16s %-8s %-10s %-12s %-8s\n" cfg exit ttfs_ft deployed_SCM wall_s
for cfg in exp_pste_off exp_pste_on; do
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 python run.py --headless experiments/$cfg.json > /tmp/$cfg.log 2>&1; ec=$?
  t1=$(date +%s)
  printf "%-16s %-8s %-10s %-12s %-8s\n" "$cfg" "$ec" "$(read_step 'TTFS Cycle Fine-Tuning')" "$(read_step 'Soft Core Mapping')" "$((t1-t0))"
  grep -E "Traceback|RuntimeError|AssertionError" /tmp/$cfg.log | tail -2
done
echo "===== PROXY STE A/B DONE ====="
