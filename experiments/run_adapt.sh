#!/usr/bin/env bash
set -u
cd /home/yigit/repos/research_stuff/mimarsinan
source env/bin/activate
WD=generated/exp_g_s8_phased_deployment_run
rd(){ python3 -c "import json;print(json.load(open('$WD/_GUI_STATE/steps.json'))['steps'].get('$1',{}).get('target_metric'))" 2>/dev/null; }
echo "===== ADAPTIVE gradual ramp (midpoint controller) + rate-gated gain correction ====="
printf "%-16s %-6s %-10s %-12s %-8s\n" cfg exit ttfs_ft deployed_SCM wall_s
for cfg in exp_adapt_base exp_adapt_gain; do
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 python run.py --headless experiments/$cfg.json > /tmp/$cfg.log 2>&1; ec=$?
  t1=$(date +%s)
  printf "%-16s %-6s %-10s %-12s %-8s\n" "$cfg" "$ec" "$(rd 'TTFS Cycle Fine-Tuning')" "$(rd 'Soft Core Mapping')" "$((t1-t0))"
  grep -E "gain_correction.*mode|NF.SCM cascaded decision agreement: [0-9.]+|NfScmParityError|Traceback" /tmp/$cfg.log | tail -2
done
echo "===== ADAPT DONE ====="
