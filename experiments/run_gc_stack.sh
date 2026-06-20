#!/usr/bin/env bash
set -u
cd /home/yigit/repos/research_stuff/mimarsinan
source env/bin/activate
WD=generated/exp_g_s8_phased_deployment_run
rd(){ python3 -c "import json;print(json.load(open('$WD/_GUI_STATE/steps.json'))['steps'].get('$1',{}).get('target_metric'))" 2>/dev/null; }
echo "===== G + G->F STACK (mmixcore 9-deep, ANN=0.982; baseline~0.926-0.937, G_geo=0.9392) ====="
printf "%-14s %-6s %-10s %-12s %-8s\n" cfg exit ttfs_ft deployed_SCM wall_s
for cfg in exp_gc_off2 exp_gc_geo2 exp_gc_stack exp_gc_stack2; do
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 python run.py --headless experiments/$cfg.json > /tmp/$cfg.log 2>&1; ec=$?
  t1=$(date +%s)
  printf "%-14s %-6s %-10s %-12s %-8s\n" "$cfg" "$ec" "$(rd 'TTFS Cycle Fine-Tuning')" "$(rd 'Soft Core Mapping')" "$((t1-t0))"
  grep -E "NF.SCM cascaded decision agreement: [0-9.]+" /tmp/$cfg.log | tail -1
done
echo "===== STACK DONE ====="
