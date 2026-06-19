#!/usr/bin/env bash
set -u
cd /home/yigit/repos/research_stuff/mimarsinan
source env/bin/activate
echo "===== DEPTH (num_blocks) x S -> cascaded TTFS deployed accuracy (MNIST) ====="
printf "%-14s %-6s %-8s %-10s %-12s %-8s\n" cfg exit ANN ttfs_ft deployed_SCM wall_s
for cfg in exp_nb2_s8 exp_nb1_s8 exp_nb1_s16; do
  WD=generated/${cfg}_phased_deployment_run
  rd(){ python3 -c "import json;print(json.load(open('$WD/_GUI_STATE/steps.json'))['steps'].get('$1',{}).get('target_metric'))" 2>/dev/null; }
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 python run.py --headless experiments/$cfg.json > /tmp/$cfg.log 2>&1; ec=$?
  t1=$(date +%s)
  printf "%-14s %-6s %-8s %-8s %-10s %-12s %-8s\n" "$cfg" "$ec" "$(rd Pretraining)" "$(rd 'TTFS Cycle Fine-Tuning')" "$(rd 'Soft Core Mapping')" "$((t1-t0))"
  grep -E "NF.SCM cascaded decision agreement: [0-9.]+|Traceback|Error" /tmp/$cfg.log | tail -1
done
echo "===== BLOCKS SWEEP DONE ====="
