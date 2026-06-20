#!/usr/bin/env bash
set -u
cd /home/yigit/repos/research_stuff/mimarsinan
source env/bin/activate
WD=generated/exp_g_s8_phased_deployment_run
rd(){ python3 -c "import json;print(json.load(open('$WD/_GUI_STATE/steps.json'))['steps'].get('$1',{}).get('target_metric'))" 2>/dev/null; }
echo "===== CLEAN fixed-ANN(0.982) PROXY S-sweep (does timing resolution help on the SAME model?) ====="
printf "%-12s %-8s %-8s %-10s %-12s %-8s\n" cfg exit ANN ttfs_ft SCM wall_s
for cfg in exp_csS8 exp_csS16; do
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 python run.py --headless experiments/$cfg.json > /tmp/$cfg.log 2>&1; ec=$?
  t1=$(date +%s)
  printf "%-12s %-8s %-8s %-8s %-10s %-12s %-8s\n" "$cfg" "$ec" "$(rd Pretraining)" "$(rd 'TTFS Cycle Fine-Tuning')" "$(rd 'Soft Core Mapping')" "$((t1-t0))"
  grep -E "NfScmParityError|RuntimeError|AssertionError|TypeError" /tmp/$cfg.log | tail -1
done
echo "===== CLEAN S-SWEEP DONE ====="
