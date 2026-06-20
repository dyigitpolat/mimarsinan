#!/usr/bin/env bash
set -u
cd /home/yigit/repos/research_stuff/mimarsinan
source env/bin/activate
echo "===== PROXY-PATH S-SWEEP (does timing resolution close the cascaded gap?) ====="
printf "%-16s %-8s %-8s %-10s %-12s %-8s\n" cfg exit ANN ttfs_ft deployed_SCM wall_s
for cfg in exp_proxyS16 exp_proxyS32; do
  WD=generated/${cfg}_phased_deployment_run
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 python run.py --headless experiments/$cfg.json > /tmp/$cfg.log 2>&1; ec=$?
  t1=$(date +%s)
  rd(){ python3 -c "import json;print(json.load(open('$WD/_GUI_STATE/steps.json'))['steps'].get('$1',{}).get('target_metric'))" 2>/dev/null; }
  printf "%-16s %-8s %-8s %-8s %-10s %-12s %-8s\n" "$cfg" "$ec" "$(rd Pretraining)" "$(rd 'TTFS Cycle Fine-Tuning')" "$(rd 'Soft Core Mapping')" "$((t1-t0))"
  grep -E "NfScmParityError|RuntimeError|AssertionError" /tmp/$cfg.log | tail -1
done
echo "===== S-SWEEP DONE ====="
