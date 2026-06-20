#!/usr/bin/env bash
set -u
cd /home/yigit/repos/research_stuff/mimarsinan
source env/bin/activate
WD=generated/exp_g_s8_phased_deployment_run
rd(){ python3 -c "import json;print(json.load(open('$WD/_GUI_STATE/steps.json'))['steps'].get('$1',{}).get('target_metric'))" 2>/dev/null; }
echo "===== S=16 + extended genuine training (fixed ANN 0.982); S8+steps600 was 0.9488 ====="
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python run.py --headless experiments/exp_s16_st600.json > /tmp/exp_s16_st600.log 2>&1; ec=$?
t1=$(date +%s)
printf "exp_s16_st600 exit=%s ttfs_ft=%s deployed_SCM=%s wall=%ss\n" "$ec" "$(rd 'TTFS Cycle Fine-Tuning')" "$(rd 'Soft Core Mapping')" "$((t1-t0))"
grep -E "NF.SCM cascaded decision agreement: [0-9.]+|NfScmParityError|Traceback" /tmp/exp_s16_st600.log | tail -1
echo DONE
