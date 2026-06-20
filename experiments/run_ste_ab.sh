#!/usr/bin/env bash
# Offload-boundary STE A/B for the genuine cascaded TTFS ramp, resumed from the
# exp_g_s8 cache (ANN=0.982, Activation Analysis intact). Each resumes TTFS Cycle
# Fine-Tuning -> Soft Core Mapping; the STE lets the genuine cascade backward
# train every segment (not just the last). Reports deployed Soft Core Mapping
# (full-test) + the TTFS Cycle Fine-Tuning metric.
set -u
cd /home/yigit/repos/research_stuff/mimarsinan
source env/bin/activate
WD=generated/exp_g_s8_phased_deployment_run

read_step() {  # $1 = step name
  python3 -c "import json;print(json.load(open('$WD/_GUI_STATE/steps.json'))['steps'].get('$1',{}).get('target_metric'))" 2>/dev/null
}

echo "===== GENUINE CASCADE STE A/B (exp_g_s8 cache, ANN=0.982, baseline SCM~0.9365) ====="
printf "%-16s %-8s %-10s %-12s %-8s\n" cfg exit ttfs_ft deployed_SCM wall_s
for cfg in exp_gste_off exp_gste_on; do
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 python run.py --headless experiments/$cfg.json > /tmp/$cfg.log 2>&1
  ec=$?
  t1=$(date +%s)
  ft=$(read_step "TTFS Cycle Fine-Tuning")
  scm=$(read_step "Soft Core Mapping")
  printf "%-16s %-8s %-10s %-12s %-8s\n" "$cfg" "$ec" "$ft" "$scm" "$((t1-t0))"
  grep -E "Traceback|RuntimeError|AssertionError" /tmp/$cfg.log | tail -2
done
echo "===== STE A/B DONE ====="
