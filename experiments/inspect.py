import json, sys
d = sys.argv[1]
s = json.load(open(f"generated/{d}_phased_deployment_run/_GUI_STATE/steps.json"))["steps"]
for k, v in s.items():
    w = v.get("end_time", 0) - v.get("start_time", 0)
    print(f"  {k:30} wall={w:8.1f}s  metric={v.get('target_metric')}")
