""
import json, re, sys, os, glob
work, plat, nick = sys.argv[1], sys.argv[2], sys.argv[3]
logs = os.path.join(work, 'logs', plat, nick, 'base')
rows = []
for f in sorted(glob.glob(os.path.join(logs, '*.log'))):
    txt = open(f, errors='replace').read()
    m = re.findall(r'Elapsed time: (\S+)\[h:\]min:sec\. CPU time: user (\S+) sys (\S+) \((\S+)%\)\. Peak memory: (\d+)KB', txt)
    if m:
        el, us, sy, pc, pk = m[-1]
        parts = [float(x) for x in el.split(':')]
        sec = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
        rows.append((os.path.basename(f)[:-4], sec, float(us) + float(sy), int(pk) / 1024))
stage = {'synth': 0, 'floorplan': 0, 'place': 0, 'cts': 0, 'route': 0, 'finish': 0}
peak = {k: 0 for k in stage}
for name, sec, cpu, pk in rows:
    k = {'1': 'synth', '2': 'floorplan', '3': 'place', '4': 'cts', '5': 'route', '6': 'finish'}.get(name[0])
    if k:
        stage[k] += sec; peak[k] = max(peak[k], pk)
d = json.load(open(os.path.join(logs, '6_report.json')))
r = json.load(open(os.path.join(logs, '5_2_route.json'))) if os.path.exists(os.path.join(logs, '5_2_route.json')) else {}
g = lambda k: d.get(k)
out = {
  "confirmatory": "CONFIRMATORY (pre-registered, EXPLORATORY study): a routed held-out tile of the confirmatory run (plan_r5.json, RA/13_geometry_study); not a record",
  'stage_elapsed_s': {k: round(v, 1) for k, v in stage.items()},
  'stage_peak_mb': {k: round(v) for k, v in peak.items()},
  'total_elapsed_s': round(sum(stage.values()), 1),
  'die_area_um2': g('finish__design__die__area'),
  'core_area_um2': g('finish__design__core__area'),
  'instance_area_um2': g('finish__design__instance__area'),
  'macro_area_um2': g('finish__design__instance__area__macros'),
  'stdcell_area_um2': g('finish__design__instance__area__stdcell'),
  'seq_area_um2': g('finish__design__instance__area__class:sequential_cell'),
  'comb_area_um2': g('finish__design__instance__area__class:multi_input_combinational_cell'),
  'macro_count': g('finish__design__instance__count__macros'),
  'stdcell_count': g('finish__design__instance__count__stdcell'),
  'seq_count': g('finish__design__instance__count__class:sequential_cell'),
  'utilization': g('finish__design__instance__utilization'),
  'setup_ws': g('finish__timing__setup__ws'), 'setup_tns': g('finish__timing__setup__tns'),
  'hold_ws': g('finish__timing__hold__ws'),
  'fmax_hz': g('finish__timing__fmax'),
  'power_total_w': g('finish__power__total'), 'power_internal_w': g('finish__power__internal__total'),
  'power_switching_w': g('finish__power__switching__total'), 'power_leakage_w': g('finish__power__leakage__total'),
  'route_drc_errors': r.get('detailedroute__route__drc_errors'),
  'route_wirelength_um': r.get('detailedroute__route__wirelength'),
}
ia = out['instance_area_um2'] or 0
if ia:
    out['macro_share_of_instance_area'] = round(out['macro_area_um2'] / ia, 4)
    out['logic_share_of_instance_area'] = round(out['stdcell_area_um2'] / ia, 4)
if out['die_area_um2']:
    out['macro_share_of_die'] = round(out['macro_area_um2'] / out['die_area_um2'], 4)
print(json.dumps(out, indent=1))
