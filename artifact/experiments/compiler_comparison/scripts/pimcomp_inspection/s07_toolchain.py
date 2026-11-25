""
import sys, json, importlib.util, torch, torchvision.models as tvm

EX, W = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("requirement_census", f"{EX}/scripts/requirement_census.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
out = {"torchvision": __import__("torchvision").__version__, "mimarsinan": rc.mimarsinan.__file__}
for name, ckpt in (("regnet_y_400mf", "regnet_y_400mf-e6988f5f.pth"), ("regnet_y_800mf", "regnet_y_800mf-58fc7688.pth")):
    model = tvm.get_model(name)
    model.load_state_dict(torch.load(f"{W}/torch_home/hub/checkpoints/{ckpt}", map_location="cpu"))
    rec = rc.census_record(model.eval(), (3, 224, 224), 1000)
    c0 = rec["rows"][0] if rec.get("rows") else {}
    out[name] = {"verdict": rec.get("verdict"), "loud_failure": rec.get("loud_failure"),
                 "c0": {k: c0.get(k) for k in ("level", "title", "host_ops", "host_kinds", "neural_segments",
                                                "host_segments", "element_ops_by_kind", "onchip_mac_share")},
                 "mac_counters": rec.get("mac_counters"), "wall_s": rec.get("wall_s")}
    print(name, json.dumps(out[name], default=str)[:3000], flush=True)
json.dump(out, open(f"{W}/toolchain_census.json", "w"), indent=1, default=str)
