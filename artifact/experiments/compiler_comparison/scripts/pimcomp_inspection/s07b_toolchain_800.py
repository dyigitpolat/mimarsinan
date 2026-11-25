""
import sys, json, importlib.util, torch, torchvision.models as tvm

EX, W = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("requirement_census", f"{EX}/scripts/requirement_census.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
from mimarsinan.mapping.capability.member import convert_for_census
from mimarsinan.mapping.capability.census import ladder_census
model = tvm.get_model("regnet_y_800mf")
model.load_state_dict(torch.load(f"{W}/torch_home/hub/checkpoints/regnet_y_800mf-58fc7688.pth", map_location="cpu"))
conv = convert_for_census(model.eval(), (3, 224, 224), 1000, packaging=rc.MVM_PACKAGING)
row = ladder_census(conv.ir, value_bits=rc.VALUE_BITS, levels=rc.LADDER[:1])[0]
out = {"pretrained_c0": {"level": row.level, "host_ops": row.host_ops, "host_kinds": row.host_kinds,
                         "neural_segments": row.neural_segments}}
print(json.dumps(out), flush=True)
torch.manual_seed(0)
rec = rc.census_record(rc.zoo_builder("regnet_y_800mf")().eval(), (3, 224, 224), 1000)
c0 = rec["rows"][0] if rec.get("rows") else {}
out["census_form_seed0"] = {"verdict": rec.get("verdict"), "loud_failure": rec.get("loud_failure"),
                            "host_kinds": c0.get("host_kinds"), "host_ops": c0.get("host_ops")}
print(json.dumps(out["census_form_seed0"], default=str), flush=True)
json.dump(out, open(f"{W}/toolchain_800_extra.json", "w"), indent=1, default=str)
