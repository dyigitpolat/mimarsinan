""
import sys, json, hashlib
import numpy as np, torch, torchvision, torchvision.models as tvm, onnx

W = sys.argv[1]
OPSET = 11
info = {"torch": torch.__version__, "torchvision": torchvision.__version__, "onnx": onnx.__version__, "opset": OPSET}
for name in ("regnet_y_400mf", "regnet_y_800mf"):
    weights = tvm.get_model_weights(name).DEFAULT
    model = tvm.get_model(name, weights=weights).eval()
    path = f"{W}/pimcomp/models/ONNX/{name}.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), path, opset_version=OPSET, do_constant_folding=True,
                      input_names=["input"], output_names=["output"], dynamo=False)
    g = onnx.load(path).graph
    ops = {}
    for n in g.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    info[name] = {"weights": str(weights), "url": weights.url, "ops": ops,
                  "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
gen = torch.Generator().manual_seed(20261120)
x = torch.rand(200, 3, 224, 224, generator=gen)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
np.save(f"{W}/inputs_raw.npy", x.numpy())
np.save(f"{W}/inputs_norm.npy", ((x - mean) / std).numpy())
info["inputs"] = {"seed": 20261120, "n": 200, "raw": "uniform [0,1)", "norm": "(raw - mean) / std, ImageNet"}
json.dump(info, open(f"{W}/export_info.json", "w"), indent=1)
print(json.dumps(info, indent=1))
