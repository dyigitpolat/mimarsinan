""
import sys, json, numpy as np, onnx, onnxruntime as ort

W = sys.argv[1]
so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.inter_op_num_threads = 1

def session(model):
    return ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])

def stats(a, b):
    rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-30)
    norm = (np.abs(a - b).reshape(len(a), -1).max(1) / np.abs(b).reshape(len(b), -1).max(1))
    return {"elements": int(a.size), "differ_1e-6": int((np.abs(a - b) > 1e-6 * np.abs(b)).sum()),
            "max_normwise_rel": float(norm.max()), "median_normwise_rel": float(np.median(norm)),
            "max_elementwise_rel": float(rel.max())}

out = {}
for name in ("regnet_y_400mf", "regnet_y_800mf"):
    base = onnx.load(f"{W}/onnx_orig/{name}.onnx")
    ctrl = onnx.load(f"{W}/onnx_orig/{name}.onnx")
    muls = [n for n in ctrl.graph.node if n.op_type == "Mul"]
    for n in muls:
        n.op_type = "Add"
    first = muls[0].output[0]
    for m in (base, ctrl):
        m.graph.output.extend([onnx.ValueInfoProto(name=first)])
    sb, sc = session(base), session(ctrl)
    out[name] = {"mul_nodes": len(muls), "first_product": first}
    for rng in ("norm", "raw"):
        x = np.load(f"{W}/inputs_{rng}.npy")
        lb, fb, lc, fc = [], [], [], []
        for i in range(len(x)):
            ob = sb.run(None, {"input": x[i:i + 1]}); oc = sc.run(None, {"input": x[i:i + 1]})
            lb.append(ob[0]); fb.append(ob[1]); lc.append(oc[0]); fc.append(oc[1])
        lb, lc, fb, fc = map(np.concatenate, (lb, lc, fb, fc))
        out[name][rng] = {"logits": {**stats(lc, lb), "top1_changed": int((lc.argmax(1) != lb.argmax(1)).sum()),
                                     "n": len(x)},
                          "first_product": stats(fc, fb)}
        np.save(f"{W}/ort_logits_{name}_{rng}.npy", lb)
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open(f"{W}/control.json", "w"), indent=1)
