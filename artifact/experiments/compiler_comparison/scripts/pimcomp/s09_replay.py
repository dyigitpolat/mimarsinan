""
import sys, os, json, time, importlib.util, contextlib, io
import numpy as np, onnx

W, name, rng, start, stop = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
VD = f"{W}/pimcomp/verification"
spec = importlib.util.spec_from_file_location("pimcomp_verification", f"{VD}/verification.py")
vm = importlib.util.module_from_spec(spec); spec.loader.exec_module(vm)
os.chdir(VD)
onnx_path = f"../models/ONNX/{name}.onnx"
x = np.load(f"{W}/inputs_{rng}.npy")[start:stop]

v = vm.Verification(onnx_path, "batch")
with contextlib.redirect_stdout(io.StringIO()):
    v.load_weight()
    v.img = x[0:1].astype(np.float32); v.nn_input = v.img.transpose((0, 2, 3, 1)).flatten()
    v.eliminate_no_consider_OP(); v.get_ground_truth(); v.load_compilation()
idx = {n["name"]: n["new_node_index"] for n in v.FinalInfo["node_list"]}
fc, prod = idx["output"], [n["new_node_index"] for n in v.FinalInfo["node_list"] if n["name"].endswith("se/Mul_output_0")][0]
nfc = v.node_name_2_output_element_num["output"]; nprod = v.node_name_2_output_element_num[v.FinalInfo["node_list"][prod]["name"]]
rows = []
for i in range(len(x)):
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        v.onnx_model = onnx.load(onnx_path)
        v.img = x[i:i + 1].astype(np.float32); v.nn_input = v.img.transpose((0, 2, 3, 1)).flatten()
        v.eliminate_no_consider_OP(); v.get_ground_truth()
        v.visited_single = np.zeros((1000000), dtype=np.int16); v.comm_index_2_index_in_core = {}
        v.comm_index_2_core_index = {}; v.inst_num_traversal = np.zeros((v.core_num), dtype=np.int16)
        v.CoreMemory = vm.Memory(v.core_num)
        v.simulation()
    a = v.CoreMemory.global_memory[fc * v.max_output_element_num: fc * v.max_output_element_num + nfc].copy()
    b = v.onnx_runtime_outs["output"].flatten()
    p = v.CoreMemory.global_memory[prod * v.max_output_element_num: prod * v.max_output_element_num + nprod]
    rows.append({"i": start + i, "replay_logits": a.tolist(), "ort_top1": int(b.argmax()),
                 "first_product_slot_nonzero": int(np.count_nonzero(p)), "s": round(time.time() - t0, 1)})
    del v.CoreMemory
json.dump(rows, open(f"{W}/replay/{name}_{rng}_{start:03d}_{stop:03d}.json", "w"))
print(name, rng, start, stop, "done", flush=True)
