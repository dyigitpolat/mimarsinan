"""Count ON-CHIP (NeuralCore) vs HOST-SIDE (ComputeOp) weight+bias params for a deployed run."""
import sys, pickle
sys.path.append("./src"); sys.path.append("./spikingjelly")
import numpy as np, torch

run_dir = sys.argv[1].rstrip("/")

def _numel(x):
    if isinstance(x, np.ndarray): return int(x.size)
    if torch.is_tensor(x):        return int(x.numel())
    return 0

ir = pickle.load(open(f"{run_dir}/Soft Core Mapping.ir_graph.pickle", "rb"))

# ON-CHIP: NeuralCore weights (banks deduped) + biases
onchip_w = onchip_b = 0
seen_banks = set()
for nc in ir.get_neural_cores():
    if nc.core_matrix is not None:
        onchip_w += _numel(nc.core_matrix); onchip_b += _numel(nc.hardware_bias)
    elif nc.weight_bank_id is not None and nc.weight_bank_id not in seen_banks:
        seen_banks.add(nc.weight_bank_id)
        bank = ir.get_weight_bank(nc.weight_bank_id)
        onchip_w += _numel(bank.core_matrix)
        onchip_b += _numel(getattr(bank, "hardware_bias", None))
onchip_params = onchip_w + onchip_b

# HOST-SIDE: ComputeOp module params + bound constant tensors
host_params = 0
for op in ir.get_compute_ops():
    m = op.params.get("module"); bound = op.params.get("bound_tensors") or []
    if m is not None and hasattr(m, "parameters"):
        host_params += sum(_numel(p) for p in m.parameters())
    host_params += sum(_numel(t) for t in bound)

# TOTAL: deployed (fused) model parameter count
fused = torch.load(f"{run_dir}/Normalization Fusion.fused_model.pt",
                   map_location="cpu", weights_only=False)
if isinstance(fused, (tuple, list)):
    fused = next(e for e in fused if hasattr(e, "parameters"))
total_params = sum(p.numel() for p in fused.parameters())

print("run_dir       =", run_dir)
print("n_neural_cores=", len(ir.get_neural_cores()))
print("n_compute_ops =", len(ir.get_compute_ops()))
print("onchip_params =", onchip_params, f"(w={onchip_w}+b={onchip_b})")
print("host_params   =", host_params)
print("total_params  =", total_params)
print("onchip frac   = %.4f%%  -> %s" % (
    100.0*onchip_params/total_params,
    "VALID (>=50%)" if onchip_params >= 0.5*total_params else "INVALID (<50%)"))
