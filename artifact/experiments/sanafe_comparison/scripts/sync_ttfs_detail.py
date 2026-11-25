import contextlib, io, sys
sys.path.insert(0, "records/11_wave2/exact/scripts")
import w33_cross_backend as H
import numpy as np, torch
from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
import mimarsinan.chip_simulation.sanafe.runner.neural_stage as NS
from mimarsinan.chip_simulation.ttfs.ttfs_executor import run_ttfs_hybrid_contract
from mimarsinan.chip_simulation.ttfs.ttfs_recorder import compare_ttfs_records
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.packing.pass_timing import executed_core_latencies
from unit.mapping.residency.executed import vehicles as V
from mimarsinan.chip_simulation.ttfs.ttfs_cycle_genuine import latency_groups
name = sys.argv[1]
ir = H._census_ir(name)
prog = build_identity_hybrid_mapping(ir_graph=ir)
st = [s for s in prog.stages if s.kind == "neural"][0]
hcm = st.hard_core_mapping
lats = executed_core_latencies(hcm)
_, groups = latency_groups(lats)
for i, c in enumerate(hcm.cores):
    srcs = sorted({("in" if s.is_input_ else ("on" if s.is_always_on_ else ("off" if s.is_off_ else int(s.core_)))) for s in c.axon_sources}, key=str)
    print("core", i, "lat", lats[i], "stored", c.latency, "group", groups[i], "reads", srcs)
beh = NeuralBehaviorConfig(spiking_mode="ttfs_cycle_based", firing_mode="TTFS", thresholding_mode="<=", spike_generation_mode="TTFS")


for label in ("final tree",):
    r = SanafeRunner(mapping=prog, simulation_length=V.T, behavior=beh, ttfs_cycle_schedule="synchronized", pass_transfer="collapse")
    x = torch.rand(1, V.IN, generator=torch.Generator().manual_seed(0)).numpy().astype(np.float64)
    with contextlib.redirect_stdout(io.StringIO()):
        ref = run_ttfs_hybrid_contract(prog, x, simulation_length=V.T, spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized").record
        rec = r.run(x, sample_index=0)
    d = compare_ttfs_records(ref, rec.to_ttfs_hardware_subset(spiking_mode="ttfs_cycle_based"), atol=0.0, rtol=0.0)
    print(label, "diffs", len(d), "cores", sorted({x.core_index for x in d}))
    for x_ in d[:5]: print("  ", x_)
