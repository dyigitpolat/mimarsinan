""
import contextlib, io, sys
sys.path.insert(0, "records/11_wave2/exact/scripts")
import w33_cross_backend as H
import numpy as np, torch
from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
from mimarsinan.chip_simulation.ttfs.ttfs_executor import run_ttfs_hybrid_contract
from mimarsinan.chip_simulation.ttfs.ttfs_recorder import compare_ttfs_records, compare_ttfs_hardware_records, compare_ttfs_contract_records
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from unit.mapping.residency.executed import vehicles as V
for name in ["chain", "skew3", "skip", "shift_probe", "seed0", "seed5"]:
    ir = H._census_ir(name)
    prog = build_identity_hybrid_mapping(ir_graph=ir)
    facts = H.timing_facts(prog)
    beh = NeuralBehaviorConfig(spiking_mode="ttfs_cycle_based", firing_mode="TTFS", thresholding_mode="<=", spike_generation_mode="TTFS")
    r = SanafeRunner(mapping=prog, simulation_length=V.T, behavior=beh, ttfs_cycle_schedule="synchronized", pass_transfer="collapse")
    hw = ct = hw0 = 0
    for b in range(3):
        x = torch.rand(1, V.IN, generator=torch.Generator().manual_seed(b)).numpy().astype(np.float64)
        with contextlib.redirect_stdout(io.StringIO()):
            ref = run_ttfs_hybrid_contract(prog, x, simulation_length=V.T, spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized").record
            rec = r.run(x, sample_index=0)
        hw += len(compare_ttfs_hardware_records(ref, rec.to_ttfs_hardware_subset(spiking_mode="ttfs_cycle_based")))
        hw0 += len(compare_ttfs_records(ref, rec.to_ttfs_hardware_subset(spiking_mode="ttfs_cycle_based"), atol=0.0, rtol=0.0))
        ct += len(compare_ttfs_contract_records(ref, rec.to_ttfs_contract_subset(spiking_mode="ttfs_cycle_based")))
    print(name, "host_late", facts["host_late"], "contract diffs", ct, "hardware diffs (1e-6)", hw, "hardware diffs atol0", hw0, flush=True)
