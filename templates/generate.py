"""SSOT for the tiered integration-run matrices: (re)generates all configs + manifests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# [D7] the row schema validates against the config registry (the key SSOT),
# so the generator needs the package importable when run standalone.
sys.path.insert(0, str(ROOT.parent / "src"))

TRAINING_RECIPE = {
    "optimizer": "adamw",
    "scheduler": "cosine",
    "weight_decay": 0.0001,
    "warmup_ratio": 0.05,
    "grad_clip_norm": 1,
    "layer_wise_lr_decay": 1,
    "label_smoothing": 0,
    "betas": [0.9, 0.999],
}
TUNING_RECIPE = {**TRAINING_RECIPE, "warmup_ratio": 0}

PLATFORMS = {
    "A": {"cores": [{"max_axons": 256, "max_neurons": 512, "count": 60, "has_bias": True},
                    {"max_axons": 512, "max_neurons": 256, "count": 60, "has_bias": True}],
          "max_axons": 512, "max_neurons": 512},
    "B": {"cores": [{"max_axons": 784, "max_neurons": 512, "count": 60, "has_bias": True},
                    {"max_axons": 512, "max_neurons": 256, "count": 60, "has_bias": True}],
          "max_axons": 784, "max_neurons": 512},
    "C": {"cores": [{"max_axons": 1024, "max_neurons": 512, "count": 180, "has_bias": True},
                    {"max_axons": 512, "max_neurons": 256, "count": 180, "has_bias": True}],
          "max_axons": 1024, "max_neurons": 512},
    "D": {"cores": [{"max_axons": 576, "max_neurons": 256, "count": 512, "has_bias": True},
                    {"max_axons": 256, "max_neurons": 576, "count": 512, "has_bias": True}],
          "max_axons": 576, "max_neurons": 576},
    "E": {"cores": [{"max_axons": 3072, "max_neurons": 768, "count": 69, "has_bias": True},
                    {"max_axons": 768, "max_neurons": 3072, "count": 69, "has_bias": True}],
          "max_axons": 3072, "max_neurons": 3072},
    "F": {"cores": [{"max_axons": 2304, "max_neurons": 512, "count": 256, "has_bias": True},
                    {"max_axons": 512, "max_neurons": 256, "count": 256, "has_bias": True}],
          "max_axons": 2304, "max_neurons": 512},
    "G": {"cores": [{"max_axons": 4608, "max_neurons": 2048, "count": 512, "has_bias": True}],
          "max_axons": 4608, "max_neurons": 2048},
    # [wsm V4] scheduling-scale pool: B's core shapes at 1/5 the count, so a
    # lenet5-class vehicle EXCEEDS the pool and must map through scheduled
    # passes (the weight-programming boundary exercised at tier-0 wall cost).
    "H": {"cores": [{"max_axons": 784, "max_neurons": 512, "count": 12, "has_bias": True},
                    {"max_axons": 512, "max_neurons": 256, "count": 12, "has_bias": True}],
          "max_axons": 784, "max_neurons": 512},
}

VEHICLES = {
    "mmixcore": {"model_type": "mlp_mixer_core", "platform": "A", "axis": "mlp_mixer_core",
                 "model_config": {"base_activation": "ReLU", "normalization": "batch",
                                  "patch_n_1": 4, "patch_m_1": 4,
                                  "patch_c_1": 32, "fc_w_1": 128, "fc_w_2": 128}},
    "lenet5": {"model_type": "lenet5", "platform": "B", "axis": "lenet5",
               "model_config": {"variant": "lenet5"}},
    "deepcnn": {"model_type": "deep_cnn", "platform": "C", "axis": "deep_cnn",
                "model_config": {"depth": 8, "width": 16}},
    "deepmlp": {"model_type": "deep_mlp", "platform": "B", "axis": "deep_mlp",
                "model_config": {"depth": 8, "width": 64}},
    "simplemlp": {"model_type": "simple_mlp", "platform": "B", "axis": "deep_mlp",
                  "model_config": {"mlp_width_1": 256, "mlp_width_2": 128}},
    # [P4] the streamed-lif conv vehicle: stride-2 blocks, no pooling —
    # streamable by construction; fc1 needs 1024 axons (platform C).
    "stream_cnn": {"model_type": "stream_cnn", "platform": "C", "axis": "deep_cnn",
                   "model_config": {"width": 16, "blocks": 3, "fc_width": 128}},
    # [BA-P4] the NON-core mixer converts through the torch path (bare fc2
    # Linears + a no-activation patch-embed conv => signed plain-host seams),
    # exposing the boundary algebra's offload failure modes at unit wall cost.
    "mmix": {"model_type": "mlp_mixer", "platform": "A", "axis": "mlp_mixer",
             "model_config": {"base_activation": "ReLU",
                              "patch_n_1": 4, "patch_m_1": 4,
                              "patch_c_1": 32, "fc_w_1": 64, "fc_w_2": 64}},
}

MODES = {
    # Configs author the (spiking_family, spiking_variant) axes; the "axis"
    # tuple keeps the historical hypervolume-cell ids for scoreboard continuity.
    # Windowed lif (the historical 'lif' cells keep their hypervolume axis).
    "lifsync": {"spiking_family": "lif", "spiking_variant": "synchronized",
                "firing_mode": "Default", "spike_generation_mode": "Uniform",
                "thresholding_mode": "<", "axis": ("lif", "none")},
    # [P4] end-to-end event-streamed lif: the canonical 'lif' discipline.
    "lifs": {"spiking_family": "lif", "spiking_variant": "streamed",
             "firing_mode": "Default", "spike_generation_mode": "Uniform",
             "thresholding_mode": "<", "axis": ("lif", "streamed")},
    "ttfs": {"spiking_family": "ttfs", "spiking_variant": "analytical",
             "firing_mode": "TTFS", "spike_generation_mode": "TTFS",
             "thresholding_mode": "<=", "axis": ("ttfs", "none")},
    "ttfsq": {"spiking_family": "ttfs", "spiking_variant": "quantized",
              "firing_mode": "TTFS", "spike_generation_mode": "TTFS",
              "thresholding_mode": "<=", "axis": ("ttfs_quantized", "none")},
    "casc": {"spiking_family": "ttfs", "spiking_variant": "cascaded",
             "firing_mode": "TTFS", "spike_generation_mode": "TTFS", "thresholding_mode": "<=",
             "axis": ("ttfs_cycle_based", "cascaded")},
    "sync": {"spiking_family": "ttfs", "spiking_variant": "synchronized",
             "firing_mode": "TTFS", "spike_generation_mode": "TTFS", "thresholding_mode": "<=",
             "axis": ("ttfs_cycle_based", "synchronized")},
    # [mvm] value-domain MVM cores: no spiking axis, no temporal grid (no S);
    # event-domain keys are unauthorable so the row carries only the domain.
    "mvm": {"core_semantics": "mvm", "axis": ("mvm", "none")},
}

# Quant axis reflects RUNTIME truth (SSOT: config_schema/deployment_derivation.py):
# activation quantization is derived from the mode (ON for lif/casc/sync/ttfsq,
# OFF for analytical ttfs), so configs carry only the WQ declaration and never pin
# activation_quantization. fp = the vanilla float assembly (pipeline_mode vanilla).
QUANT = {
    "fp": {"weight_quantization": False},
    "wq": {"weight_quantization": True},
}
AQ_DERIVED_MODES = {"lifsync", "lifs", "ttfsq", "casc", "sync"}


def _quant_axis(row):
    """Resolved hypervolume quantization coordinate (runtime truth, not config fiction)."""
    if not QUANT[row["quant"]]["weight_quantization"]:
        return "none"
    return "wq_aq" if row["mode"] in AQ_DERIVED_MODES else "wq"


# [reproducibility] endpoint_floor_steps is the RUN-total training-STEP budget
# shared by every armed endpoint stage (the endpoint_steps ledger). Training
# budgets are denominated in optimizer steps, NEVER wall seconds: identical
# configs train identical step counts on any hardware (same config + same
# seed => same step trajectory, modulo GPU nondeterminism); wall time is a
# pure MEASUREMENT, judged per hardware context at harvest. BASE is the
# validated full floor budget (t01_23: full 16k steps => the honest 0.97 fbu
# ceiling). Modes whose pipelines carry intermediate armed endpoints get the
# per-mode EXTRA so a crater draw's intermediate recovery cannot starve the
# final WQ floor: the mode conversion endpoint and the AQ endpoint both fund
# from the recipe's endpoint_recovery_steps (conversion_policy.py — lif 1560
# at the LIF and AQ endpoints; sync 600 at the AQ endpoint). Freed-ladder
# bonuses on stalled rungs may shave the WQ floor by at most one planned
# ladder (bounded, deterministic). The casc extra (2x600) left with the
# 2026-07-12 casc removal from the tier-0 family.
ENDPOINT_FLOOR_STEPS_BASE = 16000
ENDPOINT_MODE_EXTRA_STEPS = {
    "lifsync": 2 * 600,
    "lifs": 2 * 600,
    "sync": 600,
}


def _endpoint_floor_steps(row):
    """The run-total endpoint step budget for one tier-0/0.1 row (steps)."""
    return ENDPOINT_FLOOR_STEPS_BASE + ENDPOINT_MODE_EXTRA_STEPS.get(row["mode"], 0)

# [BN-mixer respec 2026-07-12, probe env_probe_bn_fc128_e8; user-authorized]
# every mmixcore cell in both matrices runs the BN+width envelope at e8;
# supersedes the M1 mixer-e4 respec AND the ttfsq e2 revert — the BN+width
# envelope is a different regime, so the e4/e2 evidence does not carry over.
MIXER_BN_NOTE = (
    "BN-mixer respec 2026-07-12, probe env_probe_bn_fc128_e8 "
    "(user-authorized): every mmixcore cell runs normalization=batch, "
    "fc_w_1/fc_w_2 64 -> 128, training_epochs 8 — supersedes the M1 "
    "mixer-e4 respec and the ttfsq e2 revert (the BN+width envelope is a "
    "different regime; the e4/e2 evidence does not carry over). Measured "
    "basis: envelope probes saturate 0.954-0.981 without BN+width (fc64 e4 "
    "0.970; fc128_c48 e12 0.9785; fc192 e8 0.9805) vs BN+fc128 e8 = 0.982."
)

T0 = [
    # S-respec 2026-07-14 (user-directed): the mixer AQ-capacity fix — the mixer
    # examples deploy <0.97 at low S because the activation grid (Tq==S) is too
    # coarse; the measured minimal passing S is baked in (lif S32 -> 0.9751).
    dict(n=1, mode="lifsync", quant="wq", wb=5, s=32, vehicle="mmixcore", epochs=8,
         note=MIXER_BN_NOTE),
    dict(n=2, mode="lifsync", quant="fp", wb=5, s=8, vehicle="lenet5", firing="Novena",
         encoding="offload", pruned=0.5, tags=["novena", "offload", "pruned"]),
    # W2: the 360-core pool packs t0_03 only scheduled (111/360 peak over 4 phases).
    dict(n=3, mode="lifsync", quant="wq", wb=4, s=16, vehicle="deepcnn", depth=8,
         scheduling=True, tags=["sched"]),
    # W3c respec: was the fictional aq form (wq=False + weight_bits ran as a de-facto
    # float deployment, X4 passed that form); now a real WQ deployment.
    dict(n=4, mode="lifsync", quant="wq", wb=5, s=32, vehicle="deepmlp", depth=8,
         note="W3c respec 2026-07-06: fictional aq form (weight_quantization=false + "
              "weight_bits ran float) -> real WQ deployment; X4 passed the old form."),
    dict(n=5, mode="lifsync", quant="wq", wb=5, s=4, vehicle="simplemlp", seed=1),
    dict(n=6, mode="ttfs", quant="wq", wb=5, s=8, vehicle="mmixcore", epochs=8,
         note=MIXER_BN_NOTE),
    # W3c respec: same fictional-aq class as t0_04.
    dict(n=7, mode="ttfs", quant="wq", wb=5, s=16, vehicle="lenet5",
         note="W3c respec 2026-07-06: fictional aq form (weight_quantization=false + "
              "weight_bits ran float) -> real WQ deployment; X4 passed the old form."),
    dict(n=8, mode="ttfs", quant="fp", wb=5, s=32, vehicle="deepcnn", depth=8,
         scheduling=True, sim_samples=25, tags=["wall_risk", "sched"],
         note="Sim-sample respec 2026-07-07 (user-directed): the analytic "
              "per-core GEMM nevresim step is the one sample-bound sim wall "
              "(~7 s/sample, 712 s at N=100, X4); accuracy read 1.00."),
    dict(n=9, mode="ttfs", quant="wq", wb=5, s=4, vehicle="deepmlp", depth=4, width=128, pruned=0.5, tags=["pruned"]),
    dict(n=10, mode="ttfs", quant="fp", wb=5, s=16, vehicle="simplemlp",
         coalescing=False, splitting=False, tags=["identity"]),
    # S-respec 2026-07-14: ttfsq mixer needs S=64 (S16 0.9596, S32 0.9689,
    # S64 0.9795) — the analytical grid heals slower than lif.
    dict(n=11, mode="ttfsq", quant="wq", wb=5, s=64, vehicle="mmixcore",
         encoding="offload", tags=["offload"], epochs=8,
         note=MIXER_BN_NOTE),
    dict(n=12, mode="ttfsq", quant="wq", wb=8, s=32, vehicle="lenet5", tags=["wall_risk"]),
    dict(n=13, mode="ttfsq", quant="wq", wb=5, s=4, vehicle="deepcnn", depth=4),
    dict(n=14, mode="ttfsq", quant="wq", wb=5, s=8, vehicle="deepmlp", depth=8),
    dict(n=15, mode="ttfsq", quant="wq", wb=5, s=8, vehicle="simplemlp", pruned=0.10,
         tags=["pruned10"],
         note="W3c respec 2026-07-06: pruning 0.5 -> 0.10 (user-directed; 50% is "
              "too strong for this cell)."),
    # casc removed from tier-0 2026-07-12 (user directive): mode marked
    # not-fully-supported pending the cascaded-gap research program; casc
    # coverage continues in tier1/2 and the advisory framework warns on
    # selection. The five casc rows (t0_16-t0_20) are gone; survivors keep
    # their load-bearing names (no renumbering).
    dict(n=21, mode="sync", quant="wq", wb=5, s=8, vehicle="mmixcore", pruned=0.10,
         tags=["pruned10"], epochs=8,
         note="W3c respec 2026-07-06: pruning 0.5 -> 0.10 (user-directed; 50% is "
              "too strong for this cell). " + MIXER_BN_NOTE),
    dict(n=22, mode="sync", quant="wq", wb=5, s=4, vehicle="lenet5", scheduling=True, tags=["sched"]),
    dict(n=23, mode="sync", quant="wq", wb=8, s=16, vehicle="deepcnn", depth=4),
    dict(n=24, mode="sync", quant="wq", wb=5, s=8, vehicle="deepmlp", depth=4, width=128),
    dict(n=25, mode="sync", quant="wq", wb=5, s=32, vehicle="simplemlp"),
    # Folded from the retired tier-0.1 diagnostic matrix (2026-07-14, user
    # directive): the high-value distinct-dimension cells kept as tier-0
    # examples — a depth-6 deepcnn, the two mixer weight-bits variants (mixers
    # respec'd to their passing S), and a wb4 deepcnn. The tier-0.1 minimal-pair
    # matrix (and its anchor test) is retired; its S-capacity hypothesis is now
    # baked into the respec'd mixer cells above.
    dict(n=26, mode="lifsync", quant="wq", wb=4, s=16, vehicle="deepcnn", depth=6,
         scheduling=True, tags=["sched"], note="folded tier-0.1 D3: depth-6 deepcnn wall frontier"),
    dict(n=27, mode="sync", quant="wq", wb=8, s=8, vehicle="mmixcore", pruned=0.10,
         tags=["pruned10", "wb8"], epochs=8,
         note="folded tier-0.1 E1: sync mixer weight-bits variant. " + MIXER_BN_NOTE),
    dict(n=28, mode="lifsync", quant="wq", wb=8, s=32, vehicle="mmixcore",
         tags=["wb8"], epochs=8,
         note="folded tier-0.1 E2: lif mixer weight-bits variant. " + MIXER_BN_NOTE),
    dict(n=29, mode="ttfsq", quant="wq", wb=4, s=4, vehicle="deepcnn", depth=4,
         tags=["wb4"], note="folded tier-0.1 E3: ttfsq deepcnn low-weight-bits variant"),
    # [BA-P4 2026-07-17] boundary-algebra failure-mode cells
    # (conversion_boundary_algebra.md sec.3): the torch-converted mixer's
    # signed host seams. t0_30 = the offload repro (pre-arming the seam
    # craters), t0_32 = the TTFS-family (sync) temporal exposure. The planned
    # subsume minimal pair (t0_31) is structurally impossible for this
    # vehicle — under subsume every torch-mixer perceptron becomes a host op
    # (measured: 0% on-chip, the majority validity gate fires) — so the
    # pre/post-fix A/B on t0_30 itself carries the isolation.
    # S=32 per the measured mixer AQ-capacity family (the t0_01 respec):
    # S=8 activation grids crater mixers regardless of the seam algebra.
    # The sync exposure cell (t0_32) is retired 2026-07-18 with a measured
    # verdict: torch<->deployed parity 0.0 on this vehicle WITH arming
    # mode-gated off — the TTFS value-op path's own convention gap on
    # per-instance host Linears (conversion_boundary_algebra.md sec.10);
    # sync/torch-mixer support is an open program item, not a tier-0 cell.
    dict(n=30, mode="lifsync", quant="wq", wb=5, s=32, vehicle="mmix",
         encoding="offload", tags=["offload"],
         note="BA-P4 repro: offloaded torch-mixer signed host seams"),
    # [mvm W3] the value-domain (MVM) family: no conversion ladder; the R/C
    # value certificates are FATAL and the deployed read is the packed value
    # census. Fresh numbering block (t0_41+) — t0_31/32 are burned labels.
    # [P4] streamed-lif cells: end-to-end event streaming, binary spikes on
    # every wire, one resident program; NF-SCM window counts hold at atol=0.
    dict(n=45, mode="lifs", quant="wq", wb=5, s=4, vehicle="simplemlp", seed=1),
    dict(n=46, mode="lifs", quant="wq", wb=5, s=8, vehicle="deepmlp", depth=4, width=128),
    dict(n=47, mode="lifs", quant="wq", wb=5, s=32, vehicle="mmixcore", epochs=8,
         note=MIXER_BN_NOTE),
    dict(n=48, mode="lifs", quant="wq", wb=5, s=16, vehicle="stream_cnn",
         encoding="offload", tags=["offload"]),
    dict(n=49, mode="lifs", quant="fp", wb=5, s=8, vehicle="stream_cnn"),
    dict(n=41, mode="mvm", quant="wq", wb=8, vehicle="lenet5",
         pruned=0.05, tags=["pruned"],
         note="mvm flagship: quantized weights, float I/O, twin certs FATAL"),
    dict(n=42, mode="mvm", quant="fp", wb=8, vehicle="mmixcore",
         pruned=0.05, tags=["pruned"],
         note="mvm float assembly: pure packing/boundary exercise"),
    # Platform F: the value family maps MORE than lif on this vehicle (the
    # encoder conv and the bare final Linear join the chip), so C's pool of
    # 180+180 cores exhausts — an honest capacity statement, not a defect.
    dict(n=43, mode="mvm", quant="wq", wb=8, vehicle="deepcnn", platform="F",
         pruned=0.05, tags=["pruned"],
         note="mvm conv/weight-bank exercise (shared-bank cores)"),
    # [wsm V4 F5+AQ] the weight-programming boundary row: platform H's pool
    # forces scheduled passes; bank_clustered streams same-bank instances
    # over a resident core-set (weights program once, W_prog report armed via
    # allow_weight_reuse) and the boundary grid (activation_bits) composes
    # with scheduling. The value twin cert (FATAL) certifies the scheduled
    # packed program bit-exactly against the unscheduled identity program.
    # Pass budget law: bank_clustered is feasible only when the minimal
    # resident set fits the pool — loads(b)=ceil(n_b/passes) ≤ pool — so
    # conv1's 784 instances on H's 12-core pool need passes ≥ 66 (measured:
    # an under-declared budget silently falls back to pool, reuse 0.16).
    dict(n=44, mode="mvm", quant="wq", wb=8, vehicle="lenet5", platform="H",
         scheduling=True, tags=["sched", "pruned"], pruned=0.05,
         extra_dp={"schedule_policy": "bank_clustered"},
         extra_pc={"activation_bits": 8, "allow_weight_reuse": True,
                   "max_schedule_passes": 128},
         note="wsm flagship: bank-clustered scheduled passes on a "
              "constrained pool + boundary AQ, twin certs FATAL"),
]


T1 = [
    # 96px preprocessing lifted this backbone off chance (0.1000 -> 0.3925 at
    # 2 epochs). Longer finetuning at the DEFAULT lr=0.001 then collapsed it
    # to exactly chance (train 0.1001) — a pretrained backbone cannot take
    # the from-scratch LR. (An earlier fast_lr_scale attempt was a no-op:
    # that lever feeds fast_ladder only, while Weight Preloading reads
    # finetune_lr/lr — which is why its numbers were bit-identical.)
    dict(n=1, mode="lifsync", quant="wq", wb=8, s=16, vehicle="squeezenet",
         regime="pretrained", finetune_epochs=8, finetune_lr=1e-4),
    dict(n=2, mode="ttfs", quant="wq", wb=8, s=32, vehicle="vit", regime="pretrained", tags=["wall_risk"]),
    dict(n=3, mode="ttfsq", quant="wq", wb=8, s=32, vehicle="vit", regime="pretrained",
         pruned=0.05, tags=["wall_risk", "pruned"]),
    dict(n=4, mode="casc", quant="wq", wb=5, s=8, vehicle="deepcnn32", depth=8, regime="from_scratch"),
    dict(n=5, mode="sync", quant="wq", wb=5, s=8, vehicle="deepcnn32", depth=4, regime="from_scratch"),
    dict(n=6, mode="lifsync", quant="wq", wb=8, s=32, vehicle="deepcnn32", depth=8, regime="from_scratch"),
    dict(n=7, mode="casc", quant="wq", wb=8, s=16, vehicle="squeezenet", regime="pretrained",
         scheduling=True, tags=["sched"], finetune_epochs=8, finetune_lr=1e-4),
    dict(n=8, mode="ttfs", quant="fp", wb=8, s=16, vehicle="mixerc10", regime="from_scratch"),
    # [mvm W3] wider-mapping showcase: patch-embed conv + MLP fc1/fc2 + heads
    # map as affine packages; MHA/LayerNorm stay host ops. [wsm V4] armed
    # with FC banks + bank-clustered scheduling: per-token instances stream
    # over resident core-sets, so ViT-B fits platform E's 138 cores (the
    # measured un-scheduled need was 4778 instances).
    # simulation_batch_size bounds the value-census activation footprint —
    # measured: batch-512 census allocated 79 GiB on a dedicated A100 (the
    # t2 ViT rows' precedent knob).
    dict(n=9, mode="mvm", quant="wq", wb=8, vehicle="vit", regime="pretrained",
         scheduling=True, tags=["wall_risk", "sched"],
         extra_dp={"schedule_policy": "bank_clustered",
                   "simulation_batch_size": 64},
         extra_pc={"allow_weight_reuse": True, "max_schedule_passes": 64}),
    # [wsm N1] the weight-programming boundary in the EVENT domain at tier-1
    # scale — an exact minimal pair with t1_06 (same vehicle/mode/S/depth,
    # scheduling + the bank-aware policy the only difference). Measured
    # preconditions on cached spiking IRs: bank_clustered ENGAGES for conv
    # banks (t0_03 d8: programmed 9216000 -> 8239104 with 2 resident stages;
    # t0_26 d6: 5603328 -> 4626432), so the cell carries real evidence
    # rather than a pool fallback. The squeezenet vehicle is unusable here:
    # its ImageNet backbone reads chance on unpreprocessed 32x32 CIFAR10
    # (no resize/normalize declared) and the pretrain envelope aborts.
    dict(n=10, mode="lifsync", quant="wq", wb=8, s=32, vehicle="deepcnn32",
         depth=8, regime="from_scratch", scheduling=True, tags=["sched"],
         extra_dp={"schedule_policy": "bank_clustered"},
         extra_pc={"allow_weight_reuse": True, "max_schedule_passes": 128}),
    # [mvm AQ] boundary value-grid quantization at ViT scale — the minimal
    # pair with t1_09 (activation_bits absent -> declared). Exercises AQ x
    # scheduling composition and the R/C certs over ~18M neuron-windows.
    dict(n=11, mode="mvm", quant="wq", wb=8, vehicle="vit", regime="pretrained",
         scheduling=True, tags=["wall_risk", "sched", "aq8"],
         extra_dp={"schedule_policy": "bank_clustered",
                   "simulation_batch_size": 64},
         extra_pc={"allow_weight_reuse": True, "max_schedule_passes": 64,
                   "activation_bits": 8}),
    # [mvm breadth] a SECOND value-domain architecture family at tier-1: the
    # mixer's token-instanced FCs are the FC-weight-bank mechanism (V1) at
    # scale on a non-transformer family, trained from scratch (t0_42 proves
    # mlp_mixer_core packages under mvm; this is its WQ tier-1 sibling).
    dict(n=12, mode="mvm", quant="wq", wb=8, vehicle="mixerc10",
         regime="from_scratch", scheduling=True, tags=["sched"],
         extra_dp={"schedule_policy": "bank_clustered"},
         extra_pc={"allow_weight_reuse": True, "max_schedule_passes": 128}),
]

T1_VEHICLES = {
    # An ImageNet backbone needs ImageNet-shaped input: at 32x32 SqueezeNet's
    # downsampling collapses the feature map and preloading reads exactly
    # chance (0.1000), so the pretrain envelope aborts every squeezenet row.
    # 96px is the measured sweet spot on platform D: 874/1024 cores
    # unscheduled (t1_01 fits as authored) and peak 471 scheduled (t1_07),
    # where 128px overflows unscheduled (1620) and 224px needs 5244 with a
    # pool-saturating peak of 1024.
    "squeezenet": {"model_type": "torch_squeezenet11", "platform": "D", "axis": "vit_b",
                   "model_config": {}, "coalescing": False,
                   "preprocessing": {"interpolation": "bilinear", "resize_to": 96,
                                     "normalize": "imagenet"}},
    "vit": {"model_type": "torch_vit", "platform": "E", "axis": "vit_b",
            "model_config": {}, "coalescing": False,
            "preprocessing": {"interpolation": "bicubic", "resize_to": 224, "normalize": "imagenet"},
            # Cycle-accurate training memory scales S x batch: the unrolled
            # S=32 ViT graph at tuning batch 128 pins ~80 GiB (measured OOM).
            "batch_size": 512, "tuning_batch_size": 32,
            # Retry-economics lever: measured accepted LRs sat ~4x below the
            # pipeline lr (each rung wasted 1-2 wrecked attempts at 3e-3
            # before the Armijo backoff found ~7e-4).
            "fast_lr_scale": 0.25},
    "deepcnn32": {"model_type": "deep_cnn", "platform": "F", "axis": "deep_cnn",
                  "model_config": {"depth": 8, "width": 32}},
    "mixerc10": {"model_type": "mlp_mixer_core", "platform": "F", "axis": "mlp_mixer_core",
                 "model_config": {"base_activation": "ReLU", "patch_n_1": 4, "patch_m_1": 4,
                                  "patch_c_1": 256, "fc_w_1": 128, "fc_w_2": 256}},
}

T2 = [
    dict(n=1, mode="lifsync", quant="wq", wb=8, s=32, vehicle="resnet50", dataset="ImageNet",
         regime="pretrained", scheduling=True, lr=0.0001, finetune_epochs=0, budget=0.5, tags=["sched"]),
    dict(n=2, mode="ttfsq", quant="wq", wb=8, s=32, vehicle="vit", dataset="CIFAR100",
         regime="pretrained", scheduling=True, pruned=0.05, tags=["sched", "pruned"]),
    dict(n=3, mode="casc", quant="wq", wb=8, s=32, vehicle="squeezenet", dataset="CIFAR100",
         regime="pretrained", scheduling=True, tags=["sched", "wall_risk"]),
    # Offloaded ViT under LIF and TTFS-sync (user-directed 2026-07-15): the
    # patch-embed encoding layer runs on the host, the transformer stack maps to
    # chip; CIFAR100 pretrained-finetuned so the tier runs locally (ImageNet
    # scale-up is tier_3). Scheduled + light-pruned to fit platform E + wall.
    dict(n=4, mode="lifsync", quant="wq", wb=8, s=32, vehicle="vit", dataset="CIFAR100",
         regime="pretrained", scheduling=True, encoding="offload", pruned=0.05,
         tags=["sched", "offload", "pruned", "wall_risk"],
         # The AB9-proven census-graded lossless-fast set (calculus 16.9/17.8):
         # deployed LIF physics (dither + membrane guard), census-grade tuner
         # evals, the funded AA endpoint, and the metric batch cap.
         extra_dp={
             "spike_phase_dither": True,
             "lif_membrane_init": -0.25,
             "eval_subsample_target": 2048,
             "aa_endpoint_recovery_steps": 8000,
             "simulation_batch_size": 64,
         }),
    dict(n=5, mode="sync", quant="wq", wb=8, s=32, vehicle="vit", dataset="CIFAR100",
         regime="pretrained", scheduling=True, encoding="offload", pruned=0.05,
         tags=["sched", "offload", "pruned", "wall_risk"]),
]

T2_VEHICLES = {
    "resnet50": {"model_type": "torch_resnet50", "platform": "G", "axis": "vit_b",
                 "model_config": {}, "coalescing": False},
    "vit": T1_VEHICLES["vit"],
    "squeezenet": T1_VEHICLES["squeezenet"],
}

# Tier 3 (user-directed 2026-07-15): the full-scale offloaded-ViT deployment tier
# — LIF, TTFS-sync, and ttfsq on native ImageNet (pretrained torchvision, no
# finetune). Needs IMAGENET_ROOT; runs on the cluster (the "final verification"
# scale above tier_2's CIFAR100). Same offloaded encoding + scheduling.
T3 = [
    dict(n=1, mode="lifsync", quant="wq", wb=8, s=32, vehicle="vit", dataset="ImageNet",
         regime="pretrained", scheduling=True, encoding="offload", finetune_epochs=0,
         lr=0.0001, budget=0.5, tags=["sched", "offload", "wall_risk"]),
    dict(n=2, mode="sync", quant="wq", wb=8, s=32, vehicle="vit", dataset="ImageNet",
         regime="pretrained", scheduling=True, encoding="offload", finetune_epochs=0,
         lr=0.0001, budget=0.5, tags=["sched", "offload", "wall_risk"]),
    dict(n=3, mode="ttfsq", quant="wq", wb=8, s=32, vehicle="vit", dataset="ImageNet",
         regime="pretrained", scheduling=True, encoding="offload", finetune_epochs=0,
         lr=0.0001, budget=0.5, tags=["sched", "offload", "wall_risk"]),
]

T3_VEHICLES = {"vit": T1_VEHICLES["vit"]}

DATASET_AXIS = {"MNIST": "mnist", "CIFAR10": "cifar10", "CIFAR100": "cifar100", "ImageNet": "imagenet"}


def _name(tier, row, vehicles):
    prefix = f"t{tier}".replace("_", "")
    v = row["vehicle"]
    depth = f"_d{row['depth']}" if "depth" in row else ""
    tags = "".join(f"_{t}" for t in row.get("tags", []) if t in
                   ("offload", "sched", "nobias", "pruned", "pruned10", "novena",
                    "identity", "residual", "e4", "wb8", "wb4", "floor", "aq8"))
    s_part = f"_s{row['s']}" if "s" in row else ""
    return f"{prefix}_{row['n']:02d}_{row['mode']}_{v}{depth}_{row['quant']}{s_part}{tags}"


def _platform(row, vehicles):
    v = vehicles[row["vehicle"]]
    plat = json.loads(json.dumps(PLATFORMS[row.get("platform", v["platform"])]))
    has_bias = row.get("has_bias", True)
    for core in plat["cores"]:
        core["has_bias"] = has_bias
    plat["has_bias"] = has_bias
    if "s" in row:
        plat["target_tq"] = row["s"]
        plat["simulation_steps"] = row["s"]
    plat["weight_bits"] = row["wb"]
    plat["allow_coalescing"] = row.get("coalescing", v.get("coalescing", True))
    plat["allow_neuron_splitting"] = row.get("splitting", True)
    # Row-level platform passthrough (mirror of extra_dp): proven per-cell
    # hardware declarations (activation_bits, weight-reuse, pass budgets).
    plat.update(row_config_keys(row, "platform_constraints"))
    plat.update(row.get("extra_pc", {}))
    return plat


def _policy_tier(tier):
    """Tier-0.1 rows inherit tier-0's budget/recipe policy (minimal pairs)."""
    return 0 if tier == "0_1" else tier


def _deployment(tier, row, vehicles, dataset):
    tier = _policy_tier(tier)
    v = vehicles[row["vehicle"]]
    mode = MODES[row["mode"]]
    quant = QUANT[row["quant"]]
    model_config = dict(v["model_config"])
    if "depth" in row:
        model_config["depth"] = row["depth"]
    if "width" in row:
        model_config["width"] = row["width"]
    if "residual" in row:
        model_config["residual"] = row["residual"]

    dp = {
        "lr": row.get("lr", 0.003),
        "tuning_budget_scale": row.get(
            "budget", 0.25 if tier == 0 and row["mode"] in ("lifsync", "lifs", "ttfs") else 0.5 if tier == 0 else 1,
        ),
        "degradation_tolerance": 0.15 if tier == 0 else 0.1,
        "model_config_mode": "user",
        "hw_config_mode": "fixed",
        "model_type": v["model_type"],
        "model_config": model_config,
        "batch_size": v.get("batch_size", 128),
        # Simulators are PARITY probes, not accuracy reads (user directive
        # 2026-07-07): the accuracy verdict is the SCM identity read (full
        # test set, parity-certified); nevresim runs a small decision-parity
        # sample (the t0_08 N=25 respec precedent, now the default).
        "max_simulation_samples": row.get("sim_samples", 25),
        "sanafe_arch_preset": "loihi",
        "sanafe_sample_count": 1,
        "allow_scheduling": row.get("scheduling", False),
    }
    if "core_semantics" in mode:
        dp["core_semantics"] = mode["core_semantics"]
    else:
        dp["spiking_family"] = mode["spiking_family"]
        dp["spiking_variant"] = mode["spiking_variant"]
        dp["firing_mode"] = row.get("firing", mode["firing_mode"])
        dp["spike_generation_mode"] = mode["spike_generation_mode"]
        dp["thresholding_mode"] = mode["thresholding_mode"]
        dp["encoding_layer_placement"] = row.get("encoding", "subsume")
    dp["weight_quantization"] = quant["weight_quantization"]
    if "pruned" in row:
        dp["pruning"] = True
        dp["pruning_fraction"] = row["pruned"]
    if quant["weight_quantization"] and "core_semantics" not in mode:
        # M4 arming 2026-07-12: exact ReLU-homogeneous rescaling, a no-op
        # (s -> 1) when per-channel spread is small (landed 96c74e42).
        dp["scale_migration"] = True
    if tier == 0:
        dp["endpoint_floor_steps"] = _endpoint_floor_steps(row)
        # WQ-cap deletion 2026-07-12 (synthesis Phase-4): the FAST respec's
        # flat per-cell wq_endpoint_recovery_steps=2000 caps are gone — the
        # C1 convergence-stop (landed 91eacc01) patience-stops the WQ
        # endpoint under the recipe's 16k ceiling, so step budgets
        # self-limit; the endpoint_floor_steps ledger budgets stay.
        if row["vehicle"] == "mmixcore" and row["mode"] == "sync":
            # [MBH-DRAWS] FAST respec 2026-07-08: draws only where the draw
            # distribution measurably crosses the bar — the sync mixer family
            # (full-budget singles read 0.944-0.968 around the 0.97 bar).
            # casc's ceiling is physical (0.88-0.91: selection cannot reach
            # the bar) and lif/ttfsq spreads are sub-pp; those families pay
            # walls without pass probability, so they stay single-draw.
            dp["conversion_draws"] = 2
    if "tuning_batch_size" in v:
        dp["tuning_batch_size"] = v["tuning_batch_size"]
    if "fast_lr_scale" in v:
        dp["fast_lr_scale"] = v["fast_lr_scale"]
    if "preprocessing" in v:
        dp["preprocessing"] = v["preprocessing"]

    regime = row.get("regime", "from_scratch")
    if regime == "pretrained":
        dp["weight_source"] = "torchvision"
        dp["finetune_epochs"] = row.get("finetune_epochs", 2)
    else:
        dp["training_epochs"] = row.get("epochs", 2 if tier == 0 else 20)
        if tier == 0:
            dp["training_recipe"] = TRAINING_RECIPE
            dp["tuning_recipe"] = TUNING_RECIPE
    # Row-level knob passthrough: proven per-cell recipes (e.g. the AB9
    # census-graded ViT set) live on the row, not as generator cases.
    dp.update(row_config_keys(row, "deployment_parameters"))
    dp.update(row.get("extra_dp", {}))
    return dp


def _cell(tier, row, vehicles, dataset):
    v = vehicles[row["vehicle"]]
    firing, sync = MODES[row["mode"]]["axis"]
    return {
        "firing": firing,
        "sync": sync,
        "quantization": _quant_axis(row),
        "S": str(row["s"]) if "s" in row else "none",
        "depth": str(row["depth"]) if "depth" in row else "any",
        "vehicle": v["axis"],
        "dataset": DATASET_AXIS[dataset],
        "regime": row.get("regime", "from_scratch"),
        "pruning": "pruned" if "pruned" in row else "dense",
        "encoding_placement": ("none" if "core_semantics" in MODES[row["mode"]]
                               else row.get("encoding", "subsume")),
    }


# Tier-0 wall budgets, MEASURED (two full sweeps, worst completed wall per
# vehicle family): deep_cnn 11.6 min, mlp_mixer_core 11.6, lenet5 6.5,
# deep_mlp 5.8, simple_mlp 4.4. The prior rule gave every mixer 6 min, so
# four mixer rows timed out at exactly 540 s (9 min at scale 1.5) in the
# 2026-07-28 verification and reported no verdict at all.
_TIER0_VEHICLE_WALL_MIN = {
    "stream_cnn": 10,
    "deep_cnn": 16,
    "mlp_mixer_core": 16,
    "mlp_mixer": 16,
    "lenet5": 10,
    "deep_mlp": 9,
    "simple_mlp": 7,
}


# [D7] Tier rows are a CLOSED schema. A key is either STRUCTURAL (this
# generator's own vocabulary) or a config key the registry knows — and the
# registry decides which section it lands in. Anything else is a typo and
# fails loud: a silently-ignored row key cost a full experiment cycle
# (finetune_lr evaporated while an LR hypothesis was believed tested).
_STRUCTURAL_ROW_KEYS = frozenset({
    "n", "mode", "quant", "wb", "s", "vehicle", "depth", "width", "platform",
    "regime", "dataset", "tags", "note", "scheduling", "pruned", "encoding",
    "firing", "epochs", "budget", "sim_samples", "wall_min", "has_bias",
    "coalescing", "splitting", "seed", "extra_dp", "extra_pc", "cell",
    "finetune_epochs", "lr",
})


def _registry_entry(key):
    from mimarsinan.config_schema.registry import REGISTRY
    return REGISTRY.get(key)


def _key_section(key) -> "str | None":
    entry = _registry_entry(key)
    if entry is None:
        return None
    return getattr(entry, "section", None) or "deployment_parameters"


def validate_row_keys(row) -> None:
    """Reject any row key that is neither structural nor registry-known."""
    import difflib

    from mimarsinan.config_schema.registry import REGISTRY

    unknown = [
        k for k in row
        if k not in _STRUCTURAL_ROW_KEYS and _registry_entry(k) is None
    ]
    if not unknown:
        return
    details = []
    for key in sorted(unknown):
        near = difflib.get_close_matches(key, list(REGISTRY), n=3, cutoff=0.6)
        hint = f" (did you mean: {', '.join(near)}?)" if near else ""
        details.append(f"{key!r}{hint}")
    raise ValueError(
        f"tier row {row.get('n')} ({row.get('vehicle')}) carries unknown "
        f"key(s): {'; '.join(details)}. A row key must be structural or a "
        f"registry config key — an unread key would be silently dropped."
    )


def row_config_keys(row, section: str) -> dict:
    """Row-authored config keys the registry assigns to ``section``."""
    return {
        k: v for k, v in row.items()
        if k not in _STRUCTURAL_ROW_KEYS and _key_section(k) == section
    }


def _wall_budget(tier, row, vehicles, default_min):
    if "wall_min" in row:
        return row["wall_min"]
    if _policy_tier(tier) != 0:
        return default_min
    model_type = vehicles[row["vehicle"]]["model_type"]
    base = _TIER0_VEHICLE_WALL_MIN.get(model_type, 10)
    # LIF pays an extra Loihi leg on top of its family's base.
    return base + 4 if row["mode"] == "lif" else base


M4_ARMING_NOTE = (
    "M4 arming 2026-07-12: scale_migration=true on every WQ cell — the "
    "step is exact ReLU-homogeneous rescaling, a no-op (s -> 1) when "
    "per-channel spread is small (landed 96c74e42)."
)

CASC_REMOVAL_NOTE = (
    "casc removed from tier-0 2026-07-12 (user directive): mode marked "
    "not-fully-supported pending the cascaded-gap research program; casc "
    "coverage continues in tier1/2 and the advisory framework warns on "
    "selection."
)

COVERAGE_NOTES = {
    0: [
        "[P4 2026-08-07] Streamed-lif respec: the historical lif rows are "
        "RE-TAGGED lifsync (windowed semantics, numbers kept: t0_01-05, 26, "
        "28, 30 — hypervolume axis ('lif','none') unchanged for scoreboard "
        "continuity) and five streamed cells land at t0_45-49 (axis "
        "('lif','streamed')): simplemlp wq s4 / deepmlp d4w128 wq s8 / "
        "mmixcore wq s32 e8 / stream_cnn wq s16 offload (fully on-chip) / "
        "stream_cnn fp s8. stream_cnn is the new spiking-native conv vehicle "
        "(stride-2 blocks, no pooling; platform C for the 1024-axon fc).",
        "Quantization axis is RUNTIME truth (SSOT: config_schema/"
        "deployment_derivation.py): activation quantization is derived from the "
        "mode (ON for lif/casc/sync/ttfsq, OFF for analytical ttfs); configs "
        "never pin activation_quantization. Names use wq (bits-quantized) or fp "
        "(float/vanilla) only.",
        "W3c respec 2026-07-06: t0_04/t0_07 were the fictional aq class "
        "(weight_quantization=false + weight_bits ran as de-facto float; X4 "
        "passed those forms) -> respecced to real WQ deployments.",
        "W3c respec 2026-07-06: t0_15/t0_21 pruning 0.5 -> 0.10 (user-directed). "
        "t0_02/t0_09 stay at 0.5: they pass and keep the heavy-pruning "
        "stressor coverage (t0_18 left with the 2026-07-12 casc removal).",
        "Sim-role respec 2026-07-07 (user-directed): simulators are parity "
        "probes — nevresim max_simulation_samples defaults to 25 (decision "
        "parity), SANA-FE/Loihi stay at 1; the ACCURACY verdict is the SCM "
        "identity read (full test set, torch<->deployed parity-certified). "
        "Historical N=100 simulator accuracy columns are not comparable to "
        "the SCM-based accuracy column.",
        "M1 mixer-e4 respec 2026-07-07 (user-mandated): every mmixcore cell "
        "trains 4 pretrain epochs (evidence t01_07: e4 + full floor passed "
        "0.9712 dedicated — envelope and training budget jointly binding on "
        "the mixer column).",
        "FAST respec 2026-07-08 (user-directed, <5 min clean / 5-10 min "
        "soft / >15 min invalid): lif endpoint cap 1560 -> 600 "
        "(convergence-grounded; healthy reaches 250-930 steps), non-mixer "
        "WQ endpoint cap 2000, draws only for the sync mixer family "
        "(best-of-2; the one family whose draw distribution crosses the "
        "bar), target-reach confirmation + fresh-run ledger reset.",
        "M2 conversion-draws 2026-07-07 (user-approved): mmixcore cells run "
        "best-of-3 D-hat-selected conversion draws on the variance-carrying "
        "stages (LIF/TTFS-cycle/AQ), torch RNG streams seed+k — the search "
        "is deterministic given the config seed and selection can only "
        "improve D-hat (each draw independently keep-best/entry-floored).",
        "Reproducibility respec 2026-07-07 (user-directed): training budgets "
        "are STEP-denominated — endpoint_floor_steps is the RUN-total step "
        "budget shared by armed endpoint stages (endpoint_steps ledger), "
        "sized 16000 (the validated full floor, t01_23) plus the mode's "
        "intermediate-endpoint recipe budgets (lif 2x1560, sync "
        "600). Wall-seconds budgets are gone: identical configs train "
        "identical step counts on any hardware (same config + same seed => "
        "same step trajectory, modulo GPU nondeterminism); wall time is a "
        "pure measurement judged per hardware context at harvest.",
        "BN-mixer respec 2026-07-12 (user-authorized, probe "
        "env_probe_bn_fc128_e8): every mmixcore cell runs the BN+width "
        "envelope — normalization=batch, fc_w_1/fc_w_2 64 -> 128, "
        "training_epochs 8 — superseding the M1 mixer-e4 respec AND the "
        "ttfsq e2 revert (the BN+width envelope is a different regime; the "
        "e4/e2 evidence does not carry over). Measured basis: envelope "
        "probes saturate 0.954-0.981 without BN+width (fc64 e4 0.970; "
        "fc128_c48 e12 0.9785; fc192 e8 0.9805) vs BN+fc128 e8 = 0.982.",
        "WQ-cap deletion 2026-07-12 (synthesis Phase-4): the FAST respec's "
        "flat per-cell wq_endpoint_recovery_steps=2000 caps are removed — "
        "the C1 convergence-stop (landed 91eacc01) patience-stops the WQ "
        "endpoint, so the recipe's 16k ceiling with convergence stop "
        "replaces the flat cap and step budgets self-limit; the "
        "endpoint_floor_steps ledger budgets stay.",
        M4_ARMING_NOTE,
        CASC_REMOVAL_NOTE,
    ],
    "0_1": [
        "Tier-0.1 (2026-07-07, user-directed): a diagnostic matrix of controlled "
        "minimal pairs derived from tier-0's remaining failure modes (theory 5t-5x, "
        "A1-A6, 6b). Every cell moves <= 2 axes off a named tier-0 anchor and "
        "carries a falsifiable hypothesis; the wave's purpose is insight - "
        "failures are the data, not defects.",
        "Families: A install-resolution law (5), B pretrain envelope e4 (3), "
        "D wall/training decomposition (3), E WQ-bit gap (3), F floor "
        "mechanics + green control (3). The C cascade-structure family left "
        "whole with the 2026-07-12 casc removal.",
        "Acceptance bar unchanged from tier-0: >= 0.97 primary deployed (N=100 "
        "pinned; t01_17 inherits the t0_08 sim-sample respec), <= 300 s artifact "
        "wall with all simulators excluded; e4 cells account their extra pretrain "
        "in the wall honestly.",
        "Sim-role respec 2026-07-07 (user-directed): identical to tier-0 — "
        "simulators are parity probes (nevresim N=25 default); the accuracy "
        "verdict is the SCM identity read.",
        "M1 mixer-e4 respec 2026-07-07 (user-mandated): every mmixcore cell "
        "trains 4 epochs, matching the lifted tier-0 anchors; the B-family "
        "mixer diagnostics (t01_07-t01_09) are replication clones now.",
        "Reproducibility respec 2026-07-07 (user-directed): step-denominated "
        "endpoint budgets exactly as tier-0's (endpoint_floor_steps = 16000 "
        "+ mode extra; wall-seconds budgets gone). The F-family 600 s "
        "floor-room diagnostic is subsumed — the full 16k floor is now the "
        "default — so t01_23/t01_24 stay as replication clones of their "
        "anchors (draw-variance controls).",
        "BN-mixer respec 2026-07-12 (user-authorized, probe "
        "env_probe_bn_fc128_e8): identical to tier-0 — every mmixcore cell "
        "(the ttfsq family included: the e2 revert is superseded with the "
        "envelope change) runs normalization=batch, fc_w_1/fc_w_2 64 -> "
        "128, training_epochs 8; anchors and clones move together, so the "
        "minimal-pair deltas are unchanged.",
        "WQ-cap deletion 2026-07-12 (synthesis Phase-4): identical to "
        "tier-0 — the flat wq_endpoint_recovery_steps=2000 caps are "
        "removed; the C1 convergence-stop (landed 91eacc01) makes step "
        "budgets self-limiting under the recipe's 16k ceiling.",
        M4_ARMING_NOTE,
        CASC_REMOVAL_NOTE,
    ],
    1: [M4_ARMING_NOTE],
    2: [M4_ARMING_NOTE],
}


def _write_json(path, payload: str) -> None:
    """Atomic write: concurrent readers (the parallel test suite) never see a
    missing or partial file while the SSOT test regenerates."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload)
    tmp.replace(path)


def _emit_tier(tier, rows, vehicles, dataset, wall_budget_min):
    out_dir = ROOT / f"tier_{tier}"
    out_dir.mkdir(exist_ok=True)
    produced = set()
    manifest = {"tier": tier, "dataset": dataset,
                "wall_budget_minutes_per_run": wall_budget_min, "runs": []}
    if tier in COVERAGE_NOTES:
        manifest["coverage_notes"] = COVERAGE_NOTES[tier]
    for row in rows:
        validate_row_keys(row)
        ds = row.get("dataset", dataset)
        name = _name(tier, row, vehicles)
        config = {
            "seed": row.get("seed", 0),
            "pipeline_mode": "vanilla" if row["quant"] == "fp" else "phased",
            "experiment_name": name,
            "generated_files_path": "./generated",
            "data_provider_name": f"{ds}_DataProvider",
            "platform_constraints": _platform(row, vehicles),
            "deployment_parameters": _deployment(tier, row, vehicles, ds),
            "target_metric_override": None,
            "start_step": None,
            "stop_step": None,
        }
        _write_json(out_dir / f"{name}.json", json.dumps(config, indent=2) + "\n")
        produced.add(f"{name}.json")
        entry = {
            "name": name,
            "config": f"{name}.json",
            "model_type": vehicles[row["vehicle"]]["model_type"],
            "cell": _cell(tier, row, vehicles, ds),
            "tags": row.get("tags", []),
            "expected_wall_min": _wall_budget(tier, row, vehicles, wall_budget_min),
        }
        # Tier-0.1 diagnostic fields: the minimal-pair provenance and the claim
        # the cell's pass/fail arbitrates.
        if "anchor" in row:
            entry["family"] = row["family"]
            entry["anchor"] = row["anchor"]
            entry["axes_moved"] = row["axes"]
            entry["hypothesis"] = row["hypothesis"]
        if "note" in row:
            entry["note"] = row["note"]
        manifest["runs"].append(entry)
    _write_json(out_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    for stale in out_dir.glob("t*.json"):
        if stale.name not in produced:
            stale.unlink()
    return len(rows)


def main():
    n0 = _emit_tier(0, T0, VEHICLES, "MNIST", 5)
    n1 = _emit_tier(1, T1, T1_VEHICLES, "CIFAR10", 120)
    n2 = _emit_tier(2, T2, T2_VEHICLES, "ImageNet", 360)
    n3 = _emit_tier(3, T3, T3_VEHICLES, "ImageNet", 480)
    print(f"tier_0={n0} tier_1={n1} tier_2={n2} tier_3={n3}")


if __name__ == "__main__":
    main()
