"""SSOT for the tiered integration-run matrices: (re)generates all configs + manifests."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

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
}

VEHICLES = {
    "mmixcore": {"model_type": "mlp_mixer_core", "platform": "A", "axis": "mlp_mixer_core",
                 "model_config": {"base_activation": "ReLU", "patch_n_1": 4, "patch_m_1": 4,
                                  "patch_c_1": 32, "fc_w_1": 64, "fc_w_2": 64}},
    "lenet5": {"model_type": "lenet5", "platform": "B", "axis": "lenet5",
               "model_config": {"variant": "lenet5"}},
    "deepcnn": {"model_type": "deep_cnn", "platform": "C", "axis": "deep_cnn",
                "model_config": {"depth": 8, "width": 16}},
    "deepmlp": {"model_type": "deep_mlp", "platform": "B", "axis": "deep_mlp",
                "model_config": {"depth": 8, "width": 64}},
    "simplemlp": {"model_type": "simple_mlp", "platform": "B", "axis": "deep_mlp",
                  "model_config": {"mlp_width_1": 256, "mlp_width_2": 128}},
}

MODES = {
    "lif": {"spiking_mode": "lif", "firing_mode": "Default", "spike_generation_mode": "Uniform",
            "thresholding_mode": "<", "axis": ("lif", "none")},
    "ttfs": {"spiking_mode": "ttfs", "firing_mode": "TTFS", "spike_generation_mode": "TTFS",
             "thresholding_mode": "<=", "axis": ("ttfs", "none")},
    "ttfsq": {"spiking_mode": "ttfs_quantized", "firing_mode": "TTFS", "spike_generation_mode": "TTFS",
              "thresholding_mode": "<=", "axis": ("ttfs_quantized", "none")},
    "casc": {"spiking_mode": "ttfs_cycle_based", "ttfs_cycle_schedule": "cascaded",
             "firing_mode": "TTFS", "spike_generation_mode": "TTFS", "thresholding_mode": "<=",
             "axis": ("ttfs_cycle_based", "cascaded")},
    "sync": {"spiking_mode": "ttfs_cycle_based", "ttfs_cycle_schedule": "synchronized",
             "firing_mode": "TTFS", "spike_generation_mode": "TTFS", "thresholding_mode": "<=",
             "axis": ("ttfs_cycle_based", "synchronized")},
}

# Quant axis reflects RUNTIME truth (SSOT: config_schema/deployment_derivation.py):
# activation quantization is derived from the mode (ON for lif/casc/sync/ttfsq,
# OFF for analytical ttfs), so configs carry only the WQ declaration and never pin
# activation_quantization. fp = the vanilla float assembly (pipeline_mode vanilla).
QUANT = {
    "fp": {"weight_quantization": False},
    "wq": {"weight_quantization": True},
}
AQ_DERIVED_MODES = {"lif", "ttfsq", "casc", "sync"}


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
# at the LIF and AQ endpoints; casc 600 at the TTFS-cycle and AQ endpoints;
# sync 600 at the AQ endpoint). Freed-ladder bonuses on stalled rungs may
# shave the WQ floor by at most one planned ladder (bounded, deterministic).
ENDPOINT_FLOOR_STEPS_BASE = 16000
ENDPOINT_MODE_EXTRA_STEPS = {
    "lif": 2 * 600,
    "casc": 2 * 600,
    "sync": 600,
}


def _endpoint_floor_steps(row):
    """The run-total endpoint step budget for one tier-0/0.1 row (steps)."""
    return ENDPOINT_FLOOR_STEPS_BASE + ENDPOINT_MODE_EXTRA_STEPS.get(row["mode"], 0)

# [M1 mixer-e4 respec 2026-07-07, user-mandated] every mmixcore cell in both
# matrices trains 4 pretrain epochs. Evidence: t01_07 (ttfs mixer, e4 + full
# floor) passed 0.9712 on a dedicated node — envelope and training budget were
# JOINTLY binding on the mixer column (the B family's e2-anchored refutation
# tested e4 against STARVED floors; with step-denominated budgets the floor
# is always full).
MIXER_E4_NOTE = (
    "M1 mixer-e4 respec 2026-07-07 (user-mandated): training_epochs 2 -> 4 "
    "on every mmixcore cell; evidence t01_07 (e4 + full floor = 0.9712 "
    "dedicated) — envelope and budget jointly binding."
)

T0 = [
    dict(n=1, mode="lif", quant="wq", wb=5, s=4, vehicle="mmixcore", epochs=4,
         note=MIXER_E4_NOTE),
    dict(n=2, mode="lif", quant="fp", wb=5, s=8, vehicle="lenet5", firing="Novena",
         encoding="offload", pruned=0.5, tags=["novena", "offload", "pruned"]),
    # W2: the 360-core pool packs t0_03 only scheduled (111/360 peak over 4 phases).
    dict(n=3, mode="lif", quant="wq", wb=4, s=16, vehicle="deepcnn", depth=8,
         scheduling=True, tags=["sched"]),
    # W3c respec: was the fictional aq form (wq=False + weight_bits ran as a de-facto
    # float deployment, X4 passed that form); now a real WQ deployment.
    dict(n=4, mode="lif", quant="wq", wb=5, s=32, vehicle="deepmlp", depth=8,
         note="W3c respec 2026-07-06: fictional aq form (weight_quantization=false + "
              "weight_bits ran float) -> real WQ deployment; X4 passed the old form."),
    dict(n=5, mode="lif", quant="wq", wb=5, s=4, vehicle="simplemlp", seed=1),
    dict(n=6, mode="ttfs", quant="wq", wb=5, s=8, vehicle="mmixcore", epochs=4,
         note=MIXER_E4_NOTE),
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
    dict(n=11, mode="ttfsq", quant="wq", wb=5, s=16, vehicle="mmixcore",
         encoding="offload", tags=["offload"], epochs=4, note=MIXER_E4_NOTE),
    dict(n=12, mode="ttfsq", quant="wq", wb=8, s=32, vehicle="lenet5", tags=["wall_risk"]),
    dict(n=13, mode="ttfsq", quant="wq", wb=5, s=4, vehicle="deepcnn", depth=4),
    dict(n=14, mode="ttfsq", quant="wq", wb=5, s=8, vehicle="deepmlp", depth=8),
    dict(n=15, mode="ttfsq", quant="wq", wb=5, s=8, vehicle="simplemlp", pruned=0.10,
         tags=["pruned10"],
         note="W3c respec 2026-07-06: pruning 0.5 -> 0.10 (user-directed; 50% is "
              "too strong for this cell)."),
    dict(n=16, mode="casc", quant="wq", wb=5, s=8, vehicle="mmixcore", encoding="offload",
         scheduling=True, has_bias=False, tags=["offload", "sched", "nobias"],
         epochs=4, note=MIXER_E4_NOTE),
    dict(n=17, mode="casc", quant="wq", wb=5, s=32, vehicle="lenet5", tags=["wall_risk"]),
    dict(n=18, mode="casc", quant="wq", wb=5, s=4, vehicle="deepcnn", depth=4, pruned=0.5,
         tags=["pruned", "known_collapse_candidate"]),
    # W2: plain d16 is recipe-unreachable (5/5 runs at chance); residual is the
    # trainable deep backbone (USER DECISION 2026-07-06: residual, depth kept).
    dict(n=19, mode="casc", quant="wq", wb=4, s=16, vehicle="deepmlp", depth=16,
         residual=True, tags=["wall_risk", "known_collapse_candidate", "residual"]),
    dict(n=20, mode="casc", quant="wq", wb=5, s=4, vehicle="simplemlp"),
    dict(n=21, mode="sync", quant="wq", wb=5, s=8, vehicle="mmixcore", pruned=0.10,
         tags=["pruned10"], epochs=4,
         note="W3c respec 2026-07-06: pruning 0.5 -> 0.10 (user-directed; 50% is "
              "too strong for this cell). " + MIXER_E4_NOTE),
    dict(n=22, mode="sync", quant="wq", wb=5, s=4, vehicle="lenet5", scheduling=True, tags=["sched"]),
    dict(n=23, mode="sync", quant="wq", wb=8, s=16, vehicle="deepcnn", depth=4),
    dict(n=24, mode="sync", quant="wq", wb=5, s=8, vehicle="deepmlp", depth=4, width=128),
    dict(n=25, mode="sync", quant="wq", wb=5, s=32, vehicle="simplemlp"),
]

# Tier-0.1: 25 controlled diagnostics derived from tier-0's remaining failure
# modes. Every cell is a MINIMAL PAIR of a named tier-0 anchor (<=2 axes moved,
# enforced by test) and carries a falsifiable hypothesis; failures are the data.
T0_1 = [
    # A - install-resolution law calibration (A6, theory 5v): sweep S per mode
    # on the L=9 mixer chain to make the resolution x chain-depth boundary visible.
    dict(n=1, family="A", mode="lif", quant="wq", wb=5, s=8, vehicle="mmixcore",
         epochs=4, note=MIXER_E4_NOTE,
         anchor="t0_01_lif_mmixcore_wq_s4", axes=["S: 4 -> 8"],
         hypothesis="t0_01's binding failure is the T=4 window back-loading crater "
                    "(frozen-weights NF 0.32@T4 -> 0.82@T8): at S=8 the entry crater "
                    "clears and deployed reaches >= 0.97."),
    dict(n=2, family="A", mode="lif", quant="wq", wb=5, s=16, vehicle="mmixcore",
         epochs=4, note=MIXER_E4_NOTE,
         anchor="t0_01_lif_mmixcore_wq_s4", axes=["S: 4 -> 16"],
         hypothesis="Deployed accuracy is monotone along the T-healing curve "
                    "(NF 0.90@T16 frozen-weights): S=16 lands at or above the S=8 "
                    "read, isolating the temporal kernel from the envelope."),
    dict(n=3, family="A", mode="casc", quant="wq", wb=5, s=4, vehicle="mmixcore",
         encoding="offload", scheduling=True, has_bias=False,
         tags=["offload", "sched", "nobias"], epochs=4, note=MIXER_E4_NOTE,
         anchor="t0_16_casc_mmixcore_wq_s8_offload_sched_nobias", axes=["S: 8 -> 4"],
         hypothesis="Cascaded per-hop attenuation GROWS with S (genuine forward "
                    "0.57@S8 -> 0.31@S64 frozen-weights): at S=4 the conversion gap "
                    "shrinks and deployed exceeds the anchor's ~0.90 ceiling."),
    dict(n=4, family="A", mode="sync", quant="wq", wb=5, s=16, vehicle="mmixcore",
         pruned=0.10, tags=["pruned10"], epochs=4, note=MIXER_E4_NOTE,
         anchor="t0_21_sync_mmixcore_wq_s8_pruned10", axes=["S: 8 -> 16"],
         hypothesis="Sync level starvation shrinks with grid resolution (entry "
                    "0.10@S8 -> 0.19@S16 -> 0.69@S32 pre-fix-stack): at S=16 the fix "
                    "stack plus WQ endpoint clear 0.97."),
    dict(n=5, family="A", mode="sync", quant="wq", wb=5, s=4, vehicle="mmixcore",
         pruned=0.10, tags=["pruned10"], epochs=4, note=MIXER_E4_NOTE,
         anchor="t0_21_sync_mmixcore_wq_s8_pruned10", axes=["S: 8 -> 4"],
         hypothesis="At S=4 starvation deepens beyond the quantile+half-step+staging "
                    "stack's recovery on the L=9 chain: deployed lands below the "
                    "anchor's 0.9677 (the boundary probed from the failing side)."),
    dict(n=6, family="A", mode="ttfsq", quant="wq", wb=5, s=8, vehicle="mmixcore",
         encoding="offload", tags=["offload"], epochs=4, note=MIXER_E4_NOTE,
         anchor="t0_11_ttfsq_mmixcore_wq_s16_offload", axes=["S: 16 -> 8"],
         hypothesis="ttfsq's dense nearest-rounding install keeps per-hop error small "
                    "at any resolution: halving S from 16 to 8 stays crater-free and "
                    "passes - the install-resolution law's negative control."),
    # B - pretrain envelope: clone each envelope-tail cell at training_epochs=4
    # (all else identical); the extra pretrain is accounted in the wall honestly.
    dict(n=7, family="B", mode="ttfs", quant="wq", wb=5, s=8, vehicle="mmixcore",
         epochs=4, tags=["e4"], wall_min=10,
         anchor="t0_06_ttfs_mmixcore_wq_s8", axes=[],
         note="M1 respec 2026-07-07: the anchor was lifted to e4 on this "
              "cell's own evidence (0.9712 dedicated); now a replication clone.",
         hypothesis="Replication clone of t0_06 post-M1 (this cell WAS the "
                    "e4 evidence: envelope+budget jointly binding); reads "
                    "calibrate the mixer column's draw variance."),
    dict(n=8, family="B", mode="lif", quant="wq", wb=5, s=4, vehicle="mmixcore",
         epochs=4, tags=["e4"], wall_min=16,
         anchor="t0_01_lif_mmixcore_wq_s4", axes=[],
         note="M1 respec 2026-07-07: the anchor was lifted to e4 (evidence "
              "t01_07); now a replication clone.",
         hypothesis="Replication clone of t0_01 post-M1 (the B2 diagnostic "
                    "showed e4 alone re-opens the S=4 crater at entry 0.276); "
                    "reads calibrate the mixer column's draw variance."),
    dict(n=9, family="B", mode="sync", quant="wq", wb=5, s=8, vehicle="mmixcore",
         pruned=0.10, epochs=4, tags=["pruned10", "e4"], wall_min=10,
         anchor="t0_21_sync_mmixcore_wq_s8_pruned10", axes=[],
         note="M1 respec 2026-07-07: the anchor was lifted to e4 (evidence "
              "t01_07); now a replication clone.",
         hypothesis="Replication clone of t0_21 post-M1 (the B3 diagnostic "
                    "showed the sync AQ crater unchanged by e4); reads "
                    "calibrate the sync mixer's draw variance."),
    dict(n=10, family="B", mode="casc", quant="wq", wb=5, s=8, vehicle="mmixcore",
         encoding="offload", scheduling=True, has_bias=False, epochs=4,
         tags=["offload", "sched", "nobias", "e4"], wall_min=10,
         anchor="t0_16_casc_mmixcore_wq_s8_offload_sched_nobias",
         axes=[],
         note="M1 respec 2026-07-07: the anchor was lifted to e4 (evidence "
              "t01_07); now a replication clone.",
         hypothesis="Replication clone of t0_16 post-M1 (the B4 diagnostic "
                    "showed e4 arms the retention gate against the casc "
                    "conversion crater); reads calibrate casc draw variance."),
    dict(n=11, family="B", mode="casc", quant="wq", wb=4, s=16, vehicle="deepmlp",
         depth=16, residual=True, epochs=4, tags=["residual", "e4"], wall_min=12,
         anchor="t0_19_casc_deepmlp_d16_wq_s16_residual",
         axes=["training_epochs: 2 -> 4"],
         hypothesis="t0_19 is budget/envelope-starved (float 0.973, deployed 0.942): "
                    "4 epochs of pretrain buy the envelope and the frontier ladder "
                    "converts it to a pass."),
    # C - cascade structure isolation: which axis of t0_16/t0_18/t0_19 carries
    # the cascade gap - the tag stack, bias axons, depth, or boundaries?
    dict(n=12, family="C", mode="casc", quant="wq", wb=5, s=8, vehicle="mmixcore",
         epochs=4, note=MIXER_E4_NOTE,
         anchor="t0_16_casc_mmixcore_wq_s8_offload_sched_nobias",
         axes=["tag stack: offload+sched+nobias -> plain"],
         hypothesis="t0_16's conversion gap survives the removal of "
                    "offload+sched+nobias: the gap is intrinsic to casc x L=9 "
                    "(deployed stays ~0.90), not tag-induced."),
    dict(n=13, family="C", mode="casc", quant="wq", wb=5, s=8, vehicle="mmixcore",
         has_bias=False, tags=["nobias"], epochs=4, note=MIXER_E4_NOTE,
         anchor="t0_16_casc_mmixcore_wq_s8_offload_sched_nobias",
         axes=["tag stack: offload+sched+nobias -> nobias only"],
         hypothesis="Bias-axon removal alone does not reproduce the anchor's gap: "
                    "this cell reads within noise of the plain clone; a materially "
                    "lower read isolates nobias as the load-bearing tag."),
    dict(n=14, family="C", mode="casc", quant="wq", wb=4, s=16, vehicle="deepmlp",
         depth=8, residual=True, tags=["residual"],
         anchor="t0_19_casc_deepmlp_d16_wq_s16_residual", axes=["depth: 16 -> 8"],
         hypothesis="The boundary law is linear (~0.3-0.5 pp per boundary): halving "
                    "t0_19's depth to d8 (~half the seams) plus the lighter training "
                    "bulk yields a pass."),
    dict(n=15, family="C", mode="casc", quant="wq", wb=5, s=4, vehicle="deepcnn",
         depth=8, scheduling=True, tags=["sched"],
         anchor="t0_18_casc_deepcnn_d4_wq_s4_pruned",
         axes=["depth: 4 -> 8 (sched on: the W2-proven d8 packing prerequisite)",
               "pruning: 0.5 -> dense"],
         hypothesis="Conv cascades carry no intra-segment compounding (segment depth "
                    "0): doubling t0_18's depth to d8 unpruned stays green - the "
                    "depth cost is boundary-linear, not exponential."),
    # D - wall / training-ceiling decomposition: the three wall cells' time is
    # load-bearing training scaling with S and depth; map the passable frontier.
    dict(n=16, family="D", mode="lif", quant="wq", wb=4, s=8, vehicle="deepcnn",
         depth=8, scheduling=True, tags=["sched"],
         anchor="t0_03_lif_deepcnn_d8_wq_s16_sched", axes=["S: 16 -> 8"],
         hypothesis="t0_03's wall is training-bulk scaling with S: at S=8 the "
                    "artifact wall drops below 300 s at unchanged accuracy (~0.98)."),
    dict(n=17, family="D", mode="ttfs", quant="fp", wb=5, s=16, vehicle="deepcnn",
         depth=8, scheduling=True, sim_samples=25, tags=["sched"],
         anchor="t0_08_ttfs_deepcnn_d8_fp_s32_sched", axes=["S: 32 -> 16"],
         note="Inherits the t0_08 sim-sample respec (per-core GEMM sim is "
              "sample-bound; N=25 keeps the exclusion arithmetic comparable).",
         hypothesis="Analytic-TTFS deployment is S-invariant (SCM bit-identical "
                    "across S): at S=16 accuracy stays 1.00 while the training and "
                    "sim walls shrink - wall scales with S, semantics do not."),
    dict(n=18, family="D", mode="casc", quant="wq", wb=5, s=16, vehicle="lenet5",
         wall_min=10,
         anchor="t0_17_casc_lenet5_wq_s32", axes=["S: 32 -> 16"],
         hypothesis="t0_17's wall scales with S: at S=16 the artifact wall fits "
                    "300 s and accuracy holds >= 0.97."),
    dict(n=19, family="D", mode="lif", quant="wq", wb=4, s=16, vehicle="deepcnn",
         depth=6, scheduling=True, tags=["sched"],
         anchor="t0_03_lif_deepcnn_d8_wq_s16_sched", axes=["depth: 8 -> 6"],
         hypothesis="The lif deepcnn wall frontier lies between d6 and d8: at d6/S=16 "
                    "the artifact wall lands materially below t0_03's, within or "
                    "near the bar."),
    # E - quantization-resolution / WQ gap: move only weight_bits.
    dict(n=20, family="E", mode="sync", quant="wq", wb=8, s=8, vehicle="mmixcore",
         pruned=0.10, tags=["pruned10", "wb8"], epochs=4, note=MIXER_E4_NOTE,
         anchor="t0_21_sync_mmixcore_wq_s8_pruned10", axes=["weight_bits: 5 -> 8"],
         hypothesis="t0_21's 0.9677 is a pure WQ gap (float 0.9735): 8-bit weights "
                    "close the ~0.6 pp and the cell passes."),
    dict(n=21, family="E", mode="lif", quant="wq", wb=8, s=4, vehicle="mmixcore",
         tags=["wb8"], epochs=4, note=MIXER_E4_NOTE,
         anchor="t0_01_lif_mmixcore_wq_s4", axes=["weight_bits: 5 -> 8"],
         hypothesis="t0_01's sub-bar residual has no WQ component: wb8 leaves "
                    "deployed at the ~0.948 envelope; a lift would relocate the "
                    "binder to quantization."),
    dict(n=22, family="E", mode="ttfsq", quant="wq", wb=4, s=4, vehicle="deepcnn",
         depth=4, tags=["wb4"],
         anchor="t0_13_ttfsq_deepcnn_d4_wq_s4", axes=["weight_bits: 5 -> 4"],
         hypothesis="The ttfsq WQ boundary sits below wb4 on a shallow conv vehicle: "
                    "t0_13's clone at wb4 costs <= 1 pp and still passes."),
    # F - floor mechanics + controls. The 600 s floor-room diagnostic is
    # SUBSUMED by the reproducibility respec (2026-07-07): endpoint budgets
    # are step-denominated and every cell now carries the full validated
    # 16k-step floor by default, which is exactly what these two cells paid
    # extra wall to prove (t01_23: full 16k => 0.97; t01_24: full budget =>
    # draw-variant 0.9441). They stay as replication clones of their anchors
    # (draw-variance controls for the conversion-draws mechanism).
    dict(n=23, family="F", mode="ttfs", quant="wq", wb=5, s=8, vehicle="mmixcore",
         epochs=4, tags=["floor"], wall_min=18,
         anchor="t0_06_ttfs_mmixcore_wq_s8",
         axes=[],
         note="Reproducibility respec 2026-07-07: the 600 s floor-room "
              "diagnostic is subsumed — step-denominated budgets give every "
              "cell the full 16k floor; kept as a replication clone.",
         hypothesis="Replication clone of t0_06 (the F1 diagnostic proved the "
                    "floor budget-complete: full 16k steps => the 0.97 fbu "
                    "ceiling, now the default); reads calibrate draw variance."),
    dict(n=24, family="F", mode="sync", quant="wq", wb=5, s=8, vehicle="mmixcore",
         pruned=0.10, epochs=4, tags=["pruned10", "floor"], wall_min=18,
         anchor="t0_21_sync_mmixcore_wq_s8_pruned10",
         axes=[],
         note="Reproducibility respec 2026-07-07: the 600 s floor-room "
              "diagnostic is subsumed — step-denominated budgets give every "
              "cell the full 16k floor; kept as a replication clone.",
         hypothesis="Replication clone of t0_21 (the F2 diagnostic proved the "
                    "sync residual draw-variant at full budget: 0.9441 vs the "
                    "0.9677 fbu); reads calibrate draw variance."),
    dict(n=25, family="F", mode="ttfsq", quant="wq", wb=8, s=32, vehicle="lenet5",
         anchor="t0_12_ttfsq_lenet5_wq_s32", axes=[],
         hypothesis="Pure t0_12 replication passes at its X4 numbers; per-pack "
                    "replicas of this cell calibrate node contention for the wave."),
]

T1 = [
    dict(n=1, mode="lif", quant="wq", wb=8, s=16, vehicle="squeezenet", regime="pretrained"),
    dict(n=2, mode="ttfs", quant="wq", wb=8, s=32, vehicle="vit", regime="pretrained", tags=["wall_risk"]),
    dict(n=3, mode="ttfsq", quant="wq", wb=8, s=32, vehicle="vit", regime="pretrained",
         pruned=0.05, tags=["wall_risk", "pruned"]),
    dict(n=4, mode="casc", quant="wq", wb=5, s=8, vehicle="deepcnn32", depth=8, regime="from_scratch"),
    dict(n=5, mode="sync", quant="wq", wb=5, s=8, vehicle="deepcnn32", depth=4, regime="from_scratch"),
    dict(n=6, mode="lif", quant="wq", wb=8, s=32, vehicle="deepcnn32", depth=8, regime="from_scratch"),
    dict(n=7, mode="casc", quant="wq", wb=8, s=16, vehicle="squeezenet", regime="pretrained",
         scheduling=True, tags=["sched"]),
    dict(n=8, mode="ttfs", quant="fp", wb=8, s=16, vehicle="mixerc10", regime="from_scratch"),
]

T1_VEHICLES = {
    "squeezenet": {"model_type": "torch_squeezenet11", "platform": "D", "axis": "vit_b",
                   "model_config": {}, "coalescing": False},
    "vit": {"model_type": "torch_vit", "platform": "E", "axis": "vit_b",
            "model_config": {}, "coalescing": False,
            "preprocessing": {"interpolation": "bicubic", "resize_to": 224, "normalize": "imagenet"},
            "batch_size": 512, "tuning_batch_size": 128},
    "deepcnn32": {"model_type": "deep_cnn", "platform": "F", "axis": "deep_cnn",
                  "model_config": {"depth": 8, "width": 32}},
    "mixerc10": {"model_type": "mlp_mixer_core", "platform": "F", "axis": "mlp_mixer_core",
                 "model_config": {"base_activation": "ReLU", "patch_n_1": 4, "patch_m_1": 4,
                                  "patch_c_1": 256, "fc_w_1": 128, "fc_w_2": 256}},
}

T2 = [
    dict(n=1, mode="lif", quant="wq", wb=8, s=32, vehicle="resnet50", dataset="ImageNet",
         regime="pretrained", scheduling=True, lr=0.0001, finetune_epochs=0, budget=0.5, tags=["sched"]),
    dict(n=2, mode="ttfsq", quant="wq", wb=8, s=32, vehicle="vit", dataset="CIFAR100",
         regime="pretrained", scheduling=True, pruned=0.05, tags=["sched", "pruned"]),
    dict(n=3, mode="casc", quant="wq", wb=8, s=32, vehicle="squeezenet", dataset="CIFAR100",
         regime="pretrained", scheduling=True, tags=["sched", "wall_risk"]),
]

T2_VEHICLES = {
    "resnet50": {"model_type": "torch_resnet50", "platform": "G", "axis": "vit_b",
                 "model_config": {}, "coalescing": False},
    "vit": T1_VEHICLES["vit"],
    "squeezenet": T1_VEHICLES["squeezenet"],
}

DATASET_AXIS = {"MNIST": "mnist", "CIFAR10": "cifar10", "CIFAR100": "cifar100", "ImageNet": "imagenet"}


def _name(tier, row, vehicles):
    prefix = f"t{tier}".replace("_", "")
    v = row["vehicle"]
    depth = f"_d{row['depth']}" if "depth" in row else ""
    tags = "".join(f"_{t}" for t in row.get("tags", []) if t in
                   ("offload", "sched", "nobias", "pruned", "pruned10", "novena",
                    "identity", "residual", "e4", "wb8", "wb4", "floor"))
    return f"{prefix}_{row['n']:02d}_{row['mode']}_{v}{depth}_{row['quant']}_s{row['s']}{tags}"


def _platform(row, vehicles):
    v = vehicles[row["vehicle"]]
    plat = json.loads(json.dumps(PLATFORMS[v["platform"]]))
    has_bias = row.get("has_bias", True)
    for core in plat["cores"]:
        core["has_bias"] = has_bias
    plat["has_bias"] = has_bias
    plat["target_tq"] = row["s"]
    plat["simulation_steps"] = row["s"]
    plat["weight_bits"] = row["wb"]
    plat["allow_coalescing"] = row.get("coalescing", v.get("coalescing", True))
    plat["allow_neuron_splitting"] = row.get("splitting", True)
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
            "budget", 0.25 if tier == 0 and row["mode"] in ("lif", "ttfs") else 0.5 if tier == 0 else 1,
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
        "spiking_mode": mode["spiking_mode"],
        "firing_mode": row.get("firing", mode["firing_mode"]),
        "spike_generation_mode": mode["spike_generation_mode"],
        "thresholding_mode": mode["thresholding_mode"],
        "encoding_layer_placement": row.get("encoding", "subsume"),
        "weight_quantization": quant["weight_quantization"],
    }
    if "ttfs_cycle_schedule" in mode:
        dp["ttfs_cycle_schedule"] = mode["ttfs_cycle_schedule"]
    if "pruned" in row:
        dp["pruning"] = True
        dp["pruning_fraction"] = row["pruned"]
    if tier == 0:
        dp["endpoint_floor_steps"] = _endpoint_floor_steps(row)
        can_pass_by_floor = (
            row["vehicle"] == "mmixcore" and row["mode"] in ("ttfs", "sync")
        )
        if quant["weight_quantization"] and not can_pass_by_floor:
            # FAST respec 2026-07-08: the 16k floor stays ONLY where the
            # family measurably passes by the climb (ttfs mixers 0.9722-0.9748,
            # sync mixers at-bar). Non-mixer endpoints reach/flatten in
            # 250-930 steps, and the lif (0.947 envelope ceiling) and casc
            # (0.88-0.91 kernel ceiling) mixer families cannot reach 0.97 by
            # grinding — their floor target is above the family ceiling, so
            # the cap was the de-facto budget. Cap = ~2x the largest healthy
            # reach; accuracy on capped families is owned by draws/respec,
            # not wall.
            dp["wq_endpoint_recovery_steps"] = 2000
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
    return dp


def _cell(tier, row, vehicles, dataset):
    v = vehicles[row["vehicle"]]
    firing, sync = MODES[row["mode"]]["axis"]
    return {
        "firing": firing,
        "sync": sync,
        "quantization": _quant_axis(row),
        "S": str(row["s"]),
        "depth": str(row["depth"]) if "depth" in row else "any",
        "vehicle": v["axis"],
        "dataset": DATASET_AXIS[dataset],
        "regime": row.get("regime", "from_scratch"),
        "pruning": "pruned" if "pruned" in row else "dense",
        "encoding_placement": row.get("encoding", "subsume"),
    }


def _wall_budget(tier, row, vehicles, default_min):
    if "wall_min" in row:
        return row["wall_min"]
    if _policy_tier(tier) != 0:
        return default_min
    # Measured locally: LIF's Loihi leg and conv-model cells exceed 5 min.
    if row["mode"] == "lif" or vehicles[row["vehicle"]]["model_type"] == "deep_cnn":
        return 12
    return 6


COVERAGE_NOTES = {
    0: [
        "Quantization axis is RUNTIME truth (SSOT: config_schema/"
        "deployment_derivation.py): activation quantization is derived from the "
        "mode (ON for lif/casc/sync/ttfsq, OFF for analytical ttfs); configs "
        "never pin activation_quantization. Names use wq (bits-quantized) or fp "
        "(float/vanilla) only.",
        "W3c respec 2026-07-06: t0_04/t0_07 were the fictional aq class "
        "(weight_quantization=false + weight_bits ran as de-facto float; X4 "
        "passed those forms) -> respecced to real WQ deployments.",
        "W3c respec 2026-07-06: t0_15/t0_21 pruning 0.5 -> 0.10 (user-directed). "
        "t0_02/t0_09/t0_18 stay at 0.5: they pass and keep the heavy-pruning "
        "stressor coverage.",
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
        "intermediate-endpoint recipe budgets (lif 2x1560, casc 2x600, sync "
        "600). Wall-seconds budgets are gone: identical configs train "
        "identical step counts on any hardware (same config + same seed => "
        "same step trajectory, modulo GPU nondeterminism); wall time is a "
        "pure measurement judged per hardware context at harvest.",
    ],
    "0_1": [
        "Tier-0.1 (2026-07-07, user-directed): a diagnostic matrix of controlled "
        "minimal pairs derived from tier-0's remaining failure modes (theory 5t-5x, "
        "A1-A6, 6b). Every cell moves <= 2 axes off a named tier-0 anchor and "
        "carries a falsifiable hypothesis; the wave's purpose is insight - "
        "failures are the data, not defects.",
        "Families: A install-resolution law (6), B pretrain envelope e4 (5), "
        "C cascade structure isolation (4), D wall/training decomposition (4), "
        "E WQ-bit gap (3), F floor mechanics + green control (3).",
        "Acceptance bar unchanged from tier-0: >= 0.97 primary deployed (N=100 "
        "pinned; t01_17 inherits the t0_08 sim-sample respec), <= 300 s artifact "
        "wall with all simulators excluded; e4 cells account their extra pretrain "
        "in the wall honestly.",
        "Sim-role respec 2026-07-07 (user-directed): identical to tier-0 — "
        "simulators are parity probes (nevresim N=25 default); the accuracy "
        "verdict is the SCM identity read.",
        "M1 mixer-e4 respec 2026-07-07 (user-mandated): every mmixcore cell "
        "trains 4 epochs, matching the lifted tier-0 anchors; the B-family "
        "mixer diagnostics (t01_07-t01_10) are replication clones now.",
        "Reproducibility respec 2026-07-07 (user-directed): step-denominated "
        "endpoint budgets exactly as tier-0's (endpoint_floor_steps = 16000 "
        "+ mode extra; wall-seconds budgets gone). The F-family 600 s "
        "floor-room diagnostic is subsumed — the full 16k floor is now the "
        "default — so t01_23/t01_24 stay as replication clones of their "
        "anchors (draw-variance controls).",
    ],
}


def _write_json(path, payload: str) -> None:
    """Atomic write: concurrent readers (the parallel test suite) never see a
    missing or partial file while the SSOT test regenerates."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload)
    tmp.replace(path)


def _emit_tier(tier, rows, vehicles, dataset, wall_budget_min):
    out_dir = ROOT / f"tier{tier}"
    out_dir.mkdir(exist_ok=True)
    produced = set()
    manifest = {"tier": tier, "dataset": dataset,
                "wall_budget_minutes_per_run": wall_budget_min, "runs": []}
    if tier in COVERAGE_NOTES:
        manifest["coverage_notes"] = COVERAGE_NOTES[tier]
    for row in rows:
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
    n01 = _emit_tier("0_1", T0_1, VEHICLES, "MNIST", 5)
    n1 = _emit_tier(1, T1, T1_VEHICLES, "CIFAR10", 120)
    n2 = _emit_tier(2, T2, T2_VEHICLES, "ImageNet", 360)
    print(f"tier0={n0} tier0_1={n01} tier1={n1} tier2={n2}")


if __name__ == "__main__":
    main()
