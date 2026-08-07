"""Written rationales for the ConversionPolicy special-case recipe rows."""

_STREAMED_LIF_RATIONALE = (
    "End-to-end event streaming: no boundary normalization exists between the "
    "encode and the readout. The exact-QAT staircase refinement is inherently "
    "WINDOWED semantics (the trained staircase IS the per-hop twin), so it "
    "stays OFF; the plain cycle-accurate LIF adaptation trains the RAW "
    "streaming cascade (re-timing never arms — the train forward IS the "
    "deployed forward) and the NF-SCM gate holds bitwise at atol=0. "
    "Depth-balancing relays stay armed: the +1 latency invariant makes the NF "
    "walk cycle-isomorphic to the chip loop. Loihi/Lava stay off pending a "
    "streaming-faithfulness audit of the wave runner."
)
_LIF_RATIONALE = (
    "BN-freeze makes the QAT train-forward bit-exact to the deployed eval-forward; "
    "the value-blend ramp measures near-zero transfer alignment (rho 0.014-0.167, X1) "
    "while the realizable T-anneal family is >= it everywhere and has no hidden debt "
    "at S=32 (X2b), so the LIF ramp anneals T on genuine LIF members; the Clamp/AQ "
    "ladders are behaviorally inert under the spiking node (X1) and train 0 steps "
    "(mnist_mixer_fix_wave / per_channel_theta / mbh_x2b_lif_tanneal_readout)."
)
_TTFS_QUANTIZED_RATIONALE = (
    "Full-quantile (q=1.0) per-perceptron decode helps the quantized timing path; it "
    "is harmful for LIF, whose decode scale is per-channel (per_channel_theta). Green "
    "family: stays on the floor+half-step proxy, and the final-WQ endpoint carries "
    "the well-conditioned floor (proxy->deployed transfer measured sub-SE: t0_11 "
    "+0.0007 / t0_14 -0.0014 / t01_06 -0.0010 vs SE 0.0092, so the funded climb "
    "survives to the deployed read); the exact-kernel endpoint promotion "
    "is an X4 follow-up (mbh_t6_sync_exact_kernel)."
)
_CASCADED_RATIONALE = (
    "The controller collapses on the deep genuine cascade (rate stalls then drops to "
    "chance); the fast blend ladder is the ec=0 survivor (0.9396 @ parity 0.9961) "
    "(per_channel_theta_deployment_fidelity). Multi-segment vehicles walk the "
    "converted-prefix frontier: boundary gradients are severed, so only the P4 "
    "frontier trains every layer once, at the moment its conversion damage is live "
    "(T4 shootout: 0.9629 vs 0.9277 post-FT at equal budget, mbh_t4_depth_law)."
)
_BIT_PARITY_LOSSLESS_RATIONALE = (
    "Analytical ttfs deploys bit-exactly (parity 0.0000% per-neuron mismatch), so "
    "sup(controller targets) <= float envelope + noise and the endpoint patience-stops "
    "at exactly patience x check_interval with the budget unspent (the stagnation "
    "theorem, mbh_analytical_ttfs_stagnation). The floor = the internal acceptance "
    "target lets the endpoint spend the measured wall headroom; keep-best and the "
    "entry guard keep the stage non-destructive, and reached=False stays legal."
)
_SYNCHRONIZED_RATIONALE = (
    "synchronized IS ttfs_quantized at deploy: sync-deploy = ttfs_quantized-deploy + the "
    "free segment-input single-spike grid-snap. It rides the ttfs_quantized ladder shape "
    "but TRAINS the exact deployed composition — the ceil TTFS kernel under STE + the "
    "per-stage entry grid snap — as the QAT endpoint (T6: parity 0.9180/0.8633-abort -> "
    "1.0000/256 on t0_22/t0_21), and the mapping-time +0.5/Tq bias compensation is "
    "skipped for models so trained (marker-asserted). Its per-neuron NF↔SCM parity "
    "stays excluded from the bit-exact per-neuron gate; nevresim has no "
    "synchronized-window backend (mbh_t6_sync_exact_kernel)."
)
