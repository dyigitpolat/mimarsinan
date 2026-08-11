# W5 — two reproducibility defects the co-search guard cell exposed (2026-08-12)

Adding ONE end-to-end co-search cell (`t0_60_lifsync_simplemlp_wq_s4_search`)
turned up two independent defects that no unit test could have found, because
both are properties of a *whole run* rather than of a function.

## 1. The search step leaked the RNG stream into the deployment

`JointArchHwProblem._candidate_model` reseeds the global torch/numpy RNGs per
candidate (deliberately — candidate scoring must be reproducible), and nothing
restored the session stream `PipelineSession.apply_determinism` had set from the
config seed. So **the weights a run deployed depended on how many candidates the
search evaluated**: flipping `hw_config_mode` from `fixed` to `search` silently
re-rolled the draw, and "same config + same seed reproduces" — the program's
stated reproducibility contract — quietly stopped holding for search runs.

Fix: `pipelining/determinism.py` holds the run's randomness in one module
(`apply_determinism` beside `isolated_rng_stream`); `ArchitectureSearchStep` runs
the entire search inside the isolation, which covers every optimizer backend.

**Proof it is closed:** the search cell and its hand-declared anchor (`t0_05`)
now produce *byte-identical* models through Pretraining at the same seed
(`Model Building.model.pt` d1d0b28054648d8d, `Pretraining.model.pt`
2ad8c68ea05c1eb4) while deploying to different chips (7 vs 3 hard cores) — the
minimal pair the guard cell was designed to be.

Method note: the prescribed fix (raise the searched neuron bound so no candidate
splits) was implemented, MEASURED, and refuted — an unsplit 848x312x8 winner
failed identically. Splitting was never the trigger; the re-rolled draw was.

## 2. The host ComputeOp evaluator was the backend's choice, not the contract's

The subsumed encoding layer runs as a host ComputeOp whose activation is a LIF
staircase of step theta/T. When a pre-activation lands within ~1.8e-8 of a step
edge, two evaluators land on ADJACENT steps — a gap of exactly theta/T, which
after boundary normalization is **exactly one spike**, i.e. the reported
`1/256 differ, sum 86 vs 87`.

The two sides used different evaluators: the reference flow ran on
`config["device"]` (cuda) while `SanafeRunner._on_compute` called
`execute_compute_op_numpy` **without** `device`, defaulting to cpu. nevresim
already passed `device=self.host_compute_device` (its comment names this exact
failure class) and Lava replays the reference instead of re-deriving — SANA-FE
was the only backend re-deriving on a different device.

Fix at the SSOT: the evaluator becomes contract state
(`SpikingDeploymentContract.host_compute_device`), and `execute_compute_op_numpy`
makes `device` a **required** keyword so no backend can silently pick its own
tie-breaker. A second, independent break at the same seam was found and fixed
with it: the numpy twin cast the f64 wire rate to float32 before
`comb_spike_count`.

**Proof:** on identical artifacts, reverting only the device argument reproduces
the signature byte-for-byte (same index 5, same sums); with the fix, nevresim +
Loihi + SANA-FE all pass (`[sanafe/exact] exact=1.000000, max|dcount|=0 over
2068 neuron-windows`), and the unsplit reference cell still reads 0.9810.

## Why both matter beyond their fixes

Neither defect is a search defect. The co-search cell was simply the first thing
in the program to (a) run a full deployment whose weight draw depended on an
upstream loop, and (b) put a *discovered* chip geometry through the boundary
comparison on a knife-edge draw. Coverage of a new hypervolume region found
defects in the shared path — which is the argument for the guarded cell
existing at all.
