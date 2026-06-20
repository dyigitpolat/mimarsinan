# Phase B — Root-cause diagnosis + fixes landed

## D3a — SCM crash → FIXED ✅

**Root cause.** The NF↔SCM parity gates (`nf_scm_parity.py`) run a full mapper-graph
`model(samples)` forward, but only moved `samples` to `next(model.parameters()).device`
— never unified the *model*. Across the offload/cache seams the mapper-graph compute
modules (e.g. the `classifier` ComputeOp) and the perceptrons end up on different devices
(offload strands compute on CPU while perceptrons sit on CUDA), so the forward hit a
cross-device `addmm`. All 3 SCM crashes were this one bug at different forward sites:
- `regression`, `ttfs_60_offload` → `ComputeOpMapper(name='classifier')` (torch↔sim gate)
- `ttfs_q_30` → `PerceptronMapper` (per-neuron capture gate)
(`mat1 on cuda:2 / cpu` either direction — model-internal device split.)

**Fix.** Added `_unify_model_device(model)` (prefers a CUDA device if the model holds any
CUDA param) and routed the 3 duplicated `next(model.parameters()).device` blocks through
it — the gate now places the WHOLE model on one device before forwarding. Empirically:
`model.to(device)` reaches all 65 params/buffers with 0 stray tensors.

**Verification.**
- Unit: `tests/.../test_nf_scm_parity_gate.py::TestDeviceConsistency` (incl. a CUDA-gated
  cross-device repro that crashes on a plain forward, passes through the gate). 25/25 pass.
- End-to-end: resumed Soft Core Mapping on the crashed `ttfs_60_offload` cache →
  `NF↔SCM per-neuron parity: 0.0000%`, `torch↔deployed-sim parity: 1.0000 / 256`, exit 0
  (ran on cuda:1; previously crashed on cuda:2).

## D3b — SANA-FE crash → CLASSIFIED (root-cause pending)

**Not a hang — SIGFPE (exit 136).** Resuming the SANA-FE step on the LIF run reproduces a
C++ floating-point exception inside `sanafe.SpikingChip.sim()` ("dumped core"), on a
single sample (`sanafe_sample_count=1`). It kills the Python process directly → the run
status was never updated, which is why the 6 runs showed a stale `running`.

**The C++ binary is fine; the trigger is our generated arch/net.** `test_sanafe_runner.py`
+ `test_sanafe_simulation_step.py` (26 tests that drive the real `chip.sim()`) all pass.
The FPE is specific to `sanafe_arch_preset="loihi"` + the real mmixcore mapping (120 cores,
coalescing + neuron-splitting). Likely a degenerate value (empty neuron group / zero core
dimension) the C++ divides by. No gdb on this box → next step is an arch/net diff between
the passing test fixture and the failing run, then a fix-on-our-side guard (plugins-not-
patches rule for SANA-FE).

## D2 — Performance (quantified, fix pending)

Per-step walls on the 9 runs: TTFS Cycle FT 23–24 min, LIF Adaptation 13–18 min,
Weight-Quant up to 20 min, SCM up to 20 min; full runs 50–67 min. The 5-min/step budget
(AC5) is blown 3–5×. The runs used the full controller, NOT the ~60 s LIF fast-fold path —
i.e. the fast path was not selected by these configs.

## D5 — Improvements were OFF in all 9 runs (re-measurement required)

`ttfs_staircase_ste=false`, `ttfs_gain_correction=false`, scale-aware/theta-cotrain off in
every config → the bad accuracy is the OLD baseline. No accuracy conclusion is valid until
a re-run with the improvements ON (and on the fast path).
