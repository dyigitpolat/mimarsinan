# Integration plan — clean STE fast training path (the deployable win)

**Status:** the hedged staircase-backward STE is PROVEN on the toy (lossless deep
cascade, d=9 S=32 → 0.966 in 112 s) and WIRED + unit-tested into
`ttfs_cycle_adaptation_tuner` (`ttfs_staircase_ste`, `_StaircaseSteKDLoss`; 7 new +
93 suite tests green). But the MNIST validation caps at **0.8274** because the STE
currently rides the **rate-search controller**, not the clean training loop the toy used.

## Why it caps (diagnosed from the MNIST trajectory)

The STE rode the genuine **annealed ramp** → controller: 8 rate cycles ×~9 s, surrogate-
alpha anneal, per-cycle commit/recovery. post_acc climbed 0.60→0.82 then plateaued. The
gradient is SOUND (it learns), but the training DRIVER is wrong vs the toy recipe:

| | toy recipe (0.966) | real controller (0.827) |
|---|---|---|
| optimizer | Adam, **split LRs** (w 2e-3, θ 5e-2) | one LR from LR-find (**1.1e-2** — too high for weights) |
| schedule | cosine over fixed N steps | rate-ramp 0→1, per-cycle commit |
| depth | **progressive shallow→deep unfreeze** | none (all params from step 0) |
| loss | STE + KD + grad_clip | STE + KD (+ controller rollback) |

## The fix (well-defined)

Give the STE its OWN clean fast training path (the LIF fast-fold pattern), NOT the
rate-search controller:

1. **New opt-in** `ttfs_staircase_ste_fast` (requires `ttfs_staircase_ste`): route the
   STE through a fixed-step loop instead of the SmoothAdaptation controller. Reuse the
   shared fast-ladder machinery (`_setup_fast_ladder` / `_ensure_fast_optimizer` /
   `_fast_rate_attempt`) but with `_fast_loss` = the STE loss, OR a dedicated
   `_run`-override loop (simplest): one optimizer, fixed `ttfs_ste_steps` (~800–1500),
   warmup+cosine LR.
2. **Split LR param groups**: weights at `lr` (~2e-3), per-channel θ (from
   `ttfs_theta_cotrain`) at a higher `theta_lr` (~5e-2). Build the optimizer with two
   groups (mirror `recipe_combo`/`warm_theta_kd`). The single high LR-find LR is a prime
   suspect for the 0.82 cap.
3. **Progressive shallow→deep unfreeze**: port `recipe_combo._weight_params_through` —
   freeze deeper perceptron weights initially, unfreeze toward the output over the budget,
   rebuilding the optimizer at each step change. θ stays trainable throughout.
4. **Keep**: `_StaircaseSteKDLoss` (mix=0.5), `ttfs_theta_cotrain`, grad_clip (already in
   the tuning_recipe), KD vs the frozen teacher. Finalize deploys the pure genuine cascade
   (the STE forward IS genuine, so cliff ≈ 0).

## Validate

MNIST mmixcore, `ttfs_staircase_ste(_fast)` + `ttfs_theta_cotrain`, sanafe off:
- S=16 and S=32. Target: deployed ≥ 0.97 (above the S=16 baseline 0.958, toward ANN
  0.9812), in < 2 min of TTFS-step wall-time. Confirm the monotonic-S payoff on hardware.

## Tests-first

- The fast STE path runs end-to-end on the tiny supermodel and returns a float.
- It beats the controller path on a small deep proxy (reaches > the controller's plateau).
- Split-LR optimizer has the θ group at `theta_lr`, weight group at `lr`.
- Progressive-depth: early steps have deeper weights frozen; final step has all trainable.
- Flag-off byte-identical.

Files: `src/mimarsinan/tuning/tuners/ttfs_cycle_adaptation_tuner.py` (the loss exists;
add the fast-loop + split-LR + progressive-depth), `config_schema/defaults.py`
(`ttfs_staircase_ste_fast`, `ttfs_ste_steps`, `ttfs_ste_theta_lr`),
`tests/unit/tuning/test_staircase_ste.py` (extend).
