"""Compare TTFS quantized activation vs training staircase activation.

Produces plots showing both activation functions and the best-matching shift.
"""

import numpy as np
import matplotlib.pyplot as plt


def ttfs_quantized(V, S, threshold=1.0):
    """TTFS quantized activation: hybrid_core_flow.py lines 376-383."""
    safe_thresh = max(threshold, 1e-12)
    k_fire_raw = np.ceil(S * (1.0 - V / safe_thresh))
    fires = k_fire_raw < S
    k_fire = np.clip(k_fire_raw, 0, S - 1)
    return np.where(fires, (S - k_fire) / S, 0.0)


def training_staircase(V, act_scale, tq, shift):
    """Training-time activation: relu -> clamp -> StaircaseFunction.

    In raw space: StaircaseFunction(clamp(relu(x + shift), 0, act_scale), tq/act_scale)
    Here V is in effective domain: V = x / act_scale.
    """
    x_raw = V * act_scale
    after_relu = np.maximum(x_raw + shift, 0.0)
    after_clamp = np.clip(after_relu, 0.0, act_scale)
    Tq = tq / act_scale
    quantized = np.floor(after_clamp * Tq) / Tq
    return quantized / act_scale  # back to effective domain [0, 1]


S = 64
tq = 64
act_scale = 1.0  # effective domain, normalised
threshold = 1.0

V = np.linspace(-0.1, 1.3, 10000)

# --- TTFS quantized activation ---
y_ttfs = ttfs_quantized(V, S, threshold)

# --- Training staircase with different shifts ---
default_shift = act_scale * 0.5 / tq  # 0.5/64 = 0.0078125
shifts_to_test = np.linspace(-0.5 * default_shift, 2.5 * default_shift, 200)
# also test shift = 0 explicitly
shifts_to_test = np.sort(np.append(shifts_to_test, [0.0, default_shift]))

avg_diffs = []
for s in shifts_to_test:
    y_stair = training_staircase(V, act_scale, tq, s)
    avg_diffs.append(np.mean(np.abs(y_ttfs - y_stair)))

avg_diffs = np.array(avg_diffs)
best_idx = np.argmin(avg_diffs)
best_shift = shifts_to_test[best_idx]

print(f"S = {S}, tq = {tq}, act_scale = {act_scale}, threshold = {threshold}")
print(f"Default shift = {default_shift:.6f}")
print(f"Best shift    = {best_shift:.6f}  (avg |diff| = {avg_diffs[best_idx]:.6f})")
print(f"Zero shift    avg |diff| = {avg_diffs[np.argmin(np.abs(shifts_to_test))]:.6f}")
print(f"Default shift avg |diff| = {avg_diffs[np.argmin(np.abs(shifts_to_test - default_shift))]:.6f}")

# ─── Plots ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Plot 1: TTFS quantized vs training staircase (shift=0) ---
ax = axes[0, 0]
y_stair_0 = training_staircase(V, act_scale, tq, 0.0)
ax.plot(V, y_ttfs, label="TTFS quantized", linewidth=1.5)
ax.plot(V, y_stair_0, label="Staircase (shift=0)", linewidth=1.5, linestyle="--")
ax.set_title("TTFS quantized vs Staircase (shift = 0)")
ax.set_xlabel("V (effective domain)")
ax.set_ylabel("Output")
ax.legend()
ax.set_xlim(-0.05, 1.15)
ax.grid(True, alpha=0.3)

# --- Plot 2: TTFS quantized vs training staircase (default shift) ---
ax = axes[0, 1]
y_stair_def = training_staircase(V, act_scale, tq, default_shift)
ax.plot(V, y_ttfs, label="TTFS quantized", linewidth=1.5)
ax.plot(V, y_stair_def, label=f"Staircase (shift={default_shift:.4f})", linewidth=1.5, linestyle="--")
ax.set_title(f"TTFS quantized vs Staircase (default shift = {default_shift:.4f})")
ax.set_xlabel("V (effective domain)")
ax.set_ylabel("Output")
ax.legend()
ax.set_xlim(-0.05, 1.15)
ax.grid(True, alpha=0.3)

# --- Plot 3: TTFS quantized vs training staircase (best shift) ---
ax = axes[1, 0]
y_stair_best = training_staircase(V, act_scale, tq, best_shift)
ax.plot(V, y_ttfs, label="TTFS quantized", linewidth=1.5)
ax.plot(V, y_stair_best, label=f"Staircase (best shift={best_shift:.6f})", linewidth=1.5, linestyle="--")
ax.set_title(f"TTFS quantized vs Staircase (best shift = {best_shift:.6f})")
ax.set_xlabel("V (effective domain)")
ax.set_ylabel("Output")
ax.legend()
ax.set_xlim(-0.05, 1.15)
ax.grid(True, alpha=0.3)

# --- Plot 4: Avg |diff| vs shift ---
ax = axes[1, 1]
ax.plot(shifts_to_test, avg_diffs, linewidth=1.5)
ax.axvline(0.0, color="gray", linestyle=":", label="shift = 0")
ax.axvline(default_shift, color="red", linestyle=":", label=f"default shift = {default_shift:.4f}")
ax.axvline(best_shift, color="green", linestyle=":", label=f"best shift = {best_shift:.6f}")
ax.set_title("Avg |diff| vs shift")
ax.set_xlabel("Shift value")
ax.set_ylabel("Mean absolute difference")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("scripts/ttfs_staircase_comparison.png", dpi=150)
print("\nSaved: scripts/ttfs_staircase_comparison.png")

# ─── Zoomed overlay near a step transition ─────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Zoom around V = 0.5 (step 32/64)
ax = axes2[0]
mask = (V > 0.45) & (V < 0.55)
ax.plot(V[mask], y_ttfs[mask], label="TTFS quantized", linewidth=2)
ax.plot(V[mask], y_stair_0[mask], label="Staircase (shift=0)", linewidth=2, linestyle="--")
ax.plot(V[mask], y_stair_def[mask], label=f"Staircase (shift={default_shift:.4f})", linewidth=2, linestyle=":")
ax.plot(V[mask], y_stair_best[mask], label=f"Staircase (best={best_shift:.6f})", linewidth=2, linestyle="-.")
ax.set_title("Zoomed: around V = 0.5")
ax.set_xlabel("V")
ax.set_ylabel("Output")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Zoom around V = 0 (firing threshold)
ax = axes2[1]
mask = (V > -0.05) & (V < 0.1)
ax.plot(V[mask], y_ttfs[mask], label="TTFS quantized", linewidth=2)
ax.plot(V[mask], y_stair_0[mask], label="Staircase (shift=0)", linewidth=2, linestyle="--")
ax.plot(V[mask], y_stair_def[mask], label=f"Staircase (shift={default_shift:.4f})", linewidth=2, linestyle=":")
ax.plot(V[mask], y_stair_best[mask], label=f"Staircase (best={best_shift:.6f})", linewidth=2, linestyle="-.")
ax.set_title("Zoomed: near firing threshold (V ~ 0)")
ax.set_xlabel("V")
ax.set_ylabel("Output")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("scripts/ttfs_staircase_zoom.png", dpi=150)
print("Saved: scripts/ttfs_staircase_zoom.png")
plt.show()
