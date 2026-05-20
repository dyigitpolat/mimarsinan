# Firing modes — cross-backend contract

| `firing_mode` | Membrane reset on fire | nevresim policy | Training (`LIFActivation`) | SANA-FE `reset_mode` |
|---------------|------------------------|-----------------|------------------------------|----------------------|
| `Default` | Subtractive (`v -= θ`) | `DefaultFirePolicy` | `v_reset=None` | `soft` |
| `Novena` | Zero-reset (`v = 0`) | `NovenaFirePolicy` | `v_reset=0.0` | `hard` |
| `TTFS` | Analytical TTFS path | N/A (use `spiking_mode`) | N/A | N/A |

Parse deployment config once via `chip_simulation.firing_strategy.FiringStrategyFactory.from_config`.
Call `strategy.require_backend("lava")` (etc.) before parity steps when enabling Novena.

LIF-only: `spiking_mode` in `{lif}` → `firing_mode` ∈ `{Default, Novena}`.
TTFS: `spiking_mode` in `{ttfs, ttfs_quantized}` → `firing_mode` must be `TTFS`.
