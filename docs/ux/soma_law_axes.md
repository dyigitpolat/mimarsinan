# The soma-law axes in the wizard — authoring spec (ODIN P1)

The written UX contract for the two new deployment axes and their two platform
widths. Everything below renders from the config-key registry: there is no
hand-written widget, no second table, and no JS copy of any rule.

## What the four keys are

| Key | Section | Card | Values (default first) | Category |
|---|---|---|---|---|
| `firing_granularity` | deployment_parameters | Core semantics (`spiking`) | `per_cycle` \| `per_event` | ADVANCED |
| `membrane_arithmetic` | deployment_parameters | Core semantics (`spiking`) | `unbounded` \| `saturating_unsigned` | ADVANCED |
| `membrane_bits` | platform_constraints | Hardware | int 0–64, default `0` | ADVANCED |
| `weight_sign_granularity` | platform_constraints | Hardware | `per_synapse` \| `per_axon` | ADVANCED |

`firing_granularity` answers WHEN the threshold is evaluated inside one cycle:
`per_cycle` is the law every deployment runs today (one compare on the reduced
contribution); `per_event` checks after every arriving event occurrence in
canonical order, so a neuron may emit 0, 1, or more spikes in a cycle.

`membrane_arithmetic` answers what the membrane accumulator IS: today's
unbounded signed accumulator, or a register that clamps to
`[0, 2**membrane_bits - 1]` on every update.

## What the author sees

1. **Both axes render as ordinary enum fields with a green derived value.**
   Empty means derived, and the field shows the concrete value the run would
   use — `per_cycle`, and `unbounded` while `membrane_bits` is `0`.
2. **`firing_granularity` LOCKS to `per_cycle` unless the resolved point is
   (lif, streamed).** The legal set is a singleton everywhere else — under the
   synchronized variant, under every TTFS variant, and in a value-domain
   document — so the wizard renders it read-only exactly as it already does for
   `firing_mode` under TTFS. A windowed hop collapses counts at every boundary
   and would destroy the multiplicity the per-event law produces, so the point
   is refused at authoring time rather than deployed and mis-reported.
3. **`membrane_arithmetic` is BITS-DRIVEN, exactly like weight quantization.**
   Declaring a platform `membrane_bits` width IS declaring a saturating
   register; the derived value follows the width with no second declaration.
   An explicit `unbounded` against a declared width, or an explicit
   `saturating_unsigned` with no width, is a keyed error on
   `membrane_arithmetic` with one-click remedies (clear the arithmetic, or drop
   the width) — the same shape the `weight_bits` contract already uses.
4. **A declared `per_event` point carries two more requirements**, each a keyed
   error naming its own key: the platform must deliver a `param_encoded` bias
   (declare `has_bias: false` on the core grid), and `lif_membrane_init` must
   sit in `[0, 1)`.
5. **Both deployment axes are event-domain**, so the whole group leaves the
   wizard when `core_semantics` is `mvm`; a draft that already declared them
   keeps them as dormant keys and gets them back on switch-back.
6. **Nothing executes the new point yet.** Every backend refuses a declared
   `per_event` or saturating law BY NAME, and the derived simulator enables all
   resolve OFF under it — an explicit enable is a keyed capability error. The
   wizard must therefore never present the point as deployable: it is
   declarable so the axis can be authored, gated so it cannot silently run a
   different physics.

## What must NOT appear

- No hardcoded value list anywhere in the wizard or the frontend. The
  `firing_modes_by_spiking` literal in `gui/wizard/schema.py` — the last
  duplicate of `legal_firing_modes` — is gone; the dict is now the legality
  SSOT's own answer, pinned by
  `tests/unit/gui/test_wizard_soma_surface.py`.
- No "advanced/experimental" badge invented per key: the ADVANCED category
  already places both axes behind the existing drawer.
- No absolute claim about the target in the field text. The keys are generic
  vocabulary; a specific chip is a set of declared values, never a named mode.
