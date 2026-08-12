# encoding_placement — what the coverage ledger claimed, and what was actually executed (2026-08-13)

`encoding_layer_placement` was a silent no-op on the only `native`-category
vehicle (`simple_mlp`): the builder baked `is_encoding_layer = True`
unconditionally and the only site that applied the CONFIGURED placement,
`convert_torch_model`, is reachable only from `TorchMappingStep`, which
`applies_to` `model_category == "torch"`. Fixed on branch
`fix-encoding-placement-noop`.

This note records what that means for the coverage the program CLAIMS, because
the answer is not "some rows were mislabelled" — it is subtler and worth
keeping.

## The direct question: was any tier cell tagged `offload` silently running `subsume`?

**No.** Audited every `templates/**/*.json` (54 rows pin
`encoding_layer_placement`):

| placement | vehicles that pin it | category |
|---|---|---|
| `offload` (9 rows) | `lenet5`, `mlp_mixer`, `mlp_mixer_core`, `stream_cnn`, `torch_vit` | all **torch** |
| `subsume` (45 rows) | incl. all 6 `simple_mlp` rows (t0_05, t0_10, t0_15, t0_25, t0_45, t0_60) | mixed |

`simple_mlp` is the ONLY `native` builder in the registry (every other
registered builder is `torch`). Every `simple_mlp` row pins `subsume`, and the
no-op's effect under `subsume` is nil — the builder's baked marking and the
configured marking coincide. So no row's deployed mapping differed from its
label, and no published tier number is retro-actively wrong.

## The real coverage hole: the `offload` arm was never EXECUTABLE on the native vehicle

`encoding_placement` is a `SCREENED_COLLAPSED` hypervolume axis
(`chip_simulation/hypervolume_axes.py`), representative `subsume`, collapsed on
a *fidelity* screen — "offload ≡ subsume to ~1e-6 under signed
integrate-and-fire". Fidelity equivalence is a claim about the deployed VALUES.
It is not a claim that the `offload` arm runs, and the ledger's own justification
already says so for cost/utilization ("FIDELITY-ONLY ... not collapsed for
cost/utilization").

What the collapse quietly also bought was that nothing ever ran `offload` on the
one vehicle where the knob was broken. The screen was performed on torch
vehicles, where the knob worked; the native vehicle only ever ran the
representative. So a whole config value was unreachable on a whole model
category, for as long as the axis stayed collapsed — and the collapse is exactly
what made that unobservable.

Measured, on `simple_mlp` (784→256→64) at the moment of the fix:

* configured `offload`, deployed mapping: the SUBSUMED one, static on-chip
  parameter gate **15.23%** (`host=201478` — the encoder perceptron itself);
* configured `offload`, deployed mapping after the fix: **100.00%** on chip.

A 6.6x error in the reported on-chip fraction, on a config the tier suite could
author but never did.

## Two more instances of the same defect class, found while fixing it

1. **The joint search's layout hook never carried the placement.** Both halves —
   `build_model` for a native builder and `convert_torch_model` for a torch one —
   resolved the default, so every candidate was scored on the subsumed mapping
   regardless of the run's configuration. Only `t0_60` searches, and it pins
   `subsume`, so again no tier row scored the wrong chip; but any `offload`
   search cell would have. The layout difference is large: on `lenet5`, 198
   softcores under `subsume` vs **982** under `offload`.
2. **The wizard's layout preview applied the placement while the pipeline did
   not**, so the GUI's predicted mapping and the run's actual mapping disagreed
   for native models.

Both are closed on the same branch, the second by making `build_model` the one
native flow-birth site.

## Why the value-domain family looked affected and is not

`encoding_layer_placement` is registry `domain="event"`: a `core_semantics="mvm"`
document cannot author the key at all, and the tier manifest already tags the MVM
rows `encoding_placement: "none"`. The placement stamp introduced with the fix
initially left value-domain flows unstamped, which the guard could not
distinguish from "never resolved" — that refused t0_41/43/44 and t1_09/11/12
until it was given its own explicit value (`PLACEMENT_NOT_APPLICABLE`). No MVM
row's mapping was ever wrong; the stamp was.

## The lesson for the ledger

A `SCREENED_COLLAPSED` axis stops being executed, so it also stops being a
smoke test for its own plumbing. The screen justifies collapsing the axis in the
RESULT space; it does not justify never running the other arm. Two cheap
counter-measures now exist in the code rather than in this note:

* `tests/unit/models/test_encoding_placement_every_builder.py` — for EVERY
  registered builder, two independent builds under `subsume` vs `offload` must
  differ in the marked-encoder set (or the model must have no encoding layer at
  all). The spec table must cover the registry exactly, so a new builder cannot
  be added without answering the question.
* `tests/unit/search/test_joint_layout_honors_encoding_placement.py` — the
  search's own flow-birth site, pinned on candidates whose layout genuinely
  differs between placements.

Whether the tier suite should also carry one native `offload` row is a coverage
decision for the ledger owner, not something to fix quietly here: adding it
changes what the tier claims.
