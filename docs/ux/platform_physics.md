# Platform Physics panel — UX spec (pixel review)

Scope: the **Co-Design** tab's hardware column gains the target's PHYSICS — the
per-unit area/energy/timing constants a vendor declares, which the cost model
multiplies by the quantities a run seals. Without them a run reports no absolute
Area/Energy/Latency/Throughput at all; with them those become optimizable axes.

The panel exists to answer three questions **before launch**, not after:

1. Which target am I pricing against?
2. What did that target actually declare, and on what evidence?
3. Which objectives can this declaration back — and if one cannot, what is missing?

## Conventions the panel inherits

- The wizard renders from the config-key registry, so the two keys
  (`platform_physics_profile`, `platform_physics_overrides`) are ordinary
  `platform_constraints` keys in the `hardware` group. They render in the Co-Design
  tab's right column with no section plumbing.
- Absence is meaningful everywhere in this program: an undeclared constant is ABSENT,
  never a default. The panel must never show a number the profile did not declare.
- Evidence travels with every value (`published` / `datasheet` / `derived` /
  `estimated`). A number without its evidence is not a number this program shows.

## State 1 — no profile selected (the default)

1. **Profile selector** — a labelled select whose options come from the live
   registry (`/api/physics_profiles`), plus an explicit empty option reading
   **"None — no absolute cost"**. Empty is selected.
2. **A single explanatory line**, not an empty panel: *"No target physics declared:
   area, energy and latency objectives are unavailable."*
3. **No configurator rows.** There is nothing to configure.
4. The objective picker (`arch_search`) shows the vendor-priced chips **greyed and
   unselectable**, each with the reason on hover — never hidden. A user must be able
   to see that Area is a thing this tool can optimize, and why it is off.

## State 2 — a profile is selected

1. **Profile header** — display name, technology node, supply voltage, and the
   measurement kind (`silicon` / `simulation` / `projection` / `mixed`) as a badge.
   A `simulation` or `projection` badge is visually distinct from `silicon`: a
   comparison resting on projected numbers must say so at a glance.
2. **Completeness readout** — one line per absolute objective:
   - available: the axis name with a check;
   - unavailable: the axis name greyed, followed by the FIRST missing constant, e.g.
     `energy_per_inference_mj — needs e_synaptic_event_total`.
   This is driven by the same availability predicates the objectives registry uses,
   so the readout and the launch behaviour cannot disagree.
3. **Constant rows, grouped** by the vocabulary's own groups, in vocabulary order
   (`array`, `periphery`, `neuron`, `interconnect`, `programming`, `sync`, `global`,
   `host`, `aggregate`). Each group is a collapsible sub-section with its name and
   the count of constants the profile declares in it.
4. **Each declared row** shows, left to right:
   - the constant key (monospace) and its label;
   - the value in the profile's own declared unit — the band as `low – high` when it
     is not a point value, otherwise the single number;
   - an **evidence badge**: `published` and `datasheet` in a confident colour,
     `derived` neutral, `estimated` in a warning colour;
   - the citation (or derivation, or note) on hover as a `title` tooltip;
   - an **override input**, empty by default, placeholder showing the profile value.
5. **Undeclared constants are listed too**, in their group, greyed, with an em dash
   for the value and the text `not declared`. This is the point: the operator must
   see what the target has *not* said, because that is what disables objectives.
6. The objective picker now offers the vendor-priced chips normally.

## State 3 — an override is entered

1. The row is **visibly marked as deviating** (accent left border + an `overridden`
   badge), so a departure from the vendor's declaration is never mistaken for it.
2. Clearing the input removes the override entirely (delete-on-empty), and the row
   returns to its profile appearance — no empty-string entry is left behind.
3. Overrides on an **undeclared** constant are allowed and are how an operator
   completes a partial profile; the row stops being greyed.
4. The completeness readout updates live: overriding the missing constant flips its
   objective from unavailable to available in the same render.

## What this panel deliberately does NOT do

- It does not edit the shipped profile. A profile is data on disk with citations;
  the panel produces `platform_physics_overrides`, which the run records separately
  so a report can always state what deviated from the vendor's declaration.
- It does not invent units. The override is entered in the constant's own declared
  unit and is stored that way.
- It shows no absolute cost estimate. That is a run's output, not a form's.

## Acceptance (pixel review)

- [x] State 1: selector reads "None", the explanatory line is visible, no rows,
      priced objective chips greyed with a reason.
- [x] State 2 (`truenorth`): header shows 28 nm / 0.775 V / `silicon`; the
      completeness readout lists all four axes with their verdicts; groups render in
      vocabulary order; declared rows show value+unit+evidence badge; undeclared
      rows are greyed with `not declared`.
- [x] State 3: an override marks its row, and clearing it restores the row.
- [x] Zero console errors in every state.

## Review record (2026-08-13)

Captured by `scripts/wizard_screenshots_c5_physics.py` into
`generated/_wizard_review/c5_physics/`, reviewed against the states above. Five
defects were visible only in pixels, and all five were fixed before sign-off:

1. **An empty override read as a set one.** The profile's value is the input's
   placeholder, but it rendered at full contrast, so a user could not tell an
   untouched field from an edited one. Placeholders are now dimmed well below an
   entered number, and an entered override renders in the accent colour.
2. **The override column truncated its content** — `columns` rendered as `columr`,
   `10.254` and `93600` were clipped. Widened; an undeclared row's placeholder now
   reads `declare` rather than repeating the unit the value column already shows.
3. **The vendor-priced chips showed raw keys** (`chip_area_mm2`) beside human ones
   ("Total Parameters") — the catalog leaking its schema into the UI. Human labels
   added, and a test now refuses any registered axis whose label is its own key.
4. **Selecting a profile did not refresh the objective picker.** The panel
   re-rendered itself but not the group hosts, so the chips stayed greyed after the
   physics that unblocks them had been declared. Panel edits now re-render
   everything, because declaring a profile changes what the picker may offer.
5. **The completeness readout named raw keys** while the picker named labels,
   making the user match them up by hand. Unified on the picker's own label.
