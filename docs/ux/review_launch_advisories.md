# UX spec — Deployment advisories in Review & Launch + acknowledge-to-launch

Written reference for the owner pixel review (browser screenshots are judged
against THIS document). Covers the advisory surfaces the wizard renders from
`POST /api/config/resolve` `advisories` rows and the acknowledge-to-launch
gate. Code: `gui/static/wizard.html`, `gui/static/js/wizard/{advisories,
review,workbench,main,state}.js`; structural pins:
`tests/unit/gui/test_wizard_advisories.py::TestReviewLaunchLayout`; gate
logic pins (node-executed): `tests/unit/gui/test_wizard_advisory_gate.py`.

## Target layout

- **Review & Launch section** (`data-section-id="review"`): the advisory
  block `#advisoryBlock` is the section's FIRST card — full width, directly
  under the section head, above the two-column area (Derived values /
  Differs-from-defaults / Emitted config). Amber `⚠` section icon, title
  "Deployment advisories", subtitle naming the gate ("UNSUPPORTED and
  mandate-violating rows must be acknowledged before launch"). The card is
  hidden (`display:none`) while the resolve serves zero advisories.
- **Advisory cards** (`#advisoryRail`, one per resolve row, server order):
  severity tag + title head, one-paragraph detail, id + `tentative theory` +
  `lossless-mandate violation` badges, "Levers: …" line. Severity colors:
  UNSUPPORTED rose, RISK amber, INFO cyan — unchanged from the rail era.
- **Gating rows only** (`mandate_violation == true` OR severity ==
  `UNSUPPORTED`): an acknowledge row at the card foot — checkbox + label,
  amber dashed border while unacknowledged reading "Acknowledge this advisory
  to enable launch"; green solid border once ticked reading "Acknowledged —
  launch unblocked for this advisory". RISK/INFO rows without a mandate
  violation never render a checkbox.
- **Live rail**: the old "Deployment advisories" rail block is GONE. In its
  place a compact amber pill button `#advisoryCountBadge` sits directly under
  the verdict pill `#statusPill`: text `⚠ N advisories` (`⚠ 1 advisory`),
  extended with ` · M to acknowledge` (bolder, `.pending`) while gating rows
  await acknowledgment. Hidden at zero advisories. Clicking it jumps to
  Review & Launch (the advisory card is the first thing on screen).
- **Section nav**: the Review & Launch item carries an amber count badge
  (total advisories) beside the rose error badge when both apply; tooltip
  "N deployment advisor(y|ies) — review before launch". Errors stay rose;
  advisories are always amber.

## Launch gate

`#runBtn` disables when `resolve.errors.length > 0` OR any gating advisory is
unacknowledged; both are re-checked inside the click handler (errors →
first-error section, pending acks → Review & Launch). Launch-status line
precedence, top to bottom:

1. resolving — "Resolving the draft…" (dim), button disabled;
2. errors — `✖ N error(s) block(s) launch — review` (rose button-line →
   first error section);
3. pending acks — `⚠ N advisor(y|ies) need(s) acknowledgment — review`
   (amber button-line → Review & Launch), button disabled;
4. ready — `✓ N-step pipeline ready`, plus the non-blocking warn lines:
   `⚠ planned mapping does not fit …` (hwStats infeasible) and, whenever
   advisories exist (all-acknowledged gating + RISK/INFO), `⚠ N deployment
   advisor(y|ies) — see Review & Launch`. Button enabled.

## Acknowledgment semantics (client-side, per advisory id)

- State lives in `state.advisoryAcks = { gatingIds, acked }`
  (`advisories.js`, pure + node-tested). Acks are granted per advisory id.
- On every resolve round-trip the gating id SET is recomputed; if it changed
  in ANY way (id added, removed, or swapped), every stored acknowledgment
  resets — a config edit that changes the gating advisories invalidates old
  acks. Non-gating churn (a RISK/INFO row appearing or vanishing) preserves
  acks.
- Draft reset and template/run config loads always clear acks
  (`state.js resetDraft / loadDraftFromConfig`).
- Acks are never persisted: a page reload starts unacknowledged.

## The three review states

1. **No advisories** (e.g. starter baseline, plain lif): no advisory card, no
   rail badge, no nav badge; launch status `✓ N-step pipeline ready`; Launch
   enabled. Review's first visible card is the run group / Derived values.
2. **Advisories, none gating** (e.g. `spiking_family=lif,
   spiking_variant=streamed` → ADV-STREAMED-CONTRACT, INFO): advisory card
   first in Review with NO checkbox; rail badge `⚠ 1 advisory` (not
   pending-styled); nav badge `1`; launch status `✓ … ready` + `⚠ 1
   deployment advisory — see Review & Launch`; Launch enabled throughout.
3. **Gating advisories** (e.g. `spiking_family=ttfs,
   spiking_variant=cascaded` → ADV-CASC-UNSUPPORTED, UNSUPPORTED; or lif +
   `firing_mode=Novena` → ADV-NOVENA-CHARGE with `mandate_violation`):
   - *pending*: card shows the amber dashed acknowledge row; rail badge
     `⚠ 1 advisory · 1 to acknowledge` (pending style); launch status `⚠ 1
     advisory needs acknowledgment — review`; Launch DISABLED, and a click
     lands on the advisory card.
   - *acknowledged*: checkbox ticked, row turns green-solid "Acknowledged —
     launch unblocked for this advisory"; rail badge back to `⚠ 1 advisory`;
     launch status `✓ … ready` + the non-blocking advisory warn line; Launch
     ENABLED.

## Manual verification steps (DOM wiring the node tests cannot cover)

From a fresh `run.py --ui` wizard:

1. Fresh starter draft → state 1: assert no advisory card/badges, Launch
   enabled.
2. Semantics → `spiking_variant=streamed` → state 2: INFO card first in
   Review, no checkbox, badges show 1, Launch stays enabled.
3. Semantics → ttfs family, cascaded variant → state 3 pending: UNSUPPORTED
   card + checkbox, rail badge pending style, Launch disabled; click Launch →
   lands on the advisory card, nothing submits.
4. Tick the checkbox → state 3 acknowledged: Launch enables, status shows
   ready + advisory warn line.
5. Invalidation: flip variant to synchronized (gating set changes → acks
   reset), then back to cascaded → checkbox is UNTICKED again and Launch is
   disabled until re-acknowledged.
6. Rail badge click and status-line click both land on Review & Launch with
   the advisory card at the top of the viewport.
7. Introduce a validation error while acks are pending → error line + rose
   badges take precedence; fixing the error returns to the pending-ack line.
8. Reload the page mid-acknowledgment → acks are gone (never persisted).

Screenshots for the owner review should capture: states 1/2/3-pending/
3-acknowledged, the rail badge in both styles, the nav badge beside a rose
error badge, and the reset-on-edit flow (step 5).
