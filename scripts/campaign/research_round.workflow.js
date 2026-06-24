export const meta = {
  name: 'research-round',
  description: 'One never-idle research round: decompose -> isolated study -> adversarial verify -> central synthesis, returning only a compact result',
  phases: [
    { title: 'Survey', detail: 'decompose into independent research items from harvest_todo/ledger' },
    { title: 'Study', detail: 'one isolated read-only analyst per item' },
    { title: 'Verify', detail: 'adversarial refutation per headline' },
    { title: 'Synthesize', detail: 'sole writer: ledger + findings + next batches' },
  ],
}

const SURVEY_SCHEMA = { type: 'object', additionalProperties: false, properties: {
  items: { type: 'array', items: { type: 'object', additionalProperties: false, properties: {
    id: { type: 'string' }, question: { type: 'string' },
    kind: { type: 'string' }, cluster: { type: 'string' },
    findings_doc: { type: 'string' },
    run_id_prefixes: { type: 'array', items: { type: 'string' } },
  }, required: ['id', 'question', 'kind', 'cluster', 'run_id_prefixes'] } },
  notes: { type: 'string' },
}, required: ['items'] }

const STUDY_SCHEMA = { type: 'object', additionalProperties: false, properties: {
  item_id: { type: 'string' }, claim: { type: 'string' },
  cells: { type: 'array', items: { type: 'object' } },
  ledger_records: { type: 'array', items: { type: 'object' } },
  verdict: { type: 'string' }, confounds: { type: 'string' },
}, required: ['item_id', 'claim', 'ledger_records'] }

const VERIFY_SCHEMA = { type: 'object', additionalProperties: false, properties: {
  item_id: { type: 'string' }, refuted: { type: 'boolean' }, reason: { type: 'string' },
}, required: ['item_id', 'refuted'] }

const SYNTH_SCHEMA = { type: 'object', additionalProperties: false, properties: {
  appended_ledger_rows: { type: 'integer' },
  docs_updated: { type: 'array', items: { type: 'string' } },
  proposed_batches: { type: 'integer' },
  headline: { type: 'string' },
  confirmed: { type: 'array', items: { type: 'string' } },
  refuted: { type: 'array', items: { type: 'string' } },
}, required: ['headline'] }

const CONV = [
  'Conventions (working dir = repo root): deployed accuracy = the bare float in',
  'generated/<id>_phased_deployment_run/__target_metric.json. ANN reference = the FIRST',
  '"Test accuracy:" line in runs/campaign/logs/<id>.log. Finalized runs live in',
  'runs/campaign/q/{done,failed}/*.json but the FILES are NOT named by id — read each',
  'JSON and match its "id" field. A run is valid only if result.returncode==0.',
].join(' ')

phase('Survey')
const survey = await agent(
  'You are the SURVEY scout for one round of the mimarsinan ANN->SNN deployment research program. '
  + 'Identify the open research ITEMS worth studying THIS round — each a self-contained question '
  + 'answerable by reading ALREADY-FINALIZED GPU runs (never launch training). Read: '
  + 'runs/campaign/harvest_todo.json (finalized runs not yet in the ledger, grouped by batch; IGNORE '
  + 'groups "untagged", anything matching ws6_seed_matrix, AND ALL deep_mlp runs — deep_mlp w64 is an INVALID '
  + 'host-majority config (<50% params on-chip; see docs/research/VALIDITY_AUDIT.md), so NEVER consolidate '
  + 'deep_mlp deployment results; only VALID vehicles count (deep_cnn, mlp_mixer/mmixcore, lenet5)), runs/campaign/ledger.jsonl '
  + '(already consolidated — do NOT duplicate), runs/campaign/backlog.json (what is running). For each clean '
  + 'group with >=6 finalized runs, emit one item {id, question, kind(arch_dataset|depth|escalation|mode|breadth), '
  + 'cluster(WS3|WS6|WS7|WS-mode), findings_doc(a docs/research/findings/*.md), run_id_prefixes(how to glob ids, '
  + 'e.g. ["dcnn_d","plan_dcnn_"])}. GOLDEN STANDARD: read docs/mimarsinan_closeout_analysis_v2.md (esp. §5 the '
  + 'publication-readiness matrix and §8 the WS0-WS7 roadmap) — the Definition-of-Done is THAT acceptance-criteria '
  + 'table populated with VERIFIED verdicts across breadth (architecture spectrum, dataset spectrum, pretrained + '
  + 'from-scratch regimes, baselines, energy/latency Pareto, automatic recipe selection). Prefer items that both '
  + 'consolidate finished runs AND move a specific closeout cell: dcnn_* (does the cascaded death-cascade appear on a '
  + 'TRAINABLE deep CNN? — §6 depth-risk), plan_controller_rescue / cp_*/ws7esc_* (does the controller auto-rescue '
  + 'the worst cascaded cells? — WS3/WS7), plan_dcnn_dataset (dataset breadth), theta_at_depth, per-mode landscape. '
  + 'Cap at 6, highest-value-toward-DoD first. ' + CONV,
  { schema: SURVEY_SCHEMA, label: 'survey', phase: 'Survey' })

const items = ((survey && survey.items) || []).slice(0, 6)
if (!items.length) return { headline: 'no clean unanalyzed science with >=6 finalized runs this round', appended_ledger_rows: 0, proposed_batches: 0 }
log(items.length + ' research items: ' + items.map(i => i.id).join(', '))

const studied = await pipeline(items,
  it => agent(
    'You are an ISOLATED READ-ONLY analyst. Study EXACTLY this item; touch nothing else, never src/ or git. '
    + 'Item: ' + JSON.stringify(it) + '. Method (deterministic, no fabrication): enumerate finalized run ids by '
    + 'reading runs/campaign/q/{done,failed}/*.json and matching the JSON "id" against run_id_prefixes. Pair '
    + 'cascaded-vs-synchronized (or conversion_policy true-vs-false) at matched (depth,dataset,seed). Compute '
    + '3-seed means, the cascaded->sync (or cp) gap in pp, and the ANN gap. FLAG confounds explicitly: ANN at/near '
    + 'chance (untrained -> NOT a firing-gain result), crashed/non-finalized runs (returncode!=0), <3 seeds, '
    + 'max_simulation_samples<=50 (read gaps, not 3rd decimals). Return a one-sentence claim, the per-cell numbers '
    + 'in cells, and ready-to-append ledger_records each matching the existing schema (cluster, kind, model, dataset, '
    + 'depth, cascaded_deployed_mean, synchronized_deployed_mean, ann_test_acc_mean, cascaded_to_sync_gap_pp, '
    + 'n_seeds, cascaded_run_ids, synchronized_run_ids, verdict, note). Write NOTHING — the synthesizer writes. ' + CONV,
    { schema: STUDY_SCHEMA, label: 'study:' + it.id, phase: 'Study' }),
  st => st ? agent(
    'You are an ADVERSARIAL verifier — try to REFUTE this analyst. Claim: ' + st.claim + '. Records: '
    + JSON.stringify(st.ledger_records).slice(0, 4000) + '. Re-derive 2-3 cited numbers from source and check: are '
    + 'the run_ids real and finalized (returncode 0)? Is a "gap" actually an ANN-untrained-floor confound (ANN near '
    + 'chance)? Are means over the claimed seed count? Is a "cascaded collapse" actually a crashed run mislabeled? '
    + 'Default refuted=true if you cannot independently confirm the headline. ' + CONV,
    { schema: VERIFY_SCHEMA, label: 'verify:' + st.item_id, phase: 'Verify' }).then(v => ({ st, v })) : null)

const ok = studied.filter(Boolean).filter(x => x.v && x.v.refuted === false)
const confirmed = ok.map(x => x.st)
const refuted = studied.filter(Boolean).filter(x => !x.v || x.v.refuted !== false).map(x => x.st.item_id)
log('confirmed ' + confirmed.length + ' / refuted ' + refuted.length)

phase('Synthesize')
const synth = await agent(
  'You are the CENTRAL SYNTHESIZER and the SOLE writer this round (working dir = repo root). You MUST actually '
  + 'PERSIST — describing is failure. CONFIRMED (verified, consolidate): ' + JSON.stringify(confirmed).slice(0, 12000)
  + '. REFUTED ids (do NOT consolidate, just note): ' + JSON.stringify(refuted) + '. Steps: '
  + '(1) Flatten every confirmed item ledger_records into ONE JSON array and WRITE it to /tmp/mim_round_records.json '
  + 'with the Write tool (each record must cite ALL its run_ids so the director per-run coverage drops them from '
  + 'harvest_todo). Run `wc -l runs/campaign/ledger.jsonl` (note N0), then '
  + '`source env/bin/activate && python scripts/campaign/research_loop.py ledger-append /tmp/mim_round_records.json` '
  + '(the CLI accepts a JSON-array file path), then `wc -l runs/campaign/ledger.jsonl` again (N1). VERIFY N1>N0; if '
  + 'not, fix the records and retry until the ledger actually grows. Do NOT return until it has. '
  + '(2) UPDATE each confirmed item findings_doc with a concise dated subsection (table + verdict + confounds) via '
  + 'Edit/Write; do not rewrite existing sections. NEVER write to docs/mimarsinan_closeout_analysis_v2.md — it is '
  + 'the user READ-ONLY north star; findings go under docs/research/findings/*.md, and AC-table evidence goes to '
  + 'docs/research/AC_EVIDENCE.md (create if absent). (3) PROPOSE next-round GPU work that advances the golden-standard '
  + 'AC table (you MAY READ docs/mimarsinan_closeout_analysis_v2.md §8 but never edit it): append DISABLED plan_stage batches '
  + '(enabled:false, new plan_stage numbers) to runs/campaign/backlog.json reusing existing templates with grid '
  + 'overrides; do NOT enable them, do NOT touch runner src/ or resolve git conflicts. '
  + 'Return the schema with appended_ledger_rows = the VERIFIED count increase (N1-N0).',
  { schema: SYNTH_SCHEMA, label: 'synthesize', phase: 'Synthesize' })

return synth || { headline: 'synthesis returned nothing', appended_ledger_rows: 0, proposed_batches: 0 }
