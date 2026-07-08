/* Live-rail + review surfaces: verdict, honest pipeline assembly, compact
   mapping, launch; derived chips with WHY, vehicle status, inline errors,
   diff-vs-defaults, unknown tray, JSON preview. All render from
   /api/config/resolve — nothing here is a static copy of the pipeline. */

import { groups, keySchema, keysGatedBy, vehicleEnableKeys } from './schema.js';
import { clearKey, setKey, state } from './state.js';
import { el, notifyChange, renderField } from './fields.js';

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function resolveErrors() {
  return (state.resolve && state.resolve.errors) || [];
}

/* ── Derived chips with WHY (read-only, never editable) ────────────────── */

function formatDerivedValue(value) {
  if (value === true) return 'on';
  if (value === false) return 'off';
  if (value === null || value === undefined) return '—';
  return String(value);
}

function derivedChip(key, info) {
  const ks = keySchema(key);
  const chip = el('span', 'derived-chip');
  chip.append(el('span', 'derived-chip-lock', '⛭'));
  chip.append(el('span', 'derived-chip-key', ks ? ks.label : key));
  chip.append(el('span', 'derived-chip-value', formatDerivedValue(info.value)));
  if (info.why) chip.append(el('span', 'derived-chip-why', info.why));
  chip.title = [
    info.why || '',
    info.derived_from && info.derived_from.length
      ? 'derived from: ' + info.derived_from.join(', ') : '',
    ks ? ks.doc : '',
  ].filter(Boolean).join('\n\n');
  return chip;
}

/** Simulator enables render in the vehicles card, core maxima inline in the
    hardware section; the review strip shows the remaining semantics. */
function chipHostKeys(hostId, derived) {
  const vehicles = new Set(vehicleKeys());
  const entries = Object.entries(derived)
    .filter(([key]) => !vehicles.has(key));
  if (hostId === 'deploymentDerived') {
    return entries.filter(([key]) => {
      const ks = keySchema(key);
      return ks && (ks.group === 'spiking' || ks.group === 'conversion' || ks.group === 'run');
    });
  }
  return entries;
}

export function renderDerivedChips() {
  const derived = (state.resolve && state.resolve.derived) || {};
  const errors = resolveErrors();
  for (const hostId of ['derivedChips', 'deploymentDerived']) {
    const host = document.getElementById(hostId);
    if (!host) continue;
    host.replaceChildren();
    if (errors.length) {
      /* Never show hypothetical derived values while the draft is invalid. */
      host.append(el('span', 'derived-blocked',
        `derivation blocked — fix ${errors.length} error${errors.length > 1 ? 's' : ''} to see derived values`));
      continue;
    }
    const entries = chipHostKeys(hostId, derived);
    for (const [key, info] of entries) host.append(derivedChip(key, info));
    if (!entries.length) {
      host.append(el('span', 'note', 'Derived values appear once the draft resolves.'));
    }
  }
  renderVehiclesStatus();
}

/* ── Simulation vehicles (recipe-defaulted toggles; unsupported = muted) ── */

function vehicleKeys() {
  return vehicleEnableKeys('deployment_target');
}

/** Supported vehicle: an honest toggle — ON is the recipe default, clicking
    stores an explicit off (a legitimate declarable override); OFF clears
    back to the recipe default. Its gated settings co-locate with the row. */
function supportedVehicleRow(key, ks, info) {
  const on = !!info.value;
  const row = el('div', 'vehicle-row' + (on ? ' on' : ' user-off'));
  const head = el('div', 'vehicle-head');
  head.append(el('span', 'vehicle-dot'));
  const name = el('span', 'vehicle-name', ks ? ks.label : key);
  name.title = ks ? ks.doc : '';
  head.append(name);
  const toggle = el('div', 'toggle-row vehicle-toggle' + (on ? ' on' : ''));
  toggle.append(el('span', 'toggle-label', on ? 'runs' : 'off'));
  toggle.append(el('div', 'toggle-switch'));
  toggle.title = on
    ? 'Switch off for this run (stored as an explicit off in the config)'
    : 'Switch back on (returns to the recipe default)';
  toggle.addEventListener('click', () => {
    if (on) setKey(key, false);
    else clearKey(key);
    notifyChange(key);
  });
  head.append(toggle);
  row.append(head);
  if (info.why) row.append(el('div', 'vehicle-why', info.why));
  if (on) {
    const gatedKeys = keysGatedBy(key);
    if (gatedKeys.length) {
      const settings = el('div', 'vehicle-settings field-grid cols-2');
      for (const gatedKey of gatedKeys) {
        const gatedSchema = keySchema(gatedKey);
        if (gatedSchema) settings.append(renderField(gatedSchema));
      }
      row.append(settings);
    }
  }
  return row;
}

/** Unsupported vehicle: one muted unavailability line, never a knob. */
function unsupportedVehicleRow(key, ks, info) {
  const row = el('div', 'vehicle-row unsupported');
  const head = el('div', 'vehicle-head');
  head.append(el('span', 'vehicle-dot'));
  const name = el('span', 'vehicle-name', ks ? ks.label : key);
  name.title = ks ? ks.doc : '';
  head.append(name);
  head.append(el('span', 'vehicle-state', 'unavailable'));
  row.append(head);
  if (info.why) row.append(el('div', 'vehicle-why', info.why));
  return row;
}

export function renderVehiclesStatus() {
  const host = document.getElementById('vehiclesStatus');
  if (!host) return;
  host.replaceChildren();
  const errors = resolveErrors();
  if (!state.resolve || errors.length) {
    host.append(el('div', 'note', errors.length
      ? 'Vehicle selection appears once the draft resolves.'
      : 'Resolving…'));
    return;
  }
  const derived = state.resolve.derived || {};
  for (const key of vehicleKeys()) {
    const ks = keySchema(key);
    const info = derived[key] || { value: null, why: null, meta: null };
    const supported = !info.meta || info.meta.supported !== false;
    host.append(supported
      ? supportedVehicleRow(key, ks, info)
      : unsupportedVehicleRow(key, ks, info));
  }
}

/* ── Launch (persistent rail: status + strong disabled semantics) ──────── */

export function renderLaunchStatus() {
  const host = document.getElementById('launchStatus');
  const runBtn = document.getElementById('runBtn');
  if (!host) return;
  host.replaceChildren();
  const errors = resolveErrors();
  if (!state.resolve) {
    host.append(el('div', 'launch-status-line pending', 'Resolving the draft…'));
    if (runBtn) runBtn.disabled = true;
    return;
  }
  const steps = (state.resolve.pipeline && state.resolve.pipeline.steps) || [];
  if (errors.length) {
    const line = el('button', 'launch-status-line error',
      `✖ ${errors.length} error${errors.length > 1 ? 's' : ''} `
      + `block${errors.length > 1 ? '' : 's'} launch — review`);
    line.type = 'button';
    line.addEventListener('click', () => {
      document.dispatchEvent(new CustomEvent('wizard:go-first-error'));
    });
    host.append(line);
  } else {
    host.append(el('div', 'launch-status-line ok',
      `✓ ${steps.length}-step pipeline ready`));
    if (state.hwStats && state.hwStats.status === 'infeasible') {
      host.append(el('div', 'launch-status-line warn',
        '⚠ planned mapping does not fit — the run will fail at Hard Core Mapping'));
    }
  }
  const name = state.draft.experiment_name;
  if (name) host.append(el('div', 'launch-status-sub', name));
  if (runBtn) runBtn.disabled = errors.length > 0;
}

/* ── Inline + global errors (with rule-prescribed one-click remedies) ──── */

/* Each rule's error text prescribes its remedies; these buttons apply them. */
const RULE_REMEDIES = {
  quantization_assembly: [
    {
      label: 'Declare vanilla (float weights)',
      apply: () => { state.draft.pipeline_mode = 'vanilla'; },
    },
    {
      label: 'Drop weight_bits',
      apply: () => { clearKey('weight_bits'); },
    },
  ],
};

function errorCard(error) {
  const note = el('div', 'field-error', error.message);
  note.dataset.ruleId = error.rule_id || '';
  const remedies = RULE_REMEDIES[error.rule_id] || [];
  if (remedies.length) {
    const row = el('div', 'error-remedies');
    for (const remedy of remedies) {
      const btn = el('button', 'btn-sm', remedy.label);
      btn.type = 'button';
      btn.addEventListener('click', () => {
        remedy.apply();
        document.dispatchEvent(new CustomEvent('wizard:rerender'));
        notifyChange(error.key || '');
      });
      row.append(btn);
    }
    note.append(row);
  }
  return note;
}

export function renderErrors() {
  document.querySelectorAll('.field-error').forEach((node) => node.remove());
  document.querySelectorAll('.field.has-error').forEach((node) => node.classList.remove('has-error'));
  const global = document.getElementById('globalErrors');
  const errors = resolveErrors();
  const unattached = [];
  for (const error of errors) {
    const host = error.key
      ? document.querySelector(`.field[data-key="${CSS.escape(error.key)}"]`)
      : null;
    if (host) {
      host.classList.add('has-error');
      host.append(errorCard(error));
    } else {
      unattached.push(error);
    }
  }
  if (global) {
    global.style.display = unattached.length ? '' : 'none';
    global.replaceChildren(...unattached.map(errorCard));
  }
}

/* ── Diff-vs-defaults (the template exposure mechanism) ────────────────── */

export function renderDiffPanel() {
  const host = document.getElementById('diffPanel');
  if (!host) return;
  const rows = ((state.resolve && state.resolve.diff_vs_defaults) || [])
    .filter((row) => row.differs);
  host.replaceChildren();
  if (!rows.length) {
    host.append(el('div', 'note', 'Nothing differs from defaults.'));
    return;
  }
  /* Compact group tags for the table column; the full title is the tooltip. */
  const groupTitles = Object.fromEntries(groups().map(
    (g) => [g.id, g.id.replace(/_/g, ' ').replace('deployment target', 'target')],
  ));
  const count = el('div', 'note', `${rows.length} knob(s) differ from defaults`);
  host.append(count);
  const table = el('div', 'diff-table');
  for (const row of rows) {
    const line = el('div', 'diff-row');
    line.append(el('span', 'diff-group', groupTitles[row.group] || row.group));
    const link = el('span', 'diff-key', row.label);
    link.title = row.key;
    line.append(link);
    const valueText = JSON.stringify(row.value);
    const value = el('span', 'diff-value', valueText);
    value.title = valueText;
    line.append(value);
    const hasDefault = row.default !== null && row.default !== undefined;
    line.append(el(
      'span', 'diff-default',
      hasDefault ? `default: ${JSON.stringify(row.default)}` : '',
    ));
    const revert = el('button', 'field-revert', '↺');
    revert.type = 'button';
    revert.title = 'Revert to default';
    revert.addEventListener('click', () => {
      clearKey(row.key);
      document.dispatchEvent(new CustomEvent('wizard:rerender'));
      notifyChange(row.key);
    });
    line.append(revert);
    table.append(line);
  }
  host.append(table);
}

/* ── Unrecognized-keys tray (schema-drift alarm; never silent) ─────────── */

export function renderUnknownTray() {
  const host = document.getElementById('unknownTray');
  if (!host) return;
  const unknown = (state.resolve && state.resolve.unknown_keys) || [];
  host.style.display = unknown.length ? '' : 'none';
  host.replaceChildren();
  if (!unknown.length) return;
  host.append(el('div', 'unknown-tray-title',
    `⚠ ${unknown.length} unrecognized key(s) — kept verbatim, check for typos or schema drift`));
  for (const path of unknown) host.append(el('div', 'unknown-tray-item', path));
}

/* ── Pipeline assembly rail (honest vertical list from the resolve) ────── */

export function renderAssemblyRail() {
  const list = document.getElementById('pipelineStepsList');
  const hint = document.getElementById('pipelineStepsHint');
  if (!list) return;
  const pipeline = (state.resolve && state.resolve.pipeline) || { steps: [], semantic_groups: [] };
  const steps = pipeline.steps || [];
  const errors = resolveErrors();
  if (hint) {
    hint.textContent = steps.length
      ? `${steps.length} steps` + (state.editContinueRunId ? ' · click to restart from' : '')
      : '';
  }
  list.replaceChildren();
  if (!steps.length) {
    list.append(el('div', 'assembly-empty', errors.length
      ? 'No resolvable pipeline — fix the flagged errors'
      : 'Resolving…'));
    return;
  }
  const selectable = !!state.editContinueRunId;
  const completed = state.prevCompleted;
  let suggestedIdx = -1;
  if (selectable && completed) {
    suggestedIdx = steps.findIndex((s) => !completed.has(s));
  }
  const startStep = state.draft.start_step || null;

  steps.forEach((name, i) => {
    const group = pipeline.semantic_groups[i] || 'other';
    const isStart = selectable && startStep === name;
    let clickable = false;
    let status = '';
    if (selectable && completed) {
      if (isStart) { status = 'start'; clickable = true; }
      else if (completed.has(name)) { status = 'completed'; clickable = true; }
      else if (i === suggestedIdx) { status = 'suggested'; clickable = true; }
    }
    const row = el(clickable ? 'button' : 'div',
      'assembly-step' + (isStart ? ' selected' : '') + (clickable ? ' selectable' : ''));
    if (clickable) row.type = 'button';
    row.dataset.group = group;
    if (status) row.dataset.status = status;
    row.append(el('span', 'assembly-dot'));
    row.append(el('span', 'assembly-name', name));
    if (isStart) row.append(el('span', 'assembly-flag', 'start'));
    else if (status === 'completed') row.append(el('span', 'assembly-flag done', '✓'));
    if (clickable) {
      row.addEventListener('click', () => {
        if (state.draft.start_step === name) delete state.draft.start_step;
        else state.draft.start_step = name;
        renderAssemblyRail();
        notifyChange('start_step');
      });
    }
    list.append(row);
  });
}

/* ── Compact mapping summary (rail mirror of the co-design panel) ──────── */

function railStat(value, label) {
  const cell = el('div', 'rail-stat');
  cell.append(el('div', 'rail-stat-value', value));
  cell.append(el('div', 'rail-stat-label', label));
  return cell;
}

export function renderRailMapping() {
  const host = document.getElementById('railMapping');
  if (!host) return;
  host.replaceChildren();
  const mapping = state.hwStats;
  if (!mapping || mapping.status === 'pending') {
    host.append(el('div', 'note', 'Awaiting a mapping plan…'));
    return;
  }
  if (mapping.status === 'search') {
    host.append(el('div', 'note', 'Search mode — hardware co-search plans the mapping.'));
    return;
  }
  if (mapping.status === 'no-metadata') {
    host.append(el('div', 'note', 'Pick a data provider to plan the mapping.'));
    return;
  }
  if (mapping.status === 'infeasible') {
    host.append(el('div', 'rail-mapping-verdict bad', '✖ does not fit the core grid'));
    return;
  }
  if (mapping.status === 'error') {
    host.append(el('div', 'rail-mapping-verdict bad', '✖ mapping check failed — see Co-Design'));
    return;
  }
  const stats = mapping.stats || {};
  const grid = el('div', 'rail-stats');
  grid.append(railStat(String(Math.round(stats.total_cores ?? 0)), 'cores'));
  const barriers = (stats.host_side_segment_count || 0)
    + (stats.schedule_sync_count || (stats.layout_preview && stats.layout_preview.schedule_sync_count) || 0);
  grid.append(railStat(String(barriers), 'barriers'));
  grid.append(railStat(`${(stats.total_wasted_axons_pct ?? 0).toFixed(0)}%`, 'axon waste'));
  grid.append(railStat(`${(stats.total_wasted_neurons_pct ?? 0).toFixed(0)}%`, 'neuron waste'));
  host.append(el('div', 'rail-mapping-verdict ok', '✓ fits the core grid'));
  host.append(grid);
}

/* ── JSON preview ──────────────────────────────────────────────────────── */

function highlightJson(obj) {
  return JSON.stringify(obj, null, 2)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/("(?:\\.|[^"\\])*")\s*:/g, (m, key) => `<span class="key">${key}</span>:`)
    .replace(/:\s*("(?:\\.|[^"\\])*")/g, (m, val) => `: <span class="str">${val}</span>`)
    .replace(/:\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)/gi, (m, val) => `: <span class="num">${val}</span>`)
    .replace(/:\s*(true|false)/g, (m, val) => `: <span class="bool">${val}</span>`)
    .replace(/:\s*(null)/g, (m, val) => `: <span class="null">${val}</span>`);
}

export function renderJsonPreview() {
  const host = document.getElementById('jsonOutput');
  if (!host) return;
  /* The server-emitted document (explicit keys only, canonical order) —
     exactly what Launch submits; the raw draft is the pre-resolve fallback. */
  const doc = (state.resolve && state.resolve.emitted) || state.draft;
  host.innerHTML = highlightJson(doc);
}

/* ── Template banner ───────────────────────────────────────────────────── */

export function renderTemplateBanner() {
  const banner = document.getElementById('templateBanner');
  if (!banner) return;
  if (!state.templateName && !state.editContinueRunId) {
    banner.style.display = 'none';
    return;
  }
  banner.style.display = '';
  banner.replaceChildren();
  const label = state.editContinueRunId
    ? `Edit & continue: ${state.editContinueRunId}`
    : `Template: ${state.templateName}`;
  banner.append(el('span', 'template-banner-label', label));
  banner.append(el('span', 'note',
    ' — every differing knob is listed under Review; unknown keys land in the tray.'));
}

export { escapeHtml };
