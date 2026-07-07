/* Review surfaces: derived chips, inline errors, diff-vs-defaults, unknown tray,
   pipeline step bar, and the JSON preview. All render from /api/config/resolve. */

import { keySchema } from './schema.js';
import { clearKey, state } from './state.js';
import { el, notifyChange } from './fields.js';

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

/* ── Derived chips with WHY ────────────────────────────────────────────── */

export function renderDerivedChips() {
  const host = document.getElementById('derivedChips');
  if (!host) return;
  host.replaceChildren();
  const derived = (state.resolve && state.resolve.derived) || {};
  for (const [key, info] of Object.entries(derived)) {
    const ks = keySchema(key);
    const chip = el('span', 'dep-chip triggered');
    chip.append(el('span', 'dot'));
    const label = ks ? ks.label : key;
    chip.append(`${label}: ${JSON.stringify(info.value)}`);
    chip.title = (info.why ? info.why + '\n\n' : '') + (ks ? ks.doc : '');
    host.append(chip);
  }
  if (!Object.keys(derived).length) {
    host.append(el('span', 'note', 'Derived values appear after the draft resolves.'));
  }
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
  const errors = (state.resolve && state.resolve.errors) || [];
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
  const runBtn = document.getElementById('runBtn');
  if (runBtn) runBtn.disabled = errors.length > 0;
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
  const count = el('div', 'note', `${rows.length} knob(s) differ from defaults`);
  host.append(count);
  const table = el('div', 'diff-table');
  for (const row of rows) {
    const line = el('div', 'diff-row');
    line.append(el('span', 'diff-group', row.group));
    const link = el('span', 'diff-key', row.label);
    link.title = row.key;
    line.append(link);
    line.append(el('span', 'diff-value', JSON.stringify(row.value)));
    if (row.default !== null && row.default !== undefined) {
      line.append(el('span', 'diff-default', `(default: ${JSON.stringify(row.default)})`));
    }
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

/* ── Pipeline step bar (+ edit-&-continue start-step selection) ────────── */

export function renderStepsBar() {
  const list = document.getElementById('pipelineStepsList');
  const hint = document.getElementById('pipelineStepsHint');
  if (!list) return;
  const pipeline = (state.resolve && state.resolve.pipeline) || { steps: [], semantic_groups: [] };
  const steps = pipeline.steps || [];
  if (hint) {
    hint.textContent = steps.length
      ? `${steps.length} steps` + (state.editContinueRunId ? ' — click a step to restart from it' : '')
      : '';
  }
  if (!steps.length) {
    list.innerHTML = '<span class="pipeline-steps-error-msg">No resolvable pipeline yet</span>';
    return;
  }
  const selectable = !!state.editContinueRunId;
  const completed = state.prevCompleted;
  let suggestedIdx = -1;
  if (selectable && completed) {
    suggestedIdx = steps.findIndex((s) => !completed.has(s));
  }
  const startStep = state.draft.start_step || null;

  const cols = steps.map((name, i) => {
    const group = pipeline.semantic_groups[i] || 'other';
    const isStart = selectable && startStep === name;
    let status = 'pending';
    let clickable = false;
    if (selectable && completed) {
      if (isStart) { status = 'running'; clickable = true; }
      else if (completed.has(name)) { status = 'completed'; clickable = true; }
      else if (i === suggestedIdx) { status = 'pending'; clickable = true; }
    }
    const cls = 'psb-col' + (isStart ? ' selected' : '') + (clickable ? ' psb-col--ec-selectable' : '');
    const data = clickable ? ` data-start-step="${encodeURIComponent(name)}"` : '';
    return `<div class="${cls}" data-status="${status}" data-group="${escapeHtml(group)}"${data}>` +
      '<div class="psb-bar"></div>' +
      `<span class="psb-label">${escapeHtml(name)}</span></div>`;
  });
  list.innerHTML = `<div class="psb-list psb-list--preview">${cols.join('')}</div>`;
}

export function bindStepsBar() {
  const list = document.getElementById('pipelineStepsList');
  if (!list) return;
  list.addEventListener('click', (event) => {
    const col = event.target.closest('[data-start-step]');
    if (!col) return;
    const name = decodeURIComponent(col.getAttribute('data-start-step'));
    if (state.draft.start_step === name) delete state.draft.start_step;
    else state.draft.start_step = name;
    renderStepsBar();
    notifyChange('start_step');
  });
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
  if (host) host.innerHTML = highlightJson(state.draft);
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
