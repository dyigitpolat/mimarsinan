/* The dedicated Pretrained-weights panel (Training & Tuning section).

   A single "Pretrained" switch beside the panel; the body expands ONLY when the
   switch is on and reveals EVERY registered/derived fact of the chosen weight
   set plus the fine-tune knobs promoted to primary. The switch is DISABLED with
   its reason when the builder registers no applicable set (|legal(preload)|
   admits only off), and LOCKED ON when a source is declared — the legal-value-
   set law, reused. Registered as the custom renderer for `preload_weights`, so
   the framework treats the whole regime as one field; the fine-tune knobs and
   the weight-set selector co-locate here instead of the generic training grid.

   Every fact the builder derives reads the GREEN derived language; a value the
   user owns (the switch they turned on, the set they picked) reads the owned
   theme — the same contract as every other widget. */

import { keySchema } from './schema.js';
import {
  clearKey, getKey, isExplicit, legalValues, setKey, state,
} from './state.js';
import { el, notifyChange, renderField } from './fields.js';

/** The always-served panel block (like the vehicles block): switch state, the
    reason, the registered records, the selection. `{}` before the first
    resolve so the panel still renders its switch. */
function block() {
  return (state.resolve && state.resolve.pretrained) || {};
}

function preloadOn() {
  const legal = legalValues('preload_weights');
  const value = getKey('preload_weights');
  /* |legal|==[true] locks the regime on (an explicit source declares it); the
     draft value wins otherwise; default off. */
  if (legal && legal.length === 1) return !!legal[0];
  return value === undefined ? false : !!value;
}

function switchEnabled() {
  const legal = legalValues('preload_weights');
  /* Disabled exactly when the only legal value is `false` — the builder
     registers no applicable weight set. Undefined legal set (builder not
     consulted) leaves the switch usable. */
  return !(legal && legal.length === 1 && legal[0] === false);
}

function locked() {
  const legal = legalValues('preload_weights');
  return legal && legal.length === 1;
}

/* ── The switch row ────────────────────────────────────────────────────── */

function switchRow() {
  const pretrained = block();
  const on = preloadOn();
  const enabled = switchEnabled();
  const derivedLock = locked();

  const row = el('div', 'pretrained-switch-row');
  const head = el('div', 'pretrained-switch-head');
  head.append(el('span', 'pretrained-switch-title', 'Pretrained weights'));
  const sub = derivedLock
    ? (on ? 'source declared — fine-tune from pretrained' : 'from-scratch')
    : (on ? 'fine-tune from pretrained' : 'train from scratch');
  head.append(el('span', 'pretrained-switch-sub', sub));
  row.append(head);

  const toggle = el('div', 'toggle-row' + (on ? ' on' : '') + (enabled ? '' : ' disabled'));
  toggle.append(el('span', 'toggle-label', on ? 'on' : 'off'), el('div', 'toggle-switch'));
  /* A derivation-forced state (locked) reads green; a user choice reads cyan. */
  toggle.classList.toggle('is-derived', !!derivedLock);
  if (enabled && !derivedLock) {
    toggle.addEventListener('click', () => {
      const next = !preloadOn();
      if (next) setKey('preload_weights', true);
      else { clearKey('preload_weights'); clearKey('pretrained_weight_set'); }
      notifyChange('preload_weights');
    });
  } else {
    toggle.title = pretrained.reason
      || (derivedLock ? 'Locked by the current configuration' : '');
  }
  row.append(toggle);

  if (!enabled && pretrained.reason) {
    row.append(el('div', 'pretrained-reason', '⚠ ' + pretrained.reason));
  }
  return row;
}

/* ── The chosen set's revealed facts ───────────────────────────────────── */

const FACTS = [
  ['task', 'Task'],
  ['dataset', 'Dataset'],
  ['input_shape', 'Input geometry'],
  ['num_classes', 'Classes'],
  ['source', 'Source'],
  ['expected_accuracy', 'Expected top-1'],
  ['num_parameters', 'Parameters'],
  ['license', 'License'],
];

function formatFact(key, value) {
  if (value === null || value === undefined) return '—';
  if (key === 'input_shape' && Array.isArray(value)) return value.join(' × ');
  if (key === 'expected_accuracy') return (value * 100).toFixed(2) + '%';
  if (key === 'num_parameters') return Number(value).toLocaleString();
  if (Array.isArray(value)) return value.join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function selectedRecord() {
  const pretrained = block();
  const sets = pretrained.sets || [];
  const chosen = getKey('pretrained_weight_set') || pretrained.selected;
  return sets.find((record) => record.id === chosen) || sets[0] || null;
}

function factsGrid(record) {
  const grid = el('div', 'pretrained-facts');
  for (const [key, label] of FACTS) {
    if (record[key] === undefined || record[key] === null) continue;
    const cell = el('div', 'pretrained-fact');
    cell.append(el('span', 'pretrained-fact-label', label));
    cell.append(el('span', 'pretrained-fact-value', formatFact(key, record[key])));
    grid.append(cell);
  }
  if (record.preprocessing && typeof record.preprocessing === 'object') {
    const cell = el('div', 'pretrained-fact span-2');
    cell.append(el('span', 'pretrained-fact-label', 'Preprocessing'));
    cell.append(el('span', 'pretrained-fact-value',
      Object.entries(record.preprocessing)
        .map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join('/') : v}`).join(' · ')));
    grid.append(cell);
  }
  return grid;
}

/* ── The weight-set selector (the legal-value-set law) ─────────────────── */

function selector() {
  const legal = legalValues('pretrained_weight_set') || [];
  const pretrained = block();
  const byId = Object.fromEntries((pretrained.sets || []).map((r) => [r.id, r]));
  const wrap = el('div', 'pretrained-selector');
  wrap.append(el('span', 'pretrained-selector-label', 'Weight set'));

  const owned = isExplicit('pretrained_weight_set');
  if (legal.length <= 1) {
    /* |legal| == 1 → LOCKED to the derived value (green content + lock glyph). */
    const only = legal[0];
    const cell = el('div', 'locked-value');
    cell.append(el('span', 'locked-value-icon', '⛭'));
    cell.append(el('span', 'locked-value-text',
      byId[only] ? byId[only].label : (only || '—')));
    wrap.classList.add('field-derived', 'field-locked');
    wrap.append(cell);
    return wrap;
  }

  const seg = el('div', 'seg-control');
  const selected = getKey('pretrained_weight_set') || pretrained.selected;
  for (const id of legal) {
    const btn = el('button', 'seg-btn', byId[id] ? byId[id].label : id);
    btn.type = 'button';
    btn.classList.toggle('active', id === selected && owned);
    btn.classList.toggle('is-derived', id === pretrained.selected && !owned);
    btn.title = byId[id]
      ? `${byId[id].dataset} · ${(byId[id].expected_accuracy * 100 || 0).toFixed(1)}% top-1`
      : id;
    btn.addEventListener('click', () => {
      setKey('pretrained_weight_set', id);
      notifyChange('pretrained_weight_set');
    });
    seg.append(btn);
  }
  wrap.append(seg);
  return wrap;
}

/* ── The panel ─────────────────────────────────────────────────────────── */

function panel() {
  const field = el('div', 'field span-2 pretrained-panel');
  field.dataset.key = 'preload_weights';
  field.append(switchRow());

  if (!preloadOn()) return field;

  const body = el('div', 'pretrained-body');
  const record = selectedRecord();
  if (record) {
    const legalCount = (legalValues('pretrained_weight_set') || []).length;
    if (legalCount > 1) body.append(selector());
    body.append(factsGrid(record));
    if (legalCount <= 1) {
      /* Single set: name it (locked) beneath the facts, not as a chooser. */
      body.append(selector());
    }
  }

  const finetune = el('div', 'pretrained-finetune field-grid cols-2');
  for (const key of ['finetune_epochs', 'finetune_lr']) {
    const ks = keySchema(key);
    if (ks) finetune.append(renderField(ks));
  }
  body.append(el('div', 'pretrained-finetune-label', 'Fine-tuning'), finetune);
  field.append(body);
  return field;
}

export function installPretrainedPanel(registerCustomRenderer) {
  registerCustomRenderer('preload_weights', panel);
}
