/* Generic schema-driven widgets: one renderer per FieldType, zero per-key forms.
   Widget quality rules (all generic, from the schema record alone):
   - numerics with finite bounds render as slider + numeric combos;
   - enums with ≤ 5 options render as segmented buttons, larger as dropdowns;
   - empty semantics live INSIDE the control: a faded BLUE placeholder states
     the wizard default (baseline first), a faded GREEN "derived: <value>"
     states the derivation-owned concrete value; under-field text is reserved
     for docs/units. */

import { keySchema } from './schema.js';
import {
  clearKey, differsFromDefault, getKey, isExplicit,
  resolvedValue, setKey, state, wizardDefault,
} from './state.js';

/* Structured widgets (cores, recipes, model_config, ...) register here;
   everything else renders by type. */
const customRenderers = new Map();

export function registerCustomRenderer(key, renderFn) {
  customRenderers.set(key, renderFn);
}

export function notifyChange(key) {
  document.dispatchEvent(new CustomEvent('wizard:change', { detail: { key } }));
}

export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function boundsText(ks) {
  if (!ks.bounds) return '';
  const [lo, hi] = ks.bounds;
  if (lo !== null && hi !== null) return `Range: ${lo} – ${hi}`;
  if (lo !== null) return `Min: ${lo}`;
  if (hi !== null) return `Max: ${hi}`;
  return '';
}

function helpText(ks) {
  const parts = [ks.doc];
  if (ks.effect) parts.push('Effect: ' + ks.effect);
  if (ks.unit) parts.push('Unit: ' + ks.unit);
  const bounds = boundsText(ks);
  if (bounds) parts.push(bounds);
  parts.push('Empty → ' + emptyMeansText(ks));
  parts.push('Consumed by: ' + ks.owner);
  return parts.filter(Boolean).join('\n');
}

/** What an empty value means for this key — always answerable. */
export function emptyMeansText(ks) {
  const base = wizardDefault(ks);
  const derived = resolvedValue(ks.key);
  if (derived !== undefined && !base.has) {
    return `derived: ${formatValue(derived)}`
      + (ks.empty_means ? ` (${ks.empty_means})` : '');
  }
  if (ks.empty_means) return ks.empty_means;
  if (base.has && base.value !== null && base.value !== undefined) {
    return `${base.fromBaseline ? 'baseline' : 'default'} (${formatValue(base.value)})`;
  }
  return `unset — ${ks.owner} decides at resolve`;
}

/** Faded in-field statement of the empty behavior: the wizard default where
    one exists (blue), else the CONCRETE derivation-owned value (green,
    "derived: <value>"), else the registry's empty_means compressed to fit
    (the full sentence stays in the field's tooltip). */
export function placeholderText(ks) {
  const base = wizardDefault(ks);
  if (base.has && base.value !== null && base.value !== undefined) {
    return formatValue(base.value);
  }
  const derived = resolvedValue(ks.key);
  if (derived !== undefined) return 'derived: ' + formatValue(derived);
  const meaning = ks.empty_means || `${ks.owner} decides at resolve`;
  return meaning.length > 36 ? meaning.slice(0, 35).trimEnd() + '…' : meaning;
}

/** Whether this key's placeholder is derivation-owned (green, not blue). */
export function placeholderIsDerived(ks) {
  const base = wizardDefault(ks);
  return !(base.has && base.value !== null && base.value !== undefined)
    && resolvedValue(ks.key) !== undefined;
}

/** Refresh every rendered placeholder from the latest resolve payload —
    called after each resolve round-trip so derived placeholders show the
    CURRENT concrete value without a disruptive full re-render. */
export function refreshPlaceholders() {
  document.querySelectorAll('[data-placeholder-key]').forEach((input) => {
    const ks = keySchema(input.dataset.placeholderKey);
    if (!ks) return;
    input.placeholder = placeholderText(ks);
    input.classList.toggle('placeholder-derived', placeholderIsDerived(ks));
    attachPlaceholderReveal(input, ks);
  });
}

/** The value a control DISPLAYS when the draft is silent: the wizard default
    (baseline first) — controls with no text surface render pre-filled. */
function displayValue(ks) {
  const explicit = getKey(ks.key);
  if (explicit !== undefined) return explicit;
  const base = wizardDefault(ks);
  return base.has ? base.value : undefined;
}

function formatValue(value) {
  if (Array.isArray(value)) return value.join(', ');
  if (typeof value === 'object' && value !== null) return JSON.stringify(value);
  return String(value);
}

function revertButton(ks, rerender) {
  const btn = el('button', 'field-revert', '↺');
  btn.type = 'button';
  const base = wizardDefault(ks);
  btn.title = base.has
    ? `Revert to ${base.fromBaseline ? 'baseline' : 'default'} (${JSON.stringify(base.value)})`
    : 'Remove explicit value';
  btn.addEventListener('click', () => {
    revertToWizardDefault(ks.key);
    rerender();
    notifyChange(ks.key);
  });
  return btn;
}

/** Revert semantics: a baseline-pinned key is RESTORED to the baseline value
    (the document stays runnable-equal to the starter); anything else clears
    so the schema default / derivation applies. */
export function revertToWizardDefault(key) {
  const ks = keySchema(key);
  const base = wizardDefault(ks);
  if (base.fromBaseline) setKey(key, JSON.parse(JSON.stringify(base.value)));
  else clearKey(key);
}

function fieldShell(ks) {
  const field = el('div', 'field' + (ks.important ? ' field-important' : ''));
  field.dataset.key = ks.key;
  const label = el('label', 'field-label');
  label.append(el('span', 'field-label-text', ks.label));
  if (ks.unit) label.append(el('span', 'field-unit', ks.unit));
  const marker = el('span', 'field-explicit-slot');
  label.append(marker);
  field.append(label);
  return { field, marker };
}

export function fieldDoc(ks) {
  const doc = el('div', 'field-doc', ks.effect || ks.doc);
  doc.title = helpText(ks);
  return doc;
}

/** A truncated faded placeholder reveals its FULL text on hover (through the
    immediate app tooltip); short placeholders need no tooltip. */
function attachPlaceholderReveal(input, ks) {
  const full = emptyMeansText(ks);
  if (input.placeholder && (input.placeholder.endsWith('…')
      || input.placeholder.length > 36)) {
    input.title = 'Empty → ' + full;
  }
}

/** The revert affordance marks TRUE user deltas: an explicit value equal to
    the wizard default (baseline first) is not an edit (defect 7). */
function markExplicit(ks, marker, rerender) {
  marker.replaceChildren();
  if (differsFromDefault(ks.key)) marker.append(revertButton(ks, rerender));
}

/* ── Numeric widgets ───────────────────────────────────────────────────── */

function parseNumeric(ks, raw) {
  const parsed = ks.type === 'int' ? parseInt(raw, 10) : parseFloat(raw);
  return Number.isNaN(parsed) ? undefined : parsed;
}

function commitNumeric(ks, raw, marker, rerender) {
  if (raw === '') {
    clearKey(ks.key);
  } else {
    const parsed = parseNumeric(ks, raw);
    if (parsed === undefined) return;
    setKey(ks.key, parsed);
  }
  markExplicit(ks, marker, rerender);
  notifyChange(ks.key);
}

function numberBox(ks) {
  const input = el('input');
  input.type = 'number';
  if (ks.type === 'float') input.step = 'any';
  if (ks.bounds) {
    if (ks.bounds[0] !== null) input.min = String(ks.bounds[0]);
    if (ks.bounds[1] !== null) input.max = String(ks.bounds[1]);
  }
  const value = displayValue(ks);
  input.value = value === undefined || value === null ? '' : String(value);
  input.dataset.placeholderKey = ks.key;
  input.placeholder = placeholderText(ks);
  input.classList.toggle('placeholder-derived', placeholderIsDerived(ks));
  attachPlaceholderReveal(input, ks);
  return input;
}

/** Bounded numerics get the knob feel: slider + numeric, two-way synced.
    Tiny-magnitude defaults on a wide range (lr-like log knobs) stay numeric —
    a linear slider cannot express them. */
function sliderEligible(ks) {
  if (!ks.bounds || ks.bounds[0] === null || ks.bounds[1] === null) return false;
  const span = ks.bounds[1] - ks.bounds[0];
  if (span <= 0) return false;
  if (ks.type === 'int') return span <= 64;
  const hasDefault = 'default' in ks && ks.default !== null;
  return !hasDefault || Math.abs(ks.default - ks.bounds[0]) >= span / 50;
}

function sliderCombo(ks, marker, rerender) {
  const wrap = el('div', 'slider-combo');
  const slider = el('input', 'slider');
  slider.type = 'range';
  const [lo, hi] = ks.bounds;
  slider.min = String(lo);
  slider.max = String(hi);
  slider.step = ks.type === 'int' ? '1' : String((hi - lo) / 100);
  const box = numberBox(ks);
  box.classList.add('slider-combo-box');

  const current = displayValue(ks);
  slider.value = current === undefined || current === null ? String(lo) : String(current);
  wrap.classList.toggle('unset', current === undefined || current === null);

  slider.addEventListener('input', () => {
    box.value = slider.value;
    wrap.classList.remove('unset');
  });
  slider.addEventListener('change', () => {
    commitNumeric(ks, slider.value, marker, rerender);
  });
  box.addEventListener('change', () => {
    const raw = box.value.trim();
    if (raw !== '') {
      const parsed = parseNumeric(ks, raw);
      if (parsed !== undefined) slider.value = String(parsed);
    }
    wrap.classList.toggle('unset', raw === '');
    commitNumeric(ks, raw, marker, rerender);
  });
  wrap.append(slider, box);
  return wrap;
}

export function numberInput(ks, marker, rerender) {
  if (sliderEligible(ks)) return sliderCombo(ks, marker, rerender);
  const input = numberBox(ks);
  input.addEventListener('change', () => {
    commitNumeric(ks, input.value.trim(), marker, rerender);
  });
  return input;
}

/* ── Bool toggle ───────────────────────────────────────────────────────── */

function boolToggle(ks, marker, rerender) {
  const row = el('div', 'toggle-row');
  const label = el('span', 'toggle-label', displayValue(ks) ? 'on' : 'off');
  row.append(label, el('div', 'toggle-switch'));
  const sync = () => {
    const on = !!displayValue(ks);
    row.classList.toggle('on', on);
    label.textContent = on ? 'on' : 'off';
  };
  sync();
  row.addEventListener('click', () => {
    setKey(ks.key, !displayValue(ks));
    sync();
    markExplicit(ks, marker, rerender);
    notifyChange(ks.key);
  });
  return row;
}

/* ── Enum widgets: segmented for ≤ 5 options, dropdown beyond ──────────── */

const SEGMENTED_MAX_OPTIONS = 5;

function segmentedEnum(ks, marker, rerender) {
  const row = el('div', 'seg-control');
  const value = displayValue(ks);
  const base = wizardDefault(ks);
  for (const option of ks.options || []) {
    const btn = el('button', 'seg-btn', option);
    btn.type = 'button';
    if (base.has && option === base.value) {
      btn.title = `${option} (${base.fromBaseline ? 'baseline' : 'default'})`;
      btn.classList.add('is-default');
    }
    btn.classList.toggle('active', String(value) === option);
    btn.addEventListener('click', () => {
      setKey(ks.key, option);
      row.querySelectorAll('.seg-btn').forEach((b) => {
        b.classList.toggle('active', b === btn);
      });
      markExplicit(ks, marker, rerender);
      notifyChange(ks.key);
    });
    row.append(btn);
  }
  return row;
}

function enumSelect(ks, marker, rerender) {
  if ((ks.options || []).length <= SEGMENTED_MAX_OPTIONS) {
    return segmentedEnum(ks, marker, rerender);
  }
  const select = el('select');
  const base = wizardDefault(ks);
  if (!base.has && !isExplicit(ks.key)) {
    select.append(new Option('— unset —', '__unset__'));
  }
  for (const option of ks.options || []) {
    const label = base.has && option === base.value
      ? `${option} (${base.fromBaseline ? 'baseline' : 'default'})` : option;
    select.append(new Option(label, option));
  }
  const value = displayValue(ks);
  select.value = value === undefined || value === null ? '__unset__' : String(value);
  select.addEventListener('change', () => {
    if (select.value === '__unset__') clearKey(ks.key);
    else setKey(ks.key, select.value);
    markExplicit(ks, marker, rerender);
    notifyChange(ks.key);
  });
  return select;
}

/* ── Text-ish widgets ──────────────────────────────────────────────────── */

function textInput(ks, marker, rerender, parse, format) {
  const input = el('input');
  input.type = 'text';
  const value = getKey(ks.key);
  input.value = value === undefined || value === null ? '' : format(value);
  /* Type-faithful default text: what the placeholder shows must parse back. */
  const base = wizardDefault(ks);
  input.placeholder = base.has && base.value !== null && base.value !== undefined
    ? format(base.value) : placeholderText(ks);
  input.dataset.placeholderKey = ks.key;
  input.classList.toggle('placeholder-derived', placeholderIsDerived(ks));
  attachPlaceholderReveal(input, ks);
  input.addEventListener('change', () => {
    const raw = input.value.trim();
    if (raw === '') {
      clearKey(ks.key);
    } else {
      const parsed = parse(raw);
      if (parsed === undefined) {
        input.classList.add('field-invalid');
        return;
      }
      input.classList.remove('field-invalid');
      setKey(ks.key, parsed);
    }
    markExplicit(ks, marker, rerender);
    notifyChange(ks.key);
  });
  return input;
}

function dynamicOptionSelect(ks, marker, rerender) {
  const select = el('select');
  const options = state.dynamicOptions[ks.key] || [];
  for (const option of options) select.append(new Option(option.label, option.id));
  const value = displayValue(ks);
  if (value !== undefined && value !== null) select.value = String(value);
  select.addEventListener('change', () => {
    setKey(ks.key, select.value);
    markExplicit(ks, marker, rerender);
    notifyChange(ks.key);
  });
  return select;
}

const _PARSE = {
  str: (raw) => raw,
  path: (raw) => raw,
  int_list: (raw) => {
    const items = raw.split(',').map((s) => parseInt(s.trim(), 10));
    return items.some(Number.isNaN) ? undefined : items;
  },
  json: (raw) => {
    try { return JSON.parse(raw); } catch (_) { return undefined; }
  },
};

const _FORMAT = {
  str: String,
  path: String,
  int_list: (v) => (Array.isArray(v) ? v.join(', ') : String(v)),
  json: (v) => JSON.stringify(v),
};

/** Render one key's field row (or its registered structured widget). */
export function renderField(ks) {
  const custom = customRenderers.get(ks.key);
  if (custom) return custom(ks);

  const { field, marker } = fieldShell(ks);
  const rerender = () => {
    const fresh = renderField(ks);
    field.replaceWith(fresh);
  };
  let control;
  if (state.dynamicOptions[ks.key]) control = dynamicOptionSelect(ks, marker, rerender);
  else if (ks.type === 'int' || ks.type === 'float') control = numberInput(ks, marker, rerender);
  else if (ks.type === 'bool') control = boolToggle(ks, marker, rerender);
  else if (ks.type === 'enum') control = enumSelect(ks, marker, rerender);
  else if (ks.type in _PARSE) {
    control = textInput(ks, marker, rerender, _PARSE[ks.type], _FORMAT[ks.type]);
  } else {
    control = el('div', 'note', `unsupported field type: ${ks.type}`);
  }
  field.append(control, fieldDoc(ks));
  markExplicit(ks, marker, rerender);
  return field;
}

/** Ownership cell: renders where a hand field would be while another
    concern (registry provided_by) produces the key's value — the card
    states WHO owns the value instead of silently dropping the concern. */
export function renderOwnershipCell(ks, provider, reason) {
  const { field } = fieldShell(ks);
  field.classList.add('field-owned');
  const chip = el('div', 'ownership-chip');
  if (provider && provider.accent) {
    chip.style.setProperty('--owner-accent', provider.accent);
  }
  chip.append(el('span', 'ownership-chip-icon', '⌖'));
  chip.append(el('span', 'ownership-chip-text',
    `${provider ? provider.title : ks.provided_by} owns this`));
  chip.title = [reason, ks.doc].filter(Boolean).join('\n\n');
  field.append(chip, fieldDoc(ks));
  return field;
}
