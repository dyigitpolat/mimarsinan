/* Generic schema-driven widgets: one renderer per FieldType, zero per-key forms.
   Widget quality rules (all generic, from the schema record alone):
   - numerics with finite bounds render as slider + numeric combos;
   - enums with ≤ 5 options render as segmented buttons, larger as dropdowns;
   - empty semantics live INSIDE the control. A key the registry says is
     DERIVATION-OWNED (it names a `provenance` source) shows, in faded GREEN,
     the CONCRETE prospective value the SSOT deriver produces for the current
     config state (served per resolve as `derived_values`), or a muted "—" when
     the derivation is blocked. Everything else shows its plain concrete
     DEFAULT in faded BLUE. Prose about WHERE a value comes from is never the
     field text — the provenance badge + tooltip carry that.
   - THE LEGAL-VALUE-SET LAW: the resolve payload's `legal_values[key]` drives
     rendering by its SIZE alone. |legal| == 1 → the field is LOCKED (read-only,
     rendered as the derived value); |legal| > 1 → the widget offers ONLY those
     options. No per-mode special case exists here.
   - ONE DERIVED VISUAL LANGUAGE. `isDerivedNow(ks)` — the registry names an
     SSOT source AND the draft has not taken ownership — is the SINGLE predicate
     every widget hangs off. It stamps `.field-derived` on the field root, and
     the stylesheet turns that into green for EVERY widget type (in-field value,
     toggle, slider thumb + numeric box, segmented ghost, locked content,
     structured editor). A user-OWNED value reads the normal active theme
     everywhere. No widget opts out and no per-key CSS exists.
   - PROVENANCE OCCUPIES NO LAYOUT. The SSOT source is tooltip text (helpText),
     never a chip: chrome above every field was noise. */

import { keySchema } from './schema.js';
import {
  clearKey, differsFromDefault, getKey, isExplicit, legalValues, lockedValue,
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
  if (ks.provenance) parts.push('SSOT source: ' + ks.provenance);
  parts.push('Consumed by: ' + ks.owner);
  return parts.filter(Boolean).join('\n');
}

/** Whether the registry says an SSOT source produces this key's value while the
    draft is silent. `isDerivedNow` adds "and the draft still is silent". */

/** What an empty value means for this key — always answerable (tooltip text,
    the ONE place the registry's empty_means prose is allowed to surface). */
export function emptyMeansText(ks) {
  if (isDerivationOwned(ks)) {
    const derived = resolvedValue(ks.key);
    const value = derived === undefined ? 'blocked by an active error' : formatValue(derived);
    return `derived: ${value}` + (ks.empty_means ? ` — ${ks.empty_means}` : '');
  }
  if (ks.empty_means) return ks.empty_means;
  const base = wizardDefault(ks);
  if (base.has && base.value !== null && base.value !== undefined) {
    return `${base.fromBaseline ? 'baseline' : 'default'} (${formatValue(base.value)})`;
  }
  return `unset — ${ks.owner} decides at resolve`;
}

/** A key whose value, while the draft is silent, is produced by a named SSOT:
    the registry says so with `provenance`. Its in-field text is that VALUE. */
export function isDerivationOwned(ks) {
  return !!ks.provenance;
}

/** The faded in-field text. GREEN = the concrete value the SSOT deriver
    produces right now ("—" while a contract error blocks the derivation: never
    a stale or hypothetical value). BLUE = a plain concrete default. Prose only
    survives for a key that is neither (no default, no source). */
export function placeholderText(ks) {
  if (isDerivationOwned(ks)) {
    const derived = resolvedValue(ks.key);
    return derived === undefined ? '—' : formatValue(derived);
  }
  const base = wizardDefault(ks);
  if (base.has && base.value !== null && base.value !== undefined) {
    return formatValue(base.value);
  }
  const meaning = ks.empty_means || `${ks.owner} decides at resolve`;
  return meaning.length > 36 ? meaning.slice(0, 35).trimEnd() + '…' : meaning;
}

/** THE predicate: the derivation currently owns this key's value — the registry
    names an SSOT source and the draft has not declared one. Every widget type
    reads its green treatment from this, and only from this; an explicit
    declaration takes ownership back and the field returns to the normal theme. */
export function isDerivedNow(ks) {
  return isDerivationOwned(ks) && !isExplicit(ks.key);
}

/** Stamp the one derived-state class on a field root (works for the generic
    widgets and for any registered structured editor alike). */
function markDerived(node, ks) {
  if (node && node.classList) node.classList.toggle('field-derived', isDerivedNow(ks));
}

/** Refresh every rendered placeholder from the latest resolve payload —
    called after each resolve round-trip so derived placeholders show the
    CURRENT concrete value without a disruptive full re-render. */
export function refreshPlaceholders() {
  document.querySelectorAll('[data-placeholder-key]').forEach((input) => {
    const ks = keySchema(input.dataset.placeholderKey);
    if (!ks) return;
    input.placeholder = placeholderText(ks);
    attachPlaceholderReveal(input, ks);
  });
  /* helpText carries the doc, the bounds, what empty resolves to and the SSOT
     source. It quotes the derived value, so it goes stale exactly like a
     placeholder does — refresh it (on the field root AND its doc line) on the
     same round-trip, and re-stamp the one derived-state class. */
  document.querySelectorAll('.field[data-key]').forEach((field) => {
    const ks = keySchema(field.dataset.key);
    if (ks) { markDerived(field, ks); setTip(field, helpText(ks)); }
  });
  document.querySelectorAll('[data-doc-key]').forEach((doc) => {
    const ks = keySchema(doc.dataset.docKey);
    if (ks) setTip(doc, helpText(ks));
  });
  document.querySelectorAll('[data-derived-enum-key]').forEach((row) => {
    const ks = keySchema(row.dataset.derivedEnumKey);
    if (ks) syncDerivedSegment(row, ks);
  });
  /* Toggles and slider thumbs carry no placeholder — they render the derived
     value itself, so a resolve that moves it must move them. */
  document.querySelectorAll('[data-prefill-key]').forEach((node) => {
    const ks = keySchema(node.dataset.prefillKey);
    if (!ks) return;
    if (node.classList.contains('toggle-row')) syncToggle(node, ks);
    else syncSlider(node, ks);
  });
}

/** The value a TEXT control displays when the draft is silent: the wizard
    default (baseline first). A derivation-owned key stays empty here — its
    concrete RESOLVED value renders as the green placeholder that vanishes on
    typing. Prefilling the schema default would be a lie whenever the deriver
    overrides it (kd_ce_alpha's default is 0.3; the lif recipe folds 0.5). */
function displayValue(ks) {
  const explicit = getKey(ks.key);
  if (explicit !== undefined) return explicit;
  if (isDerivationOwned(ks)) return undefined;
  const base = wizardDefault(ks);
  return base.has ? base.value : undefined;
}

/** The value a control with NO TEXT SURFACE shows (toggles, slider thumbs):
    they have no empty state, so a derivation-owned key renders PRE-FILLED with
    the concrete value the deriver produces — an `off` toggle on a key the run
    resolves to `true` would be a lie. */
function prefillValue(ks) {
  const value = displayValue(ks);
  if (value !== undefined && value !== null) return value;
  return isDerivedNow(ks) ? resolvedValue(ks.key) : undefined;
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
  /* The SSOT source lives in the field's IMMEDIATE tooltip beside the doc,
     effect, bounds and empty-semantics text — provenance occupies no layout. */
  field.title = helpText(ks);
  markDerived(field, ks);
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
  doc.dataset.docKey = ks.key;
  doc.title = helpText(ks);
  return doc;
}

/** Every hover surface of a field reveals the one help text: doc, effect,
    bounds, what empty resolves to, the SSOT source, the consumer. It quotes the
    derived value, so it is re-read on every resolve — a stale value in a tooltip
    is the same lie as a stale one in the field. */
function attachPlaceholderReveal(input, ks) {
  setTip(input, helpText(ks));
}

/** Write a tooltip through whichever attribute tooltip.js has claimed. */
function setTip(node, text) {
  if (node.dataset.tip !== undefined) node.dataset.tip = text;
  else node.title = text;
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

  wrap.dataset.prefillKey = ks.key;
  syncSlider(wrap, ks);

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
    commitNumeric(ks, raw, marker, rerender);
    syncSlider(wrap, ks);
  });
  wrap.append(slider, box);
  return wrap;
}

/** The thumb states the value the run will use: explicit, default, or the
    derivation's concrete value. Only a key with none of those is unset; the
    green treatment comes from the field root's one derived-state class. */
function syncSlider(wrap, ks) {
  const slider = wrap.querySelector('.slider');
  if (!slider) return;
  const current = prefillValue(ks);
  const unset = current === undefined || current === null;
  slider.value = unset ? slider.min : String(current);
  wrap.classList.toggle('unset', unset);
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
  const label = el('span', 'toggle-label', '');
  row.append(label, el('div', 'toggle-switch'));
  row.dataset.prefillKey = ks.key;
  syncToggle(row, ks);
  row.addEventListener('click', () => {
    setKey(ks.key, !prefillValue(ks));
    syncToggle(row, ks);
    markExplicit(ks, marker, rerender);
    notifyChange(ks.key);
  });
  return row;
}

function syncToggle(row, ks) {
  const value = prefillValue(ks);
  const blocked = isDerivedNow(ks) && value === undefined;
  row.classList.toggle('on', !!value);
  const label = row.querySelector('.toggle-label');
  /* An active error blocks the derivation: a toggle cannot honestly read
     "off" for a key whose resolved state is unknown. */
  if (label) label.textContent = blocked ? '—' : (value ? 'on' : 'off');
  setTip(row, helpText(ks));
}

/* ── Enum widgets: segmented for ≤ 5 options, dropdown beyond ──────────── */

const SEGMENTED_MAX_OPTIONS = 5;

/** The options a widget may offer: the legal set where the derivation exposes
    one (illegal options never render), else every declared option. Registry
    order is preserved — the legal set is a subset, not a reordering. */
function offeredOptions(ks) {
  const legal = legalValues(ks.key);
  const options = ks.options || [];
  return legal ? options.filter((option) => legal.includes(option)) : options;
}

function segmentedEnum(ks, marker, rerender) {
  const row = el('div', 'seg-control');
  const value = displayValue(ks);
  const base = wizardDefault(ks);
  /* The segment the derivation currently resolves to renders as a GREEN ghost —
     the user sees the derived value, and an explicit click takes ownership. */
  row.dataset.derivedEnumKey = ks.key;
  for (const option of offeredOptions(ks)) {
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
  syncDerivedSegment(row, ks);
  return row;
}

/** The segment the derivation CURRENTLY resolves to renders as a green ghost —
    the user sees the derived value, and an explicit click takes ownership.
    Re-synced on every resolve round-trip, so a mode switch moves the ghost.
    Generic: any derivation-owned enum, with or without a schema default. */
function syncDerivedSegment(row, ks) {
  const derived = isDerivedNow(ks) ? resolvedValue(ks.key) : undefined;
  row.querySelectorAll('.seg-btn').forEach((btn) => {
    const ghost = derived !== undefined && String(derived) === btn.textContent;
    btn.classList.toggle('is-derived', ghost);
    if (ghost) btn.title = `${btn.textContent} (derived — ${ks.provenance})`;
  });
}

function enumSelect(ks, marker, rerender) {
  const options = offeredOptions(ks);
  if (options.length <= SEGMENTED_MAX_OPTIONS) {
    return segmentedEnum(ks, marker, rerender);
  }
  const select = el('select');
  const base = wizardDefault(ks);
  if (!base.has && !isExplicit(ks.key)) {
    select.append(new Option('— unset —', '__unset__'));
  }
  for (const option of options) {
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

/** |legal| == 1: the field is LOCKED. It renders read-only as the value the
    derivation resolves to — its CONTENT, not a placeholder — with the reason
    the current config admits nothing else. Editing it could only produce an
    error, so no control is offered. The document may still carry the value
    explicitly (it round-trips); the lock is a UI truth, not an emission rule. */
function renderLockedField(ks, value) {
  const { field, marker } = fieldShell(ks);
  field.classList.add('field-locked');
  /* A locked value the DOCUMENT pins is user-owned (normal theme + revert);
     otherwise the derivation owns it and it reads green, like every other
     derivation-owned widget. */
  const cell = el('div', 'locked-value');
  cell.append(el('span', 'locked-value-icon', '⛭'));
  cell.append(el('span', 'locked-value-text', formatValue(value)));
  cell.title = `Locked: the current config admits only ${formatValue(value)}.\n`
    + helpText(ks);
  field.append(cell, fieldDoc(ks));
  markExplicit(ks, marker, () => {});
  return field;
}

/** Render one key's field row (or its registered structured widget). */
export function renderField(ks) {
  const custom = customRenderers.get(ks.key);
  if (custom) {
    /* A structured editor builds its own shell; the ONE derived-state class is
       still stamped on it, so it reads the same green language as every other
       widget without knowing anything about provenance. */
    const node = custom(ks);
    markDerived(node, ks);
    return node;
  }

  const locked = lockedValue(ks.key);
  if (locked !== undefined) return renderLockedField(ks, locked);

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
