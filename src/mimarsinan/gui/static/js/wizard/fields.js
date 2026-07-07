/* Generic schema-driven widgets: one renderer per FieldType, zero per-key forms. */

import { clearKey, effectiveValue, getKey, isExplicit, setKey, state } from './state.js';

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

function helpText(ks) {
  const parts = [ks.doc];
  if (ks.effect) parts.push('Effect: ' + ks.effect);
  if (ks.unit) parts.push('Unit: ' + ks.unit);
  parts.push('Consumed by: ' + ks.owner);
  return parts.join('\n');
}

function revertButton(ks, rerender) {
  const btn = el('button', 'field-revert', '↺');
  btn.type = 'button';
  btn.title = 'default' in ks
    ? `Revert to default (${JSON.stringify(ks.default)})`
    : 'Remove explicit value';
  btn.addEventListener('click', () => {
    clearKey(ks.key);
    rerender();
    notifyChange(ks.key);
  });
  return btn;
}

function fieldShell(ks) {
  const field = el('div', 'field');
  field.dataset.key = ks.key;
  const label = el('label', 'field-label');
  label.append(ks.label);
  if (ks.unit) label.append(el('span', 'field-unit', ` (${ks.unit})`));
  const help = el('span', 'field-help', '?');
  help.title = helpText(ks);
  label.append(help);
  const marker = el('span', 'field-explicit-slot');
  label.append(marker);
  field.append(label);
  return { field, marker };
}

function markExplicit(ks, marker, rerender) {
  marker.replaceChildren();
  if (isExplicit(ks.key)) marker.append(revertButton(ks, rerender));
}

function numberInput(ks, marker, rerender) {
  const input = el('input');
  input.type = 'number';
  if (ks.type === 'float') input.step = 'any';
  if (ks.bounds) {
    if (ks.bounds[0] !== null) input.min = String(ks.bounds[0]);
    if (ks.bounds[1] !== null) input.max = String(ks.bounds[1]);
  }
  const value = effectiveValue(ks.key);
  input.value = value === undefined || value === null ? '' : String(value);
  if ('default' in ks && ks.default !== null) input.placeholder = String(ks.default);
  input.addEventListener('change', () => {
    const raw = input.value.trim();
    if (raw === '') {
      clearKey(ks.key);
    } else {
      const parsed = ks.type === 'int' ? parseInt(raw, 10) : parseFloat(raw);
      if (Number.isNaN(parsed)) return;
      setKey(ks.key, parsed);
    }
    markExplicit(ks, marker, rerender);
    notifyChange(ks.key);
  });
  return input;
}

function boolToggle(ks, marker, rerender) {
  const row = el('div', 'toggle-row');
  const label = el('span', 'toggle-label', effectiveValue(ks.key) ? 'on' : 'off');
  row.append(label, el('div', 'toggle-switch'));
  const sync = () => {
    const on = !!effectiveValue(ks.key);
    row.classList.toggle('on', on);
    label.textContent = on ? 'on' : 'off';
  };
  sync();
  row.addEventListener('click', () => {
    setKey(ks.key, !effectiveValue(ks.key));
    sync();
    markExplicit(ks, marker, rerender);
    notifyChange(ks.key);
  });
  return row;
}

function enumSelect(ks, marker, rerender) {
  const select = el('select');
  const hasDefault = 'default' in ks;
  if (!hasDefault && !isExplicit(ks.key)) {
    select.append(new Option('— unset —', '__unset__'));
  }
  for (const option of ks.options || []) {
    const label = hasDefault && option === ks.default ? `${option} (default)` : option;
    select.append(new Option(label, option));
  }
  const value = effectiveValue(ks.key);
  select.value = value === undefined || value === null ? '__unset__' : String(value);
  select.addEventListener('change', () => {
    if (select.value === '__unset__') clearKey(ks.key);
    else setKey(ks.key, select.value);
    markExplicit(ks, marker, rerender);
    notifyChange(ks.key);
  });
  return select;
}

function textInput(ks, marker, rerender, parse, format) {
  const input = el('input');
  input.type = 'text';
  const value = getKey(ks.key);
  input.value = value === undefined || value === null ? '' : format(value);
  if ('default' in ks && ks.default !== null && ks.default !== undefined) {
    input.placeholder = format(ks.default);
  }
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
  const value = effectiveValue(ks.key);
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
  field.append(control);
  markExplicit(ks, marker, rerender);
  return field;
}
