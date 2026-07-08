/* Structured sub-schema widgets: each renders from a SERVED schema, never a
   hardcoded field list (cores, recipes, model_config, arch_search, preprocessing). */

import { schema } from './schema.js';
import { clearKey, effectiveValue, getKey, setKey, state } from './state.js';
import { el, fieldDoc, notifyChange, numberInput, registerCustomRenderer } from './fields.js';
import { installPretrainedPanel } from './pretrained.js';

function subField(labelText, control) {
  const field = el('div', 'field');
  const label = el('label', 'field-label', labelText);
  field.append(label, control);
  return field;
}

function numeric(value, onChange, { step = 'any', min = null, placeholder = '' } = {}) {
  const input = el('input');
  input.type = 'number';
  input.step = step;
  if (min !== null) input.min = String(min);
  input.placeholder = placeholder;
  input.value = value === undefined || value === null ? '' : String(value);
  input.addEventListener('change', () => {
    const raw = input.value.trim();
    onChange(raw === '' ? undefined : parseFloat(raw));
  });
  return input;
}

/* ── cores: the core-type grid editor (aligned columns, per-row labels) ── */

const CORE_DIMS = [
  ['max_axons', 'Axons'],
  ['max_neurons', 'Neurons'],
  ['count', 'Count'],
];

function coreBiasToggle(core, onChange) {
  const effective = core.has_bias !== undefined
    ? core.has_bias
    : state.draft.platform_constraints?.has_bias !== false;
  const toggle = el('button', 'cores-bias' + (effective ? ' on' : ''), effective ? 'B' : '–');
  toggle.type = 'button';
  toggle.title = (effective ? 'Bias lane: on' : 'Bias lane: off')
    + ' — click to flip this core type\'s bias lane';
  toggle.addEventListener('click', () => onChange(!effective));
  return toggle;
}

function coresWidget(ks) {
  const field = el('div', 'field span-2');
  field.dataset.key = ks.key;
  const label = el('label', 'field-label');
  label.append(el('span', 'field-label-text', ks.label));
  field.append(label);
  const host = el('div', 'cores-editor');
  field.append(host);
  field.append(fieldDoc(ks));

  const cores = () => {
    const list = getKey('cores');
    return Array.isArray(list) ? list : [];
  };
  const write = (list) => {
    setKey('cores', list);
    render();
    notifyChange('cores');
  };
  const patch = (i, mutate) => {
    const next = cores().map((c) => ({ ...c }));
    mutate(next[i]);
    write(next);
  };

  function render() {
    host.replaceChildren();
    if (cores().length) {
      const head = el('div', 'cores-editor-head');
      head.append(el('span', 'cores-col-label', 'Type'));
      for (const [, title] of CORE_DIMS) head.append(el('span', 'cores-col-label', title));
      head.append(el('span', 'cores-col-label cores-col-center', 'Bias'));
      head.append(el('span', 'cores-col-label', ''));
      host.append(head);
    }
    cores().forEach((core, i) => {
      const row = el('div', 'cores-editor-row');
      row.append(el('span', 'cores-row-label', `T${i + 1}`));
      for (const [dim, title] of CORE_DIMS) {
        const input = numeric(core[dim], (v) => {
          patch(i, (c) => {
            if (v === undefined) delete c[dim];
            else c[dim] = Math.round(v);
          });
        }, { step: '1', min: 1 });
        input.title = `${title} for core type ${i + 1}`;
        row.append(input);
      }
      row.append(coreBiasToggle(core, (on) => patch(i, (c) => { c.has_bias = on; })));
      const remove = el('button', 'btn-sm cores-remove', '✕');
      remove.type = 'button';
      remove.title = `Remove core type ${i + 1}`;
      remove.addEventListener('click', () => write(cores().filter((_, j) => j !== i)));
      row.append(remove);
      host.append(row);
    });
    const add = el('button', 'btn-sm cores-add', '+ Add core type');
    add.type = 'button';
    add.addEventListener('click', () => {
      const bias = state.draft.platform_constraints?.has_bias;
      write([...cores().map((c) => ({ ...c })),
             { max_axons: 256, max_neurons: 256, count: 100,
               ...(bias === undefined ? {} : { has_bias: bias }) }]);
    });
    host.append(add);
  }
  render();
  return field;
}

/* ── weight precision: float vs quantized-N-bits (the fp/WQ contract) ──── */

function floatWeightsActive() {
  return state.draft.pipeline_mode === 'vanilla'
    || state.draft.deployment_parameters?.weight_quantization === false;
}

/** First-class weight precision, authoring exactly the tier-config fp form:
    float = pipeline_mode 'vanilla' + weight_quantization false (weight_bits
    stays in the document, inert); quantized = bits-driven WQ (the explicit
    float declarations are cleared so the derivation owns the flags). */
function weightPrecisionWidget(ks) {
  const field = el('div', 'field');
  field.dataset.key = ks.key;
  const label = el('label', 'field-label');
  label.append(el('span', 'field-label-text', ks.label));
  field.append(label);

  const marker = el('span', 'field-explicit-slot');
  label.append(marker);
  const isFloat = floatWeightsActive();
  const seg = el('div', 'seg-control');
  const mkBtn = (text, active, title, apply) => {
    const btn = el('button', 'seg-btn' + (active ? ' active' : ''), text);
    btn.type = 'button';
    btn.title = title;
    btn.addEventListener('click', () => {
      if (active) return;
      apply();
      notifyChange('weight_quantization');
    });
    return btn;
  };
  seg.append(mkBtn('quantized', !isFloat,
    'Quantized artifact: weight_bits declares the precision (bits-driven WQ contract)',
    () => {
      clearKey('weight_quantization');
      if (state.draft.pipeline_mode === 'vanilla') clearKey('pipeline_mode');
    }));
  seg.append(mkBtn('float', isFloat,
    'Float weights: the vanilla assembly (pipeline_mode=\'vanilla\' + '
    + 'weight_quantization=false; no quantization steps run)',
    () => {
      setKey('weight_quantization', false);
      setKey('pipeline_mode', 'vanilla');
    }));
  field.append(seg);

  if (isFloat) {
    const hint = el('div', 'precision-float-hint',
      'float32 — vanilla assembly, no quantization steps');
    hint.title = 'The document keeps weight_bits (inert under the vanilla '
      + 'assembly), matching the tier-config fp form.';
    field.append(hint);
  } else {
    const bits = numberInput(ks, marker, () => notifyChange('weight_bits'));
    bits.classList.add('precision-bits');
    field.append(bits);
  }
  field.append(fieldDoc(ks));
  return field;
}

/* ── search_space: the structured hw co-search bounds editor ───────────── */

function rangePair(spec, current, onChange) {
  const wrap = el('div', 'range-pair');
  const mk = (idx, ph) => {
    const input = el('input');
    input.type = 'number';
    input.step = spec.step ? String(spec.step) : '1';
    input.min = '1';
    input.placeholder = String(ph);
    const value = Array.isArray(current) ? current[idx] : undefined;
    input.value = value === undefined || value === null ? '' : String(value);
    input.addEventListener('change', () => {
      const raw = [wrap.children[0].value.trim(), wrap.children[2].value.trim()];
      if (raw[0] === '' && raw[1] === '') { onChange(undefined); return; }
      const lo = raw[0] === '' ? spec.default[0] : parseInt(raw[0], 10);
      const hi = raw[1] === '' ? spec.default[1] : parseInt(raw[1], 10);
      if (Number.isNaN(lo) || Number.isNaN(hi)) return;
      onChange([lo, hi]);
    });
    return input;
  };
  wrap.append(mk(0, spec.default[0]));
  wrap.append(el('span', 'range-pair-sep', '–'));
  wrap.append(mk(1, spec.default[1]));
  return wrap;
}

/** Structured editor for the hw co-search bounds (never a plain-text field);
    the field schema is served by /api/config_schema (hw_search_space_fields). */
function searchSpaceWidget(ks) {
  const fieldsSchema = schema().hw_search_space_fields || {};
  const field = el('div', 'field span-full');
  field.dataset.key = ks.key;
  const label = el('label', 'field-label');
  label.append(el('span', 'field-label-text', ks.label));
  field.append(label);
  const grid = el('div', 'search-space-editor');
  field.append(grid);
  field.append(fieldDoc(ks));

  const current = () => ({ ...(getKey('search_space') || {}) });
  const write = (ss) => {
    if (Object.keys(ss).length === 0) clearKey('search_space');
    else setKey('search_space', ss);
    notifyChange('search_space');
  };

  for (const [name, spec] of Object.entries(fieldsSchema)) {
    let control;
    if (spec.type === 'int_range') {
      control = rangePair(spec, current()[name], (value) => {
        const ss = current();
        if (value === undefined) delete ss[name];
        else ss[name] = value;
        write(ss);
      });
    } else {
      control = numeric(current()[name], (v) => {
        const ss = current();
        if (v === undefined) delete ss[name];
        else ss[name] = Math.round(v);
        write(ss);
      }, { step: '1', min: spec.min ?? 1, placeholder: String(spec.default ?? '') });
    }
    const sub = subField(name.replace(/_/g, ' '), control);
    sub.title = spec.doc || '';
    grid.append(sub);
  }
  return field;
}

/* ── recipes: fields served by /api/config_schema recipe_fields ────────── */

function recipeWidget(ks) {
  const fieldsSchema = schema().recipe_fields || {};
  const defaultsKey = ks.key === 'tuning_recipe' ? 'default_tuning' : 'default_training';
  const field = el('div', 'field span-2');
  field.dataset.key = ks.key;
  const label = el('label', 'field-label', ks.label);
  field.append(label);

  const grid = el('div', 'field-grid cols-2');
  const current = () => ({ ...(getKey(ks.key) || {}) });
  const write = (recipe) => {
    if (Object.keys(recipe).length === 0) clearKey(ks.key);
    else setKey(ks.key, recipe);
    notifyChange(ks.key);
  };

  for (const [name, spec] of Object.entries(fieldsSchema)) {
    const fallback = spec[defaultsKey];
    const value = current()[name];
    let control;
    if (spec.type === 'float_list') {
      control = el('input');
      control.type = 'text';
      control.placeholder = Array.isArray(fallback) ? fallback.join(', ') : '';
      control.value = Array.isArray(value) ? value.join(', ') : '';
      control.addEventListener('change', () => {
        const recipe = current();
        const raw = control.value.trim();
        if (raw === '') delete recipe[name];
        else {
          const items = raw.split(',').map((s) => parseFloat(s.trim()));
          if (items.some(Number.isNaN)) return;
          recipe[name] = items;
        }
        write(recipe);
      });
    } else if (spec.type === 'str') {
      control = el('input');
      control.type = 'text';
      control.placeholder = fallback === undefined || fallback === null ? '' : String(fallback);
      control.value = value === undefined ? '' : String(value);
      control.addEventListener('change', () => {
        const recipe = current();
        if (control.value.trim() === '') delete recipe[name];
        else recipe[name] = control.value.trim();
        write(recipe);
      });
    } else {
      control = numeric(value, (v) => {
        const recipe = current();
        if (v === undefined) delete recipe[name];
        else recipe[name] = v;
        write(recipe);
      }, { placeholder: fallback === undefined || fallback === null ? '' : String(fallback) });
    }
    grid.append(subField(name.replace(/_/g, ' '), control));
  }
  field.append(grid);
  return field;
}

/* ── model_config: fields from the per-builder schema ──────────────────── */

export async function ensureModelSchema(modelType) {
  if (!modelType || state.modelSchemas[modelType]) return;
  const res = await fetch('/api/model_config_schema/' + encodeURIComponent(modelType));
  state.modelSchemas[modelType] = res.ok ? await res.json() : [];
}

/** The builder's default model_config (from its served schema) — the honest
    draft state after a model_type switch; a stale config from the previous
    builder would not even build. */
export function defaultModelConfig(modelType) {
  const config = {};
  for (const spec of state.modelSchemas[modelType] || []) {
    if (spec.default !== undefined && spec.default !== null) config[spec.key] = spec.default;
  }
  return config;
}

function modelConfigWidget(ks) {
  const field = el('div', 'field span-2');
  field.dataset.key = ks.key;
  field.append(el('label', 'field-label', ks.label));
  const grid = el('div', 'field-grid cols-2');
  field.append(grid);

  const modelType = state.draft.deployment_parameters?.model_type;
  const fields = state.modelSchemas[modelType] || [];
  if (!modelType) {
    grid.append(el('div', 'note', 'Pick a model type first.'));
    return field;
  }
  if (fields.length === 0) {
    grid.append(el('div', 'note', 'No configurable parameters for this model type.'));
    return field;
  }

  const current = () => ({ ...(getKey('model_config') || {}) });
  const write = (config) => { setKey('model_config', config); notifyChange('model_config'); };

  for (const spec of fields) {
    const value = current()[spec.key] ?? spec.default;
    let control;
    if (spec.type === 'select') {
      control = el('select');
      for (const option of spec.options || []) control.append(new Option(option, option));
      control.value = String(value);
      control.addEventListener('change', () => {
        const config = current();
        config[spec.key] = control.value;
        write(config);
      });
    } else if (spec.type === 'toggle') {
      control = el('div', 'toggle-row' + (value ? ' on' : ''));
      control.append(el('span', 'toggle-label', spec.label), el('div', 'toggle-switch'));
      control.addEventListener('click', () => {
        control.classList.toggle('on');
        const config = current();
        config[spec.key] = control.classList.contains('on');
        write(config);
      });
      grid.append(control);
      continue;
    } else if (spec.type === 'text') {
      control = el('input');
      control.type = 'text';
      control.value = Array.isArray(value) ? value.join(', ') : String(value ?? '');
      control.addEventListener('change', () => {
        const config = current();
        if (spec.key === 'hidden_dims') {
          const items = control.value.split(',').map((s) => parseInt(s.trim(), 10))
            .filter((n) => !Number.isNaN(n));
          config[spec.key] = items;
        } else {
          config[spec.key] = control.value;
        }
        write(config);
      });
    } else {
      control = numeric(value, (v) => {
        const config = current();
        if (v !== undefined) config[spec.key] = v;
        write(config);
      }, { step: spec.step ? String(spec.step) : 'any' });
    }
    grid.append(subField(spec.label, control));
  }
  return field;
}

/* ── preprocessing: served sub-schema ──────────────────────────────────── */

function preprocessingWidget(ks) {
  const fieldsSchema = schema().preprocessing_fields || {};
  const field = el('div', 'field span-2');
  field.dataset.key = ks.key;
  field.append(el('label', 'field-label', ks.label));
  const grid = el('div', 'field-grid cols-3');
  field.append(grid);

  const current = () => ({ ...(getKey('preprocessing') || {}) });
  const write = (pp) => {
    if (Object.keys(pp).length === 0) clearKey('preprocessing');
    else setKey('preprocessing', pp);
    notifyChange('preprocessing');
  };

  for (const [name, spec] of Object.entries(fieldsSchema)) {
    let control;
    if (spec.type === 'enum') {
      control = el('select');
      control.append(new Option('— none —', ''));
      for (const option of spec.options || []) control.append(new Option(option, option));
      control.value = current()[name] ?? '';
      control.addEventListener('change', () => {
        const pp = current();
        if (control.value === '') delete pp[name];
        else pp[name] = control.value;
        write(pp);
      });
    } else {
      control = numeric(current()[name], (v) => {
        const pp = current();
        if (v === undefined) delete pp[name];
        else pp[name] = Math.round(v);
        write(pp);
      }, { step: '1', min: 1, placeholder: 'native' });
    }
    grid.append(subField(name.replace(/_/g, ' '), control));
  }
  return field;
}

/* ── arch_search: NAS schema (optimizer, budget, objectives) ───────────── */

function archSearchWidget(ks) {
  const nas = schema().nas || {};
  const field = el('div', 'field span-2');
  field.dataset.key = ks.key;
  field.append(el('label', 'field-label', ks.label));

  const current = () => ({ ...(getKey('arch_search') || {}) });
  const write = (arch) => { setKey('arch_search', arch); notifyChange('arch_search'); };

  const optimizerRow = el('div', 'seg-control');
  const optimizers = nas.optimizer_options || [];
  const activeOptimizer = () => current().optimizer || 'nsga2';
  for (const opt of optimizers) {
    const btn = el('div', 'seg-btn' + (activeOptimizer() === opt.id ? ' active' : ''), opt.label);
    btn.dataset.val = opt.id;
    btn.addEventListener('click', () => {
      const arch = current();
      arch.optimizer = opt.id;
      write(arch);
      rerender();
    });
    optimizerRow.append(btn);
  }
  field.append(optimizerRow);

  const grid = el('div', 'field-grid cols-2');
  field.append(grid);

  const fieldGroups = { ...(nas.common_fields || {}) };
  if (activeOptimizer() === 'agent_evolve') Object.assign(fieldGroups, nas.agent_evolve_fields || {});
  if (activeOptimizer() === 'compilagent') Object.assign(fieldGroups, nas.compilagent_fields || {});

  for (const [name, spec] of Object.entries(fieldGroups)) {
    let control;
    const value = current()[name];
    if (spec.type === 'int' || spec.type === 'float') {
      control = numeric(value ?? spec.default, (v) => {
        const arch = current();
        if (v === undefined) delete arch[name];
        else arch[name] = spec.type === 'int' ? Math.round(v) : v;
        write(arch);
      }, { step: spec.type === 'int' ? '1' : 'any' });
    } else if (spec.options) {
      control = el('select');
      for (const option of spec.options) control.append(new Option(option, option));
      control.value = String(value ?? spec.default ?? '');
      control.addEventListener('change', () => {
        const arch = current();
        arch[name] = control.value;
        write(arch);
      });
    } else {
      control = el(spec.type === 'textarea' ? 'textarea' : 'input');
      if (spec.type !== 'textarea') control.type = 'text';
      control.value = String(value ?? spec.default ?? '');
      control.addEventListener('change', () => {
        const arch = current();
        if (control.value.trim() === '') delete arch[name];
        else arch[name] = control.value;
        write(arch);
      });
    }
    control.title = spec.doc || '';
    grid.append(subField(name.replace(/_/g, ' '), control));
  }

  const objectivesTitle = el('div', 'sub-section-title', 'Optimization Objectives');
  const chips = el('div', 'objective-checkboxes');
  const selected = new Set(current().objectives || (nas.objective_options || []).map((o) => o.id));
  for (const objective of nas.objective_options || []) {
    const chip = el('div', 'objective-chip' + (selected.has(objective.id) ? ' active' : ''));
    chip.append(el('span', '', objective.label), el('span', 'goal-badge', objective.goal));
    chip.addEventListener('click', () => {
      chip.classList.toggle('active');
      const arch = current();
      arch.objectives = [...chips.querySelectorAll('.objective-chip.active')]
        .map((c) => c.dataset.objective);
      write(arch);
    });
    chip.dataset.objective = objective.id;
    chips.append(chip);
  }
  field.append(objectivesTitle, chips);

  function rerender() {
    field.replaceWith(archSearchWidget(ks));
  }
  return field;
}

export function installStructuredWidgets() {
  registerCustomRenderer('cores', coresWidget);
  registerCustomRenderer('training_recipe', recipeWidget);
  registerCustomRenderer('tuning_recipe', recipeWidget);
  registerCustomRenderer('model_config', modelConfigWidget);
  registerCustomRenderer('preprocessing', preprocessingWidget);
  registerCustomRenderer('arch_search', archSearchWidget);
  registerCustomRenderer('weight_bits', weightPrecisionWidget);
  registerCustomRenderer('search_space', searchSpaceWidget);
  installPretrainedPanel(registerCustomRenderer);
}
