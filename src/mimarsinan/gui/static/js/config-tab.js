/* Pipeline configuration tab — structured server-driven display. */
import { esc } from './util.js';
import { renderConfig as renderConfigLegacy } from './overview.js';

const GROUP_GLOW = {
  configuration: '168,85,247',
  model_building: '139,92,246',
  pretraining: '56,189,248',
  torch_mapping: '14,165,233',
  pruning: '244,63,94',
  activation: '74,222,128',
  activation_quantization: '52,211,153',
  weight_quantization: '251,191,36',
  normalization: '34,211,238',
  soft_mapping: '96,165,250',
  core_verification: '129,140,248',
  coreflow_tuning: '99,102,241',
  hardware: '249,115,22',
  simulation: '103,232,249',
  other: '107,114,128',
};

let _uiState = {
  hideDefaults: false,
  filter: '',
  collapsed: {},
};

export function renderConfigView(configView, legacyConfig) {
  const el = document.getElementById('config-body');
  if (!el) return;
  if (!configView) {
    renderConfigLegacy(legacyConfig);
    return;
  }

  el.innerHTML = buildConfigViewHtml(configView);
  bindConfigViewEvents(el, configView);
}

function buildConfigViewHtml(view) {
  const summary = view.summary || {};
  let html = '<div class="config-view">';

  html += '<div class="cv-hero">';
  for (const [label, key] of [
    ['Experiment', 'experiment_name'],
    ['Pipeline', 'pipeline_mode'],
    ['Spiking', 'spiking_mode'],
    ['Model', 'model_type'],
  ]) {
    const val = summary[key];
    if (val == null || val === '') continue;
    html += `<div class="cv-hero-item"><div class="cv-hero-label">${esc(label)}</div>`;
    html += `<div class="cv-hero-value">${esc(String(val))}</div></div>`;
  }
  if (summary.data_provider_name) {
    html += `<div class="cv-hero-item cv-hero-item-wide"><div class="cv-hero-label">Data Provider</div>`;
    html += `<div class="cv-hero-value cv-hero-value-sm">${esc(summary.data_provider_name)}</div></div>`;
  }
  html += '</div>';

  html += `<div class="cv-toolbar">
    <input type="search" class="cv-filter" placeholder="Filter fields…" value="${esc(_uiState.filter)}" aria-label="Filter configuration fields">
    <label class="cv-toggle"><input type="checkbox" class="cv-hide-defaults"${_uiState.hideDefaults ? ' checked' : ''}> Hide defaults</label>
    <button type="button" class="cv-btn cv-expand-all">Expand all</button>
    <button type="button" class="cv-btn cv-collapse-all">Collapse all</button>
  </div>`;

  html += buildPipelinePreviewHtml(view.pipeline_preview);

  for (const section of view.sections || []) {
    html += buildSectionHtml(section, view.nested || {});
  }

  html += buildNestedStandaloneHtml(view.nested || {}, view.sections || []);
  html += buildRawJsonFooter(view.raw_resolved);
  html += '</div>';
  return html;
}

function buildPipelinePreviewHtml(preview) {
  if (!preview?.steps?.length) return '';
  let html = '<div class="cv-preview"><div class="cv-preview-label">Pipeline path from config</div><div class="cv-preview-chips">';
  preview.steps.forEach((step, i) => {
    const group = (preview.semantic_groups || [])[i] || 'other';
    const glow = GROUP_GLOW[group] || GROUP_GLOW.other;
    html += `<span class="cv-chip" data-group="${esc(group)}" style="--g:${glow}" title="${esc(step)}">${esc(shortStepName(step))}</span>`;
  });
  html += '</div></div>';
  return html;
}

function shortStepName(name) {
  if (!name) return '';
  const parts = name.split(' ');
  if (parts.length <= 2) return name;
  return parts.slice(0, 2).join(' ') + '…';
}

function buildSectionHtml(section, nested) {
  const collapsed = _uiState.collapsed[section.id] === true;
  const visibleFields = filterFields(section.fields || []);
  if (!visibleFields.length) return '';

  let html = `<section class="cv-section${collapsed ? ' collapsed' : ''}" data-section-id="${esc(section.id)}" style="--g:${section.accent || '107,114,128'}">`;
  html += `<button type="button" class="cv-section-header" aria-expanded="${collapsed ? 'false' : 'true'}">`;
  html += `<span class="cv-section-accent"></span>`;
  html += `<span class="cv-section-titles"><span class="cv-section-title">${esc(section.title)}</span>`;
  html += `<span class="cv-section-subtitle">${esc(section.subtitle || '')}</span></span>`;
  html += `<span class="cv-section-count">${visibleFields.length}</span>`;
  html += `<span class="cv-section-chevron">▾</span></button>`;
  html += '<div class="cv-section-body">';

  for (const field of visibleFields) {
    html += buildFieldRowHtml(field, nested);
  }

  if (section.id === 'training' && nested.training_recipe) {
    html += buildRecipeBlock('Training Recipe', nested.training_recipe);
  }
  if (section.id === 'tuning' && nested.tuning_recipe) {
    html += buildRecipeBlock('Tuning Recipe', nested.tuning_recipe);
  }
  if (section.id === 'model' && nested.model_config) {
    html += buildModelConfigBlock(nested.model_config);
  }
  if (section.id === 'search' && nested.arch_search) {
    html += buildArchSearchBlock(nested.arch_search);
  }
  if (section.id === 'hardware' && nested.cores) {
    html += buildCoresBlock(nested.cores);
  }

  html += '</div></section>';
  return html;
}

function buildNestedStandaloneHtml(nested, sections) {
  const sectionIds = new Set(sections.map(s => s.id));
  let html = '';
  if (!sectionIds.has('hardware') && nested.cores) {
    html += `<section class="cv-section" style="--g:249,115,22"><div class="cv-section-header-static"><span class="cv-section-title">Core Types</span></div><div class="cv-section-body">${buildCoresBlock(nested.cores)}</div></section>`;
  }
  return html;
}

function filterFields(fields) {
  const q = (_uiState.filter || '').trim().toLowerCase();
  return fields.filter(f => {
    if (_uiState.hideDefaults && f.source === 'default') return false;
    if (!q) return true;
    return (f.key || '').toLowerCase().includes(q)
      || (f.label || '').toLowerCase().includes(q)
      || String(f.value ?? '').toLowerCase().includes(q);
  });
}

function buildFieldRowHtml(field, nested) {
  if (field.type === 'cores_ref' && nested.cores) return '';
  let html = `<div class="cv-field" data-source="${esc(field.source || 'default')}">`;
  html += `<div class="cv-field-label">${esc(field.label || field.key)}`;
  if (field.effect) html += `<span class="cv-field-tip" title="${esc(field.effect)}">?</span>`;
  html += '</div>';
  html += `<div class="cv-field-value">${renderFieldValue(field)}</div>`;
  html += `<div class="cv-field-source">${renderSourceBadge(field.source)}</div>`;
  html += '</div>';
  return html;
}

function renderFieldValue(field) {
  const v = field.value;
  const t = field.type || 'str';
  if (v == null) return '<span class="cv-null">—</span>';
  if (t === 'bool') {
    const on = v === true;
    return `<span class="cv-pill ${on ? 'cv-pill-on' : 'cv-pill-off'}">${on ? 'ON' : 'OFF'}</span>`;
  }
  if (t === 'enum' || t === 'select') {
    return `<span class="cv-enum">${esc(String(v))}</span>`;
  }
  if (t === 'shape' && Array.isArray(v)) {
    return v.map(d => `<span class="cv-dim">[${esc(String(d))}]</span>`).join('');
  }
  if (t === 'int' || t === 'float') {
    return `<span class="cv-mono">${esc(formatNumber(v))}</span>`;
  }
  if (t === 'path' || t === 'str') {
    return `<span class="cv-mono cv-str">${esc(String(v))}</span>`;
  }
  if (typeof v === 'object') {
    return `<pre class="cv-inline-json">${esc(JSON.stringify(v, null, 2))}</pre>`;
  }
  return `<span class="cv-mono">${esc(String(v))}</span>`;
}

function formatNumber(v) {
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v)) return String(v);
  if (Math.abs(v) < 0.01 || Math.abs(v) >= 10000) return v.toExponential(2);
  return String(v);
}

function renderSourceBadge(source) {
  const labels = {
    explicit: 'set',
    default: 'default',
    derived: 'derived',
    preset: 'preset',
    runtime: 'resolved',
  };
  return `<span class="cv-src cv-src-${esc(source || 'default')}">${esc(labels[source] || source || 'default')}</span>`;
}

function buildRecipeBlock(title, recipe) {
  let html = `<div class="cv-subblock"><div class="cv-subblock-title">${esc(title)}</div><div class="cv-recipe-grid">`;
  for (const f of recipe.fields || []) {
    html += `<div class="cv-recipe-item"><span class="cv-recipe-key">${esc(f.label)}</span>`;
    html += `<span class="cv-recipe-val">${renderFieldValue(f)}</span></div>`;
  }
  html += '</div></div>';
  return html;
}

function buildModelConfigBlock(block) {
  let html = '<div class="cv-subblock"><div class="cv-subblock-title">Model Parameters</div><div class="cv-recipe-grid">';
  for (const f of block.fields || []) {
    html += `<div class="cv-recipe-item"><span class="cv-recipe-key">${esc(f.label)}</span>`;
    html += `<span class="cv-recipe-val">${renderFieldValue(f)}</span></div>`;
  }
  html += '</div></div>';
  return html;
}

function buildArchSearchBlock(block) {
  let html = '<div class="cv-subblock"><div class="cv-subblock-title">Search Configuration</div><div class="cv-recipe-grid">';
  for (const f of block.fields || []) {
    html += `<div class="cv-recipe-item"><span class="cv-recipe-key">${esc(f.label)}</span>`;
    html += `<span class="cv-recipe-val">${renderFieldValue(f)}</span></div>`;
  }
  html += '</div></div>';
  return html;
}

function buildCoresBlock(cores) {
  let html = '<div class="cv-cores">';
  for (let i = 0; i < (cores.items || []).length; i++) {
    const c = cores.items[i];
    html += `<div class="cv-core-card"><div class="cv-core-title">Type ${i + 1}</div>`;
    html += `<div class="cv-core-stats">`;
    html += `<div class="cv-core-stat"><span class="cv-core-stat-label">Axons</span><span class="cv-core-stat-val">${esc(String(c.max_axons ?? '—'))}</span></div>`;
    html += `<div class="cv-core-stat"><span class="cv-core-stat-label">Neurons</span><span class="cv-core-stat-val">${esc(String(c.max_neurons ?? '—'))}</span></div>`;
    html += `<div class="cv-core-stat"><span class="cv-core-stat-label">Count</span><span class="cv-core-stat-val">${esc(String(c.count ?? '—'))}</span></div>`;
    html += '</div></div>';
  }
  html += '</div>';
  return html;
}

function buildRawJsonFooter(raw) {
  const json = raw ? JSON.stringify(raw, null, 2) : '{}';
  return `<details class="cv-raw"><summary>Raw resolved JSON</summary>
    <div class="cv-raw-actions"><button type="button" class="cv-btn cv-copy-json">Copy JSON</button></div>
    <pre class="cv-raw-pre">${esc(json)}</pre></details>`;
}

function bindConfigViewEvents(el, view) {
  el.querySelector('.cv-filter')?.addEventListener('input', e => {
    _uiState.filter = e.target.value;
    renderConfigView(view, null);
  });
  el.querySelector('.cv-hide-defaults')?.addEventListener('change', e => {
    _uiState.hideDefaults = e.target.checked;
    renderConfigView(view, null);
  });
  el.querySelector('.cv-expand-all')?.addEventListener('click', () => {
    _uiState.collapsed = {};
    renderConfigView(view, null);
  });
  el.querySelector('.cv-collapse-all')?.addEventListener('click', () => {
    for (const s of view.sections || []) _uiState.collapsed[s.id] = true;
    renderConfigView(view, null);
  });
  el.querySelectorAll('.cv-section-header').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = btn.closest('.cv-section')?.dataset.sectionId;
      if (!id) return;
      _uiState.collapsed[id] = !_uiState.collapsed[id];
      renderConfigView(view, null);
    });
  });
  el.querySelector('.cv-copy-json')?.addEventListener('click', () => {
    const pre = el.querySelector('.cv-raw-pre');
    if (!pre) return;
    navigator.clipboard.writeText(pre.textContent || '').catch(() => {});
  });
}

export function renderConfigTab(pipeline) {
  renderConfigView(pipeline?.config_view, pipeline?.config);
}
