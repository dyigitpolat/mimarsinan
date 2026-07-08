/* Workbench controller: schema-driven rendering, resolve loop, deploy actions.
   The draft IS the config document; every widget renders from /api/config_schema;
   sections (workbench.js) host whole concern groups; the live rail renders the
   honest assembly + mapping from every resolve round-trip. */

import {
  enableAssignments, groups, keySchema, loadSchema, providedAwayKeys, schema,
  unavailabilityReason, visibleKeys,
} from './schema.js';
import {
  differsFromDefault, effectiveConfig, effectiveValue, isExplicit,
  loadDraftFromConfig, resetDraft, setKey, state,
} from './state.js';
import { el, notifyChange, renderField, renderOwnershipCell } from './fields.js';
import { defaultModelConfig, ensureModelSchema, installStructuredWidgets } from './structured.js';
import {
  renderAssemblyRail, renderDerivedChips, renderDiffPanel, renderErrors,
  renderJsonPreview, renderLaunchStatus, renderRailMapping,
  renderTemplateBanner, renderUnknownTray,
} from './review.js';
import { autoSuggestHardware, fetchMetadata, scheduleHwVerify } from './hw.js';
import { goToFirstError, goToSection, renderSectionNav } from './workbench.js';

const GROUP_ICONS = {
  run: '⬡', workload: '◈', model: '▣', spiking: '⚡', conversion: '◎',
  tuning: '☈', training: '▶', hardware: '▦', deployment_target: '✈',
  co_search: '⌖',
};

/* ── Section rendering ─────────────────────────────────────────────────── */

function fieldList(groupId, category, cfg) {
  /* Relevance controls EXISTENCE; promote_when controls prominence. */
  return visibleKeys(groupId, category, cfg).map((key) => keySchema(key));
}

/** The group's primary grid cells in registry order: hand fields where keys
    exist, ownership chips where another concern (provided_by) produces the
    value — the chip renders exactly where the hand field would be. */
function primaryCells(group, cfg, basicKeys, ownedKeys) {
  const byId = Object.fromEntries(groups().map((g) => [g.id, g]));
  const basic = new Set(basicKeys);
  const owned = new Set(ownedKeys);
  const cells = [];
  for (const ks of Object.values(schema().keys)) {
    if (ks.group !== group.id) continue;
    if (basic.has(ks.key)) cells.push(renderField(ks));
    else if (owned.has(ks.key)) {
      cells.push(renderOwnershipCell(
        ks, byId[ks.provided_by], unavailabilityReason(ks.relevant, cfg),
      ));
    }
  }
  return cells;
}

function sectionHeader(group) {
  const header = el('div', 'section-header');
  header.append(el('div', 'section-icon', GROUP_ICONS[group.id] || '◇'));
  const titles = el('div', 'section-title-group');
  titles.append(el('div', 'section-title', group.title));
  titles.append(el('div', 'section-subtitle', group.subtitle || ''));
  header.append(titles);
  return header;
}

/** The quiet dormant-group card (registry empty_state): one status line plus
    the enable path derived from the group's own relevance predicates. */
function renderGroupOffCard(group, cfg) {
  const section = el('div', 'section section-off');
  section.dataset.section = group.id;
  section.style.setProperty('--section-accent', group.accent || '91,141,245');
  const row = el('div', 'section-off-row');
  row.append(el('div', 'section-icon', GROUP_ICONS[group.id] || '◇'));
  row.append(el('span', 'section-off-title', group.title));
  row.append(el('span', 'section-off-state', group.empty_state));
  const actions = el('span', 'section-off-actions');
  for (const { key, value } of enableAssignments(group.id, cfg)) {
    const ks = keySchema(key);
    const btn = el('button', 'btn-sm', `${ks ? ks.label : key} → ${value}`);
    btn.type = 'button';
    btn.addEventListener('click', () => {
      setKey(key, value);
      notifyChange(key);
    });
    actions.append(btn);
  }
  row.append(actions);
  section.append(row);
  return section;
}

function renderGroupSection(group, cfg) {
  const basicKeys = visibleKeys(group.id, 'basic', cfg);
  const advanced = fieldList(group.id, 'advanced', cfg);
  const ownedKeys = providedAwayKeys(group.id, cfg);
  if (!basicKeys.length && !advanced.length && !ownedKeys.length) {
    return group.empty_state ? renderGroupOffCard(group, cfg) : null;
  }

  const section = el('div', 'section');
  section.dataset.section = group.id;
  section.style.setProperty('--section-accent', group.accent || '91,141,245');

  const header = sectionHeader(group);
  if (group.id === 'hardware' && basicKeys.includes('cores')) {
    /* The suggester writes the hand core grid — it only exists while the
       grid is hand-owned. */
    const suggest = el('button', 'btn-sm primary suggest-hw', '⚙ Suggest hardware for this model');
    suggest.type = 'button';
    suggest.addEventListener('click', () => autoSuggestHardware());
    header.append(suggest);
  }
  section.append(header);

  const body = el('div', 'section-body');
  const grid = el('div', 'field-grid cols-2');
  for (const cell of primaryCells(group, cfg, basicKeys, ownedKeys)) grid.append(cell);
  body.append(grid);

  if (advanced.length) {
    const edited = advanced.filter((ks) => differsFromDefault(ks.key)).length;
    const drawer = el('div', 'advanced-drawer' + (edited ? ' open' : ''));
    const toggle = el('button', 'advanced-toggle');
    toggle.type = 'button';
    toggle.append(
      el('span', 'advanced-toggle-chevron', '▸'),
      el('span', '', `Advanced (${advanced.length})`),
      edited ? el('span', 'advanced-toggle-explicit', `${edited} edited`) : '',
    );
    toggle.addEventListener('click', () => drawer.classList.toggle('open'));
    const drawerBody = el('div', 'advanced-drawer-body field-grid cols-2');
    for (const ks of advanced) drawerBody.append(renderField(ks));
    drawer.append(drawerBody);
    body.append(toggle, drawer);
  }

  section.append(body);
  return section;
}

function renderGroupHosts() {
  const cfg = effectiveConfig();
  const byId = Object.fromEntries(groups().map((g) => [g.id, g]));
  document.querySelectorAll('[data-groups]').forEach((host) => {
    host.replaceChildren();
    for (const groupId of host.dataset.groups.split(',')) {
      const group = byId[groupId.trim()];
      if (!group) continue;
      const section = renderGroupSection(group, cfg);
      if (section) host.append(section);
    }
  });
}

/* ── Dataset facts (workload section) ──────────────────────────────────── */

async function renderDatasetFacts() {
  const host = document.getElementById('datasetFacts');
  if (!host) return;
  const provider = effectiveValue('data_provider_name');
  const md = provider ? await fetchMetadata() : null;
  state.metadata = md;
  if (!md || !Array.isArray(md.input_shape)) {
    host.style.display = 'none';
    return;
  }
  host.style.display = '';
  host.replaceChildren();
  host.append(el('span', 'dataset-facts-label', md.label || provider));
  const dims = md.input_shape.join(' × ');
  host.append(el('span', 'dataset-fact', `input ${dims}`));
  if (Number.isFinite(md.num_classes)) {
    host.append(el('span', 'dataset-fact', `${md.num_classes} classes`));
  }
  host.title = 'Resolved from the data provider; the pipeline reads these at runtime.';
}

function renderResolveViews() {
  renderDerivedChips();
  renderDiffPanel();
  renderUnknownTray();
  renderAssemblyRail();
  renderRailMapping();
  renderErrors();
  renderJsonPreview();
  renderLaunchStatus();
  renderSectionNav((state.resolve && state.resolve.errors) || []);
  renderStatusPill();
}

function renderAll() {
  renderGroupHosts();
  renderTemplateBanner();
  renderResolveViews();
  renderDatasetFacts();
}

/* ── Rail verdict pill ─────────────────────────────────────────────────── */

function renderStatusPill() {
  const pill = document.getElementById('statusPill');
  if (!pill) return;
  if (!state.resolve || state.resolving) {
    pill.className = 'rail-verdict pending';
    pill.textContent = 'resolving…';
    return;
  }
  const errors = state.resolve.errors || [];
  if (errors.length) {
    pill.className = 'rail-verdict error';
    pill.textContent = `${errors.length} error${errors.length > 1 ? 's' : ''}`;
  } else {
    const n = (state.resolve.pipeline && state.resolve.pipeline.steps.length) || 0;
    pill.className = 'rail-verdict ok';
    pill.textContent = `ready · ${n} steps`;
  }
}

/* ── Resolve loop (debounced; the rail shows stale while pending) ──────── */

let _resolveTimer = null;
let _resolveInFlight = false;
let _resolveQueued = false;

function markStale(stale) {
  state.resolving = stale;
  const rail = document.getElementById('liveRail');
  if (rail) rail.classList.toggle('stale', stale);
}

function scheduleResolve() {
  if (_resolveTimer) clearTimeout(_resolveTimer);
  markStale(true);
  renderStatusPill();
  _resolveTimer = setTimeout(runResolve, 250);
}

async function runResolve() {
  if (_resolveInFlight) { _resolveQueued = true; return; }
  _resolveInFlight = true;
  try {
    const res = await fetch('/api/config/resolve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state.draft),
    });
    if (res.ok) {
      state.resolve = await res.json();
      markStale(false);
      renderResolveViews();
    }
  } catch (_) { /* transient network failure; next change retries */ }
  _resolveInFlight = false;
  if (_resolveQueued) { _resolveQueued = false; scheduleResolve(); }
}

/* ── Actions ───────────────────────────────────────────────────────────── */

function emittedConfig() {
  return (state.resolve && state.resolve.emitted) || state.draft;
}

function currentErrors() {
  return (state.resolve && state.resolve.errors) || [];
}

function bindActions() {
  document.getElementById('runBtn')?.addEventListener('click', async () => {
    const errors = currentErrors();
    if (errors.length) { goToFirstError(errors); return; }
    const btn = document.getElementById('runBtn');
    btn.disabled = true;
    try {
      const res = await fetch('/api/run?validate=1', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.draft),
      });
      const body = await res.json();
      if (res.status === 202 || res.ok) {
        window.location.href = body.run_id
          ? '/monitor?run_id=' + encodeURIComponent(body.run_id) : '/';
      } else {
        alert(body.error + (body.field_errors ? '\n' + body.field_errors.join('\n') : ''));
      }
    } catch (err) {
      alert(err.message || 'Run failed');
    } finally {
      btn.disabled = false;
    }
  });

  document.getElementById('statusPill')?.addEventListener('click', () => {
    const errors = currentErrors();
    if (errors.length) goToFirstError(errors);
    else goToSection('review');
  });

  document.getElementById('railMappingJump')?.addEventListener('click', () => {
    goToSection('codesign');
    document.getElementById('hwStatsPanel')?.scrollIntoView({ block: 'center' });
  });

  document.addEventListener('wizard:go-first-error', () => {
    goToFirstError(currentErrors());
  });

  document.getElementById('saveTemplateBtn')?.addEventListener('click', async () => {
    const name = window.prompt('Template name:', state.draft.experiment_name || 'config');
    if (!name || !name.trim()) return;
    const res = await fetch('/api/templates', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name.trim(), config: state.draft }),
    });
    if (res.ok) await loadTemplateOptions();
    alert(res.ok ? 'Template saved.' : 'Failed to save template');
  });

  document.getElementById('resetBtn')?.addEventListener('click', async () => {
    if (!window.confirm('Reset to the baseline pipeline?')) return;
    await seedStarterDraft();
    window.history.replaceState({}, '', '/wizard');
    state.editContinueRunId = null;
    state.prevCompleted = null;
    await ensureModelSchema(state.draft.deployment_parameters?.model_type);
    renderAll();
    scheduleResolve();
    scheduleHwVerify();
  });

  document.getElementById('templateSelect')?.addEventListener('change', (event) => {
    const id = event.target.value;
    if (id) window.location.href = '/wizard?template_id=' + encodeURIComponent(id);
  });

  document.getElementById('copyJsonBtn')?.addEventListener('click', () => {
    navigator.clipboard.writeText(JSON.stringify(emittedConfig(), null, 2));
  });

  document.getElementById('downloadJsonBtn')?.addEventListener('click', () => {
    const anchor = document.createElement('a');
    anchor.href = URL.createObjectURL(new Blob(
      [JSON.stringify(emittedConfig(), null, 2)], { type: 'application/json' },
    ));
    anchor.download = (state.draft.experiment_name || 'config') + '.json';
    anchor.click();
  });
}

/* ── Change plumbing ───────────────────────────────────────────────────── */

function bindChangeEvents() {
  document.addEventListener('wizard:change', async (event) => {
    const key = event.detail?.key;
    if (key === 'model_type') {
      const modelType = state.draft.deployment_parameters?.model_type;
      await ensureModelSchema(modelType);
      /* The previous builder's config would not even build — reseed from the
         new builder's served defaults. */
      if (modelType) {
        state.draft.deployment_parameters.model_config = defaultModelConfig(modelType);
      }
    }
    if (key === 'model_config_mode' || key === 'hw_config_mode') {
      /* Entering a search mode: arch_search must exist (the resolve demands
         a declaration); seed the served schema's first optimizer. */
      const dp = state.draft.deployment_parameters;
      const searchActive = dp?.model_config_mode === 'search' || dp?.hw_config_mode === 'search';
      const optimizers = (schema().nas || {}).optimizer_options || [];
      if (searchActive && !dp.arch_search && optimizers.length) {
        dp.arch_search = { optimizer: optimizers[0].id };
      }
    }
    renderGroupHosts();
    renderJsonPreview();
    scheduleResolve();
    scheduleHwVerify();
    if (key === 'data_provider_name' || key === 'preprocessing') renderDatasetFacts();
  });
  document.addEventListener('wizard:rerender', () => {
    renderAll();
    scheduleResolve();
    scheduleHwVerify();
  });
}

/* ── Boot ──────────────────────────────────────────────────────────────── */

async function loadDynamicOptions() {
  /* Which keys are registry-served (and from where) comes from the schema
     payload itself — no endpoint knowledge is hardcoded here. */
  const endpoints = schema().dynamic_options || {};
  await Promise.all(Object.entries(endpoints).map(async ([key, url]) => {
    const items = await fetch(url).then((r) => (r.ok ? r.json() : [])).catch(() => []);
    state.dynamicOptions[key] = items.map(
      (item) => ({ id: item.id, label: item.label || item.id }),
    );
  }));
}

async function loadTemplateOptions() {
  const select = document.getElementById('templateSelect');
  if (!select) return;
  const templates = await fetch('/api/templates')
    .then((r) => (r.ok ? r.json() : [])).catch(() => []);
  while (select.options.length > 1) select.remove(1);
  for (const template of templates) {
    select.append(new Option(template.name || template.id, template.id));
  }
  select.style.display = templates.length ? '' : 'none';
}

/** Seed the fresh wizard with the server's ready-to-launch baseline draft
    (the fresh-state contract); an empty draft is the offline fallback. */
async function seedStarterDraft() {
  const starter = await fetch('/api/config/starter')
    .then((r) => (r.ok ? r.json() : null)).catch(() => null);
  if (starter) loadDraftFromConfig(starter);
  else resetDraft();
}

async function loadFromUrlParams() {
  const params = new URLSearchParams(window.location.search);
  const runId = params.get('run_id');
  const templateId = params.get('template_id');
  if (!runId && !templateId) {
    await seedStarterDraft();
    return;
  }
  if (runId) {
    const config = await fetch('/api/runs/' + encodeURIComponent(runId) + '/config')
      .then((r) => (r.ok ? r.json() : null)).catch(() => null);
    if (config) {
      loadDraftFromConfig(config);
      state.draft._continue_from_run_id = runId;
      state.editContinueRunId = runId;
      document.body.classList.add('edit-continue-mode');
      const pipeline = await fetch('/api/runs/' + encodeURIComponent(runId) + '/pipeline')
        .then((r) => (r.ok ? r.json() : null)).catch(() => null);
      if (pipeline && pipeline.steps) {
        state.prevCompleted = new Set(
          pipeline.steps.filter((s) => s.status === 'completed').map((s) => s.name),
        );
      }
    } else {
      await seedStarterDraft();
    }
  } else if (templateId) {
    const config = await fetch('/api/templates/' + encodeURIComponent(templateId))
      .then((r) => (r.ok ? r.json() : null)).catch(() => null);
    if (config) loadDraftFromConfig(config, { templateName: templateId });
    else await seedStarterDraft();
  }
}

async function boot() {
  await loadSchema();
  installStructuredWidgets();
  await loadDynamicOptions();
  await loadFromUrlParams();
  await ensureModelSchema(state.draft.deployment_parameters?.model_type);
  bindActions();
  bindChangeEvents();
  goToSection('codesign');
  loadTemplateOptions();
  renderAll();
  scheduleResolve();
  scheduleHwVerify();
}

boot().catch((err) => {
  console.error('Wizard boot failed', err);
  const wizard = document.getElementById('wizard');
  if (wizard) {
    wizard.prepend(el('div', 'global-errors', 'Wizard failed to load: ' + err.message));
  }
});
