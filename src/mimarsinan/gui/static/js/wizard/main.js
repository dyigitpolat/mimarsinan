/* Wizard controller: schema-driven rendering, resolve loop, deploy actions.
   The draft IS the config document; every widget renders from /api/config_schema. */

import { editableKeys, groups, keySchema, loadSchema, relevant } from './schema.js';
import {
  effectiveConfig, isExplicit, loadDraftFromConfig, resetDraft, state,
} from './state.js';
import { el, renderField } from './fields.js';
import { ensureModelSchema, installStructuredWidgets } from './structured.js';
import {
  bindStepsBar, renderDerivedChips, renderDiffPanel, renderErrors,
  renderJsonPreview, renderStepsBar, renderTemplateBanner, renderUnknownTray,
} from './review.js';
import { autoSuggestHardware, scheduleHwVerify } from './hw.js';

const ALTITUDES = {
  intent: ['run', 'workload', 'spiking', 'conversion', 'training', 'tuning'],
  platform: ['hardware', 'deployment_target'],
};

const GROUP_ICONS = {
  run: '⬡', workload: '◈', spiking: '⚡', conversion: '◎',
  tuning: '☈', training: '▶', hardware: '▦', deployment_target: '✈',
};

/* ── Section rendering ─────────────────────────────────────────────────── */

function fieldList(groupId, category, cfg) {
  return editableKeys(groupId, category)
    .map((key) => keySchema(key))
    .filter((ks) => relevant(ks.relevant, cfg) || isExplicit(ks.key));
}

function renderGroupSection(group, cfg) {
  const basic = fieldList(group.id, 'basic', cfg);
  const advanced = fieldList(group.id, 'advanced', cfg);
  if (!basic.length && !advanced.length) return null;

  const section = el('div', 'section open');
  section.dataset.section = group.id;

  const header = el('div', 'section-header');
  header.append(el('div', 'section-icon', GROUP_ICONS[group.id] || '◇'));
  const titles = el('div', 'section-title-group');
  titles.append(el('div', 'section-title', group.title));
  titles.append(el('div', 'section-subtitle', group.subtitle || ''));
  header.append(titles, el('span', 'section-chevron', '▾'));
  header.addEventListener('click', () => section.classList.toggle('open'));
  section.append(header);

  const body = el('div', 'section-body');
  const grid = el('div', 'field-grid cols-2');
  for (const ks of basic) grid.append(renderField(ks));
  body.append(grid);

  if (group.id === 'hardware') {
    const suggest = el('button', 'btn-sm', 'Suggest hardware for this model');
    suggest.type = 'button';
    suggest.addEventListener('click', () => autoSuggestHardware());
    body.append(suggest);
  }

  if (advanced.length) {
    const hasExplicit = advanced.some((ks) => isExplicit(ks.key));
    const drawer = el('div', 'advanced-drawer' + (hasExplicit ? ' open' : ''));
    const toggle = el('button', 'advanced-toggle', `⚙ Advanced (${advanced.length})`);
    toggle.type = 'button';
    toggle.addEventListener('click', () => drawer.classList.toggle('open'));
    const drawerBody = el('div', 'advanced-drawer-body field-grid cols-2');
    for (const ks of advanced) drawerBody.append(renderField(ks));
    drawer.append(drawerBody);
    body.append(toggle, drawer);
  }

  section.append(body);
  return section;
}

function renderAltitudes() {
  const cfg = effectiveConfig();
  for (const [altitude, groupIds] of Object.entries(ALTITUDES)) {
    const host = document.getElementById('altitude-' + altitude);
    if (!host) continue;
    host.replaceChildren();
    for (const group of groups()) {
      if (!groupIds.includes(group.id)) continue;
      const section = renderGroupSection(group, cfg);
      if (section) host.append(section);
    }
  }
}

function renderAll() {
  renderAltitudes();
  renderTemplateBanner();
  renderDerivedChips();
  renderDiffPanel();
  renderUnknownTray();
  renderStepsBar();
  renderErrors();
  renderJsonPreview();
}

/* ── Resolve loop ──────────────────────────────────────────────────────── */

let _resolveTimer = null;
let _resolveInFlight = false;
let _resolveQueued = false;

function scheduleResolve() {
  if (_resolveTimer) clearTimeout(_resolveTimer);
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
      renderDerivedChips();
      renderDiffPanel();
      renderUnknownTray();
      renderStepsBar();
      renderErrors();
    }
  } catch (_) { /* transient network failure; next change retries */ }
  _resolveInFlight = false;
  if (_resolveQueued) { _resolveQueued = false; scheduleResolve(); }
}

/* ── Actions ───────────────────────────────────────────────────────────── */

function bindActions() {
  document.getElementById('runBtn')?.addEventListener('click', async () => {
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

  document.getElementById('saveTemplateBtn')?.addEventListener('click', async () => {
    const name = window.prompt('Template name:', state.draft.experiment_name || 'config');
    if (!name || !name.trim()) return;
    const res = await fetch('/api/templates', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name.trim(), config: state.draft }),
    });
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

  const showJson = document.getElementById('showJsonToggle');
  showJson?.addEventListener('click', () => {
    showJson.classList.toggle('on');
    const on = showJson.classList.contains('on');
    document.getElementById('appShell').classList.toggle('show-json', on);
    document.getElementById('copyJsonBtn').style.display = on ? '' : 'none';
    document.getElementById('downloadJsonBtn').style.display = on ? '' : 'none';
  });

  document.getElementById('copyJsonBtn')?.addEventListener('click', () => {
    navigator.clipboard.writeText(JSON.stringify(state.draft, null, 2));
  });

  document.getElementById('downloadJsonBtn')?.addEventListener('click', () => {
    const anchor = document.createElement('a');
    anchor.href = URL.createObjectURL(new Blob(
      [JSON.stringify(state.draft, null, 2)], { type: 'application/json' },
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
      await ensureModelSchema(state.draft.deployment_parameters?.model_type);
    }
    renderAltitudes();
    renderJsonPreview();
    scheduleResolve();
    scheduleHwVerify();
  });
  document.addEventListener('wizard:rerender', () => {
    renderAll();
    scheduleResolve();
    scheduleHwVerify();
  });
}

/* ── Boot ──────────────────────────────────────────────────────────────── */

async function loadDynamicOptions() {
  const [providers, modelTypes] = await Promise.all([
    fetch('/api/data_providers').then((r) => r.json()).catch(() => []),
    fetch('/api/model_types').then((r) => r.json()).catch(() => []),
  ]);
  state.dynamicOptions.data_provider_name = providers.map(
    (p) => ({ id: p.id, label: p.label }),
  );
  state.dynamicOptions.model_type = modelTypes.map(
    (m) => ({ id: m.id, label: m.label || m.id }),
  );
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
  bindStepsBar();
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
