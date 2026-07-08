/* Hardware feasibility: dataset metadata, verify (debounced), one-shot auto-suggest. */

import { buildHwStatsPanelHtml } from '../hw-stats-panel.js';
import { effectiveValue, setKey, state } from './state.js';
import { el, notifyChange } from './fields.js';

const _metadataCache = {};

export async function fetchMetadata() {
  const providerId = effectiveValue('data_provider_name');
  if (!providerId) return null;
  const pp = state.draft.deployment_parameters?.preprocessing || {};
  const params = new URLSearchParams();
  if (pp.resize_to) params.set('resize_to', String(pp.resize_to));
  if (pp.normalize) params.set('normalize', pp.normalize);
  if (pp.interpolation) params.set('interpolation', pp.interpolation);
  const url = `/api/data_providers/${encodeURIComponent(providerId)}/metadata`
    + (params.toString() ? '?' + params.toString() : '');
  if (_metadataCache[url]) return _metadataCache[url];
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const md = await res.json();
    if (md && !md.error) _metadataCache[url] = md;
    return _metadataCache[url] || null;
  } catch (_) {
    return null;
  }
}

/** True when the resolved provider metadata carries the workload facts the
    layout API requires — the framework never invents a dataset shape. */
export function hasWorkloadMetadata() {
  const md = state.metadata;
  return !!(md && Array.isArray(md.input_shape) && md.input_shape.length
    && Number.isFinite(md.num_classes));
}

function hwApiBody() {
  const dp = state.draft.deployment_parameters || {};
  const cores = state.draft.platform_constraints?.cores || [];
  const md = state.metadata;
  return {
    model_type: dp.model_type,
    input_shape: md.input_shape.slice(),
    num_classes: md.num_classes,
    model_config: dp.model_config || {},
    max_axons: cores.length ? Math.max(...cores.map((c) => c.max_axons || 1)) : 1024,
    max_neurons: cores.length ? Math.max(...cores.map((c) => c.max_neurons || 1)) : 1024,
    threshold_groups: 1,
    pruning_fraction: dp.pruning ? (dp.pruning_fraction || 0) : 0,
    threshold_seed: 0,
    allow_coalescing: !!effectiveValue('allow_coalescing'),
    hardware_bias: effectiveValue('has_bias') !== false,
    allow_neuron_splitting: !!effectiveValue('allow_neuron_splitting'),
    allow_scheduling: !!effectiveValue('allow_scheduling'),
    target_tq: effectiveValue('target_tq') || 32,
    encoding_layer_placement: dp.encoding_layer_placement || 'subsume',
  };
}

function showBanner(feasible, messages) {
  const banner = document.getElementById('hwValidationBanner');
  if (!banner) return;
  if (feasible === null) { banner.classList.add('hide'); return; }
  banner.classList.remove('hide');
  if (feasible) {
    banner.textContent = messages[0] || '✓ Hardware configuration is sufficient.';
    banner.className = 'hw-validation-banner ok';
  } else {
    banner.className = 'hw-validation-banner';
    banner.replaceChildren(el('strong', '', 'Hardware configuration issues:'));
    const list = el('ul');
    for (const message of messages) list.append(el('li', '', message));
    banner.append(list);
  }
}

function renderStats(stats, note) {
  const panel = document.getElementById('hwStatsPanel');
  if (!panel) return;
  if (!stats) {
    panel.className = 'hw-stats-panel hw-stats-state-empty';
    panel.innerHTML = '<div class="hw-stats-header">'
      + '<span class="hw-stats-title">Mapping Performance</span>'
      + `<span class="hw-stats-badge empty">${note || 'Not Verified'}</span></div>`;
    return;
  }
  panel.className = stats.feasible === false
    ? 'hw-stats-panel hw-stats-state-error' : 'hw-stats-panel';
  panel.innerHTML = buildHwStatsPanelHtml(stats, 'planned');
}

let _verifyTimer = null;

export function scheduleHwVerify() {
  if (_verifyTimer) clearTimeout(_verifyTimer);
  _verifyTimer = setTimeout(runHwVerify, 800);
}

async function runHwVerify() {
  const dp = state.draft.deployment_parameters || {};
  const searchActive = dp.model_config_mode === 'search' || dp.hw_config_mode === 'search';
  const cores = state.draft.platform_constraints?.cores || [];
  if (searchActive || !dp.model_type || !cores.length) {
    showBanner(null, []);
    renderStats(null, searchActive ? 'Search Mode' : 'Not Verified');
    return;
  }
  state.metadata = await fetchMetadata();
  if (!hasWorkloadMetadata()) {
    showBanner(null, []);
    renderStats(null, 'No Dataset Metadata');
    return;
  }
  const body = hwApiBody();
  try {
    const res = await fetch('/api/hw_config_verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_repr_json: body,
        core_types: cores.map((c) => ({
          max_axons: c.max_axons, max_neurons: c.max_neurons, count: c.count,
        })),
        allow_coalescing: body.allow_coalescing,
        allow_neuron_splitting: body.allow_neuron_splitting,
        allow_scheduling: body.allow_scheduling,
      }),
    });
    if (!res.ok) return;
    const data = await res.json();
    if (data.error) return;
    if (data.schedule_info && data.schedule_info.scheduled_feasible) {
      showBanner(true, [data.schedule_info.message]);
    } else {
      showBanner(data.feasible, data.errors || []);
    }
    renderStats(data.feasible ? data.stats : null,
      data.feasible ? null : 'Infeasible');
  } catch (_) { /* network errors: verification is advisory */ }
}

export async function autoSuggestHardware() {
  state.metadata = await fetchMetadata();
  if (!hasWorkloadMetadata()) {
    showBanner(false, ['Dataset metadata unavailable — pick a data provider first.']);
    return;
  }
  const res = await fetch('/api/hw_config_auto', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(hwApiBody()),
  });
  const data = await res.json();
  if (data.error || !data.core_types || !data.core_types.length) {
    showBanner(false, [data.error || data.rationale || 'Auto-config returned no core types.']);
    return;
  }
  const bias = effectiveValue('has_bias') !== false;
  setKey('cores', data.core_types.map((c) => ({
    max_axons: c.max_axons, max_neurons: c.max_neurons, count: c.count,
    has_bias: c.has_bias !== undefined ? c.has_bias : bias,
  })));
  showBanner(true, ['✓ Auto-configured: ' + (data.rationale || '')]);
  document.dispatchEvent(new CustomEvent('wizard:rerender'));
  notifyChange('cores');
}
