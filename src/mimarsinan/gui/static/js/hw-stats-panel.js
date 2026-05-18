/* Shared "Mapping Performance" panel renderer.
 *
 * Single source of truth for the panel rendered on:
 *   - the wizard's `#hwStatsPanel` (loaded as a non-module script)
 *   - the monitor's Model Building "Mapping" tab and HCM "Hardware"
 *     tab (loaded as an ES module)
 *
 * Dual-mode export: ESM `export` statements at the bottom for module
 * consumers, plus an assignment to `window.HwStatsPanel` so the
 * wizard (which is still a classic script using inline HTML
 * `onclick="..."` handlers, and therefore can't be `type="module"`)
 * can call the same functions. Both consumers see one implementation.
 *
 * Input stats shape matches LayoutVerificationStats.to_dict() with
 * optional layout_preview / per_segment_passes extras.
 *
 * `mode` indicates the data source:
 *   - "planned"  : stats came from a layout/verify pass on the built
 *                  model against the resolved core_types.
 *   - "real"     : stats came from the real HardCoreMapping cores.
 */

// Inlined `esc` (instead of importing from util.js) so this file has
// no ES-module dependencies and can be loaded as a classic script
// in wizard.html. util.js is a module and can't be imported by a
// classic script.
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function _fmt(v) { return v != null ? Number(v).toFixed(1) : '—'; }
function _fmtInt(v) { return v != null ? String(Math.round(Number(v))) : '—'; }
function _fmtNum(v) {
  if (v == null) return '—';
  const n = Number(v);
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

function _barColor(pct) {
  if (pct >= 70) return 'green';
  if (pct >= 40) return 'amber';
  return 'rose';
}
function _wasteBarColor(pct) {
  if (pct <= 30) return 'green';
  if (pct <= 60) return 'amber';
  return 'rose';
}

function _healthBar(label, pct, colorFn) {
  const p = Math.max(0, Math.min(100, pct != null ? pct : 0));
  const cls = colorFn(p);
  return '<div class="hw-health-bar">'
    + '<span class="hw-health-bar-label">' + esc(label) + '</span>'
    + '<div class="hw-health-bar-track"><div class="hw-health-bar-fill ' + cls + '" style="width:' + p.toFixed(1) + '%"></div></div>'
    + '<span class="hw-health-bar-value">' + _fmt(pct) + '%</span>'
    + '</div>';
}

function _miniBar(pct, colorFn) {
  const p = Math.max(0, Math.min(100, pct != null ? pct : 0));
  const cls = colorFn(p);
  return '<div class="hw-per-core-cell">'
    + '<div class="hw-per-core-bar-row">'
    + '<div class="hw-per-core-track"><div class="hw-per-core-fill ' + cls + '" style="width:' + p.toFixed(1) + '%"></div></div>'
    + '<span class="hw-per-core-value">' + _fmt(pct) + '%</span>'
    + '</div></div>';
}

function _perCoreRow(label, min, avg, max, colorFn) {
  return '<div class="hw-per-core-label">' + esc(label) + '</div>'
    + _miniBar(min, colorFn) + _miniBar(avg, colorFn) + _miniBar(max, colorFn);
}

function _detailChip(lbl, val) {
  return '<span class="hw-detail-chip"><span class="hw-detail-chip-lbl">'
    + esc(lbl) + '</span>' + esc(_fmtNum(val)) + '</span>';
}

function _renderLayoutPreview(preview) {
  if (!preview || !Array.isArray(preview.flow) || preview.flow.length === 0) return '';
  const hasScheduling = preview.schedule_sync_count > 0;
  const items = preview.flow.map(item => {
    if (item.kind === 'input' || item.kind === 'output') {
      return '<div class="hw-layout-mini-endcap ' + item.kind + '">'
        + '<span>' + item.kind.toUpperCase() + '</span></div>';
    }
    if (item.kind === 'host') {
      if (item.schedule_sync) {
        return '<div class="hw-layout-mini-host hw-layout-mini-schedule-sync">'
          + '<div class="hw-layout-mini-host-box">'
          + '<div class="hw-layout-mini-host-label">↻ sync</div>'
          + '<div class="hw-layout-mini-host-sub">pass</div>'
          + '</div></div>';
      }
      return '<div class="hw-layout-mini-host">'
        + '<div class="hw-layout-mini-host-box">'
        + '<div class="hw-layout-mini-host-label">' + _fmtInt(item.compute_op_count) + ' ops</div>'
        + '<div class="hw-layout-mini-host-sub">host</div>'
        + '</div></div>';
    }
    if (item.kind === 'neural') {
      return '<div class="hw-layout-mini-lat-group">'
        + '<div class="hw-layout-mini-lat-label">' + _fmtInt(item.latency_group_index) + '</div>'
        + '<div class="hw-layout-mini-lat-bar">' + _fmtInt(item.softcore_count) + '</div>'
        + '</div>';
    }
    return '';
  }).filter(Boolean);

  const flowHtml = items.map((itemHtml, idx) => {
    if (idx === items.length - 1) return itemHtml;
    return itemHtml + '<div class="hw-layout-mini-arrow">→</div>';
  }).join('');

  const title = hasScheduling ? 'Mapping Miniview (Scheduled)' : 'Mapping Miniview';
  return '<div class="hw-layout-mini-wrap">'
    + '<div class="hw-stats-section-label">' + title + '</div>'
    + '<div class="hw-layout-mini-flow">' + flowHtml + '</div>'
    + '</div>';
}

/**
 * Build the inner HTML for a Mapping Performance panel.
 *
 * `stats` is a LayoutVerificationStats.to_dict()-shaped dict (the same
 * shape the wizard's POST /api/hw_config_verify returns under `stats`).
 * `mode` is "planned" | "real".
 * `note` (optional) is a short caption appended next to the title.
 */
export function buildHwStatsPanelHtml(stats, mode = 'planned', note = null) {
  const modeLabel = mode === 'real' ? 'Real Mapping' : 'Planned Mapping';
  if (!stats) {
    return '<div class="hw-stats-header">'
      + '<span class="hw-stats-title">Mapping Performance</span>'
      + '<span class="hw-stats-badge empty">' + esc(modeLabel) + '</span></div>'
      + '<div class="hw-stats-empty-msg">No mapping statistics available.</div>';
  }
  if (stats.feasible === false) {
    return '<div class="hw-stats-header">'
      + '<span class="hw-stats-title">Mapping Performance</span>'
      + '<span class="hw-stats-badge error">Infeasible</span></div>'
      + '<div class="hw-stats-empty-msg">Hardware configuration cannot fit all soft cores.</div>';
  }

  let scheduleBadge = '';
  if (stats.schedule_pass_count && stats.schedule_pass_count > 1) scheduleBadge = ' • Scheduled';

  let html =
    '<div class="hw-stats-header">'
    + '<span class="hw-stats-title">Mapping Performance</span>'
    + '<span class="hw-stats-badge ok">' + esc(modeLabel) + scheduleBadge + '</span>'
    + (note ? '<span class="hw-stats-note" style="margin-left:8px;color:var(--text-muted);font-size:11px">' + esc(note) + '</span>' : '')
    + '</div>';

  const segmentCard = (stats.neural_segment_count > 0)
    ? '<div class="hw-stat-card"><div class="hw-stat-card-value">' + _fmtInt(stats.neural_segment_count) + '</div><div class="hw-stat-card-label">Neural Segments</div></div>'
    : '';
  const scheduleSyncs = stats.schedule_sync_count
    || (stats.layout_preview && stats.layout_preview.schedule_sync_count)
    || 0;
  const totalBarriers = (stats.host_side_segment_count || 0) + scheduleSyncs;
  const syncBarrierCard = (totalBarriers > 0)
    ? '<div class="hw-stat-card"><div class="hw-stat-card-value">' + _fmtInt(totalBarriers) + '</div><div class="hw-stat-card-label">Sync Barriers</div></div>'
    : '';
  const coresLabel = (stats.schedule_pass_count > 1) ? 'Cores / Pass' : 'Cores Used';

  html += '<div class="hw-stats-cards">'
    + '<div class="hw-stat-card"><div class="hw-stat-card-value">' + _fmtInt(stats.total_cores) + '</div><div class="hw-stat-card-label">' + coresLabel + '</div></div>'
    + '<div class="hw-stat-card"><div class="hw-stat-card-value">' + _fmtInt(stats.total_softcores) + '</div><div class="hw-stat-card-label">Softcores</div></div>'
    + segmentCard + syncBarrierCard
    + '</div>';

  html += '<div class="hw-stats-body">';
  html += '<div>'
    + '<div class="hw-stats-section-label">Total</div>'
    + '<div class="hw-stats-bars">'
    + _healthBar('Param Utilization', stats.mapped_params_pct, _barColor)
    + _healthBar('Wasted Axons', stats.total_wasted_axons_pct, _wasteBarColor)
    + _healthBar('Wasted Neurons', stats.total_wasted_neurons_pct, _wasteBarColor)
    + '</div>'
    + '</div>';

  html += '<div>'
    + '<div class="hw-stats-section-label">Per-Core</div>'
    + '<div class="hw-per-core-grid">'
    + '<div></div>'
    + '<div class="hw-per-core-header">Min</div>'
    + '<div class="hw-per-core-header">Avg</div>'
    + '<div class="hw-per-core-header">Max</div>'
    + _perCoreRow('Wasted Axons',
        stats.per_core_wasted_axons_pct_min,
        stats.per_core_wasted_axons_pct_avg,
        stats.per_core_wasted_axons_pct_max,
        _wasteBarColor)
    + _perCoreRow('Wasted Neurons',
        stats.per_core_wasted_neurons_pct_min,
        stats.per_core_wasted_neurons_pct_avg,
        stats.per_core_wasted_neurons_pct_max,
        _wasteBarColor)
    + _perCoreRow('Param Usage',
        stats.per_core_mapped_params_pct_min,
        stats.per_core_mapped_params_pct_avg,
        stats.per_core_mapped_params_pct_max,
        _barColor)
    + '</div>'
    + '</div>';
  html += '</div>'; // end hw-stats-body

  let detailRowsHtml = '';

  if (stats.coalescing_group_count > 0) {
    detailRowsHtml +=
      '<div class="hw-detail-row">'
      + '<span class="hw-detail-title">Coalescing</span>'
      + '<span class="hw-detail-count">' + _fmtInt(stats.coalescing_group_count) + ' groups</span>'
      + '<span class="hw-detail-stat-label">Frags/group:</span>'
      + _detailChip('Min', stats.coalescing_frags_per_group_min)
      + _detailChip('Mdn', stats.coalescing_frags_per_group_median)
      + _detailChip('Max', stats.coalescing_frags_per_group_max)
      + '</div>';
  }

  if (stats.neural_segment_count > 0) {
    detailRowsHtml +=
      '<div class="hw-detail-row">'
      + '<span class="hw-detail-title">Segment Latency</span>'
      + '<span class="hw-detail-count">' + _fmtInt(stats.neural_segment_count) + ' segments</span>'
      + '<span class="hw-detail-stat-label">Latency groups/segment:</span>'
      + _detailChip('Min', stats.segment_latency_min)
      + _detailChip('Mdn', stats.segment_latency_median)
      + _detailChip('Max', stats.segment_latency_max)
      + '</div>';
  }

  if (stats.schedule_pass_count > 1) {
    let perSegDetail = '';
    if (stats.per_segment_passes) {
      const segEntries = Object.keys(stats.per_segment_passes).sort((a, b) => a - b);
      perSegDetail = segEntries.map(sid =>
        'seg ' + sid + ': ' + stats.per_segment_passes[sid] + 'p').join(', ');
    }
    detailRowsHtml +=
      '<div class="hw-detail-row">'
      + '<span class="hw-detail-title">Scheduling</span>'
      + '<span class="hw-detail-count">' + _fmtInt(stats.schedule_pass_count) + ' total passes</span>'
      + '<span class="hw-detail-stat-label">' + (perSegDetail || 'Cores reused between passes.') + '</span>'
      + '</div>';
  }

  if (stats.split_softcore_count > 0) {
    detailRowsHtml +=
      '<div class="hw-detail-row">'
      + '<span class="hw-detail-title">Splitting</span>'
      + '<span class="hw-detail-count">' + _fmtInt(stats.split_softcore_count) + ' softcores split</span>'
      + '<span class="hw-detail-stat-label">Splits/SC:</span>'
      + _detailChip('Min', stats.splits_per_softcore_min)
      + _detailChip('Mdn', stats.splits_per_softcore_median)
      + _detailChip('Max', stats.splits_per_softcore_max)
      + '</div>';
  }

  const layoutPreviewHtml = _renderLayoutPreview(stats.layout_preview);
  if (detailRowsHtml || layoutPreviewHtml) {
    html += '<div class="hw-stats-bottom">';
    html += '<div class="hw-stats-bottom-left">' + detailRowsHtml + '</div>';
    if (layoutPreviewHtml) html += '<div class="hw-stats-bottom-right">' + layoutPreviewHtml + '</div>';
    html += '</div>';
  }
  return html;
}

/** Convenience: render into an existing element (creating a wrapper). */
export function renderHwStatsInto(targetEl, stats, mode = 'planned', note = null) {
  if (!targetEl) return;
  const wrap = document.createElement('div');
  wrap.className = 'hw-stats-panel';
  wrap.innerHTML = buildHwStatsPanelHtml(stats, mode, note);
  targetEl.appendChild(wrap);
}

/* Dual-mode shim: also attach to ``window`` for the wizard's classic-
   script consumers. ESM `export` statements above remain authoritative
   for module consumers (monitor pages). */
if (typeof window !== 'undefined') {
  window.HwStatsPanel = Object.assign(window.HwStatsPanel || {}, {
    buildHwStatsPanelHtml,
    renderHwStatsInto,
  });
}
