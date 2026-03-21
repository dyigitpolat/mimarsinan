/* Mimarsinan Welcome Page — modern active-run monitoring, searchable runs, templates. */

const _esc = (s) => { const d = document.createElement('div'); d.textContent = String(s); return d.innerHTML; };

let allRuns = [];
let allTemplates = [];
let allActiveRuns = [];
let _activeTimer = null;
let _pollMs = 4000;

// ── Tab switching ────────────────────────────────────────────────────────
document.getElementById('w-tab-bar').addEventListener('click', (e) => {
  const btn = e.target.closest('.w-tab-btn');
  if (!btn) return;
  const tab = btn.dataset.tab;
  document.querySelectorAll('.w-tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.w-tab-panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + tab));
});

// ── Search ───────────────────────────────────────────────────────────────
document.getElementById('runs-search').addEventListener('input', () => renderRuns());
document.getElementById('tpl-search').addEventListener('input', () => renderTemplates());

// ── Init ─────────────────────────────────────────────────────────────────
(async function init() {
  await Promise.all([loadRuns(), loadTemplates(), loadActiveRuns()]);
  updateStats();
  startActivePolling();
})();

// ── Data loading ─────────────────────────────────────────────────────────
async function loadRuns() {
  try { allRuns = await (await fetch('/api/runs?include_steps=true')).json(); } catch { allRuns = []; }
  renderRuns();
}

async function loadTemplates() {
  try { allTemplates = await (await fetch('/api/templates')).json(); } catch { allTemplates = []; }
  renderTemplates();
}

async function loadActiveRuns() {
  try { allActiveRuns = await (await fetch('/api/active_runs')).json(); } catch { allActiveRuns = []; }
  renderActiveRuns();
}

function startActivePolling() {
  if (_activeTimer) clearInterval(_activeTimer);
  const hasActive = allActiveRuns.some(r => r.is_alive);
  _pollMs = hasActive ? 2000 : 5000;
  _activeTimer = setInterval(async () => {
    await loadActiveRuns();
    updateStats();
    const nowActive = allActiveRuns.some(r => r.is_alive);
    if (nowActive !== hasActive) {
      clearInterval(_activeTimer);
      startActivePolling();
    }
  }, _pollMs);
}

// ── Stats ────────────────────────────────────────────────────────────────
function updateStats() {
  const el = document.getElementById('w-stats');
  const active = allActiveRuns.filter(r => r.is_alive).length;
  el.innerHTML = `
    <div class="w-stat"><div class="w-stat-value active-val">${active}</div><div class="w-stat-label">Active</div></div>
    <div class="w-stat"><div class="w-stat-value runs-val">${allRuns.length}</div><div class="w-stat-label">Past Runs</div></div>
    <div class="w-stat"><div class="w-stat-value tpl-val">${allTemplates.length}</div><div class="w-stat-label">Templates</div></div>
  `;
  document.getElementById('tab-count-active').textContent = active;
  document.getElementById('tab-count-runs').textContent = allRuns.length;
  document.getElementById('tab-count-tpl').textContent = allTemplates.length;
}

// ── Active runs ──────────────────────────────────────────────────────────
function renderActiveRuns() {
  const container = document.getElementById('active-list');

  if (allActiveRuns.length === 0) {
    container.innerHTML = '<div class="w-empty">No active experiments. <a href="/wizard">Start one &rarr;</a></div>';
    return;
  }

  // If there's an empty-state placeholder, clear it before first insert.
  if (container.querySelector('.w-empty')) container.innerHTML = '';

  const currentIds = new Set(allActiveRuns.map(r => r.run_id));

  // Remove stale cards.
  for (const el of [...container.querySelectorAll('.w-active-card[data-run-id]')]) {
    if (!currentIds.has(el.dataset.runId)) el.remove();
  }

  // Insert new or update existing, maintaining API order.
  allActiveRuns.forEach((run, idx) => {
    let el = container.querySelector(`.w-active-card[data-run-id="${CSS.escape(run.run_id)}"]`);
    if (!el) {
      el = _buildActiveCard(run, idx);
      container.appendChild(el);
    } else {
      _patchActiveCard(el, run);
    }
  });

  // Reorder nodes to match API order without re-inserting.
  allActiveRuns.forEach((run, idx) => {
    const el = container.querySelector(`.w-active-card[data-run-id="${CSS.escape(run.run_id)}"]`);
    if (el && container.children[idx] !== el) container.insertBefore(el, container.children[idx] || null);
  });

  requestAnimationFrame(() => drawSparklines());
}

function _buildActiveCard(run, idx) {
  const el = document.createElement('div');
  el.className = 'w-active-card w-active-card--enter';
  el.dataset.runId = run.run_id;
  el.style.animationDelay = (idx * 60) + 'ms';
  el.addEventListener('animationend', () => {
    el.classList.remove('w-active-card--enter');
    el.style.animationDelay = '';
  }, { once: true });
  el.innerHTML = _activeCardInnerHTML(run);
  return el;
}

function _patchActiveCard(el, run) {
  const status = run.is_alive ? 'running' : (run.status || 'unknown');
  const pct = Math.round((run.progress || 0) * 100);
  const elapsed = run.started_at ? formatElapsed(Date.now() / 1000 - run.started_at) : '';
  const completedCount = run.completed_steps || 0;
  const totalCount = run.total_steps || (run.step_names || []).length || 0;

  const statusEl = el.querySelector('.w-active-status');
  if (statusEl && (statusEl.textContent !== status || !statusEl.classList.contains(status))) {
    statusEl.className = `w-active-status ${status}`;
    statusEl.textContent = status;
  }

  const elapsedEl = el.querySelector('.w-active-elapsed');
  if (elapsedEl && elapsedEl.textContent !== elapsed) elapsedEl.textContent = elapsed;

  const fill = el.querySelector('.w-progress-fill');
  if (fill) fill.style.width = pct + '%';

  // Mini pipeline: update step status classes.
  const miniSteps = el.querySelectorAll('.w-mini-step');
  const stepsInfo = run.steps || {};
  const stepNames = run.step_names || [];
  miniSteps.forEach((stepEl, i) => {
    const sn = stepNames[i];
    if (!sn) return;
    const sd = stepsInfo[sn] || {};
    let st = sd.status || 'pending';
    if (st === 'pending' && sd.end_time != null) st = 'completed';
    if (st === 'running' && !run.is_alive) st = 'failed';
    stepEl.className = `w-mini-step ${st}`;
  });

  // Info section: metric, steps count, current step, ETA.
  const info = el.querySelector('.w-active-info');
  if (info) {
    const lastMetric = _lastTargetMetric(run);
    const metricHtml = lastMetric != null
      ? `<span class="w-metric-big">${lastMetric.toFixed(4)}</span><span>target</span>`
      : '';
    const eta = _estimateRemaining(run);
    const etaHtml = eta != null ? `<span>~${formatElapsed(eta)} remaining</span>` : '';
    info.innerHTML = `${metricHtml}
      <span>${completedCount}/${totalCount} steps</span>
      ${run.current_step ? `<span>Step: ${_esc(run.current_step)}</span>` : ''}
      ${etaHtml}`;
  }

  // Stop button: add if now alive and missing, remove if no longer alive.
  const actions = el.querySelector('.w-active-actions');
  if (actions) {
    const stopBtn = actions.querySelector('.w-card-btn.danger');
    if (run.is_alive && !stopBtn) {
      const btn = document.createElement('button');
      btn.className = 'w-card-btn danger';
      btn.onclick = () => window.stopRun(run.run_id);
      btn.textContent = 'Stop';
      actions.appendChild(btn);
    } else if (!run.is_alive && stopBtn) {
      stopBtn.remove();
    }
  }
}

function _activeCardInnerHTML(run) {
  const status = run.is_alive ? 'running' : (run.status || 'unknown');
  const pct = Math.round((run.progress || 0) * 100);
  const elapsed = run.started_at ? formatElapsed(Date.now() / 1000 - run.started_at) : '';
  const stepNames = run.step_names || [];
  const stepsInfo = run.steps || {};
  const completedCount = run.completed_steps || 0;
  const totalCount = run.total_steps || stepNames.length || 0;

  const lastMetric = _lastTargetMetric(run);
  const metricHtml = lastMetric != null
    ? `<span class="w-metric-big">${lastMetric.toFixed(4)}</span><span>target</span>`
    : '';

  const eta = _estimateRemaining(run);
  const etaHtml = eta != null ? `<span>~${formatElapsed(eta)} remaining</span>` : '';

  let miniBar = '';
  if (stepNames.length > 0) {
    miniBar = '<div class="w-mini-pipeline">';
    for (const sn of stepNames) {
      const sd = stepsInfo[sn] || {};
      let st = sd.status || 'pending';
      if (st === 'pending' && sd.end_time != null) st = 'completed';
      if (st === 'running' && !run.is_alive) st = 'failed';
      const shortName = sn.length > 10 ? sn.substring(0, 9) + '\u2026' : sn;
      miniBar += `<div class="w-mini-step ${st}" title="${_esc(sn)}">${_esc(shortName)}</div>`;
    }
    miniBar += '</div>';
  }

  return `
    <div class="w-active-header">
      <h3 title="${_esc(run.run_id)}">${_esc(run.experiment_name || run.run_id)}</h3>
      <span class="w-active-status ${status}">${status}</span>
      <span class="w-active-elapsed">${elapsed}</span>
    </div>
    ${miniBar}
    <div class="w-progress-wrap">
      <div class="w-progress-fill" style="width:${pct}%"></div>
    </div>
    <div class="w-active-body">
      <div class="w-active-info">
        ${metricHtml}
        <span>${completedCount}/${totalCount} steps</span>
        ${run.current_step ? `<span>Step: ${_esc(run.current_step)}</span>` : ''}
        ${etaHtml}
      </div>
      <div class="w-active-sparkline-wrap" id="spark-${CSS.escape(run.run_id)}"></div>
      <div class="w-active-actions">
        <a href="/monitor?run_id=${encodeURIComponent(run.run_id)}" class="w-card-btn primary">View</a>
        ${run.is_alive ? `<button class="w-card-btn danger" onclick="stopRun('${_esc(run.run_id)}')">Stop</button>` : ''}
      </div>
    </div>`;
}

function _lastTargetMetric(run) {
  const tms = run.target_metrics;
  if (tms && tms.length > 0) return tms[tms.length - 1].value;
  return null;
}

function _estimateRemaining(run) {
  if (!run.is_alive || !run.started_at) return null;
  const completed = run.completed_steps || 0;
  const total = run.total_steps || 0;
  if (completed < 1 || total < 2) return null;
  const elapsedSec = Date.now() / 1000 - run.started_at;
  const avgPerStep = elapsedSec / completed;
  return avgPerStep * (total - completed);
}

function formatElapsed(secs) {
  if (!secs || secs < 0) secs = 0;
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

// Track sparkline data fingerprints to skip unchanged series.
const _sparkFingerprints = {};

function drawSparklines() {
  for (const run of allActiveRuns) {
    const wrap = document.getElementById('spark-' + CSS.escape(run.run_id));
    if (!wrap) continue;
    const values = (run.target_metrics || []).map(m => m.value);
    if (values.length < 2) {
      if (!wrap.querySelector('span')) {
        wrap.innerHTML = '<span style="font-size:11px;color:var(--text-muted)">No metrics yet</span>';
      }
      continue;
    }
    // Only redraw when series actually changes (length + last value).
    const fingerprint = values.length + ':' + values[values.length - 1];
    if (_sparkFingerprints[run.run_id] === fingerprint) continue;
    _sparkFingerprints[run.run_id] = fingerprint;

    const divId = 'sparkdiv-' + run.run_id.replace(/[^a-zA-Z0-9_-]/g, '_');
    if (!wrap.querySelector('#' + CSS.escape(divId))) {
      wrap.innerHTML = `<div id="${divId}" style="width:100%;height:100%"></div>`;
    }
    const el = document.getElementById(divId);
    if (!el) continue;
    try {
      Plotly.react(el, [{
        y: values,
        type: 'scatter', mode: 'lines',
        line: { color: '#22d3ee', width: 2 },
        fill: 'tozeroy', fillcolor: 'rgba(34,211,238,0.08)',
      }], {
        margin: { t: 0, r: 0, b: 0, l: 0 },
        xaxis: { visible: false },
        yaxis: { visible: false, range: [0, Math.max(1, ...values) * 1.05] },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        height: 40,
        showlegend: false,
      }, { displayModeBar: false, staticPlot: true });
    } catch { /* Plotly not loaded yet */ }
  }
}

window.stopRun = async function(runId) {
  if (!confirm('Stop this run?')) return;
  await fetch('/api/active_runs/' + encodeURIComponent(runId), { method: 'DELETE' });
  await loadActiveRuns();
  updateStats();
};

// ── Past runs ────────────────────────────────────────────────────────────
function renderRuns() {
  const grid = document.getElementById('runs-grid');
  const q = (document.getElementById('runs-search').value || '').toLowerCase().trim();
  const filtered = q ? allRuns.filter(r =>
    (r.experiment_name || r.run_id).toLowerCase().includes(q) ||
    (r.pipeline_mode || '').toLowerCase().includes(q)
  ) : allRuns;
  if (filtered.length === 0) {
    grid.innerHTML = q
      ? '<div class="w-empty">No runs match your search.</div>'
      : '<div class="w-empty">No past runs found. <a href="/wizard">Create one &rarr;</a></div>';
    return;
  }
  grid.innerHTML = filtered.map((run, i) => {
    const date = new Date(run.created_at * 1000).toLocaleString();
    const total = run.total_steps || 0;
    const done = run.completed_steps || 0;
    let stepBar = '';
    if (total > 0) {
      stepBar = '<div class="w-step-bar">';
      for (let s = 0; s < total; s++) {
        stepBar += `<div class="w-step-bar-seg ${s < done ? 'done' : 'todo'}"></div>`;
      }
      stepBar += '</div>';
    }
    return `
      <div class="w-card" style="animation-delay:${i * 40}ms">
        <h3 title="${_esc(run.run_id)}">${_esc(run.experiment_name || run.run_id)}</h3>
        <div class="w-card-meta">
          <span class="w-mode-badge">${_esc(run.pipeline_mode)}</span>
          <span>${date}</span>
          <span>${done}/${total} steps</span>
        </div>
        ${stepBar}
        <div class="w-card-actions">
          <a href="/monitor?run_id=${encodeURIComponent(run.run_id)}" class="w-card-btn">View</a>
          <a href="/wizard?run_id=${encodeURIComponent(run.run_id)}" class="w-card-btn">Edit &amp; Continue</a>
          <button class="w-card-btn" onclick="saveRunAsTemplate('${_esc(run.run_id)}')">Save as Template</button>
        </div>
      </div>`;
  }).join('');
}

window.saveRunAsTemplate = async function(runId) {
  try {
    const config = await (await fetch('/api/runs/' + encodeURIComponent(runId) + '/config')).json();
    if (!config || config.error) { alert('Could not load run config.'); return; }
    const name = prompt('Template name:', config.experiment_name || runId);
    if (!name) return;
    await fetch('/api/templates', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, config }),
    });
    await loadTemplates();
    updateStats();
  } catch (err) { alert('Failed to save template: ' + err.message); }
};

// ── Templates ────────────────────────────────────────────────────────────
function renderTemplates() {
  const grid = document.getElementById('templates-grid');
  const q = (document.getElementById('tpl-search').value || '').toLowerCase().trim();
  const filtered = q ? allTemplates.filter(t =>
    (t.name || t.id).toLowerCase().includes(q) ||
    (t.pipeline_mode || '').toLowerCase().includes(q)
  ) : allTemplates;
  if (filtered.length === 0) {
    grid.innerHTML = q
      ? '<div class="w-empty">No templates match your search.</div>'
      : '<div class="w-empty">No templates saved yet. <a href="/wizard">Create one &rarr;</a></div>';
    return;
  }
  grid.innerHTML = filtered.map((t, i) => {
    const date = new Date(t.created_at * 1000).toLocaleString();
    return `
      <div class="w-card" style="animation-delay:${i * 40}ms">
        <input class="w-tpl-name" value="${_esc(t.name || t.id)}" data-tpl-id="${_esc(t.id)}" data-orig="${_esc(t.name || t.id)}"
               onblur="renameTemplate(this)" onkeydown="if(event.key==='Enter')this.blur()">
        <div class="w-card-meta">
          <span class="w-mode-badge">${_esc(t.pipeline_mode || 'unknown')}</span>
          <span>${date}</span>
        </div>
        <div class="w-card-actions">
          <a href="/wizard?template_id=${encodeURIComponent(t.id)}" class="w-card-btn primary">Use Template</a>
          <button class="w-card-btn danger" onclick="deleteTemplate('${_esc(t.id)}')">Delete</button>
        </div>
      </div>`;
  }).join('');
}

window.renameTemplate = async function(input) {
  const newName = input.value.trim();
  const origName = input.dataset.orig;
  const tplId = input.dataset.tplId;
  if (!newName || newName === origName) { input.value = origName; return; }
  try {
    const config = await (await fetch('/api/templates/' + encodeURIComponent(tplId))).json();
    if (!config || config.error) { input.value = origName; return; }
    await fetch('/api/templates/' + encodeURIComponent(tplId), { method: 'DELETE' });
    await fetch('/api/templates', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newName, config }),
    });
    await loadTemplates();
    updateStats();
  } catch { input.value = origName; }
};

window.deleteTemplate = async function(id) {
  if (!confirm('Delete this template?')) return;
  await fetch('/api/templates/' + encodeURIComponent(id), { method: 'DELETE' });
  await loadTemplates();
  updateStats();
};
