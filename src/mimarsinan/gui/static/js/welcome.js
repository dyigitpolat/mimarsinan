/* Mimarsinan Welcome Page */

const _esc = (s) => {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
};

let allRuns = [];
let allTemplates = [];
let allActiveRuns = [];
let _activeTimer = null;

// ── Tab switching ────────────────────────────────────────────────────────
document.getElementById('w-tab-bar').addEventListener('click', (e) => {
  const btn = e.target.closest('.w-tab-btn');
  if (!btn) return;
  const tab = btn.dataset.tab;
  document.querySelectorAll('.w-tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.w-tab-panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + tab));
});

// ── Init ─────────────────────────────────────────────────────────────────
(async function init() {
  await Promise.all([loadRuns(), loadTemplates(), loadActiveRuns()]);
  updateStats();
  startActivePolling();
})();

// ── Data loading ─────────────────────────────────────────────────────────
async function loadRuns() {
  try {
    allRuns = await (await fetch('/api/runs?include_steps=true')).json();
  } catch { allRuns = []; }
  renderRuns();
}

async function loadTemplates() {
  try {
    allTemplates = await (await fetch('/api/templates')).json();
  } catch { allTemplates = []; }
  renderTemplates();
}

async function loadActiveRuns() {
  try {
    allActiveRuns = await (await fetch('/api/active_runs')).json();
  } catch { allActiveRuns = []; }
  renderActiveRuns();
}

function startActivePolling() {
  if (_activeTimer) clearInterval(_activeTimer);
  _activeTimer = setInterval(async () => {
    await loadActiveRuns();
    updateStats();
  }, 4000);
}

// ── Stats ────────────────────────────────────────────────────────────────
function updateStats() {
  const el = document.getElementById('w-stats');
  const active = allActiveRuns.filter(r => r.is_alive).length;
  el.innerHTML = `
    <div class="w-stat"><div class="w-stat-value">${active}</div><div class="w-stat-label">Active</div></div>
    <div class="w-stat"><div class="w-stat-value">${allRuns.length}</div><div class="w-stat-label">Past Runs</div></div>
    <div class="w-stat"><div class="w-stat-value">${allTemplates.length}</div><div class="w-stat-label">Templates</div></div>
  `;
}

// ── Active runs ──────────────────────────────────────────────────────────
function renderActiveRuns() {
  const container = document.getElementById('active-list');
  if (allActiveRuns.length === 0) {
    container.innerHTML = '<div class="w-empty">No active experiments. <a href="/wizard" style="color:var(--accent-blue)">Start one</a>.</div>';
    return;
  }
  container.innerHTML = allActiveRuns.map(renderActiveCard).join('');
  drawSparklines();
}

function renderActiveCard(run) {
  const status = run.is_alive ? 'running' : (run.status || 'unknown');
  const pct = Math.round((run.progress || 0) * 100);
  const elapsed = run.started_at ? formatElapsed(Date.now() / 1000 - run.started_at) : '';
  const stepInfo = run.current_step ? `<span>Step: ${_esc(run.current_step)}</span>` : '';
  const progressInfo = `${run.completed_steps || 0}/${run.total_steps || '?'}`;
  return `
    <div class="w-active-card">
      <div class="w-active-header">
        <h3>${_esc(run.experiment_name || run.run_id)}</h3>
        <span class="w-active-status ${status}">${status}</span>
        <span class="w-active-elapsed">${elapsed}</span>
      </div>
      <div class="w-progress-wrap">
        <div class="w-progress-fill" style="width:${pct}%"></div>
      </div>
      <div class="w-active-bottom">
        <div class="w-active-metrics">
          ${stepInfo}
          <span>${progressInfo} steps</span>
        </div>
        <canvas class="w-active-sparkline" data-run-id="${_esc(run.run_id)}"></canvas>
        <div class="w-active-actions">
          <a href="/monitor?run_id=${encodeURIComponent(run.run_id)}" class="w-card-btn">View</a>
          ${run.is_alive ? `<button class="w-card-btn danger" onclick="stopRun('${_esc(run.run_id)}')">Stop</button>` : ''}
        </div>
      </div>
    </div>`;
}

function formatElapsed(secs) {
  if (secs < 0) secs = 0;
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function drawSparklines() {
  for (const run of allActiveRuns) {
    const canvas = document.querySelector(`.w-active-sparkline[data-run-id="${CSS.escape(run.run_id)}"]`);
    if (!canvas) continue;
    const ctx = canvas.getContext('2d');
    const values = (run.target_metrics || []).map(m => m.value);
    if (values.length < 2) { ctx.clearRect(0, 0, canvas.width, canvas.height); continue; }
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const min = Math.min(...values), max = Math.max(...values);
    const range = max - min || 1;
    ctx.beginPath();
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 1.5;
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = h - ((v - min) / range) * (h - 4) - 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
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
  if (allRuns.length === 0) {
    grid.innerHTML = '<div class="w-empty">No past runs found.</div>';
    return;
  }
  grid.innerHTML = allRuns.map(run => {
    const date = new Date(run.created_at * 1000).toLocaleString();
    const steps = run.total_steps ? `${run.completed_steps}/${run.total_steps} steps` : '';
    return `
      <div class="w-card">
        <h3>${_esc(run.experiment_name || run.run_id)}</h3>
        <div class="w-card-meta">
          <span>${_esc(run.pipeline_mode)}</span>
          <span>${date}</span>
          <span>${steps}</span>
        </div>
        <div class="w-card-actions">
          <a href="/monitor?run_id=${encodeURIComponent(run.run_id)}" class="w-card-btn">View</a>
          <a href="/wizard?run_id=${encodeURIComponent(run.run_id)}" class="w-card-btn">Edit & Continue</a>
        </div>
      </div>`;
  }).join('');
}

// ── Templates ────────────────────────────────────────────────────────────
function renderTemplates() {
  const grid = document.getElementById('templates-grid');
  if (allTemplates.length === 0) {
    grid.innerHTML = '<div class="w-empty">No templates saved yet.</div>';
    return;
  }
  grid.innerHTML = allTemplates.map(t => {
    const date = new Date(t.created_at * 1000).toLocaleString();
    return `
      <div class="w-card">
        <h3>${_esc(t.name || t.id)}</h3>
        <div class="w-card-meta">
          <span>${_esc(t.pipeline_mode)}</span>
          <span>${date}</span>
        </div>
        <div class="w-card-actions">
          <a href="/wizard?template_id=${encodeURIComponent(t.id)}" class="w-card-btn">Use Template</a>
          <button class="w-card-btn danger" onclick="deleteTemplate('${_esc(t.id)}')">Delete</button>
        </div>
      </div>`;
  }).join('');
}

window.deleteTemplate = async function(id) {
  if (!confirm('Delete this template?')) return;
  await fetch('/api/templates/' + encodeURIComponent(id), { method: 'DELETE' });
  await loadTemplates();
  updateStats();
};
