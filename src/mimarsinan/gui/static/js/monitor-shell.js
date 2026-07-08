/* Monitor workbench shell: section rail, live rail, and the vertical step
   list. Pure DOM rendering from the pipeline overview — no data reshaping. */
import { esc, fmtDuration, elapsedFromStepStart } from './util.js';

export const SECTIONS = [
  { id: 'overview', title: 'Overview', hint: 'Run health & metric', icon: '◎' },
  { id: 'steps', title: 'Steps', hint: 'Per-step instruments', icon: '☰' },
  { id: 'analysis', title: 'Analysis', hint: 'Ratchet · timeline · gauges', icon: '∿' },
  { id: 'noc', title: 'Hardware / NoC', hint: 'Spike traffic & floorplan', icon: '▦' },
  { id: 'artifacts', title: 'Artifacts', hint: 'Run directory outputs', icon: '⬡' },
  { id: 'config', title: 'Configuration', hint: 'Emitted deployment config', icon: '⚙' },
  { id: 'console', title: 'Console', hint: 'stdout / stderr', icon: '❯' },
];

let _currentSection = SECTIONS[0].id;
// The step navigator (sub-items under Steps) starts expanded; the chevron
// on the Steps item is the only collapse affordance, so scripted clicks on
// the item itself always navigate.
let _stepsExpanded = true;

export function currentSection() { return _currentSection; }

export function buildNav(onNavigate) {
  const nav = document.getElementById('section-nav');
  if (!nav) return;
  nav.innerHTML = '';
  for (const section of SECTIONS) {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'wb-nav-item' + (section.id === _currentSection ? ' active' : '');
    item.dataset.sectionId = section.id;
    item.innerHTML = `
      <span class="wb-nav-icon">${section.icon}</span>
      <span class="wb-nav-titles">
        <span class="wb-nav-title">${esc(section.title)}</span>
        <span class="wb-nav-hint">${esc(section.hint)}</span>
      </span>`;
    item.addEventListener('click', () => onNavigate(section.id));
    nav.appendChild(item);
    if (section.id === 'steps') {
      const expander = document.createElement('span');
      expander.className = 'wb-nav-expander';
      expander.title = 'Collapse step list';
      expander.textContent = '▾';
      expander.addEventListener('click', (e) => {
        e.stopPropagation();
        setStepsExpanded(!_stepsExpanded);
      });
      item.appendChild(expander);
      const sub = document.createElement('div');
      sub.className = 'nav-steps' + (_stepsExpanded ? '' : ' collapsed');
      sub.id = 'nav-step-list';
      sub.setAttribute('aria-label', 'Pipeline steps');
      sub.addEventListener('keydown', onStepListKeydown);
      nav.appendChild(sub);
    }
  }
}

function setStepsExpanded(expanded) {
  _stepsExpanded = expanded;
  const sub = document.getElementById('nav-step-list');
  if (sub) sub.classList.toggle('collapsed', !expanded);
  const expander = document.querySelector('.wb-nav-item[data-section-id="steps"] .wb-nav-expander');
  if (expander) {
    expander.textContent = expanded ? '▾' : '▸';
    expander.title = expanded ? 'Collapse step list' : 'Expand step list';
  }
}

// Arrow keys walk the navigator; Enter/Space activate (native button).
function onStepListKeydown(e) {
  if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
  const items = Array.from(e.currentTarget.querySelectorAll('.step-item'));
  const idx = items.indexOf(document.activeElement);
  if (idx === -1) return;
  e.preventDefault();
  const next = items[idx + (e.key === 'ArrowDown' ? 1 : -1)];
  if (next) next.focus();
}

export function goToSection(sectionId) {
  if (!SECTIONS.some(s => s.id === sectionId)) return;
  _currentSection = sectionId;
  if (sectionId === 'steps' && !_stepsExpanded) setStepsExpanded(true);
  document.querySelectorAll('.wb-section').forEach(panel => {
    panel.classList.toggle('active', panel.dataset.sectionId === sectionId);
  });
  document.querySelectorAll('.wb-nav-item').forEach(item => {
    item.classList.toggle('active', item.dataset.sectionId === sectionId);
  });
  window.scrollTo({ top: 0 });
}

export function setNavBadge(sectionId, count) {
  const item = document.querySelector(`.wb-nav-item[data-section-id="${sectionId}"]`);
  if (!item) return;
  let badge = item.querySelector('.wb-nav-badge');
  if (!count) { if (badge) badge.remove(); return; }
  if (!badge) {
    badge = document.createElement('span');
    badge.className = 'wb-nav-badge';
    item.appendChild(badge);
  }
  badge.textContent = String(count);
}

// ── Step list (sub-items under Steps in the section rail) ────────────────

function stepFlagHtml(step) {
  const badge = step.badge || {};
  if (badge.kind === 'verdict') {
    const cls = badge.status === 'pass' ? 'pass' : 'fail';
    return `<span class="step-item-flag ${cls}">${esc(badge.text)}</span>`;
  }
  if (badge.kind === 'metric' && step.status !== 'pending') {
    return `<span class="step-item-flag measured" title="measured metric">${esc(badge.text)}</span>`;
  }
  if (badge.kind === 'carried' && step.status === 'completed') {
    return `<span class="step-item-flag" title="carried — this step measures nothing">carried</span>`;
  }
  return '';
}

export function renderStepList(pipeline, selectedStep, onSelect) {
  const host = document.getElementById('nav-step-list');
  if (!host || !pipeline) return;
  const steps = pipeline.steps || [];
  if (!steps.length) {
    host.innerHTML = '<div class="nav-steps-empty">no steps yet</div>';
    return;
  }
  const hadFocus = host.contains(document.activeElement);
  host.innerHTML = steps.map((s, i) => {
    const dur = s.duration != null ? ` · ${fmtDuration(s.duration)}` : '';
    const title = `${s.name} [${i + 1}/${steps.length}]${dur}`;
    return `<button type="button"
      class="step-item${selectedStep === s.name ? ' selected' : ''}"
      data-step="${esc(s.name)}"
      data-status="${esc(s.status)}"
      data-group="${esc(s.semantic_group || 'other')}"
      title="${esc(title)}">
      <span class="step-dot"></span>
      <span class="step-item-name">${esc(s.name)}</span>
      ${stepFlagHtml(s)}
    </button>`;
  }).join('');
  host.querySelectorAll('.step-item').forEach(btn => {
    btn.addEventListener('click', () => onSelect(btn.dataset.step));
  });
  // Live re-renders replace the buttons; keep keyboard walkers anchored.
  if (hadFocus) host.querySelector('.step-item.selected')?.focus();
}

// ── Overview facts strip ─────────────────────────────────────────────────

export function renderOverviewFacts(pipeline, runId) {
  const host = document.getElementById('overview-facts');
  if (!host || !pipeline) return;
  const config = pipeline.config || {};
  const flat = config.deployment_parameters || config;
  const facts = [];
  if (runId) facts.push(['run', runId]);
  if (flat.experiment_name) facts.push(['experiment', flat.experiment_name]);
  if (flat.pipeline_mode) facts.push(['pipeline', flat.pipeline_mode]);
  if (flat.spiking_mode) facts.push(['spiking', flat.spiking_mode]);
  if (flat.weight_bits != null) facts.push(['weight bits', flat.weight_bits]);
  if (flat.target_tq != null) facts.push(['target tq', flat.target_tq]);
  if (flat.simulation_steps != null) facts.push(['sim steps', flat.simulation_steps]);
  const steps = pipeline.steps || [];
  facts.push(['steps', String(steps.length)]);
  if (!facts.length) { host.style.display = 'none'; return; }
  host.style.display = 'flex';
  host.innerHTML = '<span class="overview-facts-label">Run</span>'
    + facts.map(([k, v]) => `<span class="overview-fact">${esc(k)} <b>${esc(String(v))}</b></span>`).join('');
}

// ── Live rail ────────────────────────────────────────────────────────────

export function renderLiveRail(state, onSelectStep) {
  const pipeline = state.pipeline;
  if (!pipeline) return;
  const steps = pipeline.steps || [];
  const running = steps.find(s => s.status === 'running');
  const failed = steps.find(s => s.status === 'failed');
  const completed = steps.filter(s => s.status === 'completed');

  renderRailStatus(pipeline, steps, running, failed, completed, state);
  renderRailCurrent(running, failed, completed, steps, onSelectStep);
  renderRailSparkline(pipeline);
  renderRailWall(steps, state.analysis);
  renderRailVerdicts(steps, onSelectStep);
  renderRailProgress(completed.length, steps.length);
}

function renderRailStatus(pipeline, steps, running, failed, completed, state) {
  const el = document.getElementById('rail-status');
  if (!el) return;
  let cls = 'pending', text = 'waiting for pipeline…';
  if (failed || pipeline.error) {
    cls = 'error';
    text = pipeline.error ? 'run failed' : `failed — ${failed.name}`;
  } else if (running) {
    cls = 'live';
    text = `running · step ${completed.length + 1} / ${steps.length}`;
  } else if (steps.length > 0 && completed.length === steps.length) {
    cls = 'ok';
    text = `completed · ${steps.length} steps`;
  } else if (state.historicalRunId && !state.isActiveRun) {
    cls = 'ok';
    text = `finished · ${completed.length} / ${steps.length} steps`;
  }
  el.className = `rail-verdict ${cls}`;
  el.textContent = text;
}

function renderRailCurrent(running, failed, completed, steps, onSelectStep) {
  const host = document.getElementById('rail-current');
  if (!host) return;
  const title = host.closest('.rail-block')?.querySelector('.rail-block-title');
  if (title) title.textContent = running ? 'Current step' : 'Latest step';
  const step = running || failed
    || (completed.length ? completed[completed.length - 1] : null);
  if (!step) {
    host.innerHTML = '<div class="rail-empty">waiting…</div>';
    return;
  }
  const idx = steps.findIndex(s => s.name === step.name);
  const chips = [`<span class="tk-chip dim">${idx + 1} / ${steps.length}</span>`];
  if (step.status === 'running') {
    chips.push(`<span class="tk-chip mono info" id="rail-elapsed">${fmtDuration(elapsedFromStepStart(step.start_time))}</span>`);
  } else if (step.duration != null) {
    chips.push(`<span class="tk-chip mono">${fmtDuration(step.duration)}</span>`);
  }
  if (step.semantic_group) chips.push(`<span class="tk-chip dim">${esc(step.semantic_group)}</span>`);
  chips.push(`<span class="badge ${esc(step.status)}">${esc(step.status)}</span>`);
  host.innerHTML = `
    <button type="button" class="rail-verdict-row" data-step="${esc(step.name)}">
      <span class="rail-current-name">${esc(step.name)}</span>
    </button>
    <div class="rail-current-meta">${chips.join('')}</div>`;
  const btn = host.querySelector('.rail-verdict-row');
  if (btn) btn.addEventListener('click', () => onSelectStep(step.name));
}

function renderRailSparkline(pipeline) {
  const svg = document.getElementById('rail-spark');
  const label = document.getElementById('rail-spark-label');
  const value = document.getElementById('rail-spark-value');
  if (!svg) return;
  const points = (pipeline.overview_chart && pipeline.overview_chart.points) || [];
  // Backpressure: the measured series is append-only, so the deployed-metric
  // sparkline only changes when a point is ADDED. Skip the redundant SVG
  // rewrite on the many frames (metrics, step lifecycle, the 1 s tick) that
  // don't add one.
  if (svg._sparkCount === points.length) return;
  svg._sparkCount = points.length;
  if (!points.length) {
    svg.innerHTML = '';
    if (label) label.textContent = 'no measurements yet';
    if (value) value.textContent = '';
    return;
  }
  const W = 288, H = 56, PAD = 4;
  const values = points.map(p => p.value);
  const min = Math.min(...values), max = Math.max(...values);
  const span = (max - min) || 1;
  const xs = points.length === 1 ? [W / 2]
    : points.map((_, i) => PAD + (i * (W - 2 * PAD)) / (points.length - 1));
  const ys = values.map(v => H - PAD - ((v - min) / span) * (H - 2 * PAD));
  const path = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(' ');
  const last = points[points.length - 1];
  svg.innerHTML = `
    <path d="${path}" fill="none" stroke="#22d3ee" stroke-width="1.5"/>
    ${xs.map((x, i) => `<circle cx="${x.toFixed(1)}" cy="${ys[i].toFixed(1)}" r="2" fill="#22d3ee" opacity="${i === xs.length - 1 ? 1 : 0.45}"><title>${esc(points[i].step)}: ${points[i].value.toFixed(4)}</title></circle>`).join('')}`;
  if (label) label.textContent = last.step;
  if (value) value.textContent = last.value.toFixed(4);
}

function renderRailWall(steps, analysis) {
  const host = document.getElementById('rail-wall');
  if (!host) return;
  const now = Date.now() / 1000;
  let artifact = 0, total = 0, any = false;
  for (const s of steps) {
    let dur = s.duration;
    if (dur == null && s.status === 'running' && s.start_time != null) {
      dur = elapsedFromStepStart(s.start_time, now);
    }
    if (dur == null) continue;
    any = true;
    total += dur;
    if (s.semantic_group !== 'simulation') artifact += dur;
  }
  if (!any) {
    host.innerHTML = '<div class="rail-empty">no timing yet</div>';
    return;
  }
  let html = `
    <div class="rail-wall-row"><span>artifact wall</span><span class="rail-wall-num">${fmtDuration(artifact)}</span></div>
    <div class="rail-wall-row"><span>simulators</span><span class="rail-wall-num">${fmtDuration(total - artifact)}</span></div>
    <div class="rail-wall-row"><span>total</span><span class="rail-wall-num">${fmtDuration(total)}</span></div>`;
  const budget = analysis && analysis.gantt && analysis.gantt.endpoint_budget;
  if (budget && budget.budget_steps != null) {
    const frac = Math.min(1, budget.consumed_steps / Math.max(1, budget.budget_steps));
    const over = budget.consumed_steps > budget.budget_steps;
    html += `
      <div class="rail-wall-row" style="margin-top:6px"><span>endpoint steps</span>
        <span class="rail-wall-num">${budget.consumed_steps} / ${budget.budget_steps}</span></div>
      <div class="rail-budget-track"><div class="rail-budget-fill${over ? ' over' : ''}" style="width:${(frac * 100).toFixed(1)}%"></div></div>`;
  }
  host.innerHTML = html;
}

function renderRailVerdicts(steps, onSelectStep) {
  const host = document.getElementById('rail-verdicts');
  if (!host) return;
  const gated = steps.filter(s => s.verdict || s.status === 'failed');
  if (!gated.length) {
    host.innerHTML = '<div class="rail-empty">no gates yet</div>';
    return;
  }
  host.innerHTML = gated.map(s => {
    const status = s.status === 'failed' ? 'fail' : (s.verdict.status === 'pass' ? 'pass' : 'fail');
    const rule = s.status === 'failed' ? (s.error || 'step failed') : (s.verdict.rule || '');
    return `<button type="button" class="rail-verdict-row" data-step="${esc(s.name)}" title="${esc(rule)}">
      <span class="rail-verdict-name">${esc(s.name)}</span>
      <span class="rail-verdict-flag ${status}">${status === 'pass' ? 'PASS' : 'FAIL'}</span>
    </button>`;
  }).join('');
  host.querySelectorAll('.rail-verdict-row').forEach(btn => {
    btn.addEventListener('click', () => onSelectStep(btn.dataset.step));
  });
}

function renderRailProgress(done, totalSteps) {
  const fill = document.getElementById('rail-progress-fill');
  const label = document.getElementById('rail-progress-label');
  if (fill) fill.style.width = totalSteps ? `${((done / totalSteps) * 100).toFixed(1)}%` : '0%';
  if (label) label.textContent = totalSteps ? `${done} / ${totalSteps} steps` : '';
}
