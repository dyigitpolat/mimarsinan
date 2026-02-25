/* Pipeline bar + overview charts (target metric, step timing, config). */
import { esc, fmtDuration, safeReact, emptyAnnotation } from './util.js';

// ── Pipeline bar ─────────────────────────────────────────────────────────
export function renderPipelineBar(pipeline, selectedStep) {
  const bar = document.getElementById('pipeline-bar');
  if (!bar || !pipeline) return;
  bar.innerHTML = (pipeline.steps || []).map(s => {
    const cls = s.status + (selectedStep === s.name ? ' selected' : '');
    const dur = s.duration ? ` (${fmtDuration(s.duration)})` : '';
    return `<div class="step-block ${cls}" data-step="${esc(s.name)}" title="${esc(s.name)}${dur}">
      <span class="step-name">${esc(abbreviate(s.name))}</span>
      ${s.status === 'running' ? '<span class="step-running-dot"></span>' : ''}
    </div>`;
  }).join('');
}

function abbreviate(name) {
  if (name.length <= 18) return name;
  return name.split(' ').map(w => w.length > 4 ? w.substring(0, 4) + '.' : w).join(' ');
}

// ── Overview cards ───────────────────────────────────────────────────────
export function renderOverviewCards(pipeline) {
  if (!pipeline) return;
  const steps = pipeline.steps || [];
  const completed = steps.filter(s => s.status === 'completed');
  const running = steps.find(s => s.status === 'running');

  const curEl = document.getElementById('current-step-name');
  if (curEl) curEl.textContent = running ? running.name : (completed.length === steps.length && steps.length > 0 ? 'Pipeline Complete' : 'Waiting...');
  const progEl = document.getElementById('progress-text');
  if (progEl) progEl.textContent = `${completed.length} / ${steps.length}`;

  renderMetricProgression(steps);
  renderStepTiming(steps);
  renderConfig(pipeline.config);
}

function renderMetricProgression(steps) {
  const pts = steps.filter(s => s.status === 'completed' && s.target_metric != null);
  const traces = pts.length > 0 ? [{
    x: pts.map(s => s.name), y: pts.map(s => s.target_metric),
    type: 'scatter', mode: 'lines+markers', marker: { size: 8 }, line: { width: 2 },
  }] : [];
  const anno = pts.length === 0 ? emptyAnnotation('No metrics yet') : [];
  safeReact('chart-metric-progression', traces, {
    xaxis: { tickangle: -45 }, yaxis: { title: 'Target Metric' },
    height: 260, annotations: anno,
  });
}

function renderStepTiming(steps) {
  const now = Date.now() / 1000;
  const timed = [];
  for (const s of steps) {
    if (s.duration != null) timed.push({ name: s.name, duration: s.duration, running: false });
    else if (s.status === 'running' && s.start_time) timed.push({ name: s.name, duration: now - s.start_time, running: true });
  }
  const anno = timed.length === 0 ? emptyAnnotation('No timing data yet') : [];
  const traces = timed.length > 0 ? [{
    x: timed.map(s => s.duration), y: timed.map(s => s.name),
    type: 'bar', orientation: 'h',
    marker: { color: timed.map(s => s.running ? '#ff9800' : '#5b8af5') },
    text: timed.map(s => fmtDuration(s.duration) + (s.running ? ' ●' : '')),
    textposition: 'outside', textfont: { size: 10, color: '#9a9daa' },
  }] : [];
  safeReact('chart-step-timing', traces, {
    xaxis: { title: 'Duration (s)' }, yaxis: { autorange: 'reversed' },
    height: Math.max(200, timed.length * 32 + 60), annotations: anno,
  });
}

function renderConfig(config) {
  const el = document.getElementById('config-body');
  if (!el || !config) return;
  const priority = ['spiking_mode', 'pipeline_mode', 'configuration_mode', 'activation_quantization',
    'weight_quantization', 'max_axons', 'max_neurons', 'weight_bits', 'target_tq',
    'training_epochs', 'lr', 'simulation_steps', 'input_shape', 'num_classes', 'device'];
  const all = Object.keys(config);
  const sorted = [...priority.filter(k => k in config), ...all.filter(k => !priority.includes(k)).sort()];
  let html = '<table class="config-table">';
  for (const key of sorted) {
    const val = config[key];
    const display = typeof val === 'object' ? JSON.stringify(val) : String(val);
    html += `<tr><td>${esc(key)}</td><td>${esc(display)}</td></tr>`;
  }
  html += '</table>';
  el.innerHTML = html;
}
