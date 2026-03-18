/* Pipeline bar + overview charts (target metric, step timing, config). */
import { esc, fmtDuration, elapsedFromStepStart, safeReact, emptyAnnotation } from './util.js';

// ── Pipeline bar ─────────────────────────────────────────────────────────
export function renderPipelineBar(pipeline, selectedStep) {
  const bar = document.getElementById('pipeline-bar');
  if (!bar || !pipeline) return;
  const steps = pipeline.steps || [];
  bar.innerHTML = steps.map((s, i) => {
    const cls = s.status + (selectedStep === s.name ? ' selected' : '');
    const dur = s.duration ? ` (${fmtDuration(s.duration)})` : '';
    const tooltip = `${s.name}${dur} [${i + 1}/${steps.length}]`;
    const metricBadge = s.status === 'running' && s.target_metric != null
      ? `<span style="font-size:8px;color:var(--accent-cyan);font-weight:700;display:block;margin-top:1px">${s.target_metric.toFixed(3)}</span>` : '';
    return `<div class="step-block ${cls}" data-step="${esc(s.name)}" title="${esc(tooltip)}">
      <span class="step-name">${esc(abbreviate(s.name))}</span>
      ${s.status === 'running' ? '<span class="step-running-dot"></span>' : ''}
      ${metricBadge}
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
}

function renderMetricProgression(steps) {
  const pts = steps.filter(s =>
    s.target_metric != null && (s.status === 'completed' || s.end_time != null)
  );
  const lastPt = pts.length > 0 ? pts[pts.length - 1] : null;
  const annotations = pts.length === 0 ? emptyAnnotation('No metrics yet') : [];
  if (lastPt) {
    annotations.push({
      x: lastPt.name, y: lastPt.target_metric,
      text: lastPt.target_metric.toFixed(4),
      showarrow: true, arrowhead: 2, ax: 30, ay: -25,
      font: { size: 11, color: '#22d3ee' },
      bgcolor: 'rgba(15,17,23,0.85)', bordercolor: '#22d3ee', borderwidth: 1,
    });
  }
  const traces = pts.length > 0 ? [{
    x: pts.map(s => s.name), y: pts.map(s => s.target_metric),
    type: 'scatter', mode: 'lines+markers',
    marker: { size: 8, color: '#22d3ee' }, line: { width: 2, color: '#5b8af5' },
  }] : [];
  const allMetrics = pts.map(s => s.target_metric);
  const maxY = allMetrics.length > 0 ? Math.max(1, ...allMetrics) * 1.05 : 1;
  safeReact('chart-metric-progression', traces, {
    margin: { t: 40, r: 30, b: 100, l: 60 },
    xaxis: { tickangle: -45, automargin: true, tickfont: { size: 10 } },
    yaxis: { title: 'Target Metric', automargin: true, range: [0, maxY] },
    height: 260, annotations,
  });
}

function renderStepTiming(steps) {
  const now = Date.now() / 1000;
  const timed = [];
  for (const s of steps) {
    if (s.duration != null) timed.push({ name: s.name, duration: s.duration, running: false });
    else if (s.status === 'running' && s.start_time != null) timed.push({ name: s.name, duration: elapsedFromStepStart(s.start_time, now), running: true });
  }
  const anno = timed.length === 0 ? emptyAnnotation('No timing data yet') : [];
  const traces = timed.length > 0 ? [{
    x: timed.map(s => s.duration), y: timed.map(s => s.name),
    type: 'bar', orientation: 'h',
    marker: { color: timed.map(s => s.running ? '#ff9800' : '#5b8af5') },
    text: timed.map(s => fmtDuration(s.duration) + (s.running ? ' ●' : '')),
    textposition: 'inside', textfont: { size: 10, color: '#e8eaed' },
  }] : [];
  safeReact('chart-step-timing', traces, {
    margin: { t: 40, r: 30, b: 50, l: 180 },
    xaxis: { title: 'Duration (s)', automargin: true },
    yaxis: { autorange: 'reversed', automargin: true, tickfont: { size: 10 } },
    height: Math.max(200, timed.length * 32 + 60), annotations: anno,
  });
}

export function renderConfig(config) {
  const el = document.getElementById('config-body');
  if (!el || !config) return;
  const priority = ['spiking_mode', 'pipeline_mode', 'configuration_mode', 'activation_quantization',
    'weight_quantization', 'max_axons', 'max_neurons', 'weight_bits', 'target_tq',
    'training_epochs', 'lr', 'simulation_steps', 'max_simulation_samples', 'input_shape', 'num_classes', 'device'];
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
