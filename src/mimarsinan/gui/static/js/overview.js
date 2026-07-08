/* Pipeline bar + overview charts (measured points, verdict markers, timing). */
import { esc, fmtDuration, elapsedFromStepStart, safeReact, emptyAnnotation } from './util.js';

const TONE_COLORS = { pass: '#4ade80', fail: '#f87171', carried: '#8494a7' };

// ── Pipeline bar ─────────────────────────────────────────────────────────
export function renderPipelineBar(pipeline, selectedStep) {
  const bar = document.getElementById('pipeline-bar');
  if (!bar || !pipeline) return;
  const steps = pipeline.steps || [];
  const cols = steps.map((s, i) => {
    const dur = s.duration ? ` (${fmtDuration(s.duration)})` : '';
    const tooltip = `${s.name}${dur} [${i + 1}/${steps.length}]`;
    const selected = selectedStep === s.name ? ' selected' : '';
    const group = s.semantic_group || 'other';
    const badge = badgeHtml(s);
    return `<div class="psb-col${selected}" data-status="${esc(s.status)}" data-group="${esc(group)}" data-step="${esc(s.name)}" title="${esc(tooltip)}">
      <div class="psb-bar">${badge}</div>
      <span class="psb-label">${esc(s.name)}</span>
    </div>`;
  });
  bar.innerHTML = `<div class="psb-list">${cols.join('')}</div>`;
}

// The bar cell shows a VERDICT for gate steps and a metric only while a
// measured step is running — a carried value is never displayed as a number.
function badgeHtml(step) {
  const badge = step.badge;
  if (badge && badge.kind === 'verdict') {
    return `<span class="psb-metric psb-verdict-${esc(badge.status)}">${esc(badge.text)}</span>`;
  }
  if (step.status === 'running' && badge && badge.kind === 'metric') {
    return `<span class="psb-metric">${esc(badge.text)}</span>`;
  }
  return '';
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

  renderMetricProgression(steps, pipeline.overview_chart);
  renderStepTiming(steps);
}

function renderMetricProgression(steps, chart) {
  const points = (chart && chart.points) || [];
  const markers = (chart && chart.markers) || [];
  const annotations = points.length === 0 && markers.length === 0
    ? emptyAnnotation('No measured metrics yet') : [];
  const shapes = [];

  const lastPt = points.length > 0 ? points[points.length - 1] : null;
  if (lastPt) {
    annotations.push({
      x: lastPt.step, y: lastPt.value,
      text: lastPt.value.toFixed(4),
      showarrow: true, arrowhead: 2, ax: 30, ay: -25,
      font: { size: 11, color: '#22d3ee' },
      bgcolor: 'rgba(15,17,23,0.85)', bordercolor: '#22d3ee', borderwidth: 1,
    });
  }

  // Verdict/failed/carried steps: labeled vertical lines with a glyph —
  // never a fabricated data point at the carried metric. The x axis is the
  // FULL step sequence so no step is invisible.
  const categories = steps.map(s => s.name);
  for (const marker of markers) {
    const color = TONE_COLORS[marker.status] || TONE_COLORS.pass;
    shapes.push({
      type: 'line', xref: 'x', yref: 'paper',
      x0: marker.step, x1: marker.step, y0: 0, y1: 1,
      line: { color, width: 1.5, dash: 'dot' },
    });
    annotations.push({
      x: marker.step, y: 1.04, yref: 'paper',
      text: marker.glyph, showarrow: false,
      font: { size: 13, color },
      hovertext: `${marker.glyph} ${marker.label || ''}`,
    });
  }

  const traces = points.length > 0 ? [{
    x: points.map(p => p.step), y: points.map(p => p.value),
    type: 'scatter', mode: 'lines+markers',
    marker: { size: 8, color: '#22d3ee' }, line: { width: 2, color: '#5b8af5' },
    name: 'measured',
  }] : [];
  const values = points.map(p => p.value);
  const maxY = values.length > 0 ? Math.max(1, ...values) * 1.05 : 1;
  safeReact('chart-metric-progression', traces, {
    margin: { t: 40, r: 30, b: 100, l: 60 },
    xaxis: {
      tickangle: -45, automargin: true, tickfont: { size: 10 },
      type: 'category', categoryorder: 'array', categoryarray: categories,
    },
    yaxis: { title: 'Measured Metric', automargin: true, range: [0, maxY] },
    height: 260, annotations, shapes,
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
  const priority = ['spiking_mode', 'pipeline_mode', 'model_config_mode', 'hw_config_mode', 'activation_quantization',
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
