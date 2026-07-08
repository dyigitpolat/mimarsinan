/* Shared utilities and Plotly safe-wrapper for Mimarsinan GUI.
 * Markdown: pinned ESM vendored under static/vendor/ (offline monitors). */

import { marked } from '/static/vendor/marked-14.1.4.esm.js';
import DOMPurify from '/static/vendor/dompurify-3.1.7.es.mjs';

// ── HTML helpers ─────────────────────────────────────────────────────────
const _escDiv = document.createElement('div');
export function esc(s) {
  _escDiv.textContent = String(s ?? '');
  return _escDiv.innerHTML;
}

marked.setOptions({ gfm: true, breaks: false });

let _linkHookInstalled = false;
function _ensureExternalLinkHook() {
  if (_linkHookInstalled) return;
  _linkHookInstalled = true;
  DOMPurify.addHook('afterSanitizeAttributes', (node) => {
    if (node.tagName !== 'A' || !node.hasAttribute('href')) return;
    const href = node.getAttribute('href') || '';
    if (/^https?:\/\//i.test(href)) {
      node.setAttribute('target', '_blank');
      node.setAttribute('rel', 'noopener noreferrer');
    }
  });
}

/**
 * Render Markdown to safe HTML for agentic evolution prose (constraints, insights, trace).
 * Uses CommonMark + GFM via marked; output is sanitized with DOMPurify before innerHTML.
 */
export function renderMarkdown(s) {
  if (s == null || s === '') return '';
  const raw = String(s).trim();
  if (!raw) return '';
  _ensureExternalLinkHook();
  const html = marked.parse(raw);
  return DOMPurify.sanitize(html);
}

export function cssId(s) { return s.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase(); }

export function fmtDuration(sec) {
  const s = Math.max(0, Number(sec));
  if (s < 60) return s.toFixed(1) + 's';
  if (s < 3600) return Math.floor(s / 60) + 'm ' + Math.floor(s % 60) + 's';
  return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
}

/** Elapsed seconds from step start; startTime from API (seconds or ms), nowSec = Date.now()/1000. */
export function elapsedFromStepStart(startTime, nowSec = Date.now() / 1000) {
  if (startTime == null || typeof startTime !== 'number') return 0;
  const startSec = startTime > 1e12 ? startTime / 1000 : startTime;
  return Math.max(0, nowSec - startSec);
}

export function fmtNum(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

// Semantic-group accents ("r,g,b") shared by the step navigator (CSS
// mirror in monitor.css), the config pipeline preview, and the overview
// timing bars.
export const GROUP_ACCENTS = {
  configuration: '168,85,247',
  model_building: '139,92,246',
  pretraining: '56,189,248',
  torch_mapping: '14,165,233',
  pruning: '244,63,94',
  activation: '74,222,128',
  activation_quantization: '52,211,153',
  weight_quantization: '251,191,36',
  normalization: '34,211,238',
  soft_mapping: '96,165,250',
  core_verification: '129,140,248',
  coreflow_tuning: '99,102,241',
  hardware: '249,115,22',
  simulation: '103,232,249',
  other: '107,114,128',
};

export function groupAccent(group) {
  return GROUP_ACCENTS[group] || GROUP_ACCENTS.other;
}

// ── Plotly safe wrapper ──────────────────────────────────────────────────
// Every axis gets a stable uirevision so zoom/pan survives Plotly.react().

// Workbench design tokens (static/css/tokens.css): card surface, quiet
// grid lines, the shared accent set.
const DARK = {
  paper_bgcolor: '#141924',
  plot_bgcolor: '#141924',
  font: { color: '#8494a7', size: 11, family: "'Outfit', system-ui, sans-serif" },
  margin: { t: 30, r: 20, b: 40, l: 50 },
  // automargin keeps the axis titles clear of their tick labels; it only ever
  // grows the margins the callers reserve for the below-plot legend.
  xaxis: { gridcolor: '#1e2736', zerolinecolor: '#2a3547', automargin: true },
  yaxis: { gridcolor: '#1e2736', zerolinecolor: '#2a3547', automargin: true },
  colorway: ['#3b82f6', '#22d3ee', '#10b981', '#f59e0b', '#f43f5e', '#8b5cf6', '#f97316', '#4ade80'],
};
const CFG = { displayModeBar: false, responsive: true };

function deepMerge(base, over) {
  const out = { ...base };
  for (const k of Object.keys(over)) {
    if (over[k] && typeof over[k] === 'object' && !Array.isArray(over[k]) && base[k] && typeof base[k] === 'object') {
      out[k] = deepMerge(base[k], over[k]);
    } else {
      out[k] = over[k];
    }
  }
  return out;
}

export function makeLayout(overrides = {}) {
  const base = {
    ...DARK,
    uirevision: 'persist',
    xaxis: { ...DARK.xaxis, uirevision: 'persist' },
    yaxis: { ...DARK.yaxis, uirevision: 'persist' },
  };
  return deepMerge(base, overrides);
}

const LEGEND_FONT_PX = 10;
const LEGEND_CHAR_PX = 5.6;   // advance width of the 10px UI font
const LEGEND_ENTRY_PAD = 44;  // colour swatch + inter-entry gutter
const LEGEND_ROW_H = 18;
const AXIS_FOOTER_H = 48;     // tick labels + axis title

// Plotly wraps a horizontal legend once it runs out of width, and it does not
// grow the margin to fit the extra rows.
function legendRows(names, plotWidth) {
  const width = Math.max(plotWidth || 0, 240);
  const total = names.reduce(
    (acc, name) => acc + String(name).length * LEGEND_CHAR_PX + LEGEND_ENTRY_PAD, 0);
  return Math.max(1, Math.ceil(total / width));
}

// Horizontal legend pinned under the x-axis title — never to the right, where
// it eats plot width. Bottom margin and height both grow with the number of
// wrapped rows so legend entries never land on the tick labels.
// (legend.yref 'container' requires the vendored Plotly ≥ 2.26.)
export function legendBelow(names, elOrId, { height = 240, margin = {} } = {}) {
  const el = typeof elOrId === 'string' ? document.getElementById(elOrId) : elOrId;
  const showlegend = names.length > 1;
  const rows = showlegend ? legendRows(names, el && el.clientWidth) : 0;
  return {
    showlegend,
    legend: {
      orientation: 'h',
      x: 0, xanchor: 'left', xref: 'paper',
      y: 0, yanchor: 'bottom', yref: 'container',
      font: { size: LEGEND_FONT_PX },
    },
    margin: { r: 20, ...margin, b: AXIS_FOOTER_H + rows * LEGEND_ROW_H },
    height: height + rows * LEGEND_ROW_H,
  };
}

export function safeReact(elOrId, traces, layoutOverrides = {}) {
  const el = typeof elOrId === 'string' ? document.getElementById(elOrId) : elOrId;
  if (!el) return;
  const layout = makeLayout(layoutOverrides);
  Plotly.react(el, traces, layout, CFG);
}

export function plotHistogram(elId, hist, xLabel, color) {
  if (!hist || !hist.counts || hist.counts.length === 0) return;
  const mids = hist.bin_edges.slice(0, -1).map((e, i) => (e + hist.bin_edges[i + 1]) / 2);
  safeReact(elId, [{ x: mids, y: hist.counts, type: 'bar', marker: { color: color || '#5b8af5' } }],
    { height: 240, xaxis: { title: xLabel }, yaxis: { title: 'Count' } });
}

export function emptyAnnotation(text) {
  return [{ text, showarrow: false, font: { size: 14, color: '#6b6e7a' }, xref: 'paper', yref: 'paper', x: 0.5, y: 0.5 }];
}
