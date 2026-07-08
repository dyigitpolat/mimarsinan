/* Shared utilities and Plotly safe-wrapper for Mimarsinan GUI.
 * Markdown: pinned ESM from jsDelivr (no import map — works in any document). */

import { marked } from 'https://cdn.jsdelivr.net/npm/marked@14.1.4/lib/marked.esm.js';
import DOMPurify from 'https://cdn.jsdelivr.net/npm/dompurify@3.1.7/dist/purify.es.mjs';

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

// ── Plotly safe wrapper ──────────────────────────────────────────────────
// Every axis gets a stable uirevision so zoom/pan survives Plotly.react().

// Workbench design tokens (static/css/tokens.css): card surface, quiet
// grid lines, the shared accent set.
const DARK = {
  paper_bgcolor: '#141924',
  plot_bgcolor: '#141924',
  font: { color: '#8494a7', size: 11, family: "'Outfit', system-ui, sans-serif" },
  margin: { t: 30, r: 20, b: 40, l: 50 },
  xaxis: { gridcolor: '#1e2736', zerolinecolor: '#2a3547' },
  yaxis: { gridcolor: '#1e2736', zerolinecolor: '#2a3547' },
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
