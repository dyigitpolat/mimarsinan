/* Incremental Plotly line/scatter streaming.
 *
 * Live monitor curves grow one point at a time over the WebSocket. Re-plotting
 * the whole trace on every frame (Plotly.react) flashes and jumps; this module
 * appends only the NEW tail of each grown trace via Plotly.extendTraces.
 *
 * The planner (`planTraceExtension`) is pure + DOM-free so it runs under Node
 * (the unit-test pattern); the Plotly side effect lives in `applyStreamPlan`,
 * which touches the global `Plotly` only when called in the browser.
 *
 * Honesty: the planner only ever slices the TAIL of the caller-supplied REAL
 * points — it never fabricates or interpolates a sample. Any structural change
 * (a trace joined/left/renamed) or a SHRINK (a REST snapshot handing back fewer
 * points after a reconnect) forces a full redraw, so no point is doubled or
 * lost. "Smooth" is the extend easing, not invented data. */

// prev: { names: string[]|null, counts: number[] } — the trace identities and
//       point counts last drawn on the element (null before the first draw).
// next: [{ name, x:[], y:[] }] — the desired FULL trace set for this frame.
// -> { mode:'redraw'|'noop'|'extend', names, counts, [newXs, newYs, indices] }
export function planTraceExtension(prev, next) {
  const names = next.map(t => t.name);
  const counts = next.map(t => t.x.length);
  const prevNames = (prev && prev.names) || null;
  const prevCounts = (prev && prev.counts) || [];
  const sameStructure = Array.isArray(prevNames)
    && prevNames.length === names.length
    && prevNames.every((n, i) => n === names[i]);
  if (!sameStructure) return { mode: 'redraw', names, counts };

  const newXs = [];
  const newYs = [];
  const indices = [];
  for (let i = 0; i < next.length; i++) {
    const before = prevCounts[i] || 0;
    const now = counts[i];
    if (now < before) return { mode: 'redraw', names, counts };
    if (now > before) {
      newXs.push(next[i].x.slice(before));
      newYs.push(next[i].y.slice(before));
      indices.push(i);
    }
  }
  if (indices.length === 0) return { mode: 'noop', names, counts };
  return { mode: 'extend', newXs, newYs, indices, names, counts };
}

// The stream bookkeeping lives on the Plotly DOM node so it survives across
// frames without a side table.
export function readStreamState(el) {
  if (!el) return null;
  return { names: el._streamNames || null, counts: el._streamCounts || [] };
}

export function markStreamState(el, plan) {
  if (!el) return;
  el._streamNames = plan.names;
  el._streamCounts = plan.counts;
}

// Apply a plan to a live Plotly node. Returns true when the chart was advanced
// incrementally (extend or no-op) so the caller can SKIP its full redraw;
// returns false when a redraw is required (structural change / shrink / an
// out-of-sync Plotly state that made extendTraces throw).
export function applyStreamPlan(el, plan) {
  if (!el || !el.data) return false;
  if (plan.mode === 'redraw') return false;
  if (plan.mode === 'extend') {
    try {
      Plotly.extendTraces(el, { x: plan.newXs, y: plan.newYs }, plan.indices);
    } catch (_) {
      return false;
    }
  }
  markStreamState(el, plan);
  return true;
}
