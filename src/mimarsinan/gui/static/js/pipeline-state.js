/* Pipeline page-state reducer — pure, DOM-free (unit-tested under Node). */

// Fields the WS envelope adds; everything else in a frame is pipeline state.
const ENVELOPE_KEYS = ['type', 'event_seq'];

/* Merge a `pipeline_overview` frame onto the last-known pipeline state.
 * Two producers emit these frames with different key sets (the in-process
 * collector omits is_alive/status/error; the active-run tailer sends them), so
 * the merge copies whatever a frame carries and leaves every omitted field at
 * its last-known value. Rebuilding the state from a hand-listed literal instead
 * silently erased fields — that is how the Configuration tab downgraded to its
 * raw table on the first lifecycle event of a run. */
export function mergePipelineOverview(prev, frame) {
  const fields = { ...(frame || {}) };
  for (const key of ENVELOPE_KEYS) delete fields[key];
  return { steps: [], ...(prev || {}), ...fields };
}
