/* Acknowledge-to-launch gate over the resolve payload's deployment advisories.
   Pure functions only (no DOM, executed under Node by the unit suite): a
   GATING advisory — mandate_violation OR severity UNSUPPORTED — demands an
   explicit per-id acknowledgment before Launch enables, and every stored
   acknowledgment is invalidated when a config edit changes the resolved SET
   of gating advisory ids (an ack never carries over to a different set). */

export function isGatingAdvisory(advisory) {
  return !!(advisory
    && (advisory.mandate_violation || advisory.severity === 'UNSUPPORTED'));
}

/** Sorted unique ids of the advisories that gate launch. */
export function gatingAdvisoryIds(advisories) {
  const ids = (advisories || []).filter(isGatingAdvisory).map((a) => a.id);
  return [...new Set(ids)].sort();
}

function sameIdSet(a, b) {
  return a.length === b.length && a.every((id, i) => id === b[i]);
}

/** The ack-state step applied on every resolve round-trip: acknowledgments
    survive ONLY while the gating id set is unchanged (non-gating RISK/INFO
    churn never resets them; any gating-set change resets them all). */
export function nextAckState(prev, advisories) {
  const gatingIds = gatingAdvisoryIds(advisories);
  const prevIds = (prev && prev.gatingIds) || [];
  const acked = sameIdSet(gatingIds, prevIds) ? ((prev && prev.acked) || []) : [];
  return { gatingIds, acked };
}

/** Toggle one advisory's acknowledgment; ids outside the gating set never
    enter the ack list. */
export function withAck(state, advisoryId, on) {
  const acked = (state.acked || []).filter((id) => id !== advisoryId);
  if (on && (state.gatingIds || []).includes(advisoryId)) acked.push(advisoryId);
  return { gatingIds: state.gatingIds, acked };
}

/** Gating ids still awaiting acknowledgment — non-empty blocks Launch. */
export function pendingAckIds(state) {
  if (!state) return [];
  const acked = new Set(state.acked || []);
  return (state.gatingIds || []).filter((id) => !acked.has(id));
}

export function isAcked(state, advisoryId) {
  return !!(state && (state.acked || []).includes(advisoryId));
}

/** The empty client ack state (fresh page / draft reset / template load). */
export function emptyAckState() {
  return { gatingIds: [], acked: [] };
}
