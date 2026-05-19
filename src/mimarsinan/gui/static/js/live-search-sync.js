/* Routes incremental search_event updates to the active live monitor.
 *
 * WebSocket (main.js) and step-detail incremental refresh both call
 * syncActiveLiveSearch so Compilagent and AgentEvolve monitors stay in
 * sync without duplicating dispatch logic.
 */

import {
  isCompilagentEvent,
  isCompilagentLiveActive,
  syncCompilagentEventsFromState,
} from './compilagent-live.js';
import {
  isSearchLiveActive,
  syncSearchEventsFromState,
} from './search-live.js';

let _liveMonitor = null;
let _remountLiveSearch = null;

export function getLiveMonitor() {
  return _liveMonitor;
}

export function setLiveMonitor(monitor) {
  _liveMonitor = monitor;
}

export function setLiveSearchRemounter(fn) {
  _remountLiveSearch = fn;
}

const AGENT_EVOLVE_TYPES = new Set([
  'generation_start',
  'generation_complete',
  'candidate_result',
  'llm_trace',
  'search_complete',
]);

export function isAgentEvolveEvent(ev) {
  return ev && typeof ev === 'object' && typeof ev.type === 'string'
    && AGENT_EVOLVE_TYPES.has(ev.type);
}

/** Pick compilagent vs agent_evolve from buffered search_event payloads. */
export function detectLiveMonitor(events) {
  for (const ev of events) {
    if (isCompilagentEvent(ev)) return 'compilagent';
    if (isAgentEvolveEvent(ev)) return 'agent_evolve';
  }
  return 'agent_evolve';
}

function _mountedMonitor() {
  if (isCompilagentLiveActive()) return 'compilagent';
  if (isSearchLiveActive()) return 'agent_evolve';
  return null;
}

/**
 * Drain new tail events from state.searchEvents into the live DOM.
 * Remounts the tab when the detected optimizer type disagrees with
 * the currently mounted monitor (e.g. Live Search opened before the
 * first compilagent_* event arrived).
 */
export function syncActiveLiveSearch(stepName, state) {
  if (state.selectedStep !== stepName) return;
  if (state.activeTab !== 'live_search') return;

  const events = (state.searchEvents && state.searchEvents[stepName]) || [];
  const detected = detectLiveMonitor(events);
  const mounted = _mountedMonitor();
  if (!mounted) return;

  const hasTypedEvent = events.some(ev => isCompilagentEvent(ev) || isAgentEvolveEvent(ev));
  if (hasTypedEvent && detected !== mounted && _remountLiveSearch) {
    _remountLiveSearch(stepName, state);
    return;
  }

  if (detected === 'compilagent') {
    syncCompilagentEventsFromState(stepName, state);
  } else {
    syncSearchEventsFromState(stepName, state);
  }
}
