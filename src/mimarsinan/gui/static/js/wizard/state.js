/* Draft state: the config DOCUMENT itself (explicit keys only), plus UI state. */

import { keySchema, schema } from './schema.js';

export const state = {
  draft: { deployment_parameters: {}, platform_constraints: {} },
  resolve: null,          // last /api/config/resolve payload
  resolving: false,       // a resolve round-trip is pending (stale shimmer)
  hwStats: null,          // last mapping verdict: {status, stats} (hw.js)
  templateName: null,     // banner label when a template/run config is loaded
  editContinueRunId: null,
  prevCompleted: null,    // Set<string> of completed steps (edit & continue)
  dynamicOptions: {},     // flat_key -> [{id,label}]
  modelSchemas: {},       // model_type -> field schema list
  metadata: null,         // resolved data-provider metadata (input_shape, classes)
};

function sectionOf(key) {
  const ks = keySchema(key);
  return ks ? ks.section : 'top';
}

function container(key, create) {
  const section = sectionOf(key);
  if (section === 'top') return state.draft;
  if (!state.draft[section] && create) state.draft[section] = {};
  return state.draft[section] || {};
}

export function getKey(key) {
  return container(key, false)[key];
}

/** Effective value: explicit draft value, else the schema default. */
export function effectiveValue(key) {
  const explicit = getKey(key);
  if (explicit !== undefined) return explicit;
  const ks = keySchema(key);
  return ks && 'default' in ks ? ks.default : undefined;
}

export function isExplicit(key) {
  return getKey(key) !== undefined;
}

/** The wizard's presented default: the starter-baseline pin where one exists
    (the baseline IS the defaults), else the workload-neutral schema default.
    Returns { has, value, fromBaseline }. */
export function wizardDefault(ks) {
  if (!ks) return { has: false, value: undefined, fromBaseline: false };
  if ('baseline' in ks) return { has: true, value: ks.baseline, fromBaseline: true };
  if ('default' in ks) return { has: true, value: ks.default, fromBaseline: false };
  return { has: false, value: undefined, fromBaseline: false };
}

/** The concrete PROSPECTIVE value the SSOT deriver produces for the current
    draft. Precedence mirrors the run's: a provider/builder registration
    (workload facts) beats the registry's frozen fallback; the server's
    `derived_values` carries everything else, already resolved against the draft.
    Undefined while the draft errors — a hypothetical value never renders. */
export function resolvedValue(key) {
  const facts = state.metadata && state.metadata.workload_facts;
  if (facts && facts[key] !== undefined && facts[key] !== null) return facts[key];
  const derived = state.resolve && state.resolve.derived_values;
  if (derived && key in derived && derived[key] !== null) return derived[key];
  const fromResolve = state.resolve && state.resolve.resolved
    ? state.resolve.resolved[key] : undefined;
  if (fromResolve !== undefined && fromResolve !== null) return fromResolve;
  return undefined;
}

/** THE legal value set of `key` under the current config state, as the server's
    derivation computed it (always served, even while the draft errors).
    `undefined` = the key's legality does not depend on other config. */
export function legalValues(key) {
  const sets = state.resolve && state.resolve.legal_values;
  return sets && Array.isArray(sets[key]) ? sets[key] : undefined;
}

/** |legal| == 1 ⇒ the field is LOCKED and this is the value it must hold.
    `undefined` ⇒ not locked (either no legality rule, or more than one legal
    value, or the legal set has not arrived yet). */
export function lockedValue(key) {
  const legal = legalValues(key);
  return legal && legal.length === 1 ? legal[0] : undefined;
}

/** Mirrors the server diff semantics against the WIZARD default (baseline
    first): explicit null against no default (or a null default) is "unset";
    a real value against no default differs. */
export function differsFromDefault(key) {
  if (!isExplicit(key)) return false;
  const value = getKey(key);
  const base = wizardDefault(keySchema(key));
  if (value === null) return base.has && base.value !== null;
  return !base.has || JSON.stringify(value) !== JSON.stringify(base.value);
}

export function setKey(key, value) {
  container(key, true)[key] = value;
}

export function clearKey(key) {
  delete container(key, true)[key];
}

/** Flat view for relevance evaluation between resolve round-trips:
    schema defaults overlaid with the draft's flat keys and the last
    server-derived values. */
export function effectiveConfig() {
  const cfg = {};
  for (const [key, ks] of Object.entries(schema().keys)) {
    if ('default' in ks) cfg[key] = ks.default;
  }
  for (const [key, value] of Object.entries(state.draft)) {
    if (key !== 'deployment_parameters' && key !== 'platform_constraints') cfg[key] = value;
  }
  Object.assign(cfg, state.draft.deployment_parameters || {});
  Object.assign(cfg, state.draft.platform_constraints || {});
  if (state.resolve && state.resolve.derived) {
    for (const [key, info] of Object.entries(state.resolve.derived)) cfg[key] = info.value;
  }
  return cfg;
}

export function resetDraft() {
  state.draft = { deployment_parameters: {}, platform_constraints: {} };
  state.resolve = null;
  state.templateName = null;
}

export function loadDraftFromConfig(config, { templateName = null } = {}) {
  const doc = JSON.parse(JSON.stringify(config || {}));
  if (!doc.deployment_parameters) doc.deployment_parameters = {};
  if (!doc.platform_constraints) doc.platform_constraints = {};
  state.draft = doc;
  state.templateName = templateName;
}
