/* Draft state: the config DOCUMENT itself (explicit keys only), plus UI state. */

import { keySchema, schema } from './schema.js';

export const state = {
  draft: { deployment_parameters: {}, platform_constraints: {} },
  resolve: null,          // last /api/config/resolve payload
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

/** Mirrors the server diff semantics: explicit null against no default (or a
    null default) is "unset"; a real value against no default differs. */
export function differsFromDefault(key) {
  if (!isExplicit(key)) return false;
  const value = getKey(key);
  const ks = keySchema(key);
  const hasDefault = !!ks && 'default' in ks;
  if (value === null) return hasDefault && ks.default !== null;
  return !hasDefault || JSON.stringify(value) !== JSON.stringify(ks.default);
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
