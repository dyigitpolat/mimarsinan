/* Schema client: /api/config_schema payload, relevance evaluation, key catalog. */

let _schema = null;

export async function loadSchema() {
  const res = await fetch('/api/config_schema');
  if (!res.ok) throw new Error('config schema unavailable: HTTP ' + res.status);
  _schema = await res.json();
  return _schema;
}

export function schema() {
  if (!_schema) throw new Error('schema not loaded');
  return _schema;
}

export function keySchema(key) {
  return schema().keys[key] || null;
}

export function groups() {
  return schema().groups;
}

/** Evaluate a serialized Relevance tree against a resolved config object. */
export function relevant(tree, cfg) {
  if (!tree || tree.op === 'always') return true;
  switch (tree.op) {
    case 'in': return (tree.values || []).includes(cfg[tree.key]);
    case 'true': return !!cfg[tree.key];
    case 'set': return cfg[tree.key] !== undefined && cfg[tree.key] !== null;
    case 'all': return (tree.items || []).every((t) => relevant(t, cfg));
    case 'any': return (tree.items || []).some((t) => relevant(t, cfg));
    default: return true;
  }
}

/** Mode-aware prominence: an ADVANCED key whose promote_when predicate holds
    renders as PRIMARY — a knob that is the point of the current mode is never
    "advanced". */
export function effectiveCategory(ks, cfg) {
  if (ks.category === 'advanced' && ks.promote_when && relevant(ks.promote_when, cfg)) {
    return 'basic';
  }
  return ks.category;
}

/** Keys of one concern group at one EFFECTIVE altitude ('basic'|'advanced'),
    existing under the current config: relevance controls existence — an
    irrelevant key does not render anywhere (explicit-but-irrelevant values
    stay visible in the diff panel, revertible). Derived/runtime keys never
    render as fields. */
export function visibleKeys(groupId, category, cfg) {
  return Object.values(schema().keys)
    .filter((k) => k.group === groupId
      && (k.category === 'basic' || k.category === 'advanced')
      && effectiveCategory(k, cfg) === category
      && relevant(k.relevant, cfg))
    .map((k) => k.key);
}

export function groupHasKeys(groupId) {
  return Object.values(schema().keys).some(
    (k) => k.group === groupId && (k.category === 'basic' || k.category === 'advanced'),
  );
}

export function derivedKeys() {
  return Object.values(schema().keys).filter((k) => k.category === 'derived');
}

/** Human sentence for why a key is unavailable, derived from its relevance
    tree against the current config (generic; no per-key text in JS). */
export function unavailabilityReason(tree, cfg) {
  if (!tree || tree.op === 'always') return '';
  switch (tree.op) {
    case 'in': {
      const ksDep = keySchema(tree.key);
      const label = ksDep ? ksDep.label : tree.key;
      return `requires ${label} ∈ {${(tree.values || []).join(', ')}} — currently ${JSON.stringify(cfg[tree.key])}`;
    }
    case 'true': {
      const ksDep = keySchema(tree.key);
      return `requires ${ksDep ? ksDep.label : tree.key} to be on`;
    }
    case 'set': {
      const ksDep = keySchema(tree.key);
      return `requires ${ksDep ? ksDep.label : tree.key} to be set`;
    }
    case 'all':
    case 'any': {
      const parts = (tree.items || [])
        .filter((t) => !relevant(t, cfg))
        .map((t) => unavailabilityReason(t, cfg))
        .filter(Boolean);
      return parts.join(tree.op === 'all' ? ' and ' : ' or ');
    }
    default: return '';
  }
}
