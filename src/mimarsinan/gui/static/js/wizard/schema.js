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

/** Keys of one concern group at one altitude ('basic' | 'advanced'),
    editable in the given section kind. Derived/runtime keys never render. */
export function editableKeys(groupId, category) {
  return Object.values(schema().keys)
    .filter((k) => k.group === groupId && k.category === category)
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
