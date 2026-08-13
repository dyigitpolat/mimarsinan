/* The Co-Design platform-physics panel: selector, completeness, constants.

   Everything rendered here is SERVED (`/api/physics_panel`) — the profile
   registry, the constant vocabulary and the objectives registry's own
   availability predicates. The panel therefore cannot disagree with what a run
   will do: a green completeness row is exactly a run that resolves, a grey one
   exactly a run that is refused by name.

   Pure DOM over a served payload, no derived rules of its own: the ONE thing it
   decides locally is the delete-on-empty override semantics, which mirrors the
   sparse-map editors (`search_space`). */

import { el } from './fields.js';
import { clearKey, getKey, setKey } from './state.js';

const PROFILE_KEY = 'platform_physics_profile';
const OVERRIDES_KEY = 'platform_physics_overrides';

let cachedOptions = null;

/** The shipped profiles, fetched once — a registry read, not draft state. */
export async function loadPhysicsProfiles() {
  if (cachedOptions) return cachedOptions;
  cachedOptions = await fetch('/api/physics_profiles')
    .then((r) => (r.ok ? r.json() : []))
    .catch(() => []);
  return cachedOptions;
}

/** The served panel for the draft's own declaration. */
export async function fetchPhysicsPanel() {
  const body = {
    profile: getKey(PROFILE_KEY) || '',
    overrides: getKey(OVERRIDES_KEY) || {},
  };
  return fetch('/api/physics_panel', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then((r) => (r.ok ? r.json() : null)).catch(() => null);
}

function overrides() {
  return { ...(getKey(OVERRIDES_KEY) || {}) };
}

/* An override is DELETED when its input is emptied — an empty string left in the
   map would be a declaration of nothing, which the loader rightly refuses. */
function writeOverride(key, value, unit) {
  const map = overrides();
  if (value === undefined || value === null || value === '') delete map[key];
  else map[key] = { nominal: value, unit, note: 'operator override (wizard)' };
  if (Object.keys(map).length === 0) clearKey(OVERRIDES_KEY);
  else setKey(OVERRIDES_KEY, map);
}

function completenessRow(row) {
  const item = el('div', `physics-completeness-row${row.available ? ' ok' : ' missing'}`);
  item.append(el('span', 'physics-completeness-mark', row.available ? '✓' : '—'));
  item.append(el('span', 'physics-completeness-axis', row.label || row.key));
  item.append(el('span', 'physics-completeness-unit', row.unit));
  if (!row.available) {
    item.append(el('span', 'physics-completeness-reason', `needs ${row.missing}`));
  }
  return item;
}

function valueText(row) {
  if (!row.declared) return '—';
  if (row.banded) return `${row.low} – ${row.high}`;
  return String(row.nominal);
}

function constantRow(row, onEdit) {
  const line = el('div', 'physics-constant'
    + (row.declared ? '' : ' undeclared')
    + (row.overridden ? ' overridden' : ''));
  line.dataset.constant = row.key;

  const name = el('div', 'physics-constant-name');
  name.append(el('code', '', row.key));
  name.title = `${row.doc}\n\nprices: ${row.multiplicand}`;
  line.append(name);

  const value = el('div', 'physics-constant-value');
  value.append(el('span', 'physics-constant-number', valueText(row)));
  value.append(el('span', 'physics-constant-unit', row.unit));
  line.append(value);

  const badge = el('div', 'physics-constant-evidence');
  if (row.declared) {
    const chip = el('span', `evidence-badge ${row.evidence_kind}`, row.evidence_kind);
    chip.title = row.evidence_detail || '';
    badge.append(chip);
    if (row.overridden) badge.append(el('span', 'evidence-badge overridden', 'overridden'));
  } else {
    badge.append(el('span', 'evidence-badge none', 'not declared'));
  }
  line.append(badge);

  const input = el('input', 'physics-constant-override');
  input.type = 'number';
  input.step = 'any';
  input.placeholder = row.declared ? String(row.nominal) : 'declare';
  if (row.overridden) input.value = String(row.nominal);
  input.addEventListener('change', () => {
    const raw = input.value.trim();
    writeOverride(row.key, raw === '' ? undefined : Number(raw), row.unit);
    onEdit();
  });
  line.append(input);
  return line;
}

function groupBlock(group, onEdit) {
  const block = el('details', 'physics-group');
  const summary = el('summary', 'physics-group-summary');
  summary.append(el('span', 'physics-group-name', group.group));
  summary.append(el('span', 'physics-group-count',
    `${group.declared_count}/${group.total_count} declared`));
  block.append(summary);
  const rows = el('div', 'physics-group-rows');
  for (const row of group.constants) rows.append(constantRow(row, onEdit));
  block.append(rows);
  if (group.declared_count > 0) block.open = true;
  return block;
}

function header(panel) {
  const box = el('div', 'physics-header');
  box.append(el('div', 'physics-header-name', panel.display_name));
  const v = panel.validity || {};
  const facts = el('div', 'physics-header-facts');
  if (v.technology_node_nm) facts.append(el('span', 'physics-fact', `${v.technology_node_nm} nm`));
  if (v.supply_v) facts.append(el('span', 'physics-fact', `${v.supply_v} V`));
  if (v.measurement_kind) {
    facts.append(el('span', `physics-fact measurement ${v.measurement_kind}`, v.measurement_kind));
  }
  box.append(facts);
  if (v.notes) {
    const note = el('div', 'physics-header-note', v.notes);
    note.title = v.notes;
    box.append(note);
  }
  return box;
}

/** Render the panel into `host`, re-fetching after every edit. */
export async function renderPhysicsPanel(host, onEdit) {
  const options = await loadPhysicsProfiles();
  const panel = await fetchPhysicsPanel();
  host.textContent = '';

  const card = el('div', 'physics-panel');
  const title = el('div', 'physics-panel-title');
  title.append(el('span', '', 'Target Platform Physics'));
  card.append(title);

  const select = el('select', 'physics-profile-select');
  const none = el('option', '', 'None — no absolute cost');
  none.value = '';
  select.append(none);
  for (const option of options) {
    const item = el('option', '', option.label);
    item.value = option.id;
    select.append(item);
  }
  select.value = (panel && panel.selected) || '';
  select.addEventListener('change', () => {
    if (select.value) setKey(PROFILE_KEY, select.value);
    else clearKey(PROFILE_KEY);
    onEdit();
  });
  card.append(select);

  if (!panel || !panel.validity) {
    card.append(el('div', 'physics-empty',
      'No target physics declared: area, energy and latency objectives are '
      + 'unavailable.'));
  } else {
    card.append(header(panel));
  }

  if (panel && panel.completeness && panel.completeness.length) {
    const readout = el('div', 'physics-completeness');
    readout.append(el('div', 'physics-completeness-title', 'Objectives this target can back'));
    for (const row of panel.completeness) readout.append(completenessRow(row));
    card.append(readout);
  }

  if (panel && panel.groups) {
    for (const group of panel.groups) card.append(groupBlock(group, onEdit));
  }

  host.append(card);
}
