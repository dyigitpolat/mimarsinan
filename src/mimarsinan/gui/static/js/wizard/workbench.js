/* The workbench section rail: free navigation (no gated steps). Sections host
   whole registry concern GROUPS — the taxonomy is the placement — and show
   per-section error badges from the resolve payload. */

import { keySchema } from './schema.js';
import { el } from './fields.js';

export const SECTIONS = [
  { id: 'workload', title: 'Workload', hint: 'Dataset & preprocessing',
    icon: '◈', groups: ['workload'] },
  { id: 'codesign', title: 'Co-Design', hint: 'Model × hardware × mapping',
    icon: '⧉', groups: ['model', 'hardware', 'mapping_strategy', 'co_search'] },
  { id: 'semantics', title: 'Deployment semantics', hint: 'Mode · conversion · vehicles',
    icon: '⚡', groups: ['spiking', 'conversion', 'deployment_target'] },
  { id: 'training', title: 'Training & Tuning', hint: 'Pretraining × adaptation controller',
    icon: '▶', groups: ['training', 'tuning'] },
  { id: 'review', title: 'Review & Launch', hint: 'Verify & deploy',
    icon: '⬡', groups: ['run'] },
];

let _current = SECTIONS[0].id;

export function currentSection() {
  return _current;
}

export function sectionOfGroup(groupId) {
  const section = SECTIONS.find((s) => s.groups.includes(groupId));
  return section ? section.id : 'review';
}

export function sectionOfKey(key) {
  const ks = keySchema(key);
  return ks ? sectionOfGroup(ks.group) : 'review';
}

/** Per-section error counts from the resolve payload (keyless errors → review). */
export function errorCountsBySection(errors) {
  const counts = {};
  for (const error of errors || []) {
    const section = error.key ? sectionOfKey(error.key) : 'review';
    counts[section] = (counts[section] || 0) + 1;
  }
  return counts;
}

export function goToSection(sectionId) {
  if (!SECTIONS.some((s) => s.id === sectionId)) return;
  _current = sectionId;
  document.querySelectorAll('.wb-section').forEach((panel) => {
    panel.classList.toggle('active', panel.dataset.sectionId === sectionId);
  });
  document.querySelectorAll('.wb-nav-item').forEach((item) => {
    item.classList.toggle('active', item.dataset.sectionId === sectionId);
  });
  const main = document.getElementById('wizard');
  if (main) main.scrollTop = 0;
  window.scrollTo({ top: 0 });
  document.dispatchEvent(new CustomEvent('wizard:section', { detail: { section: sectionId } }));
}

export function renderSectionNav(errors) {
  const host = document.getElementById('sectionNav');
  if (!host) return;
  const counts = errorCountsBySection(errors);
  host.replaceChildren();
  for (const section of SECTIONS) {
    const item = el('button', 'wb-nav-item');
    item.type = 'button';
    item.dataset.sectionId = section.id;
    item.classList.toggle('active', section.id === _current);
    const errorCount = counts[section.id] || 0;
    item.classList.toggle('has-errors', errorCount > 0);
    item.append(el('span', 'wb-nav-icon', section.icon));
    const titles = el('span', 'wb-nav-titles');
    titles.append(el('span', 'wb-nav-title', section.title));
    titles.append(el('span', 'wb-nav-hint', section.hint));
    item.append(titles);
    if (errorCount > 0) {
      const badge = el('span', 'wb-nav-badge', String(errorCount));
      badge.title = `${errorCount} validation error(s) in this section`;
      item.append(badge);
    }
    item.addEventListener('click', () => goToSection(section.id));
    host.append(item);
  }
}

/** Jump to the first section carrying an error (the launch guard's remedy). */
export function goToFirstError(errors) {
  const counts = errorCountsBySection(errors);
  const section = SECTIONS.find((s) => counts[s.id]);
  if (section) goToSection(section.id);
}
