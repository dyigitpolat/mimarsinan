/* The wizard step flow: Workload → Model → Deployment → Tuning & Budgets →
   Review & Launch. Steps host whole concern GROUPS (the registry taxonomy is
   the step assignment); the stepper shows progress + per-step error badges. */

import { keySchema } from './schema.js';
import { el } from './fields.js';

export const STEPS = [
  { id: 'workload', title: 'Workload', hint: 'Dataset & preprocessing',
    groups: ['workload'] },
  { id: 'model', title: 'Model', hint: 'Architecture & weights',
    groups: ['model'] },
  { id: 'deployment', title: 'Deployment', hint: 'Mode · platform · precision · S',
    groups: ['spiking', 'conversion', 'hardware', 'deployment_target'] },
  { id: 'tuning', title: 'Tuning & Budgets', hint: 'Controller & training',
    groups: ['tuning', 'training'] },
  { id: 'review', title: 'Review & Launch', hint: 'Verify & deploy',
    groups: ['run'] },
];

let _current = STEPS[0].id;
const _visited = new Set([STEPS[0].id]);

export function currentStep() {
  return _current;
}

export function stepOfGroup(groupId) {
  const step = STEPS.find((s) => s.groups.includes(groupId));
  return step ? step.id : 'review';
}

export function stepOfKey(key) {
  const ks = keySchema(key);
  return ks ? stepOfGroup(ks.group) : 'review';
}

/** Per-step error counts from the resolve payload (keyless errors → review). */
export function errorCountsByStep(errors) {
  const counts = {};
  for (const error of errors || []) {
    const step = error.key ? stepOfKey(error.key) : 'review';
    counts[step] = (counts[step] || 0) + 1;
  }
  return counts;
}

export function goToStep(stepId) {
  if (!STEPS.some((s) => s.id === stepId)) return;
  _current = stepId;
  _visited.add(stepId);
  document.querySelectorAll('.step-panel').forEach((panel) => {
    panel.classList.toggle('active', panel.dataset.step === stepId);
  });
  syncNavButtons();
  window.scrollTo({ top: 0 });
  document.dispatchEvent(new CustomEvent('wizard:step', { detail: { step: stepId } }));
}

function stepIndex(stepId) {
  return STEPS.findIndex((s) => s.id === stepId);
}

function syncNavButtons() {
  const i = stepIndex(_current);
  const back = document.getElementById('navBackBtn');
  const next = document.getElementById('navNextBtn');
  if (back) back.style.visibility = i > 0 ? '' : 'hidden';
  if (next) {
    next.style.display = i < STEPS.length - 1 ? '' : 'none';
    next.textContent = `Next: ${STEPS[i + 1] ? STEPS[i + 1].title : ''} →`;
  }
}

export function renderStepper(errors) {
  const host = document.getElementById('stepper');
  if (!host) return;
  const counts = errorCountsByStep(errors);
  host.replaceChildren();
  STEPS.forEach((step, i) => {
    if (i > 0) host.append(el('span', 'stepper-link'));
    const chip = el('button', 'stepper-chip');
    chip.type = 'button';
    chip.dataset.step = step.id;
    const errorCount = counts[step.id] || 0;
    const done = _visited.has(step.id) && errorCount === 0 && step.id !== _current;
    chip.classList.toggle('active', step.id === _current);
    chip.classList.toggle('done', done);
    chip.classList.toggle('has-errors', errorCount > 0);
    const index = el('span', 'stepper-index', done ? '✓' : String(i + 1));
    const titles = el('span', 'stepper-titles');
    titles.append(el('span', 'stepper-title', step.title));
    titles.append(el('span', 'stepper-hint', step.hint));
    chip.append(index, titles);
    if (errorCount > 0) {
      const badge = el('span', 'stepper-badge', String(errorCount));
      badge.title = `${errorCount} validation error(s) on this step`;
      chip.append(badge);
    }
    chip.addEventListener('click', () => {
      goToStep(step.id);
      renderStepper(errors);
    });
    host.append(chip);
  });
  syncNavButtons();
}

export function bindStepNav(onNavigate) {
  document.getElementById('navBackBtn')?.addEventListener('click', () => {
    const i = stepIndex(_current);
    if (i > 0) { goToStep(STEPS[i - 1].id); onNavigate(); }
  });
  document.getElementById('navNextBtn')?.addEventListener('click', () => {
    const i = stepIndex(_current);
    if (i < STEPS.length - 1) { goToStep(STEPS[i + 1].id); onNavigate(); }
  });
}

/** Jump to the first step carrying an error (the launch guard's remedy). */
export function goToFirstError(errors) {
  const counts = errorCountsByStep(errors);
  const step = STEPS.find((s) => counts[s.id]);
  if (step) goToStep(step.id);
}
