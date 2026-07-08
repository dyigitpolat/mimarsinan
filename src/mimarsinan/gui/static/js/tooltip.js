/* Immediate app-wide tooltips: every `title` attribute renders through one
   custom tooltip with ZERO hover delay (the native tooltip's ~1s delay never
   fires — the title is moved to data-tip on first hover). Self-installing on
   import; styles are injected so the component stays a single file. */

const TIP_CSS = `
.app-tooltip {
  position: fixed;
  z-index: 10000;
  max-width: 380px;
  padding: 8px 10px;
  border-radius: 6px;
  border: 1px solid rgba(148, 163, 184, 0.25);
  background: #101725;
  color: #dbe4f0;
  font: 500 11.5px/1.5 'Inter', 'Segoe UI', sans-serif;
  letter-spacing: 0.01em;
  white-space: pre-line;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
  pointer-events: none;
  opacity: 0;
  transition: opacity 80ms ease;
}
.app-tooltip.visible { opacity: 1; }
`;

let _tip = null;
let _anchor = null;

function ensureTip() {
  if (_tip) return _tip;
  const style = document.createElement('style');
  style.textContent = TIP_CSS;
  document.head.append(style);
  _tip = document.createElement('div');
  _tip.className = 'app-tooltip';
  _tip.setAttribute('role', 'tooltip');
  document.body.append(_tip);
  return _tip;
}

/** Move a native title into data-tip once so the browser tooltip never fires. */
function tipText(el) {
  if (el.getAttribute('title')) {
    el.dataset.tip = el.getAttribute('title');
    el.removeAttribute('title');
  }
  return el.dataset.tip || '';
}

function position(tip, target) {
  const rect = target.getBoundingClientRect();
  const margin = 8;
  tip.style.left = '0px';
  tip.style.top = '0px';
  const tw = tip.offsetWidth;
  const th = tip.offsetHeight;
  let left = Math.min(
    Math.max(margin, rect.left), window.innerWidth - tw - margin,
  );
  let top = rect.bottom + 6;
  if (top + th > window.innerHeight - margin) top = rect.top - th - 6;
  if (top < margin) top = margin;
  tip.style.left = `${Math.round(left)}px`;
  tip.style.top = `${Math.round(top)}px`;
}

function show(target) {
  const text = tipText(target);
  if (!text) return;
  const tip = ensureTip();
  _anchor = target;
  tip.textContent = text;
  position(tip, target);
  tip.classList.add('visible');
}

function hide() {
  _anchor = null;
  if (_tip) _tip.classList.remove('visible');
}

export function installImmediateTooltips(root = document) {
  root.addEventListener('pointerover', (event) => {
    const target = event.target.closest('[title], [data-tip]');
    if (!target) return;
    if (_anchor === target) return;
    show(target);
  });
  root.addEventListener('pointerout', (event) => {
    if (!_anchor) return;
    if (event.target === _anchor || _anchor.contains(event.target)) {
      const to = event.relatedTarget;
      if (!to || !_anchor.contains(to)) hide();
    }
  });
  root.addEventListener('focusin', (event) => {
    const target = event.target.closest && event.target.closest('[title], [data-tip]');
    if (target) show(target);
  });
  root.addEventListener('focusout', hide);
  window.addEventListener('scroll', hide, true);
}

installImmediateTooltips();
