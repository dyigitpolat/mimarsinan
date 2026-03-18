/* Mimarsinan Pipeline Monitor — Console log tab rendering. */

const _container = () => document.getElementById('console-log-output');

/**
 * Returns true if the console output container is scrolled near the bottom
 * (within 40px), so we know whether to auto-scroll after appending.
 */
function _isNearBottom(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < 40;
}

/**
 * Append an array of log entry objects to the console output panel.
 * Each entry has: { stream: "stdout"|"stderr", line: string, ts: number }.
 * stderr lines receive the `log-stderr` CSS class; stdout get `log-stdout`.
 * Auto-scrolls to the bottom unless the user has scrolled up manually.
 */
export function appendConsoleLogs(entries) {
  if (!entries || entries.length === 0) return;
  const el = _container();
  if (!el) return;

  const shouldScroll = _isNearBottom(el);
  const fragment = document.createDocumentFragment();

  for (const entry of entries) {
    const div = document.createElement('div');
    div.className = 'log-line ' + (entry.stream === 'stderr' ? 'log-stderr' : 'log-stdout');
    div.textContent = entry.line;
    fragment.appendChild(div);
  }

  el.appendChild(fragment);
  if (shouldScroll) el.scrollTop = el.scrollHeight;
}

/**
 * Clear all content from the console output panel and reset scroll.
 */
export function clearConsoleLogs() {
  const el = _container();
  if (el) {
    el.innerHTML = '';
    el.scrollTop = 0;
  }
}
