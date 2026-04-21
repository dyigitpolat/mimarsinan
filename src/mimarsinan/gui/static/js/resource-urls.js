/* Resource URL resolver for lazy heatmaps and connectivity payloads.
 *
 * The backend serves resources at one of three prefixes depending on
 * run context:
 *   • live (current process):        /api/steps/{step}/resources/{kind}/{rid}
 *   • historical (finished runs):    /api/runs/{run}/steps/{step}/resources/{kind}/{rid}
 *   • active subprocess runs:        /api/active_runs/{run}/steps/{step}/resources/{kind}/{rid}
 *
 * The callers only carry a ``{kind, rid}`` hint (produced by
 * ``build_step_snapshot``). We resolve it against the currently
 * selected run via a small global context set by ``step-detail.js``
 * before tabs render, so tab modules can stay stateless.
 */

let _currentContext = null;

export function setResourceContext({ stepName, historicalRunId, isActiveRun } = {}) {
  _currentContext = { stepName, historicalRunId, isActiveRun };
}

export function getResourceContext() {
  return _currentContext;
}

export function resourceUrl(resourceRef, context = _currentContext) {
  if (!resourceRef || !resourceRef.kind || !resourceRef.rid) return null;
  if (!context || !context.stepName) return null;
  const step = encodeURIComponent(context.stepName);
  const kind = encodeURIComponent(resourceRef.kind);
  const rid = String(resourceRef.rid).split('/').map(encodeURIComponent).join('/');
  if (context.historicalRunId && context.isActiveRun) {
    return `/api/active_runs/${encodeURIComponent(context.historicalRunId)}/steps/${step}/resources/${kind}/${rid}`;
  }
  if (context.historicalRunId) {
    return `/api/runs/${encodeURIComponent(context.historicalRunId)}/steps/${step}/resources/${kind}/${rid}`;
  }
  return `/api/steps/${step}/resources/${kind}/${rid}`;
}

// Convenience for templating: returns an attribute-safe URL or empty
// string if the reference is missing. Always applies HTML entity
// escaping for double quotes so it can drop straight into <img src="…">.
export function imgSrcAttr(resourceRef, context = _currentContext) {
  const url = resourceUrl(resourceRef, context);
  if (!url) return '';
  return url.replace(/"/g, '&quot;');
}
