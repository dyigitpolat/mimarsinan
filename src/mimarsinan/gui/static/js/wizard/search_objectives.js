/* The search-objective chip RULES, as pure functions of the served payload.

   They live outside the widget because they are the load-bearing half of the
   arch_search panel: the objectives registry ABORTS a run on an axis the mode
   cannot measure, so a chip offered in the wrong mode is a dead run, not a
   cosmetic slip. A rule that exists only inside a DOM renderer cannot be
   tested without a browser — these have no DOM and no module state, so
   `tests/unit/gui/test_wizard_objective_chip_filter.py` executes THIS module
   under node, and the wizard round-trip test filters the served payload
   through the very code the browser runs instead of re-implementing it. */

/** The search mode the arch_search panel renders for.
 *
 *  Mirrors `pipelining/core/search_mode.derive_search_mode` on every mode a
 *  search can actually run in (joint / hardware / model). It DIVERGES on the
 *  one the registry has no axes for: a draft where neither card searches is
 *  `fixed` to the backend and runs no search at all, and the panel shows the
 *  model-mode chips (the widest offer) rather than an empty box the user
 *  cannot reason about. Nothing is emitted from that state — flipping a card
 *  to `search` re-renders against the real mode. */
export function deriveSearchMode(deploymentParameters) {
  const dp = deploymentParameters || {};
  const model = dp.model_config_mode === 'search';
  const hardware = dp.hw_config_mode === 'search';
  if (model && hardware) return 'joint';
  if (hardware) return 'hardware';
  return 'model';
}

/** The options this mode may offer, from the catalog's availability rows.
 *
 *  An option with NO catalog row is offered everywhere: the served payload is
 *  asserted to carry a row for every option (`test_wizard_search_objectives_
 *  round_trip`), so this hole is visible and pinned rather than silently
 *  swallowing an option the registry never described.
 *
 *  `declaresPhysics` is the draft's OWN answer to "did you select a platform
 *  physics profile": the vendor-priced axes (area in mm², energy in mJ, seconds
 *  of latency) can only be computed from declared constants, so offering one to
 *  a draft that declares none is a chip whose run aborts at objective
 *  resolution. It defaults to false — an undeclared draft is offered only what
 *  it can actually back. */
export function offeredObjectives(nas, searchMode, declaresPhysics = false) {
  const options = (nas && nas.objective_options) || [];
  const availability = new Map(
    ((nas && nas.objective_catalog) || []).map(
      (row) => [row.id, row.available_in_modes || []],
    ),
  );
  return options.filter((option) => {
    if (option.requires_physics && !declaresPhysics) return false;
    const modes = availability.get(option.id);
    return !modes || modes.includes(searchMode);
  });
}

/** The chips that start ACTIVE: the draft's declaration, pruned to what this
 *  mode still offers — an undeclared draft starts with the whole offer. */
export function seededObjectiveIds(offered, declared) {
  const offeredIds = offered.map((option) => option.id);
  return (declared || offeredIds).filter((id) => offeredIds.includes(id));
}
