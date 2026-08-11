/* The chip capability declaration, client-side — the mirror of the server's
   `ChipCapabilities.layout_kwargs()`.

   The layout answer the Mapping Performance panel previews is not a function of
   the permission bits alone: `schedule_policy` and `max_schedule_passes` decide
   which passes the hard-core builder composes, and a preview computed without
   them describes a program the chip never runs. ONE place therefore declares
   which keys travel with a layout request, so a capability added server-side
   cannot be silently dropped by this client (a test pins the two key sets
   equal). Pure — no DOM, no state — so it is executable under Node. */

/** The declared capability set, read through `read(key)` (the wizard's
    `effectiveValue`). Defaults mirror the server's `from_platform_constraints`. */
export function capabilityDeclaration(read) {
  return {
    allow_coalescing: !!read('allow_coalescing'),
    allow_neuron_splitting: !!read('allow_neuron_splitting'),
    allow_scheduling: !!read('allow_scheduling'),
    schedule_policy: read('schedule_policy') || 'pool',
    max_schedule_passes: Number(read('max_schedule_passes')) || 8,
  };
}

/** The `/api/hw_config_verify` request body: the model repr, the declared core
    grid, and the WHOLE capability declaration alongside them. */
export function hwVerifyBody(modelBody, cores, capabilities) {
  return {
    model_repr_json: modelBody,
    core_types: (cores || []).map((c) => ({
      max_axons: c.max_axons, max_neurons: c.max_neurons, count: c.count,
    })),
    ...capabilities,
  };
}
