"""The integer certificate on two executors of a multi-pass program, with the per-operator activity census."""

from __future__ import annotations

import numpy as np

from mimarsinan.deployment_record.design_point.executors import FLOW, GOLDEN, certify_program

from . import programs as P


def _certificate(hybrid):
    program = P.residual_program()
    return certify_program(program.model, program.ir, hybrid, P.inputs(), names=P.perceptron_names(program))


def test_the_multi_pass_program_is_exact_and_not_vacuous_on_every_operator():
    hybrid = P.built(3)
    assert sum(1 for s in hybrid.stages if s.kind == "neural") >= 2
    certificate = _certificate(hybrid)
    assert certificate.exact and certificate.nonvacuous
    assert all(e.compared > 0 and e.mismatches == 0 for e in certificate.edges)
    assert len(certificate.observed) == len(certificate.activity)
    assert all(a.nonzero[FLOW] == a.nonzero[GOLDEN] > 0 for a in certificate.activity)
    record = certificate.to_dict()
    assert record["exact"] and record["operators_observed"] == record["operators"]


def test_a_changed_weight_is_found_by_the_certificate():
    hybrid = P.built(3)
    stage = next(s for s in hybrid.stages if s.kind == "neural")
    placement = next(p for core in stage.hard_core_mapping.cores for p in core.matrix_placements
                     if p.is_plain() and np.count_nonzero(np.asarray(p.source)))
    changed = np.array(placement.source, dtype=np.float64, copy=True)
    changed.flat[np.flatnonzero(changed)[0]] += 1.0  # one weight of one placed block, on its own copy
    placement.source = changed
    certificate = _certificate(hybrid)
    assert not certificate.exact
    assert any(e.mismatches for e in certificate.edges if e.between[0] in (FLOW, GOLDEN))
