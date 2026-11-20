"""[W15] What the check reads of a run's declaration: the semantics each resolved config executes, and the compaction
liveness rule of the identity program (N-identity) on dead and live dropped columns."""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.certification.program_check.declaration import declaration_from_config, semantics_of_config
from mimarsinan.certification.program_check.report import ESTABLISHED, FAILED

from unit.certification.program_check import programs as P


@pytest.mark.parametrize("config,expected", [
    ({"spiking_family": "lif", "spiking_variant": "streamed"}, "streamed_lif"),
    ({"spiking_family": "lif", "spiking_variant": "synchronized", "spiking_mode": "lif",
      "lif_per_hop_retiming": True}, "windowed_lif"),
    ({"spiking_family": "lif", "spiking_variant": "synchronized", "lif_window_firing": "integrate_then_emit"},
     "synchronized_lif"),
    # Without per-hop re-timing (a Novena firing mode cannot arm it) the variant runs per-cycle segments.
    ({"spiking_family": "lif", "spiking_variant": "synchronized", "firing_mode": "Novena"}, "streamed_lif"),
    ({"spiking_family": "ttfs", "spiking_variant": "analytical"}, "analytical_ttfs"),
    ({"spiking_family": "ttfs", "spiking_variant": "quantized"}, "quantized_ttfs"),
    ({"spiking_family": "ttfs", "spiking_variant": "synchronized"}, "synchronized_ttfs"),
    ({"spiking_family": "ttfs", "spiking_variant": "cascaded"}, "cascaded_ttfs"),
    ({"core_semantics": "mvm"}, "value_float"),
    ({"core_semantics": "mvm", "readout_law": {"adc_bits": 32}}, "value_integer"),
])
def test_the_semantics_a_resolved_config_executes(config, expected, window_firing_declared):
    window_firing_declared()  # [U-LIF2] both values declared, so the integrate_then_emit row resolves
    assert semantics_of_config(config) == expected


def test_a_declaration_reads_the_run_transfer_window_and_encoder():
    decl = declaration_from_config({"spiking_family": "lif", "spiking_variant": "streamed", "simulation_steps": 12,
                                    "spike_phase_dither": True, "cores": P.V.cores(3)})
    assert (decl.semantics, decl.window, decl.pass_transfer) == ("streamed_lif", 12, "verbatim")
    assert decl.dithered_encode and decl.core_types[0]["count"] == 3
    assert declaration_from_config({"core_semantics": "mvm"}).accumulation["int64_golden"] == "int64"


def _masked_ir(dead: bool):
    """Node 1's column 0 dropped by compaction; dead (zero weights) or live (a nonzero weight); node 3 reads it."""
    from unit.mapping.residency.executed import vehicles as V

    ir = P.integer_ir(V.random_rep(0))
    producer = ir.get_neural_cores()[0]
    rows, cols = producer.get_input_count(), producer.get_output_count()
    matrix = np.array(producer.core_matrix, dtype=np.float64)
    matrix[:, 0] = 0.0 if dead else 5.0
    producer.core_matrix = matrix
    producer.pruned_row_mask = [False] * rows
    producer.pruned_col_mask = [c == 0 for c in range(cols)]
    return V.build(ir, 3, None), ir


def test_a_dead_dropped_column_is_the_networks_and_a_live_one_is_named():
    program, ir = _masked_ir(dead=True)
    assert P.run(program, ir, P.decl("synchronized_lif")).results["N-identity"].status == ESTABLISHED
    program, ir = _masked_ir(dead=False)
    result = P.run(program, ir, P.decl("synchronized_lif")).results["N-identity"]
    assert result.status == FAILED and {v.rule for v in result.violations} == {"dropped_read_column"}
