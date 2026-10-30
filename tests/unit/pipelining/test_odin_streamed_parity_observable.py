"""The streamed per-event point's parity observable and adaptation target.

Doctrine: after LIF adaptation the NF holds CYCLE-ACCURATE spiking nodes, so
NF↔SCM is exact and an agreement statistic can never legitimately read < 1.0.
Every number quoted below was measured on the shipped artifacts of
``generated/odin_narrowconv_lifs_wq_s4_nobias_phased_deployment_run`` and
``generated/t0_55_lifse_simplemlp_wq_s4_nobias_phased_deployment_run``
(streamed lif, per_event, saturating_unsigned/8, Novena, '<=', S=4).

[U-H port] Two of the lineage's three pins here are not ported, because the lineage commit that made them green
(99d54694) is dropped: it changed release-head default paths (the streamed-LIF T-anneal ladder of every run, and the
SCM torch-vs-sim check reworked into a readout-count drift report). They are named in final_units/uh/report.md as a
candidate defect (the release head's drift statistic is an argmax over a tie-dense integer readout, so it measures
float32 tie-breaking at equal counts) and an open item (the per-event ladder's ramp through the deployed composition,
to be gated on the soma point). The measurement-plane pin below holds on the release head as it stands.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.nn.lif_kernels import in_measurement_plane
from mimarsinan.pipelining.core.nf_scm_parity import (
    assert_torch_vs_deployed_sim_parity_or_raise,
)


class _PlaneProbe(nn.Module):
    """Records whether it was called inside the chip-lattice measurement plane."""

    def __init__(self, logits):
        super().__init__()
        self._logits = torch.tensor(logits, dtype=torch.float32)
        self.saw_plane = None

    def forward(self, x):
        self.saw_plane = bool(in_measurement_plane())
        return self._logits[: x.shape[0]]


# Sample 11 of the narrowconv run's own 256-sample parity batch, verbatim.
# The deployed sim reports raw window counts; the NF reports theta * counts
# accumulated in float32. Both sides carry the SAME counts — the streamed
# NF↔SCM exactness gate read 0/16640 count and 0/66560 per-cycle raster
# mismatches at atol=0 on the same model, same step, same samples.
_SIM_COUNTS = [[2.0, 2.0, 3.0, 6.0, 2.0, 5.0, 3.0, 3.0, 6.0, 3.0]]
_NF_THETA_COUNTS = [[
    2.911775, 2.911775, 4.367662, 8.735325, 2.911775,
    7.279438, 4.367662, 4.367662, 8.735326, 4.367662,
]]


class TestParityObservable:
    def test_both_twins_read_inside_the_measurement_plane(self):
        """[defect A'] The gate reads BOTH twins outside the chip-lattice plane.

        Outside ``measurement_plane()`` neither twin snaps its membrane onto
        the integer chip lattice, so neither one is the deployed computation.
        Measured on the t0_55 platform-J artifacts, 64 samples, same model,
        same step: NF vs identity-executor window counts are 0/6144, 0/7680
        and 0/640 mismatched INSIDE the plane and 91/6144, 1834/7680 and
        253/640 mismatched OUTSIDE it (max |delta| = 11 counts). That run was
        FAILED by this gate at 0.7969 while the authoritative streamed
        exactness gate was green at atol=0.
        """
        nf = _PlaneProbe(_NF_THETA_COUNTS)
        sim = _PlaneProbe(_SIM_COUNTS)
        assert_torch_vs_deployed_sim_parity_or_raise(
            nf, sim, torch.zeros(1, 1), min_agreement=0.0,
        )
        assert nf.saw_plane and sim.saw_plane, (
            "both twins must be read inside the chip-lattice measurement "
            f"plane (nf={nf.saw_plane}, sim={sim.saw_plane}); outside it "
            "neither one runs the deployed integer-lattice law"
        )


if __name__ == "__main__":
    pytest.main([__file__])
