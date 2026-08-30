"""The streamed per-event point's parity observable and adaptation target.

Doctrine: after LIF adaptation the NF holds CYCLE-ACCURATE spiking nodes, so
NF↔SCM is exact and an agreement statistic can never legitimately read < 1.0.
Every number quoted below was measured on the shipped artifacts of
``generated/odin_narrowconv_lifs_wq_s4_nobias_phased_deployment_run`` and
``generated/t0_55_lifse_simplemlp_wq_s4_nobias_phased_deployment_run``
(streamed lif, per_event, saturating_unsigned/8, Novena, '<=', S=4).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.models.nn.lif_kernels import in_measurement_plane
from mimarsinan.pipelining.core.nf_scm_parity import (
    measure_readout_decision_drift,
)


class _Fixed(nn.Module):
    """A parameter-free twin returning a fixed logit block."""

    def __init__(self, logits):
        super().__init__()
        self._logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, x):
        return self._logits[: x.shape[0]]


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
    def test_identical_counts_must_read_full_agreement(self):
        """[defect A] The gate reads argmax over a TIE-DENSE integer readout.

        Class 3 and class 8 both fired 6 spikes. The sim, holding exact
        integers, breaks the tie at the lower index; the NF, holding
        theta*6 summed in float32, lands one ULP apart (8.735325 vs
        8.735326) and breaks it the other way. All 3 of the 3 flips behind
        the run's 0.98828125 reading are of exactly this kind: recovered
        counts identical, top count tied. The statistic therefore measures
        float32 tie-breaking, not deployment fidelity, and its shortfall is
        reported as "a deployment-fidelity regression".
        """
        counts = np.asarray(_NF_THETA_COUNTS[0]) / (
            _NF_THETA_COUNTS[0][3] / _SIM_COUNTS[0][3]
        )
        assert np.array_equal(np.rint(counts), np.asarray(_SIM_COUNTS[0])), (
            "fixture invariant: both twins must carry the same window counts"
        )
        agreement = measure_readout_decision_drift(
            _Fixed(_NF_THETA_COUNTS), _Fixed(_SIM_COUNTS),
            torch.zeros(1, 1), min_agreement=0.0,
        )
        assert agreement == 1.0, (
            "two twins carrying identical window counts must read full "
            f"agreement; got {agreement}"
        )

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
        measure_readout_decision_drift(
            nf, sim, torch.zeros(1, 1), min_agreement=0.0,
        )
        assert nf.saw_plane and sim.saw_plane, (
            "both twins must be read inside the chip-lattice measurement "
            f"plane (nf={nf.saw_plane}, sim={sim.saw_plane}); outside it "
            "neither one runs the deployed integer-lattice law"
        )


class TestAdaptationTarget:
    def test_streamed_per_event_ladder_ramps_through_the_deployed_forward(self):
        """[defect B] The LIF ladder optimises a forward it never deploys.

        For the streamed per-event point the ramp strategy is a
        ``ValueDomainProxyRamp``: ``ramp_forward`` is None, so no cross-layer
        forward is installed and the rungs train the per-perceptron
        rate-mode LIF staircase (constant drive, <= 1 spike/cycle) instead of
        the deployed event-serial fold. Measured on the narrowconv artifacts
        over the tuner's own 23-batch eval set: at LIF-adaptation ENTRY the
        ladder's view reads 0.8921 while the deployed view reads 0.5084
        (+38.4 pp optimistic); at EXIT they have swapped — deployed 0.8903,
        ladder view 0.4718. The two are different functions throughout.
        """
        from mimarsinan.tuning.orchestration.lif_adaptation_plan import (
            LifAdaptationPlan,
        )
        from mimarsinan.tuning.orchestration.mbh_tanneal import (
            TAnnealRealizableRamp,
        )

        config = {
            "spiking_mode": "lif",
            "spiking_family": "lif",
            "spiking_variant": "streamed",
            "firing_granularity": "per_event",
            "firing_mode": "Novena",
            "thresholding_mode": "<=",
            "membrane_bits": 8,
            "simulation_steps": 4,
            "cycle_accurate_lif_forward": True,
            "lif_tanneal": True,
            "endpoint_recovery_steps": 600,
        }
        plan = LifAdaptationPlan.resolve(config)
        schedule = plan.tanneal_schedule(plan.blend_fast_rates)
        assert schedule is not None, "fixture invariant: the recipe T-anneals"
        ramp = TAnnealRealizableRamp(schedule)
        assert ramp.ramp_forward(None, None) is not None, (
            "the streamed per-event ladder must ramp THROUGH the deployed "
            "composition (the forward the tuner finalizes with); a "
            "value-domain proxy ramp trains a rate-mode staircase the chip "
            "never runs"
        )


if __name__ == "__main__":
    pytest.main([__file__])
