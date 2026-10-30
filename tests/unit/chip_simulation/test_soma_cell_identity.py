"""Cell identity is point-aware: an ODIN point can never wear a streamed-LIF key.

A coverage coordinate or a regression floor keyed only by the legacy mode
string would let a per_event run be certified against the streamed-LIF floor
and counted as coverage of a cell it never exercised. Every existing point must
keep its historical key byte-identically.
"""

import pytest

from mimarsinan.chip_simulation.certification import CertificationCell
from mimarsinan.chip_simulation.hypervolume_axis_encoder import (
    AxisCoordinates,
    cell_coordinates_from_row,
)
from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW, SomaLaw
from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

_STREAMED = {"spiking_family": "lif", "spiking_variant": "streamed"}
_ODIN = {**_STREAMED, "firing_mode": "Novena", "firing_granularity": "per_event",
         "membrane_bits": 8}


def _plan(**cfg) -> DeploymentPlan:
    base = {"configuration_mode": "user", "spiking_mode": "lif",
            "model_type": "mlp_mixer"}
    base.update(cfg)
    return DeploymentPlan.resolve(base)


class TestCertificationCellDiscriminates:
    def test_the_default_point_keeps_its_historical_key(self):
        cell = CertificationCell.from_mode_policy(
            policy_for_spiking_mode("lif", soma_law=DEFAULT_SOMA_LAW),
            backend="hcm",
        )
        assert cell.variant is None
        assert cell.cell_key == "lif@hcm"

    def test_a_lawless_policy_keeps_its_historical_key(self):
        cell = CertificationCell.from_mode_policy(
            policy_for_spiking_mode("lif"), backend="hcm")
        assert cell.cell_key == "lif@hcm"

    def test_the_odin_point_gets_its_own_cell_key(self):
        law = SomaLaw.resolve(_ODIN)
        cell = CertificationCell.from_mode_policy(
            policy_for_spiking_mode("lif", soma_law=law), backend="hcm")
        assert cell.variant == law.point_tag()
        assert cell.cell_key == "lif@hcm#per_event-sat8"
        assert cell.cell_key != "lif@hcm"

    def test_the_point_key_round_trips(self):
        law = SomaLaw.resolve(_ODIN)
        cell = CertificationCell.from_mode_policy(
            policy_for_spiking_mode("lif", soma_law=law), backend="hcm")
        assert CertificationCell.from_key(cell.cell_key) == cell

    def test_an_explicit_variant_still_wins(self):
        cell = CertificationCell.from_mode_policy(
            policy_for_spiking_mode("lif", soma_law=SomaLaw.resolve(_ODIN)),
            backend="hcm", variant="pruned",
        )
        assert cell.variant == "pruned"


class TestHypervolumeCoordinatesDiscriminate:
    @pytest.mark.parametrize("mode", ["lif", "ttfs", "ttfs_quantized",
                                      "ttfs_cycle_based"])
    def test_every_existing_point_keeps_its_firing_coordinate(self, mode):
        coords = AxisCoordinates.from_plan(_plan(spiking_mode=mode))
        assert coords.firing == mode

    def test_the_odin_point_is_a_distinct_firing_coordinate(self):
        streamed = AxisCoordinates.from_plan(_plan(**_STREAMED))
        odin = AxisCoordinates.from_plan(_plan(**_ODIN))
        assert streamed.firing == "lif"
        assert odin.firing == "lif+per_event-sat8"
        assert odin.firing != streamed.firing

    def test_a_ledger_row_without_the_axes_is_unchanged(self):
        coords = cell_coordinates_from_row({"spiking_mode": "lif"}, sync="none")
        assert coords.firing == "lif"

    def test_a_ledger_row_carrying_the_point_is_discriminated(self):
        coords = cell_coordinates_from_row(
            {"spiking_mode": "lif", "firing_granularity": "per_event",
             "membrane_bits": 8, "spiking_variant": "streamed"},
            sync="none",
        )
        assert coords.firing == "lif+per_event-sat8"
