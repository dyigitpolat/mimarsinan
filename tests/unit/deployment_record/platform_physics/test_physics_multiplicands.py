"""The multiplicand column is a checked contract: every token resolves somewhere."""

from mimarsinan.deployment_record.platform_physics.constants import PHYSICS_CONSTANTS
from mimarsinan.deployment_record.quantities.spec import QUANTITY_SPECS

#: A constant that multiplies nothing (a width, a noise figure, a sharing factor
#: consumed inside another constant's formula) declares itself with this marker.
_DECLARED_MARKER = "declared, not multiplied"

#: Outputs of the pricer itself that static-power constants multiply — priced
#: quantities, not recorded ones.
_PRICED_OUTPUTS = {"e2e_latency_s"}


def test_every_multiplicand_token_resolves():
    """Each token is a quantity key, a physics-constant key, or a priced output —
    so a pricing formula can never reference a number nobody produces."""
    for key, spec in PHYSICS_CONSTANTS.items():
        multiplicand = spec.multiplicand.strip()
        if multiplicand == _DECLARED_MARKER:
            continue
        # " | " separates ALTERNATIVE multiplicands (independent uses of one
        # constant); " x " separates the factors of one product.
        for alternative in multiplicand.split(" | "):
            for token in alternative.split(" x "):
                token = token.strip()
                assert (
                    token in QUANTITY_SPECS
                    or token in PHYSICS_CONSTANTS
                    or token in _PRICED_OUTPUTS
                ), f"{key}: multiplicand token {token!r} resolves to nothing"


def test_priced_energy_constants_multiply_dynamics_not_declarations():
    assert PHYSICS_CONSTANTS["e_synaptic_event_total"].multiplicand == "synaptic_events"
    assert PHYSICS_CONSTANTS["e_inter_tile_hop"].multiplicand == "noc_total_hops"
    # [E3] The inbound channel carries programming payloads and the host's
    # re-injected pass-boundary trains; the readout direction is its own
    # constant, so neither borrows the other's number.
    assert PHYSICS_CONSTANTS["e_dma_per_byte"].multiplicand == (
        "reprogrammed_bytes | carry_in_bytes"
    )
    assert PHYSICS_CONSTANTS["e_readout_per_byte"].multiplicand == "carry_out_bytes"


def test_the_time_converter_multiplies_the_latency_census():
    assert PHYSICS_CONSTANTS["t_cycle"].multiplicand == "latency_steps"


def test_core_init_counts_every_pass_programming_only_reprogram_passes():
    """A resident pass still resets its cores; only a reprogram pass pays payload."""
    assert PHYSICS_CONSTANTS["e_core_init"].multiplicand == "segment_cores"
    assert PHYSICS_CONSTANTS["t_core_init"].multiplicand == "segment_cores"
    assert PHYSICS_CONSTANTS["e_core_program"].multiplicand == "reprogrammed_cores"
    assert PHYSICS_CONSTANTS["t_program_per_byte"].multiplicand == "reprogrammed_bytes"


def test_host_constants_multiply_the_host_wall():
    assert PHYSICS_CONSTANTS["p_host"].multiplicand == "host_ops_s"
    assert PHYSICS_CONSTANTS["host_compute_rate"].multiplicand == "host_ops_s"


def test_static_power_multiplies_the_priced_latency():
    assert "e2e_latency_s" in PHYSICS_CONSTANTS["p_static_per_core"].multiplicand
    assert "e2e_latency_s" in PHYSICS_CONSTANTS["p_static_global"].multiplicand


def test_conversion_quantities_exist_awaiting_their_producer():
    """adc_conversions/adc_count are catalog quantities with no producer until the
    C6 conversion models land — declared now so analog constants name real keys."""
    assert PHYSICS_CONSTANTS["e_adc_conversion"].multiplicand == "adc_conversions"
    assert PHYSICS_CONSTANTS["area_per_adc"].multiplicand == "adc_count"
    assert "adc_conversions" in QUANTITY_SPECS
    assert "adc_count" in QUANTITY_SPECS
