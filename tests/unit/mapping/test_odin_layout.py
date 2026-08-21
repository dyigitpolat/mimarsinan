"""[ODIN4] Stock-ODIN bit layouts: neuron word (params + STATE), synapse memory, SYN_SIGN.

The layouts are transcribed from the upstream RTL (`ChFrenkel/ODIN`). These tests
are the golden pack/unpack round-trip the plan's Sec.7 row 14 demands, and they pin
the transcription itself against the numbers quoted in `doc/README.md` Sec.3.3.2 /
Sec.3.1 / Sec.4 so a table typo cannot survive.
"""

import pytest

from mimarsinan.mapping.export.odin.layout import (
    LIF_NEURON_WORD_FIELDS,
    NEURON_WORD_BITS,
    NEURON_WORD_BYTES,
    STATE_FIELD,
    OdinLayoutError,
    lif_neuron_field,
    masked_state_byte_writes,
    neuron_state_bit_mask,
    neuron_word_from_bytes,
    neuron_word_to_bytes,
    pack_neuron_word,
    unpack_neuron_word,
)
from mimarsinan.mapping.export.odin.registers import (
    CONFIG_REGISTERS,
    INFERENCE_REGISTER_VALUES,
    SYN_SIGN_BASE_ADDR,
    config_register,
    inference_register_writes,
    pack_syn_sign,
    syn_sign_register_writes,
    unpack_syn_sign,
)
from mimarsinan.mapping.export.odin.synapse import (
    SYNAPSE_WORD_COUNT,
    pack_synapse_nibble,
    pack_synapse_words,
    synapse_address,
    unpack_synapse_nibble,
    unpack_synapse_words,
)


# --------------------------------------------------------------------------
# A1. The neuron word table matches the upstream LIF port map, field by field.
# --------------------------------------------------------------------------

# doc/README.md Sec.3.3.2 table, transcribed independently of the module under test.
EXPECTED_NEURON_FIELDS = {
    "lif_izh_sel": (0, 1, "parameter"),
    "leak_str": (1, 7, "parameter"),
    "leak_en": (8, 1, "parameter"),
    "thr": (9, 8, "parameter"),
    "ca_en": (17, 1, "parameter"),
    "thetamem": (18, 8, "parameter"),
    "ca_theta1": (26, 3, "parameter"),
    "ca_theta2": (29, 3, "parameter"),
    "ca_theta3": (32, 3, "parameter"),
    "ca_leak": (35, 5, "parameter"),
    "vmem": (70, 8, "state"),
    "calcium": (78, 3, "state"),
    "caleak_cnt": (81, 5, "state"),
    "neur_disable": (127, 1, "parameter"),
}


class TestTheNeuronWordTableIsTheUpstreamTable:
    def test_every_field_matches_the_documented_bit_range(self):
        actual = {
            f.name: (f.lsb, f.width, f.kind) for f in LIF_NEURON_WORD_FIELDS
        }
        assert actual == EXPECTED_NEURON_FIELDS

    def test_no_two_fields_overlap_and_all_fit_the_word(self):
        seen = 0
        for field in LIF_NEURON_WORD_FIELDS:
            assert field.lsb + field.width <= NEURON_WORD_BITS
            assert seen & field.mask == 0, f"{field.name} overlaps an earlier field"
            seen |= field.mask

    def test_the_three_state_fields_are_the_only_state_fields(self):
        state = {f.name for f in LIF_NEURON_WORD_FIELDS if f.kind == STATE_FIELD}
        assert state == {"vmem", "calcium", "caleak_cnt"}

    def test_the_state_mask_is_bits_70_through_85(self):
        assert neuron_state_bit_mask() == ((1 << 86) - 1) ^ ((1 << 70) - 1)


# --------------------------------------------------------------------------
# A2. Golden pack/unpack round trip, INCLUDING the state fields.
# --------------------------------------------------------------------------

def _full_neuron_values(**overrides):
    values = {f.name: 0 for f in LIF_NEURON_WORD_FIELDS}
    values["lif_izh_sel"] = 1
    values.update(overrides)
    return values


class TestNeuronWordPackUnpackRoundTrip:
    def test_a_deployment_word_round_trips_exactly(self):
        values = _full_neuron_values(
            thr=200, vmem=137, calcium=5, caleak_cnt=19, neur_disable=0
        )
        word = pack_neuron_word(values)
        assert unpack_neuron_word(word) == values

    @pytest.mark.parametrize("name", sorted(EXPECTED_NEURON_FIELDS))
    def test_every_field_round_trips_at_its_maximum(self, name):
        lsb, width, _kind = EXPECTED_NEURON_FIELDS[name]
        values = _full_neuron_values(**{name: (1 << width) - 1})
        word = pack_neuron_word(values)
        assert unpack_neuron_word(word)[name] == (1 << width) - 1
        assert (word >> lsb) & ((1 << width) - 1) == (1 << width) - 1

    def test_the_state_fields_land_at_the_documented_bits(self):
        word = pack_neuron_word(
            _full_neuron_values(vmem=0xA5, calcium=0b101, caleak_cnt=0b10011)
        )
        assert (word >> 70) & 0xFF == 0xA5
        assert (word >> 78) & 0b111 == 0b101
        assert (word >> 81) & 0b11111 == 0b10011

    def test_bytes_round_trip_in_spi_byte_address_order(self):
        word = pack_neuron_word(_full_neuron_values(thr=77, vmem=250, calcium=7))
        payload = neuron_word_to_bytes(word)
        assert len(payload) == NEURON_WORD_BYTES
        assert payload[0] == word & 0xFF
        assert neuron_word_from_bytes(payload) == word

    def test_a_missing_field_is_refused_by_name(self):
        values = _full_neuron_values()
        del values["thr"]
        with pytest.raises(OdinLayoutError, match="thr"):
            pack_neuron_word(values)

    def test_an_unknown_field_is_refused_by_name(self):
        with pytest.raises(OdinLayoutError, match="v_membrane"):
            pack_neuron_word(_full_neuron_values(v_membrane=1))

    def test_an_out_of_range_value_is_refused_with_its_width(self):
        with pytest.raises(OdinLayoutError, match="thr"):
            pack_neuron_word(_full_neuron_values(thr=256))

    def test_a_negative_value_is_refused(self):
        with pytest.raises(OdinLayoutError, match="vmem"):
            pack_neuron_word(_full_neuron_values(vmem=-1))

    def test_the_field_lookup_names_the_unknown_field(self):
        assert lif_neuron_field("thr").width == 8
        with pytest.raises(OdinLayoutError, match="nope"):
            lif_neuron_field("nope")


class TestTheClearStageRewritesExactlyTheStateBytes:
    """SPI writes are byte-granular with a mask where 1 = masked = keep old."""

    def test_only_bytes_8_9_10_are_touched(self):
        word = pack_neuron_word(_full_neuron_values(vmem=0, calcium=0, caleak_cnt=0))
        writes = masked_state_byte_writes(word)
        assert [w.byte_addr for w in writes] == [8, 9, 10]

    def test_the_masks_protect_every_non_state_bit(self):
        word = pack_neuron_word(_full_neuron_values(vmem=0xFF, calcium=7, caleak_cnt=31))
        masks = {w.byte_addr: w.mask for w in masked_state_byte_writes(word)}
        # byte 8 holds bits 64..71: only 70 and 71 are state.
        assert masks[8] == 0b00111111
        # byte 9 holds bits 72..79: all state (vmem[2:7], calcium[0:1]).
        assert masks[9] == 0x00
        # byte 10 holds bits 80..87: bits 80..85 are state.
        assert masks[10] == 0b11000000

    def test_applying_the_writes_to_a_dirty_word_zeroes_only_the_state(self):
        dirty = pack_neuron_word(
            _full_neuron_values(thr=201, vmem=0xFF, calcium=7, caleak_cnt=31)
        )
        clean = pack_neuron_word(
            _full_neuron_values(thr=201, vmem=0, calcium=0, caleak_cnt=0)
        )
        payload = list(neuron_word_to_bytes(dirty))
        for write in masked_state_byte_writes(clean):
            old = payload[write.byte_addr]
            payload[write.byte_addr] = (write.value & ~write.mask) | (old & write.mask)
        assert neuron_word_from_bytes(payload) == clean


# --------------------------------------------------------------------------
# A3. Synapse memory: 8192 x 32, 8 nibbles, mapping bit + 3-bit magnitude.
# --------------------------------------------------------------------------

class TestSynapseAddressing:
    def test_the_address_split_is_the_documented_one(self):
        for pre, post in ((0, 0), (1, 7), (255, 255), (37, 130), (128, 9)):
            word_addr, byte_addr, nibble = synapse_address(pre, post)
            assert word_addr == (pre << 5) | (post >> 3)
            assert byte_addr == (post >> 1) & 0b11
            assert nibble == post & 0b1

    def test_every_pre_post_pair_maps_to_a_distinct_nibble_slot(self):
        seen = set()
        for pre in (0, 1, 128, 255):
            for post in range(256):
                seen.add(synapse_address(pre, post))
        assert len(seen) == 4 * 256

    def test_out_of_range_addresses_are_refused(self):
        with pytest.raises(OdinLayoutError):
            synapse_address(256, 0)
        with pytest.raises(OdinLayoutError):
            synapse_address(0, -1)


class TestSynapseWordPacking:
    def test_a_nibble_is_the_mapping_bit_over_a_three_bit_magnitude(self):
        assert pack_synapse_nibble(magnitude=5, mapped=False) == 0b0101
        assert pack_synapse_nibble(magnitude=5, mapped=True) == 0b1101
        assert unpack_synapse_nibble(0b1101) == (5, True)
        assert unpack_synapse_nibble(0b0101) == (5, False)

    def test_a_magnitude_above_seven_is_refused(self):
        with pytest.raises(OdinLayoutError, match="magnitude"):
            pack_synapse_nibble(magnitude=8, mapped=False)

    def test_the_image_has_one_word_per_documented_address(self):
        magnitudes = [[0] * 256 for _ in range(256)]
        words = pack_synapse_words(magnitudes, mapped=False)
        assert len(words) == SYNAPSE_WORD_COUNT
        assert set(words) == {0}

    def test_a_single_synapse_lands_in_exactly_one_nibble(self):
        magnitudes = [[0] * 256 for _ in range(256)]
        magnitudes[37][130] = 6
        words = pack_synapse_words(magnitudes, mapped=False)
        word_addr, byte_addr, nibble = synapse_address(37, 130)
        shift = 4 * (2 * byte_addr + nibble)
        assert (words[word_addr] >> shift) & 0xF == 0b0110
        assert sum(1 for w in words if w) == 1

    def test_a_dense_image_round_trips(self):
        magnitudes = [
            [(pre * 7 + post * 3) % 8 for post in range(256)] for pre in range(256)
        ]
        words = pack_synapse_words(magnitudes, mapped=True)
        recovered_magnitudes, recovered_mapped = unpack_synapse_words(words)
        assert recovered_magnitudes == magnitudes
        assert all(all(row) for row in recovered_mapped)

    def test_a_wrong_shape_is_refused(self):
        with pytest.raises(OdinLayoutError, match="256"):
            pack_synapse_words([[0] * 256] * 4, mapped=False)


class TestSynSignVector:
    def test_the_vector_is_one_bit_per_presynaptic_row(self):
        signs = [False] * 256
        signs[3] = True
        signs[255] = True
        vector = pack_syn_sign(signs)
        assert vector == (1 << 3) | (1 << 255)
        assert unpack_syn_sign(vector) == tuple(signs)

    def test_the_vector_programs_sixteen_registers_from_address_two(self):
        signs = [i % 2 == 1 for i in range(256)]
        writes = syn_sign_register_writes(pack_syn_sign(signs))
        assert [w.address for w in writes] == list(
            range(SYN_SIGN_BASE_ADDR, SYN_SIGN_BASE_ADDR + 16)
        )
        assert all(w.value == 0b1010101010101010 for w in writes)

    def test_a_short_sign_list_is_refused(self):
        with pytest.raises(OdinLayoutError, match="256"):
            pack_syn_sign([False] * 128)


# --------------------------------------------------------------------------
# A4. Config registers: the upstream address map, with the inference values.
# --------------------------------------------------------------------------

# doc/README.md Sec.4 table, transcribed independently.
EXPECTED_REGISTERS = {
    "SPI_GATE_ACTIVITY": (0, 1, 1),
    "SPI_OPEN_LOOP": (1, 1, 1),
    "SPI_SYN_SIGN": (2, 16, 16),
    "SPI_BURST_TIMEREF": (18, 20, 1),
    "SPI_AER_SRC_CTRL_nNEUR": (19, 1, 1),
    "SPI_OUT_AER_MONITOR_EN": (20, 1, 1),
    "SPI_MONITOR_NEUR_ADDR": (21, 8, 1),
    "SPI_MONITOR_SYN_ADDR": (22, 8, 1),
    "SPI_UPDATE_UNMAPPED_SYN": (23, 1, 1),
    "SPI_PROPAGATE_UNMAPPED_SYN": (24, 1, 1),
    "SPI_SDSP_ON_SYN_STIM": (25, 1, 1),
}


class TestTheConfigRegisterMap:
    def test_the_table_matches_the_documented_address_map(self):
        actual = {r.name: (r.address, r.width, r.count) for r in CONFIG_REGISTERS}
        assert actual == EXPECTED_REGISTERS

    def test_the_lookup_names_an_unknown_register(self):
        assert config_register("SPI_OPEN_LOOP").address == 1
        with pytest.raises(OdinLayoutError, match="SPI_NOPE"):
            config_register("SPI_NOPE")


class TestTheInferenceRegisterValues:
    """F16: open loop, weight freeze doubly guaranteed, monitoring off."""

    def test_open_loop_is_asserted_so_the_host_routes_every_event(self):
        assert INFERENCE_REGISTER_VALUES["SPI_OPEN_LOOP"] == 1

    def test_the_weight_freeze_is_doubly_guaranteed(self):
        # Learning never updates an unmapped synapse, and every synapse is
        # written unmapped; propagation ignores the mapping bit so the frozen
        # magnitudes still reach the soma.
        assert INFERENCE_REGISTER_VALUES["SPI_UPDATE_UNMAPPED_SYN"] == 0
        assert INFERENCE_REGISTER_VALUES["SPI_PROPAGATE_UNMAPPED_SYN"] == 1
        assert INFERENCE_REGISTER_VALUES["SPI_SDSP_ON_SYN_STIM"] == 0

    def test_monitoring_and_bursting_are_off(self):
        assert INFERENCE_REGISTER_VALUES["SPI_OUT_AER_MONITOR_EN"] == 0
        assert INFERENCE_REGISTER_VALUES["SPI_AER_SRC_CTRL_nNEUR"] == 0
        assert INFERENCE_REGISTER_VALUES["SPI_BURST_TIMEREF"] == 0

    def test_the_write_list_covers_every_register_once_syn_sign_expanded(self):
        writes = inference_register_writes(
            syn_sign=pack_syn_sign([False] * 256), gate_activity=1
        )
        addresses = [w.address for w in writes]
        assert len(addresses) == len(set(addresses))
        assert set(addresses) == set(range(26))

    def test_gate_activity_is_the_caller_declared_value(self):
        vector = pack_syn_sign([False] * 256)
        for gate in (0, 1):
            writes = inference_register_writes(syn_sign=vector, gate_activity=gate)
            gated = [w for w in writes if w.address == 0]
            assert [w.value for w in gated] == [gate]

    def test_the_sign_vector_reaches_the_sixteen_sign_registers(self):
        signs = [i >= 128 for i in range(256)]
        writes = inference_register_writes(
            syn_sign=pack_syn_sign(signs), gate_activity=0
        )
        sign_writes = [w for w in writes if 2 <= w.address <= 17]
        assert [w.value for w in sign_writes] == [0] * 8 + [0xFFFF] * 8
