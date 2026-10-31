"""The exploration foundations together (CL0, CL2, CL3 integration): where two units' rules meet on one priced point.

- CL0 x CL3: a point carries the host block (CL0) and a write census priced by the executed rotation (CL3,
  interaction finding 4 (a)); the host energy and the rotation's write energy sit on their own sides of the
  host split, and the batch-K verdict is not host-bound.
- CL0 x CL2: the SRAM-CIM macro names its backing store as an external component (CL2); the host block (CL0) does
  not declare it, CL0's demonstration block (transfer energy only) does not complete it, and a block that declares
  both its constants prices energy reading both blocks.
- CL2 x CL3: the residency step choice (CL3) prices write energy outside ``price_absolute``; on a profile whose
  external component is undeclared it refuses by name, as ``price_absolute`` does, instead of choosing a step on the
  in-array write energy alone.

Every constant here is a test constant (``(test)`` in its note or ``estimated`` with note ``test constant``).
"""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses, period_census
from mimarsinan.deployment_record.cost.absolute import (
    HOST_BOUND,
    UNJUDGED,
    dual_verdict,
    host_energy_split,
    price_absolute,
    price_write_terms,
)
from mimarsinan.deployment_record.cost.absolute.host_split import corner_order, lowest
from mimarsinan.deployment_record.cost.absolute.metrics import ABSOLUTE_PRICING
from mimarsinan.deployment_record.cost.absolute.report import ABSOLUTE_TERM_NAMES
from mimarsinan.deployment_record.cost.absolute.step_choice import StepChoiceError, cheapest_step, price_step
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs
from mimarsinan.deployment_record.platform_physics.host_calibration import calibration_overrides, host_block
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.quantities.write_law import WritePolicy

from unit.deployment_record.g3_fixtures import constant, quantities

_T = (256, 256)
_ENDS = ("low", "nominal", "high")
_P_HOST = {"low": 4.98, "nominal": 37.34, "high": 280.0, "unit": "W", "evidence_kind": "published",
           "citation": "edge-host running power (low end); CPU default TDP (high end) (test)",
           "note": "low end SECONDARY; high end a datasheet bound (test)"}
#: CL0's demonstration backing-store block of G3 acceptance 16: the transfer energy only.
_STORE_ENERGY_ONLY = {"e_dma_per_byte": constant(31.25, unit="pJ", note="HBM2 I/O per byte (test)")}
#: A backing-store block that declares both constants the macro's external component names.
_STORE = {**_STORE_ENERGY_ONLY, "t_program_per_byte": constant(0.0754, unit="ns", note="HBM channel rate (test)")}


def _host():
    """CL0's host block as the calibration builds it: measured rate and dispatch, the declared p_host band."""
    return host_block({"platform_physics_overrides": calibration_overrides(
        macs_per_s=2.0e9, op_overhead_s=(1e-5, 2e-5, 4e-5), p_host_w=None, identity="h",
        declared={"p_host": _P_HOST})})


def _row(cells, *, segment_index=0, resident=False):
    return PassRow(segment_index=segment_index, resident=resident, core_cells=tuple(cells), marks=frozenset(),
                   core_extents=tuple((c, 1) for c in cells), compute_steps=None, stage_latencies=(0,),
                   core_types=None)


#: Interaction finding 4's witness: three segments of one pass each, 33 tiles; at C = F = 33 the tiles fit.
_WITNESS = (_row([100] * 16), _row([110] * 16, segment_index=1), _row([44], segment_index=2))
#: U1's bank passes after the first: one writing pass, never displaced.
_LONE = (_row([60, 60]),) + tuple(_row([60, 60], resident=True) for _ in range(3))


def _writes(rows, cores, k):
    written = period_census(ProgramPasses(rows=tuple(rows)), weight_bits=8, capacity={_T: cores})
    policy = WritePolicy(batch_k=k, core_capacity=((_T, cores),), physical_cores=cores)
    return written, WriteInputs(census=written.census, policy=policy, written_extents=written.written_extents,
                                most_written=written.most_written_core, pass_core_types=written.pass_core_types)


def _point(physics, rows, cores, k, **extra):
    """One priced point: a LeNet-5-sized host share (118,440 host MACs over 4 host ops) and the program's census."""
    _written, inputs = _writes(rows, cores, k)
    values = dict(synaptic_events=1e6, host_macs=118440, compute_op_count=4, cores_physical=cores,
                  latency_steps=4, timesteps=1, **extra)
    pricing = price_absolute(quantities(**values), physics, writes=inputs, geometry=UNJUDGED)
    return {t.name: t for t in pricing.terms}, pricing


def _ends(band):
    return tuple(getattr(band, end) for end in _ENDS)


class TestHostEnergyAndTheRotationOnOnePricedPoint:
    """CL0 x CL3: the host block prices the host's energy, the rotation prices the writes, and both hold."""

    def _physics(self):
        return resolve_platform_physics("generic_estimated_22nm", merge_override_blocks({"host": _host()}))

    def test_the_headline_holds_the_rotation_writes_and_the_host_energy_on_their_own_sides(self):
        physics = self._physics()
        terms, pricing = _point(physics, _WITNESS, 33, 1)
        assert "energy_per_inference_mj" in terms, [r.reason for r in pricing.refusals]
        written, inputs = _writes(_WITNESS, 33, 1)
        writes = price_write_terms(written.census, physics, inputs.policy, written_extents=inputs.written_extents,
                                   most_written=inputs.most_written, pass_core_types=inputs.pass_core_types).energy
        assert writes is not None and writes.low > 0          # C = F: the rotation rewrites (finding 4 (a))
        assert _ends(terms["energy_write_mj"].band) == _ends(writes)
        host = terms["energy_host_mj"].band
        for end, value in zip(_ENDS, _ends(host)):          # p_host x (MACs / rate + dispatch x ops), end by end
            dispatch = {"low": 1e-5, "nominal": 2e-5, "high": 4e-5}[end]
            assert value == pytest.approx(getattr(physics.band("p_host"), end) * (118440 / 2.0e9 + 4 * dispatch)
                                          * 1e3, rel=1e-12)
        split = host_energy_split(pricing.terms)
        assert split is not None and _ends(split.host) == _ends(host)
        assert _ends(split.with_host) == _ends(terms["energy_per_inference_mj"].band)
        chip = [terms["energy_dynamic_mj"].band, terms["energy_static_mj"].band, writes]
        for end in _ENDS:
            assert getattr(split.with_host, end) == pytest.approx(
                sum(getattr(b, end) for b in chip) + getattr(host, end), rel=1e-12)
            assert getattr(split.without_host, end) == pytest.approx(sum(getattr(b, end) for b in chip), rel=1e-9)
            assert getattr(split.without_host, end) > getattr(writes, end)
        assert all(0 < share < 1 for share in split.host_share())

    def test_the_write_and_host_energies_are_the_same_at_c_equal_f_and_below(self):
        physics = self._physics()
        at_f, _ = _point(physics, _WITNESS, 33, 1)
        below, _ = _point(physics, _WITNESS, 32, 1)
        assert _ends(at_f["energy_write_mj"].band) == _ends(below["energy_write_mj"].band)
        assert _ends(at_f["energy_host_mj"].band) == _ends(below["energy_host_mj"].band)

    def test_a_lone_writing_pass_puts_no_write_energy_on_either_side(self):
        terms, pricing = _point(self._physics(), _LONE, 2, 1)
        assert _ends(terms["energy_write_mj"].band) == (0.0, 0.0, 0.0)
        split = host_energy_split(pricing.terms)
        chip = [terms["energy_dynamic_mj"].band, terms["energy_static_mj"].band]
        for end in _ENDS:
            assert getattr(split.without_host, end) == pytest.approx(sum(getattr(b, end) for b in chip), rel=1e-9)

    def test_the_batch_k_verdict_reads_the_same_with_and_without_the_host(self):
        """The writes shrink with K and the host energy does not: the shared host energy cancels end by end, so the
        verdict is not host-bound under the corner-aligned reading."""
        physics = self._physics()
        splits = {k: host_energy_split(_point(physics, _WITNESS, 33, k)[1].terms) for k in (1, 64)}
        verdict = dual_verdict(lowest, splits)
        assert (verdict.with_host, verdict.without_host) == (64, 64)
        assert verdict.label != HOST_BOUND
        assert corner_order(splits[64].with_host, splits[1].with_host) == "below"


class TestTheMacroWithTheHostBlockAndItsBackingStore:
    """CL0 x CL2: G3 acceptance 16 on the SRAM-CIM macro, which is read only with its backing store declared."""

    def _quantities(self):
        return quantities(synaptic_events=1e6, host_macs=118440, compute_op_count=4, cores_physical=8,
                          latency_steps=4, timesteps=1, carry_in_bytes=1000, reprogrammed_bytes=4096)

    def _refusal(self, blocks):
        physics = resolve_platform_physics("sram_cim_macro", merge_override_blocks(blocks))
        pricing = price_absolute(self._quantities(), physics, geometry=UNJUDGED)
        assert pricing.terms == ()
        refusal, *suppressed = pricing.refusals
        assert refusal.name == ABSOLUTE_PRICING
        # [U-CR C-2, declared pin amendment] each headline the gate suppresses is refused by name, by its reason
        assert [(r.name, r.reason) for r in suppressed] == [
            (headline, f"{ABSOLUTE_PRICING}: {refusal.reason}") for headline in ABSOLUTE_TERM_NAMES]
        return refusal.reason

    def test_the_host_block_does_not_declare_the_backing_store(self):
        reason = self._refusal({"host": _host()})
        assert "backing store" in reason and "e_dma_per_byte" in reason and "t_program_per_byte" in reason

    def test_a_block_with_the_transfer_energy_only_refuses_naming_the_transfer_time(self):
        reason = self._refusal({"host": _host(), "backing_store": _STORE_ENERGY_ONLY})
        assert "t_program_per_byte undeclared" in reason

    def test_both_blocks_resolve_and_price_energy_reading_both(self):
        physics = resolve_platform_physics("sram_cim_macro", merge_override_blocks(
            {"host": _host(), "backing_store": _STORE}))
        assert physics.undeclared_external() == ()
        assert {"e_dma_per_byte", "t_program_per_byte", "p_host"} <= set(physics.overridden_keys())
        pricing = price_absolute(self._quantities(), physics, geometry=UNJUDGED)
        terms = {t.name: t for t in pricing.terms}
        assert "energy_per_inference_mj" in terms, [r.reason for r in pricing.refusals]
        assert "[backing_store]" in terms["energy_per_inference_mj"].source
        assert "edge-host running power" in terms["energy_per_inference_mj"].source    # the host block's p_host
        assert "[backing_store]" in terms["energy_programming_mj"].source
        assert host_energy_split(pricing.terms).host is not None

    def test_the_rotation_writes_on_the_macro_read_the_backing_store(self):
        """CL0 x CL2 x CL3: with both blocks, the write energy of the rotation's census carries the store's
        transfer energy beside the in-array write, and the host energy stays on the host side."""
        physics = resolve_platform_physics("sram_cim_macro", merge_override_blocks(
            {"host": _host(), "backing_store": _STORE}))
        terms, pricing = _point(physics, _WITNESS, 33, 1)
        write = terms["energy_write_mj"]
        assert "e_dma_per_byte x period_rewrite_bytes / K" in write.source and "unpriced" not in write.source
        split = host_energy_split(pricing.terms)
        assert split is not None and split.host is not None
        for end in _ENDS:
            assert getattr(split.without_host, end) > getattr(write.band, end) > 0


class TestTheStepChoiceOnAProfileWithAnExternalComponent:
    """CL2 x CL3: the step choice prices writes outside ``price_absolute``, so it carries the external-component
    rule itself: no step is chosen on a price the profile says is undefined without its component."""

    def _census(self):
        return _writes(_WITNESS, 33, 1)[0].census

    def test_the_macro_without_its_backing_store_refuses_the_step_price_by_name(self):
        macro = resolve_platform_physics("sram_cim_macro", None)
        with pytest.raises(StepChoiceError) as info:
            price_step(self._census(), macro, WritePolicy(batch_k=1))
        assert ABSOLUTE_PRICING in str(info.value) and "backing store" in str(info.value)
        with pytest.raises(StepChoiceError):
            cheapest_step([self._census()], macro, WritePolicy(batch_k=1))

    def test_a_partial_backing_store_still_refuses(self):
        macro = resolve_platform_physics("sram_cim_macro", merge_override_blocks({"backing_store": _STORE_ENERGY_ONLY}))
        with pytest.raises(StepChoiceError, match="t_program_per_byte"):
            price_step(self._census(), macro, WritePolicy(batch_k=1))

    def test_the_macro_with_its_backing_store_prices_the_transfer(self):
        macro = resolve_platform_physics("sram_cim_macro", merge_override_blocks({"backing_store": _STORE}))
        price = price_step(self._census(), macro, WritePolicy(batch_k=1))
        assert "e_dma_per_byte" in price.write.basis and "unpriced" not in price.write.basis
        assert cheapest_step([self._census()], macro, WritePolicy(batch_k=1)) == 0
