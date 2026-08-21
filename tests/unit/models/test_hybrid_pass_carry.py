"""A pass boundary inside a streamed segment is semantically invisible.

The load-bearing claim of multi-pass scheduling under streamed LIF: cutting a segment
into passes must not change what it computes. That is a test, not an argument — the
same model on a grid that fits and on a grid that forces a cut must agree bit for bit.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

T = 8


#: Signed weights around a small positive bias. All-POSITIVE weights make every
#: output neuron see nearly the same sum, which collapses the vehicle to one repeated
#: row — see TestTheVehicleDiscriminates for why that silently weakens every claim here.
_WEIGHT_SCALE = 2.0
_BIAS = 0.3


def _lif(width, fan_in, scale):
    p = Perceptron(width, fan_in, normalization=nn.Identity())
    p.base_activation = LIFActivation(T=T, activation_scale=torch.tensor(1.0))
    p.activation = p.base_activation
    with torch.no_grad():
        p.layer.weight.copy_((torch.rand(width, fan_in) - 0.5) * scale)
        if p.layer.bias is not None:
            p.layer.bias.fill_(_BIAS)
    return p


def _deep_lif_ir(seed: int = 0):
    """Encode -> three LIF hops with mid-range firing, so the segment has latency
    groups to cut between AND real rhythm on the wires between them.

    Weights are set explicitly: untrained random ones saturate every neuron, and a
    saturated raster IS its own uniform re-encode, so such a vehicle cannot witness
    the carry at all.
    """
    torch.manual_seed(seed)
    inp = InputMapper((8,))
    enc = _lif(8, 8, _WEIGHT_SCALE)
    enc.is_encoding_layer = True
    enc.use_cycle_accurate_trains = True
    node = PerceptronMapper(inp, enc)
    for width in (8, 8, 4):
        node = PerceptronMapper(node, _lif(width, 8, _WEIGHT_SCALE))
    repr_ = ModelRepresentation(node)
    mark_encoding_layers(repr_)
    return IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=16, max_neurons=16,
    ).map(repr_)


def _flow(hybrid):
    return SpikingHybridCoreFlow(
        (8,), hybrid, simulation_length=T, spiking_mode="lif",
        cycle_accurate_lif_forward=True,
    thresholding_mode="<=", )


def _fused(ir):
    return build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 16, "max_neurons": 16, "count": 16}],
    )


def _scheduled(ir, count: int):
    """A grid too small for one program, so the segment must be cut into passes."""
    strategy = MappingStrategy.resolve(ChipCapabilities(allow_scheduling=True))
    return build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 16, "max_neurons": 16, "count": count}],
        strategy=strategy,
    )


def _passes(hybrid):
    return sum(
        1 for s in hybrid.stages
        if getattr(s, "kind", None) == "neural"
        and getattr(s, "schedule_pass_index", None) is not None
    )


class TestTheVehicleDiscriminates:
    """Every equivalence claim in this file is only as strong as its witness. A
    network whose outputs saturate, or which answers the same thing for every input,
    can be reproduced by almost any wrong implementation — that is exactly how the
    first version of these tests passed with the carry DISABLED. So the vehicle's
    own health is pinned here, and drifting back into degeneracy fails loudly."""

    def _outputs(self):
        with torch.no_grad():
            return _flow(_fused(_deep_lif_ir()))(torch.rand(8, 8))

    def test_different_inputs_give_different_answers(self):
        out = self._outputs()
        assert len({tuple(r.tolist()) for r in out}) >= 6, out

    def test_the_output_neurons_do_not_all_agree(self):
        """All-positive weights make every neuron see the same sum; the spread is
        what says the network actually computes something per neuron."""
        out = self._outputs()
        spread = (out.max(dim=1).values - out.min(dim=1).values).mean()
        assert float(spread) >= 2.0, out

    def test_the_counts_are_not_pinned_to_the_window(self):
        """A saturated raster IS its own uniform re-encode, so a saturated vehicle
        cannot witness the difference between the two transfer disciplines."""
        out = self._outputs()
        assert float(out.max()) < T, out


class TestTheRasterAgreesWithTheCounts:
    """The carry is recorded from the same fires the counts accumulate, so the
    raster summed over its window must BE the counts. A carry that failed this
    would hand the next pass spikes the producer never emitted."""

    def test_the_published_raster_sums_to_the_segment_counts(self):
        from mimarsinan.models.spiking.hybrid.executors.packed_cycle import (
            run_neural_segment_packed,
        )
        from mimarsinan.models.spiking.cycle_policy import cycle_neuron_policy

        hybrid = _fused(_deep_lif_ir())
        flow = _flow(hybrid)
        stage = next(s for s in hybrid.stages if getattr(s, "kind", None) == "neural")
        device = torch.device("cpu")
        seg = flow._get_segment_tensors(stage, device)
        seg.setdefault("latency", None)
        if seg["latency"] is None:
            from mimarsinan.mapping.latency.chip import ChipLatency

            seg["latency"] = int(ChipLatency(stage.hard_core_mapping).calculate())
        policy = cycle_neuron_policy(
            "lif", "cascaded", "Default", soma_law=flow.soma_law)
        train = (torch.rand(T, 2, len(stage.input_map) and
                            max(s.offset + s.size for s in stage.input_map)) > 0.5
                 ).float()
        sink: list = []
        with torch.no_grad():
            counts = run_neural_segment_packed(
                flow, train, seg=seg, stage=stage, T=T, batch_size=2,
                device=device, policy=policy, output_train=sink,
            )
        assert sink, "a requested carry must be produced"
        assert sink[0].shape == (T, 2, counts.shape[1])
        assert torch.equal(sink[0].sum(dim=0), counts)


class TestPassingIsSemanticallyInvisible:
    def test_a_scheduled_run_reproduces_the_fused_run_exactly(self):
        ir = _deep_lif_ir()
        fused, scheduled = _fused(ir), _scheduled(ir, count=2)
        assert _passes(scheduled) >= 2, (
            f"the grid must force a cut for this test to mean anything; got "
            f"{_passes(scheduled)} pass(es)")
        x = torch.rand(3, 8)
        with torch.no_grad():
            assert torch.equal(_flow(fused)(x), _flow(scheduled)(x))

    def test_more_passes_still_reproduce_the_fused_run(self):
        ir = _deep_lif_ir(seed=7)
        fused, scheduled = _fused(ir), _scheduled(ir, count=1)
        x = torch.rand(3, 8)
        with torch.no_grad():
            assert torch.equal(_flow(fused)(x), _flow(scheduled)(x))

    def test_the_cut_really_happened(self):
        """Guard the guard: if the grid stopped forcing a cut, the equivalence
        tests above would pass vacuously."""
        from mimarsinan.mapping.support.schedule.pass_carry import (
            carried_outputs_by_stage,
        )

        scheduled = _scheduled(_deep_lif_ir(), count=2)
        carried = carried_outputs_by_stage(scheduled.stages)
        assert carried, "no wire crosses a pass boundary; the test proves nothing"

    def test_the_carry_is_load_bearing(self, monkeypatch):
        """Collapsing the pass boundary to counts CHANGES the result — which is
        exactly why streamed scheduling was locked off. Without this, the
        equivalence tests above could pass with the carry disabled."""
        from mimarsinan.models.spiking.hybrid.stage_io import HybridStageIOMixin

        ir = _deep_lif_ir(seed=3)
        fused, scheduled = _fused(ir), _scheduled(ir, count=2)
        x = torch.rand(8, 8)
        with torch.no_grad():
            reference = _flow(fused)(x)
        monkeypatch.setattr(
            HybridStageIOMixin, "_publish_carried_trains",
            staticmethod(lambda *a, **k: None),
        )
        with torch.no_grad():
            collapsed = _flow(scheduled)(x)
        assert not torch.equal(reference, collapsed), (
            "the uniform re-encode reproduced the raster on this vehicle, so it "
            "cannot witness the carry; pick a vehicle with non-uniform rhythm")


class TestTheRecorderRespectsTheProducerWindow:
    """A producer emits only during ``[latency, latency + T)``. Outside it, ``fires``
    still holds whatever that core last did, so an out-of-window cycle must be
    SKIPPED — clamping it into range would overwrite a real emission with a stale
    one. Tested on the pure recorder, because a segment whose outputs all sit at the
    deepest latency can never exhibit the overhang."""

    def _run(self, latency: int, window: int, cycles: int):
        import torch as _t

        from mimarsinan.models.spiking.hybrid.carry import record_carry

        carry = _t.zeros(window, 1, 1)
        plan = [("core", 0, 1, 0, 1, latency)]
        for cycle in range(cycles):
            fires = _t.full((1, 1), float(cycle + 1))
            record_carry(carry, plan, cycle=cycle, fires=fires,
                         train=_t.zeros(window, 1, 1), T=window)
        return carry[:, 0, 0].tolist()

    def test_only_the_producers_own_window_is_recorded(self):
        # latency 1, T=3 -> cycles 1,2,3 are local 0,1,2; cycles 0 and 4 are outside.
        assert self._run(latency=1, window=3, cycles=5) == [2.0, 3.0, 4.0]

    def test_a_late_cycle_does_not_overwrite_the_last_emission(self):
        """The clamping failure mode: cycle 4 would land back on local T-1."""
        assert self._run(latency=1, window=3, cycles=5)[-1] == 4.0

    def test_a_passthrough_span_carries_the_segment_input(self):
        """An ``input``-kind output span is a segment output wired straight from the
        segment INPUT. A linear chain has none, so the branch needs pinning here or a
        carry that dropped it would look correct."""
        import torch as _t

        from mimarsinan.models.spiking.hybrid.carry import record_carry

        carry = _t.zeros(3, 1, 1)
        train = _t.tensor([[[1.0]], [[0.0]], [[1.0]]])
        for cycle in range(3):
            record_carry(carry, [("input", 0, 1, 0, 1, 0)], cycle=cycle,
                         fires=_t.zeros(1, 1), train=train, T=3)
        assert carry[:, 0, 0].tolist() == [1.0, 0.0, 1.0]

    def test_an_always_on_span_carries_a_spike_every_cycle(self):
        import torch as _t

        from mimarsinan.models.spiking.hybrid.carry import record_carry

        carry = _t.zeros(3, 1, 1)
        for cycle in range(3):
            record_carry(carry, [("on", 0, 1, 0, 1, 0)], cycle=cycle,
                         fires=_t.zeros(1, 1), train=_t.zeros(3, 1, 1), T=3)
        assert carry[:, 0, 0].tolist() == [1.0, 1.0, 1.0]

    def test_a_zero_latency_producer_starts_at_local_zero(self):
        assert self._run(latency=0, window=3, cycles=3) == [1.0, 2.0, 3.0]


class TestEveryBackendDeploysAScheduledSegment:
    """No backend refuses. A pass boundary is one this program INTRODUCES, so a
    chip that reprograms across it has to buffer the intermediate signal either
    way — the only question is WHAT it buffers. Both answers are honest
    deployments; the run reports which one it executed."""

    def test_no_backend_refuses_a_scheduled_deployment(self):
        from mimarsinan.models.spiking.hybrid.carry import (
            pass_transfer_for_backend,
        )

        for backend in ("hcm", "sanafe", "nevresim", "lava"):
            assert pass_transfer_for_backend(backend) in (VERBATIM, COLLAPSE)

    def test_the_hcm_executor_carries_the_raster_verbatim(self):
        from mimarsinan.models.spiking.hybrid.carry import (
            pass_transfer_for_backend,
        )

        assert pass_transfer_for_backend("hcm") == VERBATIM

    def test_the_cost_measuring_backend_carries_verbatim(self):
        """SANA-FE feeds the physics and energy path, so a collapsed boundary there
        would model a different computation from the one the record claims."""
        from mimarsinan.models.spiking.hybrid.carry import (
            pass_transfer_for_backend,
        )

        assert pass_transfer_for_backend("sanafe") == VERBATIM

    def test_nevresim_carries_verbatim(self):
        """SPKTRN extraction + SpikeTrain replay: the per-segment binaries record
        producer-local trains and consuming segments replay them."""
        from mimarsinan.models.spiking.hybrid.carry import (
            pass_transfer_for_backend,
        )

        assert pass_transfer_for_backend("nevresim") == VERBATIM

    def test_lava_carries_verbatim(self):
        """lava is host-scheduled: core output spikes already live host-side, so
        extraction is a windowed gather and replay overwrites the encoded train."""
        from mimarsinan.models.spiking.hybrid.carry import (
            pass_transfer_for_backend,
        )

        assert pass_transfer_for_backend("lava") == VERBATIM

    def test_an_unknown_backend_still_collapses_by_default(self):
        """The weakest-backend rule is the safety default for FUTURE backends: a
        name outside the declaration takes the cheaper discipline, never an
        unearned verbatim."""
        from mimarsinan.models.spiking.hybrid.carry import (
            pass_transfer_for_backend,
        )

        assert pass_transfer_for_backend("some_future_backend") == COLLAPSE

    def test_collapsing_is_cheaper_to_buffer_than_carrying(self):
        """Counts fit in log2(T+1) bits; a raster needs T. That is the whole
        trade the two disciplines make, and it is why both exist."""
        from mimarsinan.models.spiking.hybrid.carry import carried_wire_bytes

        verbatim = carried_wire_bytes(64, 32, VERBATIM)
        collapse = carried_wire_bytes(64, 32, COLLAPSE)
        assert collapse < verbatim
        assert verbatim == 64 * 4 and collapse == 64 * 1
class TestSchedulingIsLegalUnderStreamedLif:
    def test_streamed_lif_may_now_choose_scheduling(self):
        """The view lists only RESTRICTED keys, so an unrestricted boolean is
        absent from it — which is what 'no longer locked' looks like here."""
        from mimarsinan.config_schema.resolve import legal_values_view

        streamed = legal_values_view(
            {"spiking_family": "lif", "spiking_variant": "streamed"}
        )
        assert "allow_scheduling" not in streamed, streamed.get("allow_scheduling")

    def test_the_default_is_still_off(self):
        """Unlocking makes scheduling CHOOSABLE, not automatic: a segment that
        fits stays in one program, and nothing about existing runs changes."""
        from mimarsinan.config_schema.resolve import resolve_draft

        resolution = resolve_draft(
            {"spiking_family": "lif", "spiking_variant": "streamed"}
        )
        assert resolution.resolved["allow_scheduling"] is False


class TestTheRecordNamesWhatCrossed:
    """VERBATIM and COLLAPSE are different computations, so a record that did not
    name the one it ran would leave a reader unable to say what produced the
    numbers."""

    def _schedule(self, hybrid, **over):
        from mimarsinan.deployment_record.build.from_mapping import (
            schedule_record_from_mapping,
        )

        kwargs = dict(weight_bits=8, params_reloaded=0,
                      timesteps=T, pass_transfer=VERBATIM)
        kwargs.update(over)
        return schedule_record_from_mapping(hybrid, **kwargs)

    def test_a_single_program_carries_nothing_and_seals_none(self):
        assert self._schedule(_fused(_deep_lif_ir())).carry is None

    def test_a_scheduled_program_seals_the_discipline_and_the_census(self):
        carry = self._schedule(_scheduled(_deep_lif_ir(), count=2)).carry
        assert carry is not None
        assert carry.transfer == VERBATIM
        assert carry.carried_wires >= 1
        assert carry.carried_bytes > 0
        assert 0 < carry.peak_live_bytes <= carry.carried_bytes
        assert carry.timesteps == T

    def test_collapsing_seals_a_smaller_census_than_carrying(self):
        """At a window wide enough for the trade to bite: a raster needs T bits per
        wire, a count only log2(T+1). (At this vehicle's T=8 both round to one byte,
        which is why the window is stated explicitly here.)"""
        scheduled = _scheduled(_deep_lif_ir(), count=2)
        verbatim = self._schedule(scheduled, timesteps=32).carry
        collapse = self._schedule(scheduled, timesteps=32,
                                  pass_transfer=COLLAPSE).carry
        assert verbatim is not None and collapse is not None
        assert collapse.carried_bytes * 4 == verbatim.carried_bytes

    def test_at_a_narrow_window_the_two_disciplines_cost_the_same(self):
        """Both fit one byte per wire at T=8 — the trade is real but not free of
        rounding, and a census that pretended otherwise would be inventing precision."""
        scheduled = _scheduled(_deep_lif_ir(), count=2)
        verbatim = self._schedule(scheduled).carry
        collapse = self._schedule(scheduled, pass_transfer=COLLAPSE).carry
        assert verbatim is not None and collapse is not None
        assert collapse.carried_bytes == verbatim.carried_bytes

    def test_a_carrying_program_refuses_to_seal_an_unstated_discipline(self):
        """Sealing None here would drop the fact that the two disciplines differ."""
        with pytest.raises(ValueError, match="which discipline"):
            self._schedule(_scheduled(_deep_lif_ir(), count=2), pass_transfer=None)

    def test_the_sealed_carry_round_trips(self):
        from mimarsinan.deployment_record.schema import ScheduleRecord

        record = self._schedule(_scheduled(_deep_lif_ir(), count=2))
        assert ScheduleRecord.from_dict(record.to_dict()).carry == record.carry

    def test_an_unknown_discipline_fails_loud(self):
        with pytest.raises(ValueError, match="transfer"):
            self._schedule(_scheduled(_deep_lif_ir(), count=2),
                           pass_transfer="whatever")

    def test_wires_that_do_not_overlap_share_the_buffer(self):
        """The stage-level census is a separate function from PassCut's, and it owes
        the same discipline: a three-pass program carries two wires whose live ranges
        are disjoint, so the buffer a run needs is the worst boundary, not the sum."""
        carry = self._schedule(_scheduled(_deep_lif_ir(), count=1)).carry
        assert carry is not None
        assert carry.carried_wires >= 2
        assert carry.peak_live_bytes < carry.carried_bytes


class TestOneDisciplinePerRun:
    """A 3-pass streamed MLP found this end-to-end: HCM carrying while nevresim
    collapsed diverged on 4.4% of neuron windows by one spike each, and the
    cross-backend exactness gates — which admit no tolerance on integer arithmetic —
    were comparing two different computations."""

    def _transfer(self, **enabled):
        from mimarsinan.models.spiking.hybrid.carry import run_pass_transfer

        return run_pass_transfer(enabled)

    def test_a_run_of_carrying_backends_only_is_verbatim(self):
        assert self._transfer(enable_sanafe_simulation=True) == VERBATIM

    def test_every_shipped_backend_now_carries_so_streamed_runs_are_verbatim(self):
        assert self._transfer(enable_sanafe_simulation=True,
                              enable_nevresim_simulation=True,
                              enable_loihi_simulation=True) == VERBATIM

    def test_lava_no_longer_collapses_the_run(self):
        assert self._transfer(enable_loihi_simulation=True) == VERBATIM

    def test_a_run_with_no_chip_backend_keeps_the_verbatim_boundary(self):
        """Nothing enabled means only the HCM executor runs, and it carries."""
        assert self._transfer() == VERBATIM

    def test_the_flow_honours_the_run_discipline(self):
        """The flow must not publish rasters a collapsing run will not replay."""
        ir = _deep_lif_ir()
        scheduled = _scheduled(ir, count=2)
        x = torch.rand(4, 8)
        collapsing = SpikingHybridCoreFlow(
            (8,), scheduled, simulation_length=T, spiking_mode="lif",
            cycle_accurate_lif_forward=True, pass_transfer=COLLAPSE,
        thresholding_mode="<=", )
        with torch.no_grad():
            assert not torch.equal(_flow(scheduled)(x), collapsing(x)), (
                "a collapsing run must differ from a carrying one, or the "
                "discipline is not reaching the executor")


class TestSemanticsDecideBeforeBackends:
    """Only STREAMED execution has a rhythm to carry. A windowed or value-domain run
    that resolved verbatim would seal a record naming a computation that never
    happened, and any backend replaying a raw raster into a windowed pass would
    diverge from every peer that re-encoded — the audit found exactly that pair."""

    def test_a_windowed_run_collapses_even_with_only_carrying_backends(self):
        from mimarsinan.models.spiking.hybrid.carry import run_pass_transfer

        assert run_pass_transfer({
            "spiking_family": "lif", "spiking_variant": "synchronized",
            "enable_sanafe_simulation": True,
        }) == COLLAPSE

    def test_a_value_domain_run_collapses(self):
        from mimarsinan.models.spiking.hybrid.carry import run_pass_transfer

        assert run_pass_transfer({
            "core_semantics": "mvm", "enable_sanafe_simulation": True,
        }) == COLLAPSE

    def test_a_synchronized_twin_of_a_verbatim_run_neither_carries_nor_refuses(self):
        """The gauge certificate runs a SYNCHRONIZED twin of the streamed run:
        the run discipline says verbatim, but that instance's execution re-encodes
        at its boundaries by its own discipline — so it must run clean without
        publishing (the first verbatim probe died here on the refusal)."""
        ir = _deep_lif_ir()
        scheduled = _scheduled(ir, count=2)
        flow = SpikingHybridCoreFlow(
            (8,), scheduled, simulation_length=T, spiking_mode="lif",
            cycle_accurate_lif_forward=True, lif_execution_synchronized=True,
            pass_transfer=VERBATIM,
        thresholding_mode="<=", )
        with torch.no_grad():
            out = flow(torch.rand(2, 8))
        assert torch.isfinite(out).all()

    def test_requesting_a_raster_on_the_synchronized_executor_refuses_loudly(self):
        """The executor-level guard stays: a DIRECT caller that asks the
        synchronized path for a raster must get the refusal, never a silently
        empty list that downstream code happens to skip."""
        ir = _deep_lif_ir()
        scheduled = _scheduled(ir, count=2)
        flow = SpikingHybridCoreFlow(
            (8,), scheduled, simulation_length=T, spiking_mode="lif",
            cycle_accurate_lif_forward=True, lif_execution_synchronized=True,
            pass_transfer=VERBATIM,
        thresholding_mode="<=", )
        stage = next(s for s in scheduled.stages
                     if getattr(s, "kind", None) == "neural")
        train = torch.zeros(T, 1, max(s.offset + s.size for s in stage.input_map))
        with pytest.raises(NotImplementedError, match="synchronized"):
            flow._run_neural_segment_rate(
                stage, input_spike_train=train, output_train=[])

    def test_every_enable_key_survives_config_resolution(self):
        """run_pass_transfer reads enable_* keys with no default; DeploymentPlan
        defaults an absent nevresim key to True. The two can only agree because a
        RESOLVED config always carries the keys — pin that, or the absent-key path
        could silently disagree with the plan about which backends run."""
        from mimarsinan.config_schema.resolve import resolve_draft
        from mimarsinan.models.spiking.hybrid.carry import _BACKEND_ENABLE_KEYS

        resolved = resolve_draft(
            {"spiking_family": "lif", "spiking_variant": "streamed"}
        ).resolved
        for key in _BACKEND_ENABLE_KEYS.values():
            assert key in resolved, key


class TestTheReferenceLoopCarriesLikeThePackedExecutor:
    """The certificate paths run the RECORDING reference loop (batch 1), which the
    first verbatim probe found could not carry. It now can — and its raster must
    be bit-identical to the packed executor's, or the two execution paths of one
    flow would compute different scheduled runs."""

    def _published(self, use_packed: bool):
        ir = _deep_lif_ir(seed=11)
        scheduled = _scheduled(ir, count=2)
        flow = _flow(scheduled)
        flow.use_packed_cycle_executor = use_packed
        captured = {}
        original = SpikingHybridCoreFlow._publish_carried_trains

        def _capture(stage, output_train, carried_ids, buffer):
            for s in stage.output_map:
                if int(s.node_id) in set(carried_ids):
                    captured[int(s.node_id)] = (
                        output_train[..., s.offset:s.offset + s.size].clone())
            return original(stage, output_train, carried_ids, buffer)

        flow._publish_carried_trains = _capture
        x = torch.rand(3, 8)
        with torch.no_grad():
            out = flow(x)
        assert captured, "the scheduled run must have published a carry"
        return out, captured

    def test_reference_and_packed_rasters_are_identical(self):
        torch.manual_seed(5)
        packed_out, packed = self._published(use_packed=True)
        torch.manual_seed(5)
        reference_out, reference = self._published(use_packed=False)
        assert torch.equal(packed_out, reference_out)
        assert packed.keys() == reference.keys()
        for node in packed:
            assert torch.equal(packed[node], reference[node]), node
