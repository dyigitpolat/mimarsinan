"""[W33 integration] nevresim reads a stage input in core-local time only where the nevresim tree it compiles against
carries the read.

The read is W33's change to nevresim (``nevresim.patch`` on base 90af7d2). nevresim is a nested repository whose
commit of the change waits for its owner, so a checkout can hold either tree, and base 90af7d2 reads every stage-input
row at the global cycle. The guard therefore counts nevresim among the backends that read in core-local time only
where a probe of that tree's sources establishes the read: its declared read sites, then its own self-test
constant-evaluated by the compiler nevresim builds with. On a tree without the change every late reader and every
delayed slice is refused on nevresim by name, at plan time and before any segment compiles, instead of being read at
the global cycle. Both trees are built from this checkout's (``nevresim_trees``); the tests of the tree with the change
read this checkout's own tree, so on a checkout whose nevresim lacks the change they fail by that refusal."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from mimarsinan.chip_simulation.simulation_runner import emit as nevresim_emit
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.packing.pass_timing import REPLAY_DELAYS, late_stage_input_readers
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.models.spiking.hybrid import carry as C

from unit.chip_simulation import nevresim_trees as NT
from unit.mapping.residency.executed import vehicles as V

#: Every file of the change, named by the probe of a tree without it.
CHANGED = ("stage_input_train.hpp", "stage_input_clock.hpp", "compute_policy_base.hpp", "spiking_execution.hpp",
           "ttfs_execution.hpp")


@pytest.fixture
def nevresim_at(monkeypatch):
    """Point nevresim at a tree, as ``init`` points it at ``nevresim_root()``."""
    def point(root):
        monkeypatch.setattr(NevresimDriver, "nevresim_path", str(root))
        return root
    return point


def _streamed(**enabled):
    return {"spiking_family": "lif", "spiking_variant": "streamed", "spike_generation_mode": "Uniform", **enabled}


def _carried_late():
    """Chain C 3: pass 4 reads its carried slice at latency 1."""
    return V.build(V.ir_of(V.chain_repr(seed=0)), 3, None)


def _host_late():
    """skew3's identity program: a reader of a host-encoded slice raised to latency 2, no carried slice."""
    program = build_identity_hybrid_mapping(ir_graph=V.ir_of(V.branchy_repr("skew3")))
    assert any(late_stage_input_readers(s) for s in V.neural_passes(program))
    return program


def _delayed():
    """Fuzz seed 58 at C 2: a slice delivered 1 cycle late, read at latency 0."""
    program = V.build(V.ir_of(V.random_rep(58)), 2, None)
    assert any(getattr(s, REPLAY_DELAYS, None) for s in V.neural_passes(program))
    return program


def _lif(program, backend):
    C.require_backend_replay(program, backend, transfer=VERBATIM, spike_mode="Uniform", spiking_mode="lif")


def _cascaded(program, backend):
    C.require_backend_replay(program, backend, transfer=COLLAPSE, spike_mode="TTFS", spiking_mode="ttfs_cycle_based")


class TestATreeWithoutTheChange:
    """Base 90af7d2 reads every stage-input row at the global cycle: nevresim refuses each late reader by name."""

    def test_a_carried_and_a_host_encoded_late_reader_are_refused_by_name(self, nevresim_at, tmp_path):
        nevresim_at(NT.base_tree(tmp_path))
        for program in (_carried_late(), _host_late()):
            with pytest.raises(C.ChipClockReplayRefusal, match="nevresim reads a stage input at the global cycle"):
                _lif(program, "nevresim")
            with pytest.raises(C.ChipClockReplayRefusal, match="nevresim"):
                C.run_pass_transfer(_streamed(enable_nevresim_simulation=True), hybrid_mapping=program)
            with pytest.raises(C.ChipClockReplayRefusal, match="nevresim"):
                _cascaded(program, "nevresim")
            for backend in ("hcm", "sanafe", "lava"):          # their reads are mimarsinan's own code
                _lif(program, backend)

    def test_a_delayed_slice_is_refused_by_name(self, nevresim_at, tmp_path):
        nevresim_at(NT.base_tree(tmp_path))
        with pytest.raises(C.ChipClockReplayRefusal, match="delayed slices"):
            _cascaded(_delayed(), "nevresim")

    def test_the_refusal_names_every_part_of_the_change_the_tree_lacks(self, nevresim_at, tmp_path):
        root = nevresim_at(NT.base_tree(tmp_path))
        with pytest.raises(C.ChipClockReplayRefusal) as refused:
            _lif(_carried_late(), "nevresim")
        assert str(root) in str(refused.value)
        for name in CHANGED:
            assert name in str(refused.value)

    def test_the_refusal_comes_before_any_segment_compiles(self, nevresim_at, tmp_path):
        nevresim_at(NT.base_tree(tmp_path))
        runner = SimpleNamespace(pass_transfer=VERBATIM, spike_generation_mode="Uniform", spiking_mode="lif")
        with pytest.raises(C.ChipClockReplayRefusal):
            nevresim_emit.prepare_all_segments(runner, _carried_late())   # before it reads test data or compiles

    def test_the_runner_refuses_before_its_flat_route_compiles(self, nevresim_at, tmp_path, monkeypatch):
        """A program of one neural segment and no host op runs on nevresim's flat route, which compiles without
        ``prepare_all_segments``; the refusal comes first there too."""
        from mimarsinan.chip_simulation.simulation_runner.core import SimulationRunner

        nevresim_at(NT.base_tree(tmp_path))
        program = _host_late()
        monkeypatch.setattr(program, "get_compute_ops", lambda: [])
        assert len(program.get_neural_segments()) == 1
        runner = object.__new__(SimulationRunner)
        vars(runner).update(mapping=program, pass_transfer=VERBATIM, spike_generation_mode="Uniform",
                            spiking_mode="lif")
        monkeypatch.setattr(runner, "_run_flat_mapping", lambda mapping: pytest.fail("the flat route compiled"))
        with pytest.raises(C.ChipClockReplayRefusal, match="nevresim reads a stage input at the global cycle"):
            runner.run()

    def test_a_program_without_a_late_reader_or_a_delayed_slice_runs_on_it(self, nevresim_at, tmp_path):
        """Fuzz seed 1's identity program reads every stage input at latency 0: nothing to time, nothing refused."""
        nevresim_at(NT.base_tree(tmp_path))
        program = build_identity_hybrid_mapping(ir_graph=V.ir_of(V.random_rep(1)))
        assert not any(late_stage_input_readers(s) or getattr(s, REPLAY_DELAYS, None) for s in V.neural_passes(program))
        _lif(program, "nevresim")
        _cascaded(program, "nevresim")


class TestATreeWithTheChange:
    def test_the_same_programs_are_admitted_on_this_checkouts_tree(self, nevresim_at):
        nevresim_at(NT.checkout_tree())
        for program in (_carried_late(), _host_late()):
            _lif(program, "nevresim")
            _cascaded(program, "nevresim")
            assert C.run_pass_transfer(_streamed(enable_nevresim_simulation=True), hybrid_mapping=program) == VERBATIM
        _cascaded(_delayed(), "nevresim")

    def test_a_tree_whose_policy_reads_the_global_cycle_is_refused_by_its_self_test(self, nevresim_at, tmp_path):
        """Every declared read site holds, but the compute policy reads at the global cycle: the sites cannot tell,
        the tree's self-test can (its second core, at latency 2, fires 0 times where 2 are due)."""
        nevresim_at(NT.global_cycle_policy_tree(tmp_path))
        with pytest.raises(C.ChipClockReplayRefusal, match="does not hold"):
            _lif(_carried_late(), "nevresim")


class TestTheDeclaration:
    def test_every_backend_a_config_enables_is_declared(self):
        from mimarsinan.chip_simulation import stage_input_clock as S

        declared = set(S.IN_TREE_CORE_LOCAL_BACKENDS) | set(S.SOURCE_TREE_CORE_LOCAL_BACKENDS)
        assert set(C._BACKEND_ENABLE_KEYS) | {"hcm"} <= declared
        assert not set(S.IN_TREE_CORE_LOCAL_BACKENDS) & set(S.SOURCE_TREE_CORE_LOCAL_BACKENDS)

    def test_nevresim_is_core_local_exactly_where_its_tree_carries_the_read(self, nevresim_at, tmp_path):
        from mimarsinan.chip_simulation import stage_input_clock as S

        nevresim_at(NT.checkout_tree())
        assert S.core_local_replay_backends() == {"hcm", "sanafe", "lava", "nevresim"}
        assert "self-test" in S.stage_input_read("nevresim").evidence
        nevresim_at(NT.base_tree(tmp_path))
        assert S.core_local_replay_backends() == {"hcm", "sanafe", "lava"}

    def test_an_undeclared_backend_and_a_missing_tree_read_at_the_global_cycle(self, nevresim_at, tmp_path):
        from mimarsinan.chip_simulation import stage_input_clock as S

        assert not S.stage_input_read("future_backend").core_local
        nevresim_at(tmp_path / "absent")
        read = S.stage_input_read("nevresim")
        assert not read.core_local and "no nevresim include tree" in read.evidence

    def test_the_probe_runs_once_per_tree_fingerprint(self, nevresim_at, tmp_path, monkeypatch):
        """The self-test compiles once per include tree; an edit to the tree probes it again."""
        from mimarsinan.chip_simulation import stage_input_clock as S

        calls = []
        evaluate = S.self_test_problem
        monkeypatch.setattr(S, "self_test_problem", lambda *a: calls.append(a) or evaluate(*a))
        root = NT._copy(tmp_path, "nevresim_copy")
        (root / "include" / "probe_test_marker.txt").write_text(str(tmp_path))   # a tree no other test probed
        nevresim_at(root)
        assert S.stage_input_read("nevresim").core_local and S.stage_input_read("nevresim").core_local
        assert len(calls) == 1
        header = root / "include" / NT.ADDED[0]
        header.write_text(header.read_text() + "\n// edited\n")
        assert S.stage_input_read("nevresim").core_local and len(calls) == 2
