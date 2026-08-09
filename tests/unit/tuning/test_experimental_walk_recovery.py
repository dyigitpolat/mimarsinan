"""EXPERIMENTAL walk recovery: contained impact surface, knob-gated, host-graph-gated."""

from __future__ import annotations


def _armed_cfg(**overrides):
    cfg = {
        "spiking_mode": "lif",
        "simulation_steps": 8,
        "lif_exact_qat": True,
        "lif_per_hop_retiming": True,
        "cycle_accurate_lif_forward": True,
        "lif_exact_qat_walk_recovery": True,
    }
    cfg.update(overrides)
    return cfg


class _Repr:
    def __init__(self, nodes):
        self._exec_order = nodes

    def _ensure_exec_graph(self):
        pass


class _Model:
    def __init__(self, nodes):
        self._r = _Repr(nodes)

    def get_mapper_repr(self):
        return self._r


class _Tuner:
    def __init__(self, model, cfg):
        self.model = model
        self.pipeline = type("_P", (), {"config": cfg})()
        self._patched_forward = False

    def _install_forward(self, fwd):
        self.model.forward = fwd
        self._patched_forward = True


def _host_node():
    from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper

    return ComputeOpMapper.__new__(ComputeOpMapper)


class TestContainment:
    def test_builder_returns_the_synchronized_walk(self):
        from mimarsinan.tuning.forward_install import ChipAlignedNFForward
        from mimarsinan.tuning.orchestration.experimental_walk_recovery import (
            exact_qat_training_forward,
        )

        m = _Model([])
        fwd = exact_qat_training_forward(m, {"simulation_steps": 8})
        assert isinstance(fwd, ChipAlignedNFForward)
        assert fwd.synchronized is True and fwd.retime is False
        assert fwd.T == 8 and fwd.model is m

    def test_aq_install_is_knob_and_host_graph_gated(self):
        from mimarsinan.tuning.forward_install import ChipAlignedNFForward
        from mimarsinan.tuning.orchestration.experimental_walk_recovery import (
            install_walk_for_aq_recovery,
        )

        armed = _Tuner(_Model([object(), _host_node()]), _armed_cfg())
        assert install_walk_for_aq_recovery(armed) is True
        assert isinstance(armed.model.forward, ChipAlignedNFForward)

        no_host = _Tuner(_Model([object()]), _armed_cfg())
        assert install_walk_for_aq_recovery(no_host) is False
        assert not hasattr(no_host.model, "forward")

        knob_off = _Tuner(
            _Model([object(), _host_node()]),
            _armed_cfg(lif_exact_qat_walk_recovery=False),
        )
        assert install_walk_for_aq_recovery(knob_off) is False
        assert not hasattr(knob_off.model, "forward")

    def test_lif_budget_restore_is_knob_and_host_graph_gated(self):
        from mimarsinan.tuning.orchestration.experimental_walk_recovery import (
            restore_lif_recovery_budget,
        )
        from mimarsinan.tuning.orchestration.lif_adaptation_plan import (
            LifAdaptationPlan,
        )

        cfg = _armed_cfg(endpoint_recovery_steps=600)
        plan = LifAdaptationPlan.resolve(cfg)
        assert plan.endpoint_recovery_steps == 0

        armed = _Tuner(_Model([object(), _host_node()]), cfg)
        assert restore_lif_recovery_budget(plan, armed).endpoint_recovery_steps == 600

        knob_off = _Tuner(
            _Model([object(), _host_node()]),
            _armed_cfg(endpoint_recovery_steps=600,
                       lif_exact_qat_walk_recovery=False),
        )
        assert restore_lif_recovery_budget(plan, knob_off) is plan

        no_host = _Tuner(_Model([object()]), cfg)
        assert restore_lif_recovery_budget(plan, no_host) is plan
