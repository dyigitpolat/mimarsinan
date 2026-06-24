"""Frontier EW3 / R3 — the standing cost-extractor that mines a sim's cost record.

Locks the mechanism:

* the cost tuple ``{acc_deploy, mJ_per_sample, spikes, latency_steps, cores, mode,
  S, depth}`` is extracted CORRECTLY from a SANA-FE snapshot (the shape the step
  report already persists), keyed to the RIGHT E6 cell;
* the record round-trips through JSON losslessly (and fails loud on a format-version
  or field drift);
* a generated-run directory's persisted artifacts mine into the same record;
* the :class:`CostScatter` reader/aggregator groups by cell and computes the
  accuracy↑ × cost↓ Pareto front;
* the cost model the R3 scout confirmed — ``energy ∝ Σ_d neurons_d · S_d`` — is
  reproducible off the record's ``energy_proxy_neuron_steps`` alone.
"""

import json

import pytest

from mimarsinan.chip_simulation.certification import CertificationCell
from mimarsinan.chip_simulation.cost_extraction import (
    COST_RECORD_FILENAME,
    COST_RECORD_FORMAT_VERSION,
    CostRecord,
    CostScatter,
    extract_cost_record,
    extract_cost_record_from_run,
    load_cost_record,
    save_cost_record,
)


# --------------------------------------------------------------------------- #
# Synthetic SANA-FE snapshots — the shape ``SanafeStepReport.to_snapshot_dict``
# persists (aggregate headline + per_sample segment / per-core breakdown).
# --------------------------------------------------------------------------- #

def _segment(stage_index, stage_name, neurons_per_core, timesteps_executed, spikes):
    """One neural segment: a list of per-core neuron counts + its executed window."""
    return {
        "stage_index": stage_index,
        "stage_name": stage_name,
        "timesteps_executed": timesteps_executed,
        "spikes": spikes,
        "per_core": [{"n_neurons": n} for n in neurons_per_core],
    }


def _sanafe_snapshot(*, segments, total_energy_mj, total_spikes, T, sample_count=1):
    return {
        "arch_preset": "loihi",
        "sample_indices": list(range(sample_count)),
        "aggregate": {
            "sample_count": sample_count,
            "total_energy_mj": total_energy_mj,
            "total_spikes": total_spikes,
            "total_packets": 0,
        },
        "per_sample": [
            {"sample_index": 0, "T": T, "segments": segments}
        ],
    }


def _single_segment_snapshot():
    # Mirrors the real mmixcore LIF run: one segment, 115 cores totalling
    # 30720 neurons, executed for 13 timesteps, T (=S) = 4.
    neurons_per_core = [256] * 120  # 120 cores × 256 = 30720 neurons
    seg = _segment(0, "neural_segment_until:mean", neurons_per_core, 13, 607726)
    return _sanafe_snapshot(
        segments=[seg], total_energy_mj=0.04733, total_spikes=607726, T=4
    )


# --------------------------------------------------------------------------- #
# extract_cost_record — the cost tuple from what the sim already reports.
# --------------------------------------------------------------------------- #

class TestExtractCostRecord:
    def test_cost_tuple_extracted_correctly(self):
        cell = CertificationCell(firing="lif", sync=None, backend="sanafe")
        record = extract_cost_record(
            cell=cell,
            deployed_accuracy=0.974,
            sanafe_snapshot=_single_segment_snapshot(),
        )
        assert record.acc_deploy == pytest.approx(0.974)
        assert record.mj_per_sample == pytest.approx(0.04733)
        assert record.spikes == 607726
        assert record.latency_steps == 13
        assert record.cores == 120
        assert record.s_global == 4
        assert record.depth == 1

    def test_record_keyed_to_right_cell(self):
        cell = CertificationCell(firing="lif", sync=None, backend="sanafe")
        record = extract_cost_record(
            cell=cell, deployed_accuracy=0.97, sanafe_snapshot=_single_segment_snapshot()
        )
        assert record.cell_key == "lif@sanafe"
        assert record.mode == "lif"
        assert record.backend == "sanafe"
        assert record.cell == cell

    def test_cascaded_cell_keyed_with_schedule(self):
        cell = CertificationCell(
            firing="ttfs_cycle_based", sync="cascaded", backend="sanafe"
        )
        record = extract_cost_record(
            cell=cell, deployed_accuracy=0.95, sanafe_snapshot=_single_segment_snapshot()
        )
        assert record.cell_key == "ttfs_cycle_based/cascaded@sanafe"
        assert record.mode == "ttfs_cycle_based/cascaded"

    def test_multi_segment_sums_depth_latency_cores(self):
        # Two latency groups (the real ttfs_q two-segment run): depth=2, latency is
        # the SUM of per-segment executed windows, cores summed across segments.
        seg0 = _segment(0, "seg0", [256] * 60, 9, 200000)  # 15360 neurons
        seg1 = _segment(1, "seg1", [256] * 51, 8, 213502)  # 13056 neurons
        snap = _sanafe_snapshot(
            segments=[seg0, seg1],
            total_energy_mj=0.0299,
            total_spikes=413502,
            T=4,
        )
        cell = CertificationCell("ttfs_quantized", None, "sanafe")
        record = extract_cost_record(
            cell=cell, deployed_accuracy=0.986, sanafe_snapshot=snap
        )
        assert record.depth == 2
        assert record.latency_steps == 9 + 8
        assert record.cores == 60 + 51
        assert record.s_global == 4

    def test_energy_proxy_is_sum_neurons_times_steps(self):
        # The R3 cost model: energy ∝ Σ_d neurons_d · S_d. The proxy must equal that
        # sum so the cross-check is reproducible off the record alone.
        seg0 = _segment(0, "seg0", [256] * 60, 9, 0)  # 15360 · 9 = 138240
        seg1 = _segment(1, "seg1", [256] * 51, 8, 0)  # 13056 · 8 = 104448
        snap = _sanafe_snapshot(
            segments=[seg0, seg1], total_energy_mj=0.03, total_spikes=0, T=4
        )
        cell = CertificationCell("ttfs_quantized", None, "sanafe")
        record = extract_cost_record(
            cell=cell, deployed_accuracy=0.98, sanafe_snapshot=snap
        )
        assert record.energy_proxy_neuron_steps == 15360 * 9 + 13056 * 8

    def test_mj_per_sample_divides_by_sample_count(self):
        seg = _segment(0, "seg", [256] * 10, 4, 100)
        snap = _sanafe_snapshot(
            segments=[seg], total_energy_mj=0.08, total_spikes=100, T=4, sample_count=4
        )
        cell = CertificationCell("lif", None, "sanafe")
        record = extract_cost_record(
            cell=cell, deployed_accuracy=0.9, sanafe_snapshot=snap
        )
        assert record.mj_per_sample == pytest.approx(0.02)

    def test_cost_tuple_accessor(self):
        cell = CertificationCell("lif", None, "sanafe")
        record = extract_cost_record(
            cell=cell, deployed_accuracy=0.97, sanafe_snapshot=_single_segment_snapshot()
        )
        assert record.cost_tuple() == (record.mj_per_sample, 607726, 13, 120)


# --------------------------------------------------------------------------- #
# Serialization — round-trip + format-version / field-drift guards.
# --------------------------------------------------------------------------- #

class TestCostRecordSerialization:
    def _record(self):
        cell = CertificationCell("lif", None, "sanafe")
        return extract_cost_record(
            cell=cell,
            deployed_accuracy=0.974,
            sanafe_snapshot=_single_segment_snapshot(),
            provenance={"run_dir": "mnist_run"},
        )

    def test_round_trips_through_dict(self):
        record = self._record()
        assert CostRecord.from_dict(record.to_dict()) == record

    def test_round_trips_through_disk(self, tmp_path):
        record = self._record()
        run_dir = str(tmp_path / "run")
        path = save_cost_record(record, run_dir)
        assert path.endswith(COST_RECORD_FILENAME)
        assert load_cost_record(path) == record

    def test_rejects_format_version_drift(self):
        data = self._record().to_dict()
        data["format_version"] = COST_RECORD_FORMAT_VERSION + 1
        with pytest.raises(ValueError, match="format_version"):
            CostRecord.from_dict(data)

    def test_rejects_unknown_field(self):
        data = self._record().to_dict()
        data["bogus"] = 1
        with pytest.raises(ValueError, match="unknown fields"):
            CostRecord.from_dict(data)

    def test_saved_json_is_stable(self, tmp_path):
        record = self._record()
        path = save_cost_record(record, str(tmp_path / "run"))
        with open(path) as fh:
            data = json.load(fh)
        assert data["cell_key"] == "lif@sanafe"
        assert data["format_version"] == COST_RECORD_FORMAT_VERSION


# --------------------------------------------------------------------------- #
# extract_cost_record_from_run — mine a generated-run directory's artifacts.
# --------------------------------------------------------------------------- #

class TestExtractFromRun:
    def _write_run(self, run_dir, *, spiking_mode, schedule, snapshot, target_metric):
        import os

        os.makedirs(os.path.join(run_dir, "_GUI_STATE"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "_RUN_CONFIG"), exist_ok=True)
        with open(os.path.join(run_dir, "__target_metric.json"), "w") as fh:
            json.dump(target_metric, fh)
        steps = {
            "steps": {
                "SANA-FE Simulation": {
                    "snapshot": {"sanafe_simulation": snapshot},
                }
            }
        }
        with open(os.path.join(run_dir, "_GUI_STATE", "steps.json"), "w") as fh:
            json.dump(steps, fh)
        params = {"spiking_mode": spiking_mode}
        if schedule is not None:
            params["ttfs_cycle_schedule"] = schedule
        with open(os.path.join(run_dir, "_RUN_CONFIG", "config.json"), "w") as fh:
            json.dump({"deployment_parameters": params}, fh)

    def test_mines_run_directory(self, tmp_path):
        run_dir = str(tmp_path / "mnist_mmixcore_lif_run")
        self._write_run(
            run_dir,
            spiking_mode="lif",
            schedule=None,
            snapshot=_single_segment_snapshot(),
            target_metric=0.974,
        )
        record = extract_cost_record_from_run(run_dir)
        assert record is not None
        assert record.cell_key == "lif@sanafe"
        assert record.acc_deploy == pytest.approx(0.974)
        assert record.spikes == 607726
        assert record.cores == 120
        assert record.provenance["run_dir"] == "mnist_mmixcore_lif_run"

    def test_cascaded_run_keyed_with_schedule(self, tmp_path):
        run_dir = str(tmp_path / "mmix_cascade")
        self._write_run(
            run_dir,
            spiking_mode="ttfs_cycle_based",
            schedule="cascaded",
            snapshot=_single_segment_snapshot(),
            target_metric=0.95,
        )
        record = extract_cost_record_from_run(run_dir)
        assert record.cell_key == "ttfs_cycle_based/cascaded@sanafe"

    def test_run_without_sanafe_is_skipped(self, tmp_path):
        import os

        run_dir = str(tmp_path / "no_sanafe")
        os.makedirs(os.path.join(run_dir, "_GUI_STATE"), exist_ok=True)
        with open(os.path.join(run_dir, "_GUI_STATE", "steps.json"), "w") as fh:
            json.dump({"steps": {"Simulation": {}}}, fh)
        assert extract_cost_record_from_run(run_dir) is None


# --------------------------------------------------------------------------- #
# CostScatter — the reader/aggregator: cell-keyed grouping + Pareto front.
# --------------------------------------------------------------------------- #

def _record(cell, acc, mj, spikes, latency, cores, *, depth=1, s=4):
    return CostRecord(
        cell_key=cell.cell_key,
        mode=cell.cell_key.rsplit("@", 1)[0],
        backend=cell.backend,
        acc_deploy=acc,
        mj_per_sample=mj,
        spikes=spikes,
        latency_steps=latency,
        cores=cores,
        s_global=s,
        depth=depth,
    )


class TestCostScatter:
    def test_groups_by_cell(self):
        lif = CertificationCell("lif", None, "sanafe")
        ttfs = CertificationCell("ttfs", None, "sanafe")
        scatter = CostScatter()
        scatter.add(_record(lif, 0.97, 0.05, 100, 13, 120))
        scatter.add(_record(lif, 0.96, 0.04, 90, 12, 115))
        scatter.add(_record(ttfs, 0.98, 0.03, 80, 10, 109))
        assert scatter.cell_keys() == ["lif@sanafe", "ttfs@sanafe"]
        assert len(scatter.for_cell(lif)) == 2
        grouped = scatter.by_cell()
        assert set(grouped) == {"lif@sanafe", "ttfs@sanafe"}

    def test_pareto_front_drops_dominated(self):
        cell = CertificationCell("ttfs_cycle_based", "cascaded", "sanafe")
        # The R3 finding: genuine cascaded S=4→8→12 is DOMINATED everywhere —
        # latency and energy rise while accuracy DROPS. So only the cheapest,
        # most-accurate S=4 point is on the front.
        s4 = _record(cell, 0.972, 0.040, 100, 28, 109, depth=3, s=4)
        s8 = _record(cell, 0.956, 0.052, 200, 64, 109, depth=3, s=8)
        s12 = _record(cell, 0.914, 0.061, 300, 96, 109, depth=3, s=12)
        scatter = CostScatter([s12, s8, s4])
        front = scatter.pareto_front(cell=cell)
        assert front == [s4]

    def test_pareto_front_keeps_genuine_tradeoffs(self):
        cell = CertificationCell("lif", None, "sanafe")
        # cheaper-but-less-accurate vs pricier-but-more-accurate: BOTH on the front.
        cheap = _record(cell, 0.95, 0.03, 80, 10, 100)
        accurate = _record(cell, 0.98, 0.05, 120, 14, 115)
        scatter = CostScatter([cheap, accurate])
        front = scatter.pareto_front(cell=cell)
        assert cheap in front and accurate in front and len(front) == 2

    def test_cross_cell_pareto_front(self):
        # The headline R3 scatter: rate-ttfs (0.992 @ 0.033 mJ) DOMINATES
        # genuine-synchronized (0.976 @ 0.089 mJ) at fixed accuracy band.
        rate = CertificationCell("ttfs", None, "sanafe")
        gen = CertificationCell("ttfs_cycle_based", "synchronized", "sanafe")
        rate_pt = _record(rate, 0.992, 0.033, 80, 12, 100)
        gen_pt = _record(gen, 0.976, 0.089, 200, 40, 120, depth=4, s=8)
        scatter = CostScatter([gen_pt, rate_pt])
        front = scatter.pareto_front()
        assert front == [rate_pt]

    def test_from_runs_mines_and_skips(self, tmp_path):
        import os

        run_dir = str(tmp_path / "mmix_lif")
        os.makedirs(os.path.join(run_dir, "_GUI_STATE"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "_RUN_CONFIG"), exist_ok=True)
        with open(os.path.join(run_dir, "__target_metric.json"), "w") as fh:
            json.dump(0.97, fh)
        steps = {
            "steps": {
                "SANA-FE Simulation": {
                    "snapshot": {"sanafe_simulation": _single_segment_snapshot()},
                }
            }
        }
        with open(os.path.join(run_dir, "_GUI_STATE", "steps.json"), "w") as fh:
            json.dump(steps, fh)
        with open(os.path.join(run_dir, "_RUN_CONFIG", "config.json"), "w") as fh:
            json.dump({"deployment_parameters": {"spiking_mode": "lif"}}, fh)
        empty = str(tmp_path / "empty")
        os.makedirs(empty, exist_ok=True)
        scatter = CostScatter.from_runs([run_dir, empty])
        assert scatter.cell_keys() == ["lif@sanafe"]


# --------------------------------------------------------------------------- #
# AC5 — the per-fine-tuning-PASS wall the cost record carries (task A5).
# --------------------------------------------------------------------------- #

class TestFtPassWall:
    def test_default_is_zero(self):
        # A run with no surfaced FT-pass wall still produces a well-defined record
        # (0.0 ⇒ "no FT pass timed", never None — AC5 reads a float).
        cell = CertificationCell("lif", None, "sanafe")
        record = extract_cost_record(
            cell=cell, deployed_accuracy=0.97, sanafe_snapshot=_single_segment_snapshot()
        )
        assert record.max_ft_pass_wall_s == 0.0
        assert record.ft_pass_walls == ()

    def test_carries_max_ft_pass_wall_and_breakdown(self):
        # AC5 is judged on this exact field — the MAX single FT pass wall — plus the
        # per-pass breakdown the verdict can drill into.
        cell = CertificationCell("lif", None, "sanafe")
        passes = (
            {"label": "recover", "wall_s": 42.0},
            {"label": "stabilize", "wall_s": 180.0},
        )
        record = extract_cost_record(
            cell=cell,
            deployed_accuracy=0.97,
            sanafe_snapshot=_single_segment_snapshot(),
            max_ft_pass_wall_s=180.0,
            ft_pass_walls=passes,
        )
        assert record.max_ft_pass_wall_s == pytest.approx(180.0)
        assert record.ft_pass_walls == passes

    def test_max_ft_pass_wall_round_trips(self, tmp_path):
        cell = CertificationCell("lif", None, "sanafe")
        record = extract_cost_record(
            cell=cell,
            deployed_accuracy=0.97,
            sanafe_snapshot=_single_segment_snapshot(),
            max_ft_pass_wall_s=123.4,
            ft_pass_walls=({"label": "recover", "wall_s": 123.4},),
        )
        assert CostRecord.from_dict(record.to_dict()) == record
        path = save_cost_record(record, str(tmp_path / "run"))
        assert load_cost_record(path) == record

    def test_mines_ft_pass_walls_from_run_artifact(self, tmp_path):
        # The standing path: the tuner persists ``ft_pass_walls.json`` alongside the
        # run artifacts; ``extract_cost_record_from_run`` mines it into the record.
        import os

        run_dir = str(tmp_path / "mmix_lif")
        os.makedirs(os.path.join(run_dir, "_GUI_STATE"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "_RUN_CONFIG"), exist_ok=True)
        with open(os.path.join(run_dir, "__target_metric.json"), "w") as fh:
            json.dump(0.97, fh)
        steps = {
            "steps": {
                "SANA-FE Simulation": {
                    "snapshot": {"sanafe_simulation": _single_segment_snapshot()},
                }
            }
        }
        with open(os.path.join(run_dir, "_GUI_STATE", "steps.json"), "w") as fh:
            json.dump(steps, fh)
        with open(os.path.join(run_dir, "_RUN_CONFIG", "config.json"), "w") as fh:
            json.dump({"deployment_parameters": {"spiking_mode": "lif"}}, fh)
        from mimarsinan.chip_simulation.cost_extraction import FT_PASS_WALLS_FILENAME
        with open(os.path.join(run_dir, FT_PASS_WALLS_FILENAME), "w") as fh:
            json.dump(
                {
                    "max_ft_pass_wall_s": 73.2,
                    "passes": [
                        {"label": "recover", "wall_s": 30.0},
                        {"label": "recover", "wall_s": 73.2},
                    ],
                },
                fh,
            )
        record = extract_cost_record_from_run(run_dir)
        assert record.max_ft_pass_wall_s == pytest.approx(73.2)
        assert record.ft_pass_walls == (
            {"label": "recover", "wall_s": 30.0},
            {"label": "recover", "wall_s": 73.2},
        )

    def test_missing_ft_pass_walls_artifact_defaults_to_zero(self, tmp_path):
        import os

        run_dir = str(tmp_path / "mmix_lif_no_ft")
        os.makedirs(os.path.join(run_dir, "_GUI_STATE"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "_RUN_CONFIG"), exist_ok=True)
        with open(os.path.join(run_dir, "__target_metric.json"), "w") as fh:
            json.dump(0.97, fh)
        steps = {
            "steps": {
                "SANA-FE Simulation": {
                    "snapshot": {"sanafe_simulation": _single_segment_snapshot()},
                }
            }
        }
        with open(os.path.join(run_dir, "_GUI_STATE", "steps.json"), "w") as fh:
            json.dump(steps, fh)
        with open(os.path.join(run_dir, "_RUN_CONFIG", "config.json"), "w") as fh:
            json.dump({"deployment_parameters": {"spiking_mode": "lif"}}, fh)
        record = extract_cost_record_from_run(run_dir)
        assert record.max_ft_pass_wall_s == 0.0
        assert record.ft_pass_walls == ()


# --------------------------------------------------------------------------- #
# Cost-model cross-check — energy ∝ Σ_d neurons_d · S_d (the R3 soma-dominance).
# --------------------------------------------------------------------------- #

class TestCostModelCrossCheck:
    def test_energy_proxy_tracks_energy_across_runs(self):
        # Two runs of the SAME (firing × sync) cell: the per-neuron-step energy
        # ratio (mJ / Σ neurons·S) must be ~constant (soma-dominated cost model).
        cell = CertificationCell("lif", None, "sanafe")

        def run(neurons_per_core, timesteps, energy_mj):
            seg = _segment(0, "seg", [256] * neurons_per_core, timesteps, 0)
            snap = _sanafe_snapshot(
                segments=[seg], total_energy_mj=energy_mj, total_spikes=0, T=4
            )
            return extract_cost_record(
                cell=cell, deployed_accuracy=0.97, sanafe_snapshot=snap
            )

        small = run(60, 4, 0.020)
        big = run(120, 4, 0.040)
        # Σ neurons·S doubles (60·256·4 → 120·256·4) and energy doubles too:
        assert big.energy_proxy_neuron_steps == 2 * small.energy_proxy_neuron_steps
        ratio_small = small.mj_per_sample / small.energy_proxy_neuron_steps
        ratio_big = big.mj_per_sample / big.energy_proxy_neuron_steps
        assert ratio_big == pytest.approx(ratio_small, rel=0.05)


# --------------------------------------------------------------------------- #
# Weight-reuse scheduling cost term (round-1 GAP-R fix). The cost model gains a
# reprogram-vs-reuse distinction: a reuse pass costs activation data movement at a
# sync barrier, NOT a parameter reload. Chip coefficients default 0.0 ⇒ the whole
# term is 0 ⇒ byte-identical to a record without it.
# --------------------------------------------------------------------------- #

from mimarsinan.chip_simulation.cost_extraction import weight_reuse_mj


class TestWeightReuseCostTerm:
    def _record(self, **kw):
        cell = CertificationCell("lif", None, "sanafe")
        return extract_cost_record(
            cell=cell,
            deployed_accuracy=0.97,
            sanafe_snapshot=_single_segment_snapshot(),
            **kw,
        )

    def test_phase_fields_default_zero(self):
        # A record built without phase info still has well-defined (0) phase fields —
        # byte-identical to the pre-mode record (every pass implicitly a reprogram).
        record = self._record()
        assert record.reprogram_passes == 0
        assert record.reuse_passes == 0
        assert record.params_reloaded == 0
        assert record.activation_bytes_moved == 0

    def test_phase_fields_carried(self):
        record = self._record(
            reprogram_passes=16,
            reuse_passes=142,
            params_reloaded=1_000_000,
            activation_bytes_moved=2_000_000,
        )
        assert record.reprogram_passes == 16
        assert record.reuse_passes == 142
        assert record.params_reloaded == 1_000_000
        assert record.activation_bytes_moved == 2_000_000

    def test_total_passes_property(self):
        record = self._record(reprogram_passes=16, reuse_passes=142)
        assert record.total_passes == 158
        assert record.sync_barrier_count == 157  # total_passes - 1

    def test_sync_barrier_count_floors_at_zero(self):
        # No passes recorded ⇒ no barriers (never negative).
        assert self._record().sync_barrier_count == 0

    def test_reuse_fraction(self):
        record = self._record(reprogram_passes=16, reuse_passes=142)
        assert record.reuse_fraction == pytest.approx(142 / 158)

    def test_reuse_mj_zero_coefficients_is_byte_identical(self):
        # The keystone byte-identical guarantee: with both chip coefficients 0.0 the
        # weight-reuse mJ term is exactly 0 — adds nothing to the deployed energy.
        record = self._record(
            reprogram_passes=16, reuse_passes=142,
            params_reloaded=1_000_000, activation_bytes_moved=2_000_000,
        )
        assert record.reuse_mj() == 0.0
        assert record.reuse_mj(mj_per_reprogram=0.0, mj_per_sync=0.0) == 0.0

    def test_reuse_mj_charges_reprogram_per_param(self):
        # mj_per_reprogram is charged per PARAM RELOADED (Σ over the N reprogram passes
        # of the resident bank weight count) — reuse passes reload nothing.
        record = self._record(
            reprogram_passes=2, reuse_passes=200,
            params_reloaded=1000, activation_bytes_moved=0,
        )
        assert record.reuse_mj(mj_per_reprogram=1e-3, mj_per_sync=0.0) == pytest.approx(
            1000 * 1e-3
        )

    def test_reuse_mj_charges_sync_per_barrier_byte(self):
        # mj_per_sync is charged per ACTIVATION BYTE moved at each of the
        # (total_passes - 1) sync barriers.
        record = self._record(
            reprogram_passes=2, reuse_passes=2,
            params_reloaded=0, activation_bytes_moved=500,
        )
        # 4 passes ⇒ 3 barriers; bytes is already the total over barriers (summed by
        # the caller from the per-barrier slice walk), so charge it once per the
        # supplied total.
        assert record.reuse_mj(mj_per_reprogram=0.0, mj_per_sync=2e-6) == pytest.approx(
            500 * 2e-6
        )

    def test_reuse_mj_helper_matches_record(self):
        record = self._record(
            reprogram_passes=3, reuse_passes=10,
            params_reloaded=4000, activation_bytes_moved=900,
        )
        direct = weight_reuse_mj(
            params_reloaded=4000,
            activation_bytes_moved=900,
            mj_per_reprogram=5e-4,
            mj_per_sync=1e-5,
        )
        assert record.reuse_mj(mj_per_reprogram=5e-4, mj_per_sync=1e-5) == direct
        assert direct == pytest.approx(4000 * 5e-4 + 900 * 1e-5)

    def test_phase_fields_round_trip(self):
        record = self._record(
            reprogram_passes=16, reuse_passes=142,
            params_reloaded=1_000_000, activation_bytes_moved=2_000_000,
        )
        assert CostRecord.from_dict(record.to_dict()) == record

    def test_phase_fields_round_trip_disk(self, tmp_path):
        record = self._record(reprogram_passes=5, reuse_passes=50, params_reloaded=99)
        path = save_cost_record(record, str(tmp_path / "run"))
        assert load_cost_record(path) == record

    def test_format_version_is_three(self):
        # The phase fields bumped the cost-record format 2 -> 3.
        assert COST_RECORD_FORMAT_VERSION == 3
        assert self._record().format_version == 3

    def test_old_format_version_two_rejected(self):
        data = self._record().to_dict()
        data["format_version"] = 2
        with pytest.raises(ValueError, match="format_version"):
            CostRecord.from_dict(data)


# --------------------------------------------------------------------------- #
# P2 — the grounded BANDED mJ wired into the production cost-emit path. The
# record sources its low/nominal/high mJ from the defensible `weight_reuse_cost_
# model` (consuming `phase_cost_band`); the default `reuse_mj` (0.0) stays
# byte-identical, and the emit step attaches the band ONLY when a coefficient
# config is supplied (no config ⇒ the emitted record is byte-identical to today).
# --------------------------------------------------------------------------- #

from mimarsinan.chip_simulation.weight_reuse_cost_model import (
    DEFAULT_COEFFICIENT_BAND,
    CoefficientBand,
    DmaCostCoefficients,
    phase_cost_band,
)


class TestReuseMjBand:
    def _record(self, **kw):
        cell = CertificationCell("lif", None, "sanafe")
        return extract_cost_record(
            cell=cell,
            deployed_accuracy=0.97,
            sanafe_snapshot=_single_segment_snapshot(),
            **kw,
        )

    def test_band_matches_phase_cost_band_for_known_plan(self):
        # The grounded BANDED mJ is sourced from `weight_reuse_cost_model`: the
        # record's `reuse_mj_band` IS `phase_cost_band` evaluated over the record's
        # phase plan + the default cited coefficient band.
        record = self._record(
            reprogram_passes=16,
            reuse_passes=142,
            params_reloaded=66_600_000,
            activation_bytes_moved=3_000_000,
        )
        lo, mid, hi = record.reuse_mj_band()
        expected = phase_cost_band(
            reprogram_passes=16,
            reuse_passes=142,
            params_reloaded=66_600_000,
            activation_bytes_moved=3_000_000,
            coefficient_band=DEFAULT_COEFFICIENT_BAND,
        )
        assert lo == pytest.approx(expected.low_mj)
        assert mid == pytest.approx(expected.nominal_mj)
        assert hi == pytest.approx(expected.high_mj)

    def test_band_is_ordered_lo_mid_hi(self):
        record = self._record(
            reprogram_passes=16,
            reuse_passes=142,
            params_reloaded=66_600_000,
            activation_bytes_moved=3_000_000,
        )
        lo, mid, hi = record.reuse_mj_band()
        assert lo <= mid <= hi
        # A defensible band is a genuine RANGE — high strictly above low.
        assert hi > lo

    def test_band_accepts_a_custom_coefficient_band(self):
        record = self._record(
            reprogram_passes=2,
            reuse_passes=0,
            params_reloaded=1000,
            activation_bytes_moved=0,
        )
        flat = DmaCostCoefficients(
            e_dma_per_byte_mj=1e-6, bytes_per_param=1.0, e_sync_barrier_mj=0.0
        )
        custom = CoefficientBand(low=flat, nominal=flat, high=flat)
        lo, mid, hi = record.reuse_mj_band(coeffs=custom)
        # reprogram DMA = 1000 · 1 · 1e-6 = 1e-3, no sync charge.
        assert lo == pytest.approx(1e-3)
        assert mid == pytest.approx(1e-3)
        assert hi == pytest.approx(1e-3)

    def test_band_zero_when_no_phase_fields(self):
        # A default record (no phase plan) ⇒ a 0/0/0 band — nothing to reload.
        record = self._record()
        assert record.reuse_mj_band() == (0.0, 0.0, 0.0)

    def test_reuse_mj_default_stays_byte_identical(self):
        # The grounded band is ADDITIVE: the default-off `reuse_mj` (0.0 coeffs)
        # is untouched — still exactly 0.0 — so nothing else moves.
        record = self._record(
            reprogram_passes=16, reuse_passes=142,
            params_reloaded=66_600_000, activation_bytes_moved=3_000_000,
        )
        assert record.reuse_mj() == 0.0


# --------------------------------------------------------------------------- #
# Backend as a first-class cost coordinate — a CostRecord carries its backend so
# cost records are comparable per backend (the E6 cell keys firing × sync ×
# backend). This LOCKS the property; it is not a behavior change.
# --------------------------------------------------------------------------- #

class TestBackendCoordinate:
    def _record(self, cell):
        return extract_cost_record(
            cell=cell,
            deployed_accuracy=0.97,
            sanafe_snapshot=_single_segment_snapshot(),
        )

    def test_record_carries_backend_field(self):
        record = self._record(CertificationCell("lif", None, "sanafe"))
        assert record.backend == "sanafe"

    def test_cell_property_carries_backend(self):
        # The cost cell (re-parsed from the key) names the SAME backend coordinate.
        record = self._record(CertificationCell("lif", None, "nevresim"))
        assert record.cell.backend == "nevresim"
        assert record.cell_key.endswith("@nevresim")

    def test_different_backends_produce_different_cells(self):
        # The keystone lock: two records of the SAME (firing × sync) recipe on
        # DIFFERENT backends are DISTINCT cells — never silently merged.
        sanafe = self._record(CertificationCell("lif", None, "sanafe"))
        nevresim = self._record(CertificationCell("lif", None, "nevresim"))
        assert sanafe.cell != nevresim.cell
        assert sanafe.cell_key != nevresim.cell_key
        assert sanafe.cell.firing == nevresim.cell.firing  # same recipe
        assert sanafe.cell.backend != nevresim.cell.backend  # different coordinate


# --------------------------------------------------------------------------- #
# Wiring — the production cost-emit step OPTIONALLY attaches the grounded band,
# ONLY when a cost-coefficient config is supplied. With NO config (default) the
# emitted record is byte-IDENTICAL to today (no band attached, reuse_mj 0.0).
# --------------------------------------------------------------------------- #

class TestEmitWiring:
    def _emit(self, **kw):
        cell = CertificationCell("lif", None, "sanafe")
        return extract_cost_record(
            cell=cell,
            deployed_accuracy=0.97,
            sanafe_snapshot=_single_segment_snapshot(),
            reprogram_passes=16,
            reuse_passes=142,
            params_reloaded=66_600_000,
            activation_bytes_moved=3_000_000,
            **kw,
        )

    def test_default_emit_is_byte_identical(self):
        # No coefficient config (default) ⇒ the emitted record is byte-identical to
        # today: reuse_mj 0.0, NO band in provenance, the full dict unchanged.
        with_default = self._emit()
        no_kw = extract_cost_record(
            cell=CertificationCell("lif", None, "sanafe"),
            deployed_accuracy=0.97,
            sanafe_snapshot=_single_segment_snapshot(),
            reprogram_passes=16,
            reuse_passes=142,
            params_reloaded=66_600_000,
            activation_bytes_moved=3_000_000,
        )
        assert with_default == no_kw
        assert with_default.reuse_mj() == 0.0
        assert "reuse_cost_band_mj" not in with_default.provenance
        assert with_default.to_dict() == no_kw.to_dict()

    def test_default_emit_provenance_unchanged(self):
        # The default record's provenance is exactly what the caller passed (here
        # empty) — the band wiring adds nothing when no config is supplied.
        record = self._emit()
        assert record.provenance == {}

    def test_emit_with_band_attaches_grounded_range(self):
        # A supplied coefficient band ⇒ the emitted record carries the grounded
        # low/nominal/high mJ (sourced from `phase_cost_band`) in provenance,
        # comparable per backend.
        record = self._emit(coefficient_band=DEFAULT_COEFFICIENT_BAND)
        band = record.provenance["reuse_cost_band_mj"]
        expected = phase_cost_band(
            reprogram_passes=16,
            reuse_passes=142,
            params_reloaded=66_600_000,
            activation_bytes_moved=3_000_000,
            coefficient_band=DEFAULT_COEFFICIENT_BAND,
        )
        assert band == [
            pytest.approx(expected.low_mj),
            pytest.approx(expected.nominal_mj),
            pytest.approx(expected.high_mj),
        ]

    def test_emit_with_band_round_trips(self, tmp_path):
        record = self._emit(coefficient_band=DEFAULT_COEFFICIENT_BAND)
        assert CostRecord.from_dict(record.to_dict()) == record
        path = save_cost_record(record, str(tmp_path / "run"))
        assert load_cost_record(path) == record

    def test_emit_band_preserves_explicit_provenance(self):
        # The grounded band is added ALONGSIDE any caller provenance, not instead.
        cell = CertificationCell("lif", None, "sanafe")
        record = extract_cost_record(
            cell=cell,
            deployed_accuracy=0.97,
            sanafe_snapshot=_single_segment_snapshot(),
            reprogram_passes=2,
            reuse_passes=0,
            params_reloaded=1000,
            provenance={"run_dir": "mnist_run"},
            coefficient_band=DEFAULT_COEFFICIENT_BAND,
        )
        assert record.provenance["run_dir"] == "mnist_run"
        assert "reuse_cost_band_mj" in record.provenance
