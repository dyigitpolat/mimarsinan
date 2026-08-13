from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep

from mimarsinan.common.best_effort import best_effort
from mimarsinan.common.env import vram_probe_enabled
from mimarsinan.common.reporter import emit_reporter_event
from mimarsinan.pipelining.pipeline_steps.mapping.deployment_record_emission import (
    emit_deployment_record_hcm,
)
from mimarsinan.mapping.crossbar_utilization import (
    CrossbarUtilizationReport,
    summarize_utilization,
    write_utilization_record,
)
from mimarsinan.mapping.weight_programming import weight_programming_report
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.hybrid_mapping_consumer import load_hybrid_mapping_for_step
from mimarsinan.pipelining.core.engine.pipeline_helpers import run_optional_viz
from mimarsinan.pipelining.core.simulation_factory import run_hcm_mapping_metric
from mimarsinan.pipelining.core.spike_count_gate import (
    run_spike_count_certificate_gate,
)
from mimarsinan.pipelining.core.gates.value_gates import (
    run_value_mapping_metric,
    run_value_twin_certificate_gate,
)

import torch
import os

from mimarsinan.common.diagnostics import phase_profiler


def _vram_probe(tag: str) -> None:
    """Opt-in VRAM/RSS probe when ``MIMARSINAN_VRAM_PROBE=1``."""
    if not vram_probe_enabled():
        return
    rss = 0
    with best_effort("read process RSS via psutil"):
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alc = torch.cuda.memory_allocated()
        rsv = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()
    else:
        alc = rsv = peak = 0
    print(
        f"[VRAM::HCM] {tag:<52} "
        f"RSS={rss/1e6:8.1f} MB  "
        f"alc={alc/1e6:8.1f} MB  "
        f"rsv={rsv/1e6:8.1f} MB  "
        f"peak={peak/1e6:8.1f} MB",
        flush=True,
    )

class HardCoreMappingStep(PipelineStep):
    REQUIRES = (
        "model", "ir_graph", "platform_constraints_resolved",
        "deployment_record_scm",
    )
    PROMISES = ("hard_core_mapping", "deployment_record_hcm")

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def validate(self):
        """Return hard-core spiking simulation accuracy from the step metric run."""
        m = getattr(self, "_last_metric", None)
        if m is None:
            raise RuntimeError(
                "Hard-core spiking simulation did not produce a metric; "
                "the step must run run_hcm_mapping_metric successfully."
            )
        return m

    def process(self):
        self._last_metric = None
        _vram_probe("process_entry")
        model = self.get_entry("model")
        ir_graph = self.get_entry('ir_graph')
        sim_len = int(self.pipeline.config["simulation_steps"])
        platform_constraints = self.get_entry("platform_constraints_resolved")
        scm_fragment = self.get_entry("deployment_record_scm")
        _vram_probe("after_load_entries")

        _HCM = "HardCoreMappingStep"
        with phase_profiler(_HCM, "build_hybrid"):
            hybrid_mapping = load_hybrid_mapping_for_step(self.pipeline, self)

        neural_segs = hybrid_mapping.get_neural_segments()
        compute_ops = hybrid_mapping.get_compute_ops()
        scheduled_stages = [s for s in hybrid_mapping.stages if s.schedule_pass_index is not None]
        if scheduled_stages:
            seg_pass_counts: dict[int, int] = {}
            for s in scheduled_stages:
                si = s.schedule_segment_index or 0
                seg_pass_counts[si] = max(seg_pass_counts.get(si, 0), (s.schedule_pass_index or 0) + 1)
            detail = ", ".join(
                f"seg {si}: {pc} pass{'es' if pc > 1 else ''}"
                for si, pc in sorted(seg_pass_counts.items())
            )
            print(
                f"[HardCoreMappingStep] Hybrid program (scheduled): {len(seg_pass_counts)} neural segment(s) "
                f"({detail}), {len(compute_ops)} compute op(s)"
            )
        else:
            print(
                f"[HardCoreMappingStep] Hybrid program: {len(neural_segs)} neural segments, "
                f"{len(compute_ops)} compute ops"
            )

        _vram_probe("after_build_hybrid")
        self.add_entry("hard_core_mapping", hybrid_mapping, "pickle")
        _vram_probe("after_pickle_save")

        # [wsm V0'] the weight-programming boundary, measured on every run.
        with phase_profiler(_HCM, "reports"):
            programming = weight_programming_report(hybrid_mapping)
        print(f"[WeightProgramming] {programming.summary()}")
        emit_reporter_event(self.pipeline.reporter, "weight_programming", {
            "neural_stages": programming.neural_stages,
            "programming_events": programming.programming_events,
            "params_programmed": programming.params_programmed,
            "params_unique": programming.params_unique,
            "reuse_factor": programming.reuse_factor,
        })

        # [imc G-A] crossbar occupancy of the deployed program, measured on every run.
        utilization = CrossbarUtilizationReport.from_hybrid_mapping(
            hybrid_mapping, weight_bits=platform_constraints.get("weight_bits"),
        )
        print(summarize_utilization(utilization))
        emit_reporter_event(
            self.pipeline.reporter, "crossbar_utilization", utilization.to_dict()
        )
        write_utilization_record(utilization, self.pipeline.working_directory)

        _vram_probe("before_test")
        plan = DeploymentPlan.of(self.pipeline)
        with phase_profiler(_HCM, "spike_count_gate"):
            spike_gate_result = run_spike_count_certificate_gate(
                self.pipeline, model, ir_graph, hybrid_mapping,
            )
        # Self-guarding: the gate SKIPs unless the policy observes values.
        with phase_profiler(_HCM, "value_twin_gate"):
            run_value_twin_certificate_gate(
                self.pipeline, model, ir_graph, hybrid_mapping,
            )
        plan_cap = plan.simulation_batch_size
        _metric_phase = phase_profiler(_HCM, "mapping_metric")
        _metric_phase.__enter__()
        if plan.mode_policy().observes_values():
            acc = run_value_mapping_metric(
                self.pipeline,
                ir_graph,
                platform_constraints,
                hybrid_mapping=hybrid_mapping,
                cache_key="hybrid_mapping",
            )
        else:
            acc = run_hcm_mapping_metric(
                self.pipeline,
                ir_graph,
                platform_constraints,
                hybrid_mapping=hybrid_mapping,
                model=model,
                cache_key="hybrid_mapping",
                # An explicit simulation_batch_size bounds the PRIMARY attempt too
                # (the >=1024 eval batch demands one huge contiguous encode);
                # the designed OOM retry stays as the fallback.
                max_batch_cap=int(plan_cap) if plan_cap else None,
                retry_on_oom=True,
                outer_oom_retry=True,
            )
        _metric_phase.__exit__(None, None, None)
        _vram_probe("after_test")
        self._last_metric = float(acc)
        print(f"[HardCoreMappingStep] Hard-core Spiking Simulation Test: {acc}")

        emit_deployment_record_hcm(
            self,
            hybrid_mapping,
            platform_constraints=platform_constraints,
            scm_fragment=scm_fragment,
            programming=programming,
            crossbar_report=utilization,
            spike_gate_result=spike_gate_result,
            accuracy=float(acc),
            observes_values=plan.mode_policy().observes_values(),
            model=model,
            ir_graph=ir_graph,
            input_shape=self.pipeline.config["input_shape"],
            num_classes=int(self.pipeline.config["num_classes"]),
            encoding_placement=str(
                self.pipeline.config.get("encoding_layer_placement", "subsume")
            ),
        )

        if self.pipeline.config.get("generate_visualizations", False):
            def _viz():
              from mimarsinan.visualization.hardcore_visualization import HardCoreMappingVisualizer
              from mimarsinan.visualization.graphviz import (
                  try_render_dot,
                  write_hybrid_hardcore_mapping_dots,
                  write_hybrid_hardcore_mapping_combined_dot,
              )

              artifacts = write_hybrid_hardcore_mapping_dots(
                  hybrid_mapping,
                  self.pipeline.working_directory,
                  basename="hybrid_hardcore_mapping",
              )

              heatmaps = []
              for i, seg in enumerate(hybrid_mapping.get_neural_segments()):
                  heat_path = self.pipeline.working_directory + f"/hybrid_segment{i}_hardcore_heatmap.png"
                  HardCoreMappingVisualizer(seg).visualize(heat_path)
                  heatmaps.append(heat_path)

              rendered = try_render_dot(artifacts.program_dot, formats=("svg", "png"))
              if rendered:
                  print(f"[HardCoreMappingStep] Wrote hybrid program visualization: {artifacts.program_dot} (+ {', '.join(rendered)})")
              else:
                  print(f"[HardCoreMappingStep] Wrote hybrid program visualization: {artifacts.program_dot} (render skipped: graphviz 'dot' not found)")

              segment_pngs = []
              for i, seg_dot in enumerate(artifacts.segment_dots):
                  rendered_seg = try_render_dot(seg_dot, formats=("svg", "png"))
                  if rendered_seg:
                      print(f"[HardCoreMappingStep] Wrote hybrid segment {i} visualization: {seg_dot} (+ {', '.join(rendered_seg)})")
                  else:
                      print(f"[HardCoreMappingStep] Wrote hybrid segment {i} visualization: {seg_dot} (render skipped: graphviz 'dot' not found)")
                  segment_pngs.append(os.path.splitext(seg_dot)[0] + ".png")

              combined_dot = self.pipeline.working_directory + "/hybrid_hardcore_mapping_combined.dot"
              write_hybrid_hardcore_mapping_combined_dot(
                  hybrid_mapping,
                  combined_dot,
                  segment_graph_pngs=segment_pngs,
                  segment_heatmap_pngs=heatmaps,
                  title=f"Hybrid mapping: {getattr(model, 'name', type(model).__name__)}",
              )
              rendered_combined = try_render_dot(combined_dot, formats=("svg", "png"))
              if rendered_combined:
                  print(f"[HardCoreMappingStep] Wrote hybrid combined overview: {combined_dot} (+ {', '.join(rendered_combined)})")
              else:
                  print(f"[HardCoreMappingStep] Wrote hybrid combined overview: {combined_dot} (render skipped: graphviz 'dot' not found)")

            run_optional_viz("HardCoreMappingStep", _viz)