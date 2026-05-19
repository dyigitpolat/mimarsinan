from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.ir_latency import IRLatency
from mimarsinan.mapping.ir import NeuralCore

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.pipelining.simulation_factory import (
    build_hybrid_mapping_for_pipeline,
    build_spiking_hybrid_flow,
)
from mimarsinan.common.diagnostics import phase_profiler

import numpy as np
import torch.nn as nn
import torch

import os

class SoftCoreMappingStep(PipelineStep):

    def __init__(self, pipeline):
        # Require fused_model from Normalization Fusion.
        requires = ["fused_model", "platform_constraints_resolved"]
        promises = ["ir_graph"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)
        self.trainer = None
        self._soft_core_spiking_metric = None

    def validate(self):
        if self._soft_core_spiking_metric is not None:
            return self._soft_core_spiking_metric
        if self.trainer is not None:
            return self.trainer.validate()
        return self.pipeline.get_target_metric()

    def pipeline_metric(self):
        if self._soft_core_spiking_metric is not None:
            return self._soft_core_spiking_metric
        return super().pipeline_metric()

    def process(self):
        model = self.get_entry("fused_model")
        platform_constraints = self.get_entry("platform_constraints_resolved")

        cores = platform_constraints.get("cores", [])
        from mimarsinan.mapping.platform_constraints import resolve_platform_mapping_params

        mapping_params = resolve_platform_mapping_params(
            cores,
            allow_coalescing=bool(platform_constraints.get("allow_coalescing", False)),
        )
        resolved_hardware_bias = mapping_params.hardware_bias
        effective_max_axons = mapping_params.effective_max_axons
        resolved_max_neurons = mapping_params.effective_max_neurons
        resolved_allow_coalescing = mapping_params.allow_coalescing

        for perceptron in model.get_perceptrons():
            if isinstance(perceptron.layer, FusedLinear):
                perceptron.layer = self.bring_back_bias(perceptron.layer)

        if self.pipeline.config.get("generate_visualizations", False):
          try:
              from mimarsinan.visualization.softcore_flowchart import write_softcore_flowchart_dot

              try:
                  flowchart_device = next(model.parameters()).device
              except StopIteration:
                  flowchart_device = self.pipeline.config["device"]

              out_dot = os.path.join(self.pipeline.working_directory, "softcore_flowchart.dot")
              write_softcore_flowchart_dot(
                  model.get_mapper_repr(),
                  out_dot,
                  input_shape=tuple(self.pipeline.config["input_shape"]),
                  max_axons=int(effective_max_axons),
                  max_neurons=int(resolved_max_neurons),
                  device=flowchart_device,
              )
              print(f"[SoftCoreMappingStep] Wrote flowchart DOT: {out_dot}")
          except Exception as e:
              print(f"[SoftCoreMappingStep] Flowchart generation failed (non-fatal): {e}")
        
        _PHASE_TAG = "SoftCoreMappingStep"
        def _phase(name):
            return phase_profiler(_PHASE_TAG, name)

        with _phase("basic_trainer_ctor"):
            self.trainer = BasicTrainer(
                model,
                self.pipeline.config['device'],
                DataLoaderFactory(self.pipeline.data_provider_factory),
                self.pipeline.loss)
        validator = self.trainer

        act_q = bool(self.pipeline.config.get("activation_quantization", False))

        # ttfs_quantized requires activation_quantization for deployment parity.
        spiking_mode = self.pipeline.config.get("spiking_mode", "lif")
        if spiking_mode == "ttfs_quantized" and not act_q:
            print(
                "[SoftCoreMappingStep] Warning: ttfs_quantized is on but activation_quantization is off; "
                "deployment accuracy may drop compared to training."
            )

        self._apply_ttfs_quantized_bias_shift(model, act_q)

        from mimarsinan.transformations.quantization_bounds import quantization_bounds

        bits = self.pipeline.config['weight_bits']
        _, q_max = quantization_bounds(bits)

        ir_mapping = IRMapping(
            q_max=q_max,
            firing_mode=self.pipeline.config["firing_mode"],
            max_axons=effective_max_axons,
            max_neurons=resolved_max_neurons,
            allow_coalescing=resolved_allow_coalescing,
            hardware_bias=resolved_hardware_bias,
        )
        
        mapper_repr = model.get_mapper_repr()
        if hasattr(mapper_repr, "assign_perceptron_indices"):
            mapper_repr.assign_perceptron_indices()
        with _phase("ir_mapping.map"):
            ir_graph = ir_mapping.map(mapper_repr)

        wt_q = bool(self.pipeline.config.get("weight_quantization", False))
        from mimarsinan.mapping.chip_quantize import quantize_ir_graph

        with _phase("weight_quantization"):
            quantize_ir_graph(ir_graph, bits, weight_quantization=wt_q)

        # Calculate latencies for all neural cores in the IR graph
        with _phase("ir_latency"):
            max_latency = IRLatency(ir_graph).calculate()
        print(f"[SoftCoreMappingStep] IR Graph max latency: {max_latency}")

        # Compact zeroed rows/columns when pruning was applied.
        if self.pipeline.config.get("pruning", False):
            from mimarsinan.mapping.ir_pruning import prune_ir_graph, get_initial_pruning_masks_from_model
            try:
                perceptrons_pre = model.get_perceptrons()
                if perceptrons_pre:
                    layer0 = getattr(perceptrons_pre[0], "layer", None)
                    has_row = getattr(layer0, "prune_row_mask", None) is not None
                    has_col = getattr(layer0, "prune_col_mask", None) is not None
                    print(
                        f"[SoftCoreMappingStep] Pruning: before mask extraction — first perceptron layer "
                        f"prune_row_mask={has_row} prune_col_mask={has_col}"
                    )
            except Exception as e:
                print(f"[SoftCoreMappingStep] Pruning: could not check first perceptron buffers: {e}")
            initial_node, initial_bank = get_initial_pruning_masks_from_model(model, ir_graph)
            try:
                perceptrons = model.get_perceptrons()
                neural_cores = ir_graph.get_neural_cores()
                n_banks = len(getattr(ir_graph, "weight_banks", {}))
                print(
                    f"[SoftCoreMappingStep] Pruning: perceptrons={len(perceptrons)} neural_cores={len(neural_cores)} "
                    f"weight_banks={n_banks} initial_pruned_per_node={len(initial_node or {})} "
                    f"initial_pruned_per_bank={len(initial_bank or {})}"
                )
                if len(initial_node or {}) == 0 and len(initial_bank or {}) == 0 and len(neural_cores) != len(perceptrons):
                    print(
                        "[SoftCoreMappingStep] Pruning: no model masks applied (neural_cores != perceptrons; "
                        "ensure mapper assigns perceptron_index for tiled IR)."
                    )
            except Exception:
                pass
            store_heatmap = bool(self.pipeline.config.get("store_pre_pruning_heatmap", True))
            if store_heatmap:
                heatmap_budget_bytes = int(self.pipeline.config.get(
                    "pre_pruning_heatmap_budget_bytes", 2 * 1024**3,  # 2 GB
                ))
                est_bytes = 0
                for nc in ir_graph.get_neural_cores():
                    if nc.core_matrix is not None:
                        est_bytes += nc.core_matrix.shape[0] * nc.core_matrix.shape[1] * 4  # float32
                if est_bytes > heatmap_budget_bytes:
                    print(
                        f"[SoftCoreMappingStep] Pre-pruning heatmap would require "
                        f"{est_bytes/1e9:.1f} GB (budget {heatmap_budget_bytes/1e9:.1f} GB); "
                        f"disabling heatmap storage for this run. "
                        f"Set `pre_pruning_heatmap_budget_bytes` higher to override."
                    )
                    store_heatmap = False
            with _phase("prune_ir_graph"):
                ir_graph = prune_ir_graph(
                    ir_graph,
                    initial_pruned_per_node=initial_node if initial_node else None,
                    initial_pruned_per_bank=initial_bank if initial_bank else None,
                    store_heatmap=store_heatmap,
                )
            print(f"[SoftCoreMappingStep] Applied IR pruning (zeroed row/col elimination)")

        with _phase("pickle_save"):
            self.add_entry("ir_graph", ir_graph, 'pickle')

        if self.pipeline.config.get("generate_visualizations", False):
          try:
              from mimarsinan.visualization.mapping_graphviz import (
                  try_render_dot,
                  write_ir_graph_dot,
                  write_ir_graph_summary_dot,
              )

              node_count = len(ir_graph.nodes)
              large_graph = node_count > 500

              out_dot = os.path.join(self.pipeline.working_directory, "ir_graph.dot")
              write_ir_graph_dot(
                  ir_graph,
                  out_dot,
                  title=f"IRGraph: {getattr(model, 'name', type(model).__name__)}",
              )
              if large_graph:
                  print(f"[SoftCoreMappingStep] Wrote IRGraph DOT: {out_dot} (render skipped: {node_count} nodes)")
              else:
                  rendered = try_render_dot(out_dot, formats=("svg", "png"))
                  if rendered:
                      print(f"[SoftCoreMappingStep] Wrote IRGraph visualization: {out_dot} (+ {', '.join(rendered)})")
                  else:
                      print(f"[SoftCoreMappingStep] Wrote IRGraph visualization: {out_dot} (render skipped: graphviz 'dot' not found)")

              out_sum = os.path.join(self.pipeline.working_directory, "ir_graph_summary.dot")
              write_ir_graph_summary_dot(
                  ir_graph,
                  out_sum,
                  title=f"IRGraph: {getattr(model, 'name', type(model).__name__)}",
              )
              rendered_sum = try_render_dot(out_sum, formats=("svg", "png"))
              if rendered_sum:
                  print(f"[SoftCoreMappingStep] Wrote IRGraph summary: {out_sum} (+ {', '.join(rendered_sum)})")
              else:
                  print(f"[SoftCoreMappingStep] Wrote IRGraph summary: {out_sum} (render skipped: graphviz 'dot' not found)")
          except Exception as e:
              print(f"[SoftCoreMappingStep] IRGraph visualization failed (non-fatal): {e}")

        compute_ops = ir_graph.get_compute_ops()
        neural_cores = ir_graph.get_neural_cores()
        print(f"[SoftCoreMappingStep] IR Graph: {len(neural_cores)} neural cores, {len(compute_ops)} compute ops")
        if compute_ops:
            print(f"[SoftCoreMappingStep] Model contains {len(compute_ops)} non-neural operations:")
            for op in compute_ops:
                print(f"  - {op.name}: {op.op_type}")

        device = self.pipeline.config["device"]
        sim_batches = self.pipeline.config.get("simulation_batch_count", None)

        try:
            model.to("cpu")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        spiking_trainer = None
        attempt_cap = None
        for attempt in range(2):
            try:
                cap_tag = "default" if attempt_cap is None else str(attempt_cap)
                with _phase(f"sim_build_hybrid[cap={cap_tag}]"):
                    hybrid_mapping = build_hybrid_mapping_for_pipeline(
                        ir_graph, platform_constraints
                    )
                    self.add_entry("hybrid_mapping", hybrid_mapping, "pickle")
                with _phase(f"sim_construct[cap={cap_tag}]"):
                    flow = build_spiking_hybrid_flow(self.pipeline, hybrid_mapping)
                spiking_trainer = BasicTrainer(
                    flow,
                    device,
                    DataLoaderFactory(self.pipeline.data_provider_factory),
                    None,
                )
                if attempt_cap is not None:
                    current_bs = spiking_trainer.test_batch_size
                    spiking_trainer.set_test_batch_size(min(current_bs, attempt_cap))
                max_samples = int(self.pipeline.config.get("max_simulation_samples", 0) or 0)
                if max_samples > 0:
                    with _phase(f"sim_test[max_samples={max_samples}]"):
                        acc = spiking_trainer.test_on_subsample(
                            max_samples=max_samples,
                            seed=int(self.pipeline.config.get("seed", 0)),
                        )
                else:
                    with _phase(f"sim_test[batches={sim_batches}]"):
                        acc = spiking_trainer.test(max_batches=sim_batches)
                spiking_trainer.close()
                self._soft_core_spiking_metric = float(acc)
                print(f"[SoftCoreMappingStep] Soft-core Spiking Simulation Test: {acc}")
                break
            except RuntimeError as e:
                oom_cls = getattr(torch.cuda, "OutOfMemoryError", ())
                is_oom = isinstance(e, oom_cls) or "out of memory" in str(e).lower()
                if spiking_trainer is not None:
                    try:
                        spiking_trainer.close()
                    except Exception:
                        pass
                    spiking_trainer = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not is_oom or attempt >= 1:
                    print(f"[SoftCoreMappingStep] Soft-core simulation failed (non-fatal): {e}")
                    break
                attempt_cap = int(self.pipeline.config.get("simulation_batch_size", 8))
                print(
                    f"[SoftCoreMappingStep] Soft-core sim OOM at default batch size; "
                    f"retrying once with batch size capped at {attempt_cap}."
                )
            except Exception as e:
                if spiking_trainer is not None:
                    try:
                        spiking_trainer.close()
                    except Exception:
                        pass
                print(f"[SoftCoreMappingStep] Soft-core simulation failed (non-fatal): {e}")
                break

    def _apply_ttfs_quantized_bias_shift(self, model, act_q: bool) -> None:
        if self.pipeline.config.get("spiking_mode", "lif") != "ttfs_quantized" or not act_q:
            return
        from mimarsinan.mapping.ttfs_bias import apply_ttfs_quantized_bias_shift

        apply_ttfs_quantized_bias_shift(model, self.pipeline.config["target_tq"])

    def bring_back_bias(self, fused_linear_layer):
        assert isinstance(fused_linear_layer, FusedLinear), 'Input layer must be an instance of LinearWithoutBias'

        weights = fused_linear_layer.linear.weight.data
        main_weights, bias = weights[:, :-1], weights[:, -1]

        out_features, in_features = main_weights.shape
        new_layer = nn.Linear(in_features, out_features)
        new_layer.weight.data = main_weights
        new_layer.bias.data = bias

        for src in (fused_linear_layer, getattr(fused_linear_layer, "linear", None)):
            if src is None:
                continue
            for buf_name, buf_val in src.named_buffers():
                if not hasattr(new_layer, buf_name):
                    new_layer.register_buffer(buf_name, buf_val.clone())

        return new_layer


class FusedLinear(nn.Module):
    """Linear layer with bias fused into an extra weight column."""

    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear = nn.Linear(input_features + 1, output_features, bias=False)
        self.weight = self.linear.weight
        self.bias = self.linear.bias

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape
        bias_feature = torch.ones((batch_size, seq_len, 1), device=x.device)
        x = torch.cat([x, bias_feature], dim=-1)
        output = self.linear(x)

        if len(output.shape) == 3 and output.shape[1] == 1:
            output = output.squeeze(1)
        return output