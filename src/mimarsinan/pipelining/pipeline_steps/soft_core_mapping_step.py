from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.ir_latency import IRLatency
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.models.layers import SavedTensorDecorator
from mimarsinan.models.layers import TransformedActivation

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow
from mimarsinan.common.diagnostics import phase_profiler

import numpy as np
import torch.nn as nn
import torch

import os

class SoftCoreMappingStep(PipelineStep):

    def __init__(self, pipeline):
        # Require fused_model (only produced by Normalization Fusion) so we never load an unfused model.
        requires = ["fused_model", "platform_constraints_resolved"]
        # Unified-only: this step produces the unified IRGraph (NeuralCore + ComputeOp).
        promises = ["ir_graph"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)
        self.trainer = None
        self._soft_core_spiking_metric = None
        self._ttfs_shift_applied = False

    def validate(self):
        # When the TTFS shift was baked into biases, the fused FP model's own
        # evaluation is no longer meaningful (the decorator chain still subtracts
        # the shift, so the effective compensation is doubled).  Report the
        # spiking-sim metric instead, which matches the deployment behavior.
        if self._ttfs_shift_applied and self._soft_core_spiking_metric is not None:
            return self._soft_core_spiking_metric
        if self.trainer is not None:
            return self.trainer.validate()
        return self.pipeline.get_target_metric()

    def pipeline_metric(self):
        """Pipeline metric for Soft Core Mapping.

        In ``ttfs_quantized`` mode with activation quantization, the fused
        floating-point model's biases are modified in-place by the TTFS shift
        compensation (see ``process()`` below).  That shift aligns the
        quantization staircase for the spiking simulation but makes
        ``trainer.test()`` on the FP model meaningless (the decorator chain
        already subtracts an equivalent shift, so applying another shift to the
        bias yields a doubly-shifted output).  In that specific mode only, we
        report the spiking-simulation metric which is the actual representation
        of what will be deployed.

        In all other modes the FP model is unmodified by this step and
        ``trainer.test()`` remains the correct metric.
        """
        if self._ttfs_shift_applied and self._soft_core_spiking_metric is not None:
            return self._soft_core_spiking_metric
        return super().pipeline_metric()

    def process(self):
        model = self.get_entry("fused_model")
        platform_constraints = self.get_entry("platform_constraints_resolved")

        cores = platform_constraints.get("cores", [])
        if not cores:
            raise ValueError("platform_constraints_resolved must contain a non-empty 'cores' list")
        resolved_max_axons = max(ct["max_axons"] for ct in cores)
        resolved_max_neurons = max(ct["max_neurons"] for ct in cores)
        resolved_allow_coalescing = bool(platform_constraints.get("allow_coalescing", False))
        # hardware_bias=True only when ALL core types declare has_bias=True.
        # If any core uses the legacy always-on axon row, conservative mode is required.
        if cores:
            resolved_hardware_bias = all(bool(ct.get("has_bias", True)) for ct in cores)
        else:
            resolved_hardware_bias = False

        # When hardware_bias=False, every biased core consumes an extra always-on
        # axon slot.  Reduce effective max_axons by 1 so that IRMapping correctly
        # detects wide cores that would overflow after the bias row is appended.
        effective_max_axons = resolved_max_axons
        if not resolved_hardware_bias:
            effective_max_axons = resolved_max_axons - 1

        for perceptron in model.get_perceptrons():
            if isinstance(perceptron.layer, FusedLinear):
                perceptron.layer = self.bring_back_bias(perceptron.layer)

        if self.pipeline.config.get("generate_visualizations", False):
          # Emit a mapper/hardware flowchart for debugging.
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

        from mimarsinan.mapping.per_source_scales import compute_per_source_scales
        with _phase("compute_per_source_scales"):
            compute_per_source_scales(model.get_mapper_repr())

        act_q = bool(self.pipeline.config.get("activation_quantization", False))

        # ttfs_quantized is only relevant with activation quantization; without it, deployment may diverge from training.
        # Plain ttfs uses continuous TTFS and does not require activation_quantization.
        spiking_mode = self.pipeline.config.get("spiking_mode", "rate")
        if spiking_mode == "ttfs_quantized" and not act_q:
            print(
                "[SoftCoreMappingStep] Warning: ttfs_quantized is on but activation_quantization is off; "
                "deployment accuracy may drop compared to training."
            )

        # TTFS shift compensation for ttfs_quantized only.
        #
        # The training staircase is floor((V + shift)*tq)/tq while the
        # ttfs_quantized simulation computes floor(V*tq)/tq — the shift
        # (= 0.5*act_scale/tq) from the QuantizeDecorator's nested
        # ShiftDecorator is missing.  Baking it into the effective bias
        # aligns the two staircases exactly.
        #
        # IDEMPOTENCY: applying the shift is *not* idempotent — each call
        # overwrites ``perceptron.layer.bias.data`` with ``(old + s) * act_scale``
        # (see ``apply_effective_bias_transform``).  Historically the model
        # was re-saved after every SCM run so a resumed or repeated SCM
        # accumulated the shift on every re-run, slowly degrading soft-core
        # simulation accuracy.  Mark each perceptron after the first shift
        # and skip it on subsequent invocations.
        #
        # For ttfs (continuous) this is NOT applied: the simulation uses
        # bare relu(V)/threshold without an upper clamp, so the extra
        # shift would push some outputs above 1.0 and overflow into
        # downstream layers.  The ttfs_quantized formula naturally clamps
        # via k_fire.clamp(0, S-1).
        if self.pipeline.config.get("spiking_mode", "rate") == "ttfs_quantized" and act_q:
            from mimarsinan.tuning.shift_calculation import calculate_activation_shift
            from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
            tq = self.pipeline.config["target_tq"]
            for perceptron in model.get_perceptrons():
                # Encoding-layer Perceptrons run host-side as ComputeOps
                # wrapping the full ``Perceptron.forward`` (with its
                # ``TransformedActivation`` decorator chain, whose
                # ``QuantizeDecorator`` already applies the training
                # ``+ shift``). Adding a bias shift here would double-shift
                # them; it must only be applied to chip-side NeuralCore
                # Perceptrons that use the plain TTFS formula
                # ``floor(V*tq)/tq``.
                if getattr(perceptron, "is_encoding_layer", False):
                    continue
                if getattr(perceptron, "_ttfs_shift_baked_into_bias", False):
                    # Shift already baked in (persisted via a prior SCM save);
                    # a second application would double-shift and degrade
                    # soft-core sim accuracy.
                    continue
                shift = calculate_activation_shift(tq, perceptron.activation_scale)
                bias_shift = shift / perceptron.activation_scale
                PerceptronTransformer().apply_effective_bias_transform(
                    perceptron, lambda b, s=bias_shift: b + s)
                perceptron._ttfs_shift_baked_into_bias = True
            self._ttfs_shift_applied = True

        bits = self.pipeline.config['weight_bits']
        q_max = (2 ** (bits - 1)) - 1
        
        # Use the new IRMapping which supports both neural cores and compute ops
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

        # Apply chip quantization to NeuralCores: scale weights into [q_min, q_max],
        # round to integers, and set parameter_scale = 1.0. When weight_quantization is False,
        # skip scale/round so the flow uses the same float weights as the model and threshold=1.0
        # for exact equivalence (relu(W@x)/1).
        #
        # For bank-backed cores quantization is applied once per WeightBank.
        wt_q = bool(self.pipeline.config.get("weight_quantization", False))
        from mimarsinan.mapping.ir import WeightBank
        # After quantization, weights are integers in [q_min, q_max]; store in the
        # smallest integer dtype that fits so the IR artifact (and in-memory
        # footprint) is proportional to the quantization precision, not float64.
        # SpikingUnifiedCoreFlow upcasts to torch.float64 per-forward, so the
        # precision of spiking simulation is unchanged.
        q_dtype = np.int8 if bits <= 8 else np.int16
        with _phase("weight_quantization"):
            if not wt_q:
                for bank in ir_graph.weight_banks.values():
                    bank.parameter_scale = torch.tensor(1.0)
                for node in ir_graph.nodes:
                    if isinstance(node, NeuralCore):
                        node.threshold = 1.0
                        node.parameter_scale = torch.tensor(1.0)
            else:
                q_min = -(2 ** (bits - 1))
                eps = 1e-12
                quantized_banks: set[int] = set()
                bank_scale_used: dict[int, float] = {}
                for bank_id, bank in ir_graph.weight_banks.items():
                    ps = float(
                        bank.parameter_scale.item()
                        if hasattr(bank.parameter_scale, "item")
                        else bank.parameter_scale
                    )
                    if abs(ps) > eps:
                        scale = ps
                    else:
                        w_max = float(np.max(np.abs(bank.core_matrix)))
                        w_max = max(w_max, eps)
                        scale = q_max / w_max
                    W_q = np.clip(np.round(bank.core_matrix * scale), q_min, q_max).astype(q_dtype)
                    bank.core_matrix = W_q
                    bank.parameter_scale = torch.tensor(1.0)
                    quantized_banks.add(bank_id)
                    bank_scale_used[bank_id] = scale

                for node in ir_graph.nodes:
                    if isinstance(node, NeuralCore):
                        if node.has_weight_bank():
                            if node.weight_bank_id in quantized_banks:
                                scale_used = bank_scale_used[node.weight_bank_id]
                                node.threshold = scale_used
                                node.parameter_scale = torch.tensor(1.0)
                                if node.hardware_bias is not None:
                                    node.hardware_bias = np.round(
                                        node.hardware_bias * scale_used
                                    ).astype(q_dtype)
                        else:
                            ps = float(
                                node.parameter_scale.item()
                                if hasattr(node.parameter_scale, "item")
                                else node.parameter_scale
                            )
                            if abs(ps) > eps:
                                scale = ps
                            else:
                                w_max = float(np.max(np.abs(node.core_matrix)))
                                w_max = max(w_max, eps)
                                scale = q_max / w_max
                            W_q = np.clip(np.round(node.core_matrix * scale), q_min, q_max).astype(q_dtype)
                            node.core_matrix = W_q
                            node.threshold = scale
                            node.parameter_scale = torch.tensor(1.0)
                            # Scale hardware_bias by the same factor as the weights so that
                            # act(W_q @ inp + b_hw) / threshold = act(W_eff @ inp + b_eff).
                            # Without this, b_eff is effectively divided by `scale` (≈127 for 8-bit).
                            if node.hardware_bias is not None:
                                node.hardware_bias = np.round(
                                    node.hardware_bias * scale
                                ).astype(q_dtype)

        # Calculate latencies for all neural cores in the IR graph
        with _phase("ir_latency"):
            max_latency = IRLatency(ir_graph).calculate()
        print(f"[SoftCoreMappingStep] IR Graph max latency: {max_latency}")

        # Compact the IR graph by removing zeroed rows/columns when pruning was applied.
        # Use model pruning masks when available so compaction is driven by maps, not parameter values.
        if self.pipeline.config.get("pruning", False):
            from mimarsinan.mapping.ir_pruning import prune_ir_graph, get_initial_pruning_masks_from_model
            # Diagnostic: confirm pruning buffers on model before mask extraction
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
            # Optional diagnostic: confirm pruning provenance when tiled
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
            # ``store_pre_pruning_heatmap`` controls whether we hold a pre-compaction
            # copy of each NeuralCore's weight matrix on the node. This powers the
            # GUI monitor's "before/after pruning" views (IR Graph + Hardware tabs)
            # and is *independent* from ``generate_visualizations`` (which only
            # controls graphviz DOT/SVG dumps on disk). Default True so the
            # monitor works out of the box; memory-constrained runs can opt out.
            #
            # Memory guard: estimate the total heatmap footprint before
            # materialising it.  For large ViT-scale models this is tens of
            # GB (2364 × 768×3072 × 4 bytes = 22 GB for cifar_vit) — silently
            # overwhelms machines and bloats the IR pickle to 27 GB+.  When
            # the estimate exceeds a budget, drop heatmaps and warn.
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
          # Write IRGraph visualizations.
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
        
        # Log summary
        compute_ops = ir_graph.get_compute_ops()
        neural_cores = ir_graph.get_neural_cores()
        print(f"[SoftCoreMappingStep] IR Graph: {len(neural_cores)} neural cores, {len(compute_ops)} compute ops")
        if compute_ops:
            print(f"[SoftCoreMappingStep] Model contains {len(compute_ops)} non-neural operations:")
            for op in compute_ops:
                print(f"  - {op.name}: {op.op_type}")

        # Always report a *soft-core* spiking simulation result at this stage (pre-tuning),
        # so it's easy to compare before/after CoreFlow Tuning. Works for both neural-only
        # graphs and graphs containing ComputeOps (sync barriers handled in SpikingUnifiedCoreFlow).
        #
        # Memory hygiene: the fused FP model was moved to GPU for BasicTrainer but the sim
        # itself consumes only the IR graph; evicting the FP model frees headroom and
        # defragments the CUDA allocator before the sim's own parameter upload.
        device = self.pipeline.config["device"]
        sim_batches = self.pipeline.config.get("simulation_batch_count", None)

        try:
            model.to("cpu")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        spiking_trainer = None
        # Use the data provider's default test batch size — matches HardCoreMappingStep.
        # The SCM and HCM forwards are numerically equivalent on the same IR graph,
        # but per-operator cuBLAS kernel selection is batch-shape-dependent and can
        # produce ~1e-7 float drift.  The ``ceil(S*(1 - V/θ))`` TTFS step amplifies
        # that drift into k_fire flips, which compound over ~100 NeuralCore layers
        # into argmax flips and a 4-5 pp accuracy gap between the two reports.
        # Running sim_test at the same batch size as HCM keeps the two metrics aligned.
        #
        # On OOM, halve the batch size and retry once.
        attempt_cap = None  # None => keep provider default
        for attempt in range(2):
            try:
                cap_tag = "default" if attempt_cap is None else str(attempt_cap)
                with _phase(f"sim_construct[cap={cap_tag}]"):
                    flow = SpikingUnifiedCoreFlow(
                        self.pipeline.config["input_shape"],
                        ir_graph,
                        int(self.pipeline.config["simulation_steps"]),
                        None,
                        self.pipeline.config["firing_mode"],
                        self.pipeline.config["spike_generation_mode"],
                        self.pipeline.config["thresholding_mode"],
                        spiking_mode=self.pipeline.config.get("spiking_mode", "rate"),
                    )
                with _phase("sim_to_device"):
                    flow = flow.to(device)
                spiking_trainer = BasicTrainer(
                    flow,
                    device,
                    DataLoaderFactory(self.pipeline.data_provider_factory),
                    None,
                )
                if attempt_cap is not None:
                    current_bs = spiking_trainer.test_batch_size
                    spiking_trainer.set_test_batch_size(min(current_bs, attempt_cap))
                # Honour ``max_simulation_samples`` the same way the C++
                # ``SimulationRunner`` does: seeded numpy.choice
                # subsampling of the full test set.  This keeps the SCM
                # verification metric comparable to both the HCM
                # verification and the downstream chip-sim pass.
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
                print(f"[SoftCoreMappingStep] Soft-core (Unified IR) Spiking Simulation Test: {acc}")
                break
            except RuntimeError as e:
                # torch.cuda.OutOfMemoryError subclasses RuntimeError in recent
                # PyTorch; older versions raise a plain RuntimeError with
                # "out of memory" in the message. Detect by message to cover both.
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
                    print(f"[SoftCoreMappingStep] Soft-core (Unified IR) simulation failed (non-fatal): {e}")
                    break
                # OOM fallback cap (default 8).  Note: running at a different
                # batch size than HCM will produce a slightly different reported
                # accuracy due to cuBLAS kernel-selection drift, but this only
                # kicks in when the default batch would OOM.
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
                print(f"[SoftCoreMappingStep] Soft-core (Unified IR) simulation failed (non-fatal): {e}")
                break

    def _calculate_input_activation_scales(self, model, validator, rate):
        for perceptron in model.get_perceptrons():
            if not isinstance(perceptron.input_activation, TransformedActivation):
                perceptron.input_activation = TransformedActivation(perceptron.input_activation, [])
                
            # sample_to_cpu prevents decorators from pinning every perceptron's
            # input tensor in VRAM during the forward pass; the downstream
            # histogram-based quantile is unbiased under subsampling.
            perceptron.input_activation.decorate(SavedTensorDecorator(sample_to_cpu=True))

        validator.validate()

        max_target_scale = 0.0
        for perceptron in model.get_perceptrons():
            saved_tensor_dec = perceptron.input_activation.pop_decorator()
            if saved_tensor_dec.latest_input is None:
                raise RuntimeError(
                    "Failed to capture input activations for input scaling. "
                    f"Perceptron '{getattr(perceptron, 'name', '<unnamed>')}' did not record any inputs. "
                    "This typically happens when a Mapper's forward bypasses `perceptron.input_activation`."
                )
            # The saved tensor may live on CPU (sample_to_cpu decorator) or on
            # the pipeline device (legacy path). Keep histogram math on the
            # tensor's own device so the two paths share one code branch.
            x = saved_tensor_dec.latest_input
            in_min = x.min()
            in_max = x.max()

            bins = 1000
            activation_hist = torch.histc(x.flatten(), bins=bins, min=in_min.item(), max=in_max.item())
            bin_edges = torch.linspace(in_min.item(), in_max.item(), steps=bins+1, device=activation_hist.device)

            activation_hist *= bin_edges[1:]
            activation_hist[activation_hist < 0] = 0
            hist_sum = activation_hist.sum()
            cumulative_hist = activation_hist.cumsum(0)
            cumulative_hist /= hist_sum

            clip_rate = 0.999

            # # find the index of the bin which first exceeds the rate
            index = (cumulative_hist > clip_rate).flatten().nonzero()[0]
            clipped_act_scale = bin_edges[index].item()

            target_act_scale = (in_max * (1.0 - rate) + rate * clipped_act_scale) 

            perceptron.set_input_activation_scale(target_act_scale)
            max_target_scale = max(max_target_scale, target_act_scale)

    def bring_back_bias(self, fused_linear_layer):
        assert isinstance(fused_linear_layer, FusedLinear), 'Input layer must be an instance of LinearWithoutBias'
        
        # Get the weights from the existing layer
        weights = fused_linear_layer.linear.weight.data
        
        # Split the weights back into the main weights and the bias
        main_weights, bias = weights[:, :-1], weights[:, -1]

        # Create a new layer with the main weights and bias
        out_features, in_features = main_weights.shape
        new_layer = nn.Linear(in_features, out_features)
        new_layer.weight.data = main_weights
        new_layer.bias.data = bias

        # Preserve pruning buffers (e.g. prune_row_mask, prune_col_mask) so IR compaction can use model masks
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