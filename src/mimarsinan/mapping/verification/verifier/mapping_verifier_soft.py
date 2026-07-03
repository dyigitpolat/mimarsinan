from __future__ import annotations
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.verification.verifier.mapping_verifier_types import MappingVerificationResult
def verify_soft_core_mapping(
    model_repr,
    max_axons: int,
    max_neurons: int,
    *,
    allow_coalescing: bool = False,
    hardware_bias: bool = False,
) -> MappingVerificationResult:
    """Verify that a mapper-graph model representation can be laid out as soft cores.

    ``model_repr`` is a native ``ModelRepresentation`` or a torch-converted
    ``mapper_repr``; returns feasibility plus the collected softcores.
    """
    try:
        layout = LayoutIRMapping(
            max_axons=max_axons,
            max_neurons=max_neurons,
            allow_coalescing=allow_coalescing,
            hardware_bias=hardware_bias,
        )
        softcores = layout.collect_layout_softcores(model_repr)
    except Exception as exc:
        return MappingVerificationResult(
            feasible=False,
            softcores=[],
            num_neural_cores=0,
            max_input_size=0,
            max_output_size=0,
            total_area=0,
            host_side_segment_count=0,
            layout_preview=None,
            error=str(exc),
        )

    if not softcores:
        try:
            from mimarsinan.mapping.ir_mapping_class import IRMapping

            irm = IRMapping(
                q_max=1,
                firing_mode="Default",
                max_axons=max_axons,
                max_neurons=max_neurons,
                allow_coalescing=allow_coalescing,
                hardware_bias=hardware_bias,
            )
            ir_graph = irm.map(model_repr)
            n_nc = len(ir_graph.get_neural_cores())
            n_co = len(ir_graph.get_compute_ops())
            if n_nc > 0 or n_co > 0:
                return MappingVerificationResult(
                    feasible=True,
                    softcores=[],
                    num_neural_cores=n_nc,
                    max_input_size=0,
                    max_output_size=0,
                    total_area=0,
                    host_side_segment_count=n_co,
                    layout_preview=getattr(layout, "layout_preview", None),
                    error=None,
                )
        except Exception:
            pass

        return MappingVerificationResult(
            feasible=False,
            softcores=[],
            num_neural_cores=0,
            max_input_size=0,
            max_output_size=0,
            total_area=0,
            host_side_segment_count=getattr(layout, "host_side_segment_count", 0),
            layout_preview=getattr(layout, "layout_preview", None),
            error="No neural cores produced by mapping — model may have no perceptron layers.",
        )

    max_in = max(sc.input_count for sc in softcores)
    max_out = max(sc.output_count for sc in softcores)
    total_area = sum(sc.area for sc in softcores)

    return MappingVerificationResult(
        feasible=True,
        softcores=softcores,
        num_neural_cores=len(softcores),
        max_input_size=max_in,
        max_output_size=max_out,
        total_area=total_area,
        host_side_segment_count=getattr(layout, "host_side_segment_count", 0),
        layout_preview=getattr(layout, "layout_preview", None),
    )

