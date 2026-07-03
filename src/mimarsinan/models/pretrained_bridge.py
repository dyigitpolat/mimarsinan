"""Pretrained bridge: import stock pretrained CNN classifiers as deployable regions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _resize_head(model: nn.Module, num_classes: int) -> nn.Module:
    """Swap a torchvision classifier ``fc`` Linear to ``num_classes`` outputs, eval()."""
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, int(num_classes))
    return model.eval()


def load_pretrained_resnet18(
    num_classes: int,
    *,
    pretrained: bool = True,
) -> nn.Module:
    """Return a torchvision ResNet-18 with its ``fc`` head re-sized to ``num_classes``.

    ``pretrained=True`` loads stock ImageNet1K_V1 weights (needs network); only the ``fc``
    Linear is replaced. Raises ImportError verbatim if torchvision is not importable.
    """
    if int(num_classes) < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
    try:
        import torchvision.models as tvm
        from torchvision.models import ResNet18_Weights
    except ImportError as exc:  # pragma: no cover - dependency-availability gate
        raise ImportError(
            "load_pretrained_resnet18 requires torchvision. The pretrained bridge "
            f"could not import it: {exc}"
        ) from exc

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return _resize_head(tvm.resnet18(weights=weights), num_classes)


def load_pretrained_resnet50(
    num_classes: int,
    *,
    pretrained: bool = True,
) -> nn.Module:
    """Return a torchvision ResNet-50 with its ``fc`` head re-sized to ``num_classes``.

    Bottleneck-block sibling of :func:`load_pretrained_resnet18`; same head-resize and
    ImportError contract. Raises ImportError verbatim if torchvision is not importable.
    """
    if int(num_classes) < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
    try:
        import torchvision.models as tvm
        from torchvision.models import ResNet50_Weights
    except ImportError as exc:  # pragma: no cover - dependency-availability gate
        raise ImportError(
            "load_pretrained_resnet50 requires torchvision. The pretrained bridge "
            f"could not import it: {exc}"
        ) from exc

    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    return _resize_head(tvm.resnet50(weights=weights), num_classes)


@dataclass(frozen=True)
class DeployedEval:
    """Result of deploying a bridge model through the SNN pipeline and evaluating it.

    All fields are MEASURED on the deployed ``SpikingHybridCoreFlow`` sim, not the torch model.
    """

    accuracy: float
    num_samples: int
    num_classes: int
    simulation_length: int
    spiking_mode: str
    neural_segments: int
    hard_cores: int
    logits: torch.Tensor


def _install_lif_activations(flow, T: int) -> None:
    """Install deployable LIF activations (rate-coded, lossless-capable) on every perceptron."""
    from mimarsinan.models.nn.activations import LIFActivation

    for perceptron in flow.get_perceptrons():
        lif = LIFActivation(
            T=T, activation_scale=torch.tensor(1.0), thresholding_mode="<="
        )
        perceptron.base_activation = lif
        perceptron.activation = lif


def _mapping_fingerprint(hybrid) -> tuple[int, int]:
    """(neural_segments, hard_cores) of a packed hybrid mapping (the deployed structure)."""
    neural_segments = hard_cores = 0
    for stage in hybrid.stages:
        if stage.hard_core_mapping is None:
            continue
        neural_segments += 1
        hard_cores += len(stage.hard_core_mapping.soft_core_placements_per_hard_core)
    return neural_segments, hard_cores


def deploy_and_eval(
    model: nn.Module,
    input_shape,
    num_classes: int,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
    *,
    simulation_length: int = 4,
    spiking_mode: str = "lif",
    max_axons: int = 8192,
    max_neurons: int = 8192,
    device: str = "cpu",
) -> DeployedEval:
    """Deploy ``model`` through the real SNN pipeline and report DEPLOYED accuracy.

    Fast/subset bridge: keep ``model`` and ``eval_inputs`` small so the deployed spiking sim
    stays cheap. Only ``spiking_mode='lif'`` is wired; conversion/mapping errors propagate verbatim.
    """
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
        build_hybrid_hard_core_mapping,
    )
    from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
    from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

    eval_inputs = torch.as_tensor(eval_inputs)
    eval_targets = torch.as_tensor(eval_targets)
    if eval_inputs.shape[0] != eval_targets.shape[0]:
        raise ValueError(
            f"eval_inputs ({eval_inputs.shape[0]}) and eval_targets "
            f"({eval_targets.shape[0]}) must have the same number of samples."
        )
    if tuple(eval_inputs.shape[1:]) != tuple(input_shape):
        raise ValueError(
            f"eval_inputs sample shape {tuple(eval_inputs.shape[1:])} != "
            f"input_shape {tuple(input_shape)}."
        )
    if spiking_mode != "lif":
        raise ValueError(
            f"deploy_and_eval currently deploys spiking_mode='lif' only; got "
            f"{spiking_mode!r}. TTFS deploy is a calibration-gated follow-up."
        )

    T = int(simulation_length)
    flow = convert_torch_model(model.eval(), tuple(input_shape), int(num_classes), device=device)
    flow.eval()
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    _install_lif_activations(flow, T)
    repr_.assign_perceptron_indices()

    ir = IRMapping(
        q_max=127.0, firing_mode="Default",
        max_axons=int(max_axons), max_neurons=int(max_neurons),
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{
            "max_axons": int(max_axons),
            "max_neurons": int(max_neurons),
            "count": 4000,
        }],
        strategy=MappingStrategy.from_permissions(
            allow_neuron_splitting=False, allow_coalescing=False,
        ),
    )
    hcm = SpikingHybridCoreFlow(
        tuple(input_shape), hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode="Default", spike_mode="Uniform", thresholding_mode="<=",
        spiking_mode="lif", cycle_accurate_lif_forward=True,
    )

    with torch.no_grad():
        deployed = hcm(eval_inputs.float()).double() / float(T)
    predictions = deployed.argmax(dim=1)
    correct = int((predictions == eval_targets.to(predictions.device)).sum())
    num_samples = int(eval_targets.shape[0])
    neural_segments, hard_cores = _mapping_fingerprint(hybrid)

    return DeployedEval(
        accuracy=float(correct) / num_samples,
        num_samples=num_samples,
        num_classes=int(num_classes),
        simulation_length=T,
        spiking_mode=spiking_mode,
        neural_segments=neural_segments,
        hard_cores=hard_cores,
        logits=deployed,
    )
