#!/usr/bin/env python3

"""
Generate a Graphviz DOT flowchart describing the mapper graph and estimated SoftCore hardware usage.

Usage:
  python examples/visualize_softcore_flowchart.py examples/mnist_vgg16.json ./generated/vgg16.dot

Optional:
  - set allow_axon_tiling in the JSON under platform_constraints (or pass it via CLI later if needed)
  - render with graphviz: `dot -Tpng ./generated/vgg16.dot -o ./generated/vgg16.png`
"""

import json
import os
import sys

import torch


def main():
    if len(sys.argv) < 3:
        print("Usage: python examples/visualize_softcore_flowchart.py <config.json> <out.dot>")
        raise SystemExit(2)

    config_path = sys.argv[1]
    out_dot = sys.argv[2]

    # Ensure imports work when running from repo root
    if "./src" not in sys.path:
        sys.path.append("./src")

    from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
    import mimarsinan.data_handling.data_providers  # noqa: F401

    from mimarsinan.models.builders import PerceptronMixerBuilder, SimpleConvBuilder, SimpleMLPBuilder, VGG16Builder
    from mimarsinan.visualization.softcore_flowchart import write_softcore_flowchart_dot

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    platform = cfg["platform_constraints"]
    deploy = cfg["deployment_parameters"]

    data_provider_name = cfg["data_provider_name"]
    data_provider_factory = BasicDataProviderFactory(data_provider_name, "./datasets")
    dp = data_provider_factory.create()

    input_shape = tuple(dp.get_input_shape())
    num_classes = int(dp.get_prediction_mode().num_classes)

    device = torch.device("cpu")

    # Build minimal pipeline config dict expected by builders
    pipeline_config = {}
    pipeline_config.update(platform)
    pipeline_config.update(deploy)
    pipeline_config["device"] = device
    pipeline_config["input_shape"] = input_shape
    pipeline_config["num_classes"] = num_classes

    max_axons = int(platform["max_axons"])
    max_neurons = int(platform["max_neurons"])
    allow_axon_tiling = bool(platform.get("allow_axon_tiling", False))

    model_type = deploy["model_type"]

    builders = {
        "mlp_mixer": PerceptronMixerBuilder(device, input_shape, num_classes, max_axons, max_neurons, pipeline_config),
        "simple_mlp": SimpleMLPBuilder(device, input_shape, num_classes, max_axons, max_neurons, pipeline_config),
        "simple_conv": SimpleConvBuilder(device, input_shape, num_classes, max_axons, max_neurons, pipeline_config),
        "vgg16": VGG16Builder(device, input_shape, num_classes, max_axons, max_neurons, pipeline_config),
    }

    builder = builders[model_type]
    model_config = deploy.get("model_config", {})

    # Build model (builders may enforce platform checks; visualization should still work)
    try:
        model = builder.build(model_config)
    except ValueError as e:
        # Fall back to direct construction for visualization when platform checks fail.
        if model_type == "vgg16":
            from mimarsinan.models.vgg16 import VGG16Mapper
            model = VGG16Mapper(device, input_shape, num_classes, max_axons=max_axons, max_neurons=max_neurons)
        else:
            raise
    model.eval()

    os.makedirs(os.path.dirname(out_dot) or ".", exist_ok=True)
    write_softcore_flowchart_dot(
        model.get_mapper_repr(),
        out_dot,
        input_shape=input_shape,
        max_axons=max_axons,
        max_neurons=max_neurons,
        allow_axon_tiling=allow_axon_tiling,
        device=device,
    )

    print(f"Wrote DOT flowchart: {out_dot}")
    print("Render (if graphviz is installed):")
    print(f"  dot -Tpng {out_dot} -o {out_dot[:-4]}.png")


if __name__ == "__main__":
    main()


