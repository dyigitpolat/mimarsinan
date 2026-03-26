# models/builders/ -- Model Builder Classes

Factory classes that construct `Supermodel` instances (category "native") or native `nn.Module` instances (category "torch") for the pipeline and architecture search. **Category** determines whether `TorchMappingStep` runs: `ModelRegistry.get_category(model_type) == "torch"` adds the step.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `simple_mlp_builder.py` | `SimpleMLPBuilder` | Builds `Supermodel` with `SimpleMLP` flow (category "native", only mapper-repr example) |
| `torch_mlp_mixer_builder.py` | `TorchMLPMixerBuilder` | Builds native `TorchMLPMixer` (registered as `mlp_mixer`, category "torch") |
| `torch_mlp_mixer_core_builder.py` | `TorchMLPMixerCoreBuilder` | Builds native `TorchMLPMixerCore` (registered as `mlp_mixer_core`, category "torch") |
| `torchvision_builder_utils.py` | `parse_image_input_shape`, `resize_conv_input_weights`, `adapt_conv_in_channels` | Shared helpers for adapting torchvision image-model stems to arbitrary input shapes |
| `torch_vgg16_builder.py` | `TorchVGG16Builder` | Builds `torchvision.vgg16_bn`, adapting input channels and trimming trailing pools for small inputs while preserving the stock ImageNet layout at `3x224x224` |
| `torch_vit_builder.py` | `TorchViTBuilder` | Builds a ViT-B-style `VisionTransformer`, choosing a square patch size near `16` for the current input while preserving `vit_b_16` behavior at `3x224x224` |
| `torch_squeezenet11_builder.py` | `TorchSqueezeNet11Builder` | Builds `torchvision.squeezenet1_1`, adapting input channels while preserving the stock ImageNet stem at `3x224x224` |
| `torch_custom_builder.py` | `TorchCustomBuilder` | Builds a native model from a user-provided factory |
| `torch_sequential_linear_builder.py` | `TorchSequentialLinearBuilder` | Builds native Sequential(Flatten, Linear, …) |
| `torch_sequential_conv_builder.py` | `TorchSequentialConvBuilder` | Builds native Sequential(Conv2d, ReLU, MaxPool2d, Flatten, Linear, …); two IR segments |

Each builder implements `build(configuration) -> nn.Module`. Builders self-register via `@ModelRegistry.register(id, label=..., category=...)` and implement `get_config_schema()`. Category "torch" builders return a plain `nn.Module`; `TorchMappingStep` converts them to `Supermodel` after pretraining.

## Dependencies

- **Internal**: `models.supermodel`, `models.preprocessing.input_cq`, `models.perceptron_mixer.*`, `models.torch_mlp_mixer`, `models.torch_mlp_mixer_core`.
- **External**: `torch`, `torchvision` (for torch_* builders).

## Dependents

- `pipelining.pipeline_steps` (architecture search, model configuration) imports all builders.

## Exported API (\_\_init\_\_.py)

`SimpleMLPBuilder`, `TorchMLPMixerBuilder`, `TorchMLPMixerCoreBuilder`, `TorchVGG16Builder`, `TorchViTBuilder`, `TorchSqueezeNet11Builder`, `TorchCustomBuilder`, `TorchSequentialLinearBuilder`, `TorchSequentialConvBuilder`.
