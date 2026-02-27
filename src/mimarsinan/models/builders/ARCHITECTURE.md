# models/builders/ -- Model Builder Classes

Factory classes that construct `Supermodel` instances (or native `nn.Module`
instances for torch_* model types) for the pipeline and architecture search
systems.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `perceptron_mixer_builder.py` | `PerceptronMixerBuilder` | Builds `Supermodel` with `PerceptronMixer` flow |
| `vit_builder.py` | `VitBuilder` | Builds `Supermodel` with `VisionTransformer` flow |
| `simple_mlp_builder.py` | `SimpleMLPBuilder` | Builds `Supermodel` with `SimpleMLP` flow |
| `simple_conv_builder.py` | `SimpleConvBuilder` | Builds `Supermodel` with `SimpleConvMapper` flow |
| `vgg16_builder.py` | `VGG16Builder` | Builds `Supermodel` with `VGG16Mapper` flow |
| `torch_vgg16_builder.py` | `TorchVGG16Builder` | Builds native `torchvision.vgg16_bn` model |
| `torch_vit_builder.py` | `TorchViTBuilder` | Builds native `torchvision.vit_b_16` model |
| `torch_squeezenet11_builder.py` | `TorchSqueezeNet11Builder` | Builds native `torchvision.squeezenet1_1` model |
| `torch_custom_builder.py` | `TorchCustomBuilder` | Builds a native model from a user-provided factory |

Each builder implements `build(configuration) -> nn.Module`.  Native builders
produce plain `nn.Module` instances that are later converted to `Supermodel`
by `TorchMappingStep`.

## Dependencies

- **Internal**: `models.supermodel`, `models.preprocessing.input_cq`, `models.perceptron_mixer.*`, `models.simple_conv`, `models.vgg16`.
- **External**: `torch`, `torchvision` (for torch_* builders).

## Dependents

- `pipelining.pipeline_steps` (architecture search, model configuration) imports all builders.

## Exported API (\_\_init\_\_.py)

`PerceptronMixerBuilder`, `SimpleMLPBuilder`, `SimpleConvBuilder`, `VGG16Builder`, `VitBuilder`, `TorchVGG16Builder`, `TorchViTBuilder`, `TorchSqueezeNet11Builder`, `TorchCustomBuilder`.
