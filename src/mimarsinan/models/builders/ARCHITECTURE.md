# models/builders/ -- Model Builder Classes

Factory classes that construct `Supermodel` instances for the pipeline and
architecture search systems.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `perceptron_mixer_builder.py` | `PerceptronMixerBuilder` | Builds `Supermodel` with `PerceptronMixer` flow |
| `vit_builder.py` | `VitBuilder` | Builds `Supermodel` with `VisionTransformer` flow |
| `simple_mlp_builder.py` | `SimpleMLPBuilder` | Builds `Supermodel` with `SimpleMLP` flow |
| `simple_conv_builder.py` | `SimpleConvBuilder` | Builds `Supermodel` with `SimpleConvMapper` flow |
| `vgg16_builder.py` | `VGG16Builder` | Builds `Supermodel` with `VGG16Mapper` flow |

Each builder implements `build(configuration) -> Supermodel`.

## Dependencies

- **Internal**: `models.supermodel`, `models.preprocessing.input_cq`, `models.perceptron_mixer.*`, `models.simple_conv`, `models.vgg16`.
- **External**: `torch`.

## Dependents

- `pipelining.pipeline_steps` (architecture search, model configuration) imports all builders.

## Exported API (\_\_init\_\_.py)

`PerceptronMixerBuilder`, `SimpleMLPBuilder`, `SimpleConvBuilder`, `VGG16Builder`, `VitBuilder`.
