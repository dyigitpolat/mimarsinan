# models/perceptron_mixer/ -- Perceptron-Based Architectures

Core building blocks for all neuromorphic-deployable model architectures.
Every architecture is built from `Perceptron` modules organized as a
`PerceptronFlow`.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `perceptron.py` | `Perceptron` | Fundamental building block: Linear + Normalization + Activation + Regularization |
| `perceptron_flow.py` | `PerceptronFlow` | Abstract base for models composed of `Perceptron`s; defines `get_perceptrons()`, `get_mapper_repr()` |
| `perceptron_mixer.py` | `PerceptronMixer` | MLP-Mixer architecture (patch embed + token/channel mixing) |
| `vision_transformer.py` | `VisionTransformer` | Vision Transformer (ViT) with CLS token, positional embeddings, MHSA |
| `simple_mlp.py` | `SimpleMLP` | Simple multi-layer perceptron |
| `skip_perceptron_mixer.py` | `SkipPerceptronMixer` | MLP-Mixer variant with skip connections (experimental) |
| `perceptron_mixer_old.py` | `PerceptronMixerOld` | Legacy MLP-Mixer implementation |

## Dependencies

- **Internal**: `models.layers` (`MaxValueScaler`, `LeakyGradReLU`), `mapping.mapping_utils` (all `Mapper` classes).
- **External**: `torch`, `einops`, `math`.

## Dependents

- `models.builders` imports architectures to construct models
- `models.simple_conv`, `models.vgg16` import `Perceptron` and `PerceptronFlow`
- `mapping.mapping_utils` imports `Perceptron` for weight extraction
- `model_training` imports `Perceptron` for transform training

## Exported API (\_\_init\_\_.py)

`Perceptron`, `PerceptronFlow`.

Architecture classes (`PerceptronMixer`, `VisionTransformer`, `SimpleMLP`) are
**not** re-exported at the package level because they import from
`mapping.mapping_utils`, which itself imports `Perceptron` — creating a circular
dependency. Import them directly from their modules, e.g.
`from mimarsinan.models.perceptron_mixer.perceptron_mixer import PerceptronMixer`.
