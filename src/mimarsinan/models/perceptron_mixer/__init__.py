"""Perceptron-based model architectures: Perceptron, PerceptronFlow, SimpleMLP.

Only ``Perceptron`` and ``PerceptronFlow`` are eagerly re-exported here.
``SimpleMLP`` (mapper-repr example) is in ``simple_mlp``; other architectures
use torch-based builders and torch_mapping.
"""

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow
