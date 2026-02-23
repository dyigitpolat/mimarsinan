"""
Vision Transformer (ViT) implemented as a PerceptronFlow.

All linear operations (patch embedding, Q/K/V projections, FFN layers,
classification head) are implemented as Perceptrons and mapped to
NeuralCores in the IR.

Non-linear / data-dependent operations (LayerNorm, multi-head attention,
positional embedding, CLS token, residual addition) are mapped to
ComputeOps and executed on the host / auxiliary compute unit.

Parameters are fully exposed for architecture search:
    patch_size, d_model, num_heads, num_layers, mlp_ratio
"""

from mimarsinan.mapping.mapping_utils import (
    InputMapper,
    ModuleMapper,
    Conv2DPerceptronMapper,
    EinopsRearrangeMapper,
    MergeLeadingDimsMapper,
    SplitLeadingDimMapper,
    Ensure2DMapper,
    PerceptronMapper,
    AddMapper,
    ModelRepresentation,
    LayerNormMapper,
    GELUMapper,
    ConstantPrependMapper,
    ConstantAddMapper,
    DropoutMapper,
    SelectMapper,
    MultiHeadAttentionComputeMapper,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow

import torch
import torch.nn as nn
import math


class VisionTransformer(PerceptronFlow):
    """
    Vision Transformer (ViT) for image classification.

    Architecture:
        Conv2d patch embedding → CLS token prepend → positional embedding →
        N × (LayerNorm → MHSA → residual → LayerNorm → FFN → residual) →
        final LayerNorm → CLS token select → classification head

    Parameters:
        device:       Target device.
        input_shape:  (C, H, W) of input images.
        num_classes:  Number of output classes.
        patch_size:   Size of each non-overlapping patch (must divide H and W).
        d_model:      Embedding / hidden dimension.
        num_heads:    Number of attention heads (must divide d_model).
        num_layers:   Number of transformer encoder blocks.
        mlp_ratio:    FFN hidden dimension = d_model * mlp_ratio.
        dropout:      Dropout probability (training only).
    """

    def __init__(
        self,
        device,
        input_shape,
        num_classes: int,
        patch_size: int = 4,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(device)

        assert len(input_shape) == 3, f"Expected (C, H, W), got {input_shape}"
        C, H, W = input_shape
        assert H % patch_size == 0 and W % patch_size == 0, (
            f"patch_size={patch_size} must divide image dims ({H}, {W})"
        )
        assert d_model % num_heads == 0, (
            f"d_model={d_model} must be divisible by num_heads={num_heads}"
        )

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.dropout_p = dropout

        num_patches = (H // patch_size) * (W // patch_size)
        self.num_patches = num_patches
        seq_len = num_patches + 1  # +1 for CLS token
        self.seq_len = seq_len
        ffn_hidden = d_model * mlp_ratio

        # ---- Input activation (decorated by the pipeline) ----
        self.input_activation = nn.Identity()

        # ---- Learnable tokens / positional embeddings ----
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # ---- Transformer block components (nn.ModuleLists) ----
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.q_proj = nn.ModuleList([
            Perceptron(d_model, d_model, normalization=nn.Identity(), name=f"q_{i}")
            for i in range(num_layers)
        ])
        self.k_proj = nn.ModuleList([
            Perceptron(d_model, d_model, normalization=nn.Identity(), name=f"k_{i}")
            for i in range(num_layers)
        ])
        self.v_proj = nn.ModuleList([
            Perceptron(d_model, d_model, normalization=nn.Identity(), name=f"v_{i}")
            for i in range(num_layers)
        ])
        self.out_proj = nn.ModuleList([
            Perceptron(d_model, d_model, normalization=nn.Identity(), name=f"out_{i}")
            for i in range(num_layers)
        ])

        self.ffn1 = nn.ModuleList([
            Perceptron(ffn_hidden, d_model, normalization=nn.Identity(), name=f"ffn1_{i}")
            for i in range(num_layers)
        ])
        self.ffn2 = nn.ModuleList([
            Perceptron(d_model, ffn_hidden, normalization=nn.Identity(), name=f"ffn2_{i}")
            for i in range(num_layers)
        ])

        # ---- Final norm + classifier ----
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = Perceptron(
            num_classes, d_model, normalization=nn.Identity(), name="classifier"
        )

        # ---- Build mapper graph (single source of truth) ----
        self._build_mapper_graph()

    def _build_mapper_graph(self):
        """Construct the mapper DAG used for both forward and IR mapping."""
        C, H, W = self.input_shape
        P = self.patch_size
        N = self.num_patches
        S = self.seq_len  # N + 1
        D = self.d_model

        # Input → input activation
        inp = InputMapper(self.input_shape)
        self._input_activation_mapper = ModuleMapper(inp, self.input_activation)

        # Patch embedding: Conv2d(C, D, kernel_size=P, stride=P)
        out = Conv2DPerceptronMapper(
            self._input_activation_mapper,
            in_channels=C,
            out_channels=D,
            kernel_size=P,
            stride=P,
            padding=0,
            use_batchnorm=True,
            name="patch_embed",
        )
        self.patch_embed_perceptron = out  # Expose for owned_perceptron_groups

        # Rearrange: (D, H/P, W/P) → (N, D)  [batch-safe with ...]
        out = EinopsRearrangeMapper(out, "... d h w -> ... (h w) d")

        # CLS token prepend: (N, D) → (S, D)
        out = ConstantPrependMapper(out, self.cls_token, name="cls_prepend")

        # Positional embedding: (S, D) → (S, D)
        out = ConstantAddMapper(out, self.pos_embed, name="pos_embed")

        # Dropout (training only)
        out = DropoutMapper(out, p=self.dropout_p, name="embed_dropout")

        # Transformer blocks
        for i in range(self.num_layers):
            # ---- Pre-attention LayerNorm ----
            ln1_out = LayerNormMapper(out, self.norm1[i], name=f"ln1_{i}")

            # ---- Q / K / V projections (applied per-token) ----
            q = MergeLeadingDimsMapper(ln1_out)
            q = PerceptronMapper(q, self.q_proj[i])
            q = SplitLeadingDimMapper(q, S)

            k = MergeLeadingDimsMapper(ln1_out)
            k = PerceptronMapper(k, self.k_proj[i])
            k = SplitLeadingDimMapper(k, S)

            v = MergeLeadingDimsMapper(ln1_out)
            v = PerceptronMapper(v, self.v_proj[i])
            v = SplitLeadingDimMapper(v, S)

            # ---- Multi-head self-attention (ComputeOp) ----
            attn_out = MultiHeadAttentionComputeMapper(
                [q, k, v],
                num_heads=self.num_heads,
                attn_dropout=self.dropout_p,
                name=f"mhsa_{i}",
            )

            # ---- Output projection ----
            proj = MergeLeadingDimsMapper(attn_out)
            proj = PerceptronMapper(proj, self.out_proj[i])
            proj = SplitLeadingDimMapper(proj, S)

            # ---- Residual connection ----
            out = AddMapper(out, proj)

            # ---- Pre-FFN LayerNorm ----
            ln2_out = LayerNormMapper(out, self.norm2[i], name=f"ln2_{i}")

            # ---- FFN ----
            ffn = MergeLeadingDimsMapper(ln2_out)
            ffn = PerceptronMapper(ffn, self.ffn1[i])  # d_model → ffn_hidden (ReLU)
            ffn = PerceptronMapper(ffn, self.ffn2[i])  # ffn_hidden → d_model (ReLU)
            ffn = SplitLeadingDimMapper(ffn, S)

            # ---- Residual connection ----
            out = AddMapper(out, ffn)

        # ---- Final LayerNorm ----
        out = LayerNormMapper(out, self.final_norm, name="final_ln")

        # ---- Select CLS token: (S, D) → (D,) ----
        out = SelectMapper(out, index=0, name="cls_select")

        # ---- Ensure 2D for mapping: (D,) → (1, D) ----
        out = Ensure2DMapper(out)

        # ---- Classification head ----
        out = PerceptronMapper(out, self.classifier)

        self._mapper_repr = ModelRepresentation(out)

    # ---- PerceptronFlow interface ----

    def get_input_activation(self):
        return self.input_activation

    def set_input_activation(self, activation):
        self.input_activation = activation
        self._input_activation_mapper.module = activation

    def get_perceptrons(self):
        return self._mapper_repr.get_perceptrons()

    def get_perceptron_groups(self):
        return self._mapper_repr.get_perceptron_groups()

    def get_mapper_repr(self):
        return self._mapper_repr

    def forward(self, x):
        return self._mapper_repr(x)

