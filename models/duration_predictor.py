"""
Duration Predictor module for Supertonic TTS.
Predicts the duration of speech given text and style.

ONNX Inputs:
  - text_ids: [batch_size, text_length]
  - style_dp: [batch_size, 8, 16]
  - text_mask: [batch_size, 1, text_length]

ONNX Outputs:
  - duration: [batch_size] (in seconds)
"""

import torch
import torch.nn as nn
from typing import Optional

from .layers import (
    ConvNextStack,
    AttentionEncoder,
    LayerNorm1d,
)


class SentenceEncoder(nn.Module):
    """
    Sentence encoder for duration prediction.
    Encodes text into a sentence-level representation.

    Structure from ONNX:
    - text_embedder.char_embedder: Embedding(163, 64)
    - sentence_token: learnable token (1, 64, 1)
    - convnext: 6 ConvNext blocks
    - attn_encoder: 2-layer transformer
    - proj_out: projection layer
    """

    def __init__(
        self,
        vocab_size: int = 163,
        char_emb_dim: int = 64,
        convnext_layers: int = 6,
        convnext_kernel: int = 5,
        convnext_intermediate: int = 256,
        attn_hidden: int = 64,
        attn_filter: int = 256,
        attn_heads: int = 2,
        attn_layers: int = 2,
        out_dim: int = 64,
    ):
        super().__init__()

        # Character embedding
        self.text_embedder = TextEmbedderDP(vocab_size, char_emb_dim)

        # Learnable sentence token
        self.sentence_token = nn.Parameter(torch.randn(1, char_emb_dim, 1) * 0.01)

        # ConvNext stack
        self.convnext = ConvNextStack(
            dim=char_emb_dim,
            kernel_size=convnext_kernel,
            intermediate_dim=convnext_intermediate,
            num_layers=convnext_layers,
            dilation_list=[1] * convnext_layers,
        )

        # Attention encoder
        self.attn_encoder = AttentionEncoder(
            hidden_channels=attn_hidden,
            filter_channels=attn_filter,
            n_heads=attn_heads,
            n_layers=attn_layers,
            p_dropout=0.0,
        )

        # Output projection
        self.proj_out = ProjectionOutDP(char_emb_dim, out_dim)

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_ids: (B, T)
            text_mask: (B, 1, T)
        Returns:
            sentence_emb: (B, out_dim)
        """
        B = text_ids.shape[0]

        # Embed characters
        x = self.text_embedder(text_ids)  # (B, C, T)

        # ONNX masks embedding before concat
        x = x * text_mask

        # Prepend sentence token
        sent_token = self.sentence_token.expand(B, -1, -1)  # (B, C, 1)
        x = torch.cat([sent_token, x], dim=2)  # (B, C, T+1)

        # Extend mask
        mask_ext = torch.ones(B, 1, 1, device=text_mask.device, dtype=text_mask.dtype)
        mask = torch.cat([mask_ext, text_mask], dim=2)  # (B, 1, T+1)

        # ConvNext processing
        convnext_out = self.convnext(x, mask)
        convnext_masked = convnext_out * mask  # mask after convnext

        # Attention encoder
        attn_out = self.attn_encoder(convnext_masked, mask)
        attn_masked = attn_out * mask  # mask after attn

        # ONNX has skip connection: attn_masked + convnext_masked
        x = attn_masked + convnext_masked

        # Extract sentence representation (first position)
        first_token = x[:, :, :1]  # (B, C, 1) - keep dim

        # Project
        proj_out = self.proj_out(first_token)

        # ONNX applies mask to proj_out
        first_mask = mask[:, :, :1]
        proj_out = proj_out * first_mask

        # Squeeze to get (B, out_dim)
        return proj_out.squeeze(2)


class TextEmbedderDP(nn.Module):
    """Text embedder for duration predictor.

    ONNX path: sentence_encoder.text_embedder.char_embedder.weight
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        # Direct embedding - ONNX: text_embedder.char_embedder.weight
        self.char_embedder = nn.Embedding(vocab_size, embed_dim)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        x = self.char_embedder(text_ids)  # (B, T, C)
        return x.transpose(1, 2)  # (B, C, T)


class ProjectionOutDP(nn.Module):
    """Output projection for duration predictor - no bias in ONNX."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # ONNX: sentence_encoder.proj_out.net.weight (no bias)
        self.net = nn.Conv1d(in_dim, out_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DurationPredictor(nn.Module):
    """
    MLP predictor head for duration.

    Takes sentence embedding + style and predicts duration.
    """

    def __init__(
        self,
        sentence_dim: int = 64,
        n_style: int = 8,
        style_dim: int = 16,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()

        input_dim = sentence_dim + n_style * style_dim  # 64 + 8*16 = 192

        # Build MLP layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, 1))

        # Learnable activation parameter (PReLU-like)
        self.activation = nn.PReLU()

    def forward(
        self,
        sentence_emb: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sentence_emb: (B, sentence_dim)
            style: (B, n_style, style_dim)
        Returns:
            duration: (B,) predicted duration in seconds
        """
        # Flatten style
        B = sentence_emb.shape[0]
        style_flat = style.view(B, -1)  # (B, n_style * style_dim)

        # Concatenate
        x = torch.cat([sentence_emb, style_flat], dim=1)  # (B, input_dim)

        # MLP
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)

        x = self.layers[-1](x)  # (B, 1)

        # ONNX uses exp() for positive duration (not softplus)
        duration = torch.exp(x.squeeze(1))

        return duration


class FullDurationPredictor(nn.Module):
    """
    Complete Duration Predictor matching ONNX duration_predictor.onnx.

    Path prefix in ONNX: tts.dp.*
    """

    def __init__(self, config: dict):
        super().__init__()
        dp_cfg = config["dp"]

        # Sentence encoder
        se_cfg = dp_cfg["sentence_encoder"]
        self.sentence_encoder = SentenceEncoder(
            vocab_size=163,
            char_emb_dim=se_cfg["char_emb_dim"],
            convnext_layers=se_cfg["convnext"]["num_layers"],
            convnext_kernel=se_cfg["convnext"]["ksz"],
            convnext_intermediate=se_cfg["convnext"]["intermediate_dim"],
            attn_hidden=se_cfg["attn_encoder"]["hidden_channels"],
            attn_filter=se_cfg["attn_encoder"]["filter_channels"],
            attn_heads=se_cfg["attn_encoder"]["n_heads"],
            attn_layers=se_cfg["attn_encoder"]["n_layers"],
            out_dim=se_cfg["proj_out"]["odim"],
        )

        # Duration predictor head
        pred_cfg = dp_cfg["predictor"]
        self.predictor = DurationPredictor(
            sentence_dim=pred_cfg["sentence_dim"],
            n_style=pred_cfg["n_style"],
            style_dim=pred_cfg["style_dim"],
            hidden_dim=pred_cfg["hdim"],
            n_layers=pred_cfg["n_layer"],
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        style_dp: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_ids: (B, T) token indices
            style_dp: (B, 8, 16) style tokens for duration
            text_mask: (B, 1, T)
        Returns:
            duration: (B,) predicted duration in seconds
        """
        # Encode sentence
        sentence_emb = self.sentence_encoder(text_ids, text_mask)

        # Predict duration
        duration = self.predictor(sentence_emb, style_dp)

        return duration
