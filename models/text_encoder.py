"""
Text Encoder module for Supertonic TTS.
Reconstructed from ONNX model weights.

ONNX Inputs:
  - text_ids: [batch_size, text_length]
  - style_ttl: [batch_size, 50, 256]
  - text_mask: [batch_size, 1, text_length]

ONNX Outputs:
  - text_emb: [batch_size, 256, text_length]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .layers import (
    ConvNextStack,
    AttentionEncoder,
    LayerNorm1d,
)


class TextEmbedder(nn.Module):
    """Text embedding - matches ONNX naming exactly."""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        # ONNX path: tts.ttl.text_encoder.text_embedder.char_embedder.weight
        self.char_embedder = nn.Embedding(vocab_size, embed_dim)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        x = self.char_embedder(text_ids)  # (B, T, C)
        return x.transpose(1, 2)  # (B, C, T)


class LinearWithBias(nn.Module):
    """Linear layer matching ONNX naming: .linear.weight/.linear.bias"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TanhEmbedding(nn.Module):
    """Container for the static learned key embeddings (named 'tanh' to match ONNX)."""

    def __init__(self, n_heads: int, head_dim: int, n_style_tokens: int = 50):
        super().__init__()
        # ONNX stores as (n_heads, 1, head_dim, n_style_tokens) - already transposed for matmul
        # Named Tanh_output_0 to match ONNX initializer path
        self.register_buffer(
            'Tanh_output_0',
            torch.zeros(n_heads, 1, head_dim, n_style_tokens)
        )


class StyleCrossAttention(nn.Module):
    """
    Cross-attention for style conditioning.

    Based on ONNX structure:
    - W_query: projects text query (has bias)
    - W_value: projects style value (has bias)
    - out_fc: output projection (has bias)
    - tanh: static learned key embeddings (NOT derived from style input!)

    ONNX uses heads-first format: (n_heads, batch, seq, head_dim)
    and Split/Concat instead of view/transpose for multi-head reshape.
    """

    def __init__(self, dim: int, n_heads: int = 2, n_style_tokens: int = 50):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        # ONNX uses 1/sqrt(dim), not 1/sqrt(head_dim) for scaling
        self.scale = dim ** -0.5

        # Match ONNX naming exactly
        self.W_query = LinearWithBias(dim, dim)
        self.W_value = LinearWithBias(dim, dim)
        self.out_fc = LinearWithBias(dim, dim)

        # Static learned key embeddings - ONNX path: .../tanh/Tanh_output_0
        self.tanh = TanhEmbedding(n_heads, self.head_dim, n_style_tokens)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, C, T) text features
            key_value: (B, N, D) style tokens (used only for V, not K!)
            mask: (B, 1, T)
        """
        B, C, T = query.shape

        # Query projection: (B, C, T) -> (B, T, C)
        q = self.W_query(query.transpose(1, 2))  # (B, T, C)

        # Value projection
        v = self.W_value(key_value)  # (B, N, C)

        # ONNX-style multi-head reshape using split and stack
        # Split along channel dim into n_heads parts, then stack
        q_splits = q.split(self.head_dim, dim=-1)  # tuple of (B, T, head_dim)
        q = torch.stack(q_splits, dim=0)  # (n_heads, B, T, head_dim)

        v_splits = v.split(self.head_dim, dim=-1)  # tuple of (B, N, head_dim)
        v = torch.stack(v_splits, dim=0)  # (n_heads, B, N, head_dim)

        # K is static learned embedding: (n_heads, 1, head_dim, N) - already transposed
        k_t = self.tanh.Tanh_output_0  # (n_heads, 1, head_dim, N)

        # Attention: Q @ K^T (K is already transposed)
        # (n_heads, B, T, head_dim) @ (n_heads, 1, head_dim, N) -> (n_heads, B, T, N)
        scores = torch.matmul(q, k_t) * self.scale
        attn = F.softmax(scores, dim=-1)

        # Apply attention to values
        # (n_heads, B, T, N) @ (n_heads, B, N, head_dim) -> (n_heads, B, T, head_dim)
        out = torch.matmul(attn, v)

        # Reshape back: (n_heads, B, T, head_dim) -> (B, T, C)
        # Split heads, concat along channel dim
        out_splits = [out[i] for i in range(self.n_heads)]  # list of (B, T, head_dim)
        out = torch.cat(out_splits, dim=-1)  # (B, T, C)

        # Output projection
        out = self.out_fc(out)

        # Apply mask BEFORE residual (matches ONNX)
        if mask is not None:
            # mask is (B, 1, T), out is (B, T, C)
            out = out * mask.transpose(1, 2)  # transpose mask to (B, T, 1) for broadcast

        # Residual connection
        # Note: residual is added later in SpeechPromptedTextEncoder, not here
        # Return in (B, T, C) format for the parent to handle residual
        return out


class SpeechPromptedTextEncoder(nn.Module):
    """
    Style-conditioned text encoder.

    ONNX structure:
    - Transpose input to (B, T, C)
    - attention1: cross-attention with style -> masked output (B, T, C)
    - Add: attention1 + original transposed input
    - attention2: cross-attention with style -> masked output (B, T, C)
    - Add: attention2 + original transposed input (SAME residual!)
    - norm: LayerNorm on (B, T, C)
    - Transpose back to (B, C, T)
    - Mul with text_mask
    """

    def __init__(self, dim: int = 256, n_heads: int = 2):
        super().__init__()
        self.attention1 = StyleCrossAttention(dim, n_heads)
        self.attention2 = StyleCrossAttention(dim, n_heads)
        self.norm = LayerNorm1d(dim)

    def forward(
        self,
        text_emb: torch.Tensor,
        style: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # text_emb is (B, C, T), transpose to (B, T, C) for attention
        residual = text_emb.transpose(1, 2)  # (B, T, C)

        # Attention1: cross-attention with style
        # StyleCrossAttention now returns (B, T, C) with mask applied
        attn1_out = self.attention1(text_emb, style, text_mask)

        # Residual connection (both attentions use same original residual)
        x = attn1_out + residual  # (B, T, C)

        # Attention2: takes the result of attention1 + residual
        # Need to transpose back to (B, C, T) for StyleCrossAttention input
        attn2_out = self.attention2(x.transpose(1, 2), style, text_mask)

        # Residual connection (same original residual!)
        x = attn2_out + residual  # (B, T, C)

        # Norm on (B, T, C) format
        x = self.norm.norm(x)  # Apply layer norm directly, not with transpose

        # Transpose back to (B, C, T)
        x = x.transpose(1, 2)

        # Final mask
        if text_mask is not None:
            x = x * text_mask

        return x


class TextEncoder(nn.Module):
    """
    Text encoder core module.

    ONNX structure:
    - text_embedder.char_embedder: Embedding(163, 256)
    - convnext.convnext.{0-5}: ConvNext blocks
    - attn_encoder: AttentionEncoder
    - Skip connection: Add convnext output to attn_encoder output
    - proj_out: Just mask multiply (no learnable params)
    """

    def __init__(
        self,
        vocab_size: int = 163,
        char_emb_dim: int = 256,
        convnext_layers: int = 6,
        convnext_kernel: int = 5,
        convnext_intermediate: int = 1024,
        attn_hidden: int = 256,
        attn_filter: int = 1024,
        attn_heads: int = 4,
        attn_layers: int = 4,
        attn_dropout: float = 0.1,
    ):
        super().__init__()

        self.text_embedder = TextEmbedder(vocab_size, char_emb_dim)

        self.convnext = ConvNextStack(
            dim=char_emb_dim,
            kernel_size=convnext_kernel,
            intermediate_dim=convnext_intermediate,
            num_layers=convnext_layers,
            dilation_list=[1] * convnext_layers,
        )

        self.attn_encoder = AttentionEncoder(
            hidden_channels=attn_hidden,
            filter_channels=attn_filter,
            n_heads=attn_heads,
            n_layers=attn_layers,
            p_dropout=attn_dropout,
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.text_embedder(text_ids)
        convnext_out = self.convnext(x, text_mask)

        # Attention encoder with skip connection from convnext
        attn_out = self.attn_encoder(convnext_out, text_mask)
        if text_mask is not None:
            attn_out = attn_out * text_mask

        # Skip connection: add convnext output to attention output
        x = attn_out + convnext_out

        # Final mask (proj_out in ONNX)
        if text_mask is not None:
            x = x * text_mask
        return x


class FullTextEncoder(nn.Module):
    """
    Full text encoder matching ONNX text_encoder.onnx.

    ONNX prefix: tts.ttl.*
    """

    def __init__(self, config: dict):
        super().__init__()
        te_cfg = config["ttl"]["text_encoder"]
        spte_cfg = config["ttl"]["speech_prompted_text_encoder"]

        self.text_encoder = TextEncoder(
            vocab_size=163,
            char_emb_dim=te_cfg["text_embedder"]["char_emb_dim"],
            convnext_layers=te_cfg["convnext"]["num_layers"],
            convnext_kernel=te_cfg["convnext"]["ksz"],
            convnext_intermediate=te_cfg["convnext"]["intermediate_dim"],
            attn_hidden=te_cfg["attn_encoder"]["hidden_channels"],
            attn_filter=te_cfg["attn_encoder"]["filter_channels"],
            attn_heads=te_cfg["attn_encoder"]["n_heads"],
            attn_layers=te_cfg["attn_encoder"]["n_layers"],
            attn_dropout=te_cfg["attn_encoder"]["p_dropout"],
        )

        self.speech_prompted_text_encoder = SpeechPromptedTextEncoder(
            dim=spte_cfg["text_dim"],
            n_heads=spte_cfg["n_heads"],
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        style_ttl: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_emb = self.text_encoder(text_ids, text_mask)
        text_emb = self.speech_prompted_text_encoder(text_emb, style_ttl, text_mask)
        return text_emb
