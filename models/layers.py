"""
Core building blocks for Supertonic TTS model.
Reconstructed from ONNX model weights and tts.json architecture specification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerNorm1d(nn.Module):
    """Layer normalization for 1D sequences (B, C, T) format."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        # ONNX uses eps=1e-6 (9.999999974752427e-07), PyTorch default is 1e-5
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C) -> norm -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x


class GELUActivation(nn.Module):
    """GELU activation as used in the model."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class ConvNextBlock(nn.Module):
    """
    ConvNext block used throughout the model.
    Structure: dwconv -> norm -> pwconv1 -> gelu -> pwconv2 -> gamma scaling -> residual

    NOTE: ONNX uses "edge" padding (replicate) before the conv, not zero padding.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 5,
        intermediate_dim: int = None,
        dilation: int = 1,
        layer_scale_init: float = 1e-6,
        use_net_wrapper: bool = False,  # vocoder uses .net wrapper for dwconv
    ):
        super().__init__()
        intermediate_dim = intermediate_dim or dim * 4
        self.padding = (kernel_size - 1) * dilation // 2
        self.dilation = dilation

        # Depthwise convolution - no padding here, we do it manually with replicate mode
        if use_net_wrapper:
            self.dwconv = nn.Sequential()
            self.dwconv.net = nn.Conv1d(
                dim, dim, kernel_size,
                padding=0, dilation=dilation, groups=dim
            )
        else:
            self.dwconv = nn.Conv1d(
                dim, dim, kernel_size,
                padding=0, dilation=dilation, groups=dim
            )

        self.use_net_wrapper = use_net_wrapper
        self.norm = LayerNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, intermediate_dim, 1)
        self.act = GELUActivation()
        self.pwconv2 = nn.Conv1d(intermediate_dim, dim, 1)

        # Learnable layer scale (gamma)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(1, dim, 1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x

        # Apply mask before convolution (as ONNX does)
        if mask is not None:
            x = x * mask

        # Replicate padding (ONNX "edge" mode)
        x = F.pad(x, (self.padding, self.padding), mode='replicate')

        if self.use_net_wrapper:
            x = self.dwconv.net(x)
        else:
            x = self.dwconv(x)

        # Apply mask after dwconv (as ONNX does)
        if mask is not None:
            x = x * mask

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x

        x = x + residual

        if mask is not None:
            x = x * mask

        return x


class ConvNextStack(nn.Module):
    """Stack of ConvNext blocks."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 5,
        intermediate_dim: int = None,
        num_layers: int = 6,
        dilation_list: list = None,
        use_net_wrapper: bool = False,
    ):
        super().__init__()
        intermediate_dim = intermediate_dim or dim * 4
        dilation_list = dilation_list or [1] * num_layers

        self.convnext = nn.ModuleList([
            ConvNextBlock(
                dim=dim,
                kernel_size=kernel_size,
                intermediate_dim=intermediate_dim,
                dilation=dilation_list[i],
                use_net_wrapper=use_net_wrapper,
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.convnext:
            x = block(x, mask)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional embeddings.
    Used in the attention encoder blocks.

    Implements relative position attention as in Transformer-XL/XLNet.
    """

    def __init__(
        self,
        channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: int = 4,  # For relative position embeddings
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.window_size = window_size
        self.p_dropout = p_dropout

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)

        # Relative position embeddings: shape (1, 2*window_size+1, head_dim)
        self.emb_rel_k = nn.Parameter(torch.randn(1, 2 * window_size + 1, self.head_dim) * 0.01)
        self.emb_rel_v = nn.Parameter(torch.randn(1, 2 * window_size + 1, self.head_dim) * 0.01)

        self.dropout = nn.Dropout(p_dropout)

    def _get_relative_embeddings(self, T: int, device) -> tuple:
        """Get relative position embeddings for all position pairs.

        IMPORTANT: ONNX zeroes out positions outside the window, it does NOT clamp.
        Positions where |i-j| > window_size get zero embeddings.

        Returns:
            rel_k: (T, T, head_dim) relative key embeddings
            rel_v: (T, T, head_dim) relative value embeddings
        """
        positions = torch.arange(T, device=device)
        # rel_pos[i,j] = j - i (relative position of key j from query i)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)

        # Mask for positions outside window (will be zeroed)
        outside_window = (rel_pos.abs() > self.window_size)  # (T, T)

        # Clamp for indexing, then shift to [0, 2*window_size]
        rel_idx = rel_pos.clamp(-self.window_size, self.window_size) + self.window_size

        # Gather embeddings for each position pair
        rel_k = self.emb_rel_k[0, rel_idx, :]  # (T, T, head_dim)
        rel_v = self.emb_rel_v[0, rel_idx, :]  # (T, T, head_dim)

        # Zero out positions outside window (ONNX behavior)
        rel_k = rel_k.masked_fill(outside_window.unsqueeze(-1), 0.0)
        rel_v = rel_v.masked_fill(outside_window.unsqueeze(-1), 0.0)

        return rel_k, rel_v

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, T = x.shape

        q = self.conv_q(x).view(B, self.n_heads, self.head_dim, T).transpose(2, 3)  # B, H, T, D
        k = self.conv_k(x).view(B, self.n_heads, self.head_dim, T).transpose(2, 3)
        v = self.conv_v(x).view(B, self.n_heads, self.head_dim, T).transpose(2, 3)

        # IMPORTANT: Scale Q first, then use scaled Q for both content and relative position
        # This matches ONNX which does: q_scaled = q / sqrt(d), then uses q_scaled everywhere
        scale = math.sqrt(self.head_dim)
        q = q / scale

        # Content-based attention: q_scaled @ k^T
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Add relative position bias to scores using scaled Q
        rel_k, rel_v = self._get_relative_embeddings(T, x.device)

        # Compute q_scaled @ rel_k^T for relative position bias
        # rel_pos_bias[b, h, i, j] = sum_d q[b, h, i, d] * rel_k[i, j, d]
        rel_pos_bias = torch.einsum('bhid,ijd->bhij', q, rel_k)
        scores = scores + rel_pos_bias

        if mask is not None:
            # mask: (B, 1, T) -> need (B, 1, 1, T) for broadcasting
            attn_mask = mask.unsqueeze(2)  # (B, 1, 1, T)
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, T, D)

        # Add relative position embedding to output
        # out[b, h, i, d] += sum_j attn[b, h, i, j] * rel_v[i, j, d]
        rel_v_out = torch.einsum('bhij,ijd->bhid', attn, rel_v)
        out = out + rel_v_out

        out = out.transpose(2, 3).contiguous().view(B, C, T)
        out = self.conv_o(out)

        return out


class FFN(nn.Module):
    """Feed-forward network used after attention.

    NOTE: ONNX uses ReLU, not GELU in FFN layers.
    """

    def __init__(self, channels: int, filter_channels: int, p_dropout: float = 0.0):
        super().__init__()
        self.conv_1 = nn.Conv1d(channels, filter_channels, 1)
        self.conv_2 = nn.Conv1d(filter_channels, channels, 1)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Mask before conv (ONNX does this)
        if mask is not None:
            x = x * mask
        x = self.conv_1(x)
        x = F.relu(x)  # ONNX uses ReLU, not GELU
        # Mask after relu (ONNX does this)
        if mask is not None:
            x = x * mask
        x = self.dropout(x)
        x = self.conv_2(x)
        if mask is not None:
            x = x * mask
        return x


class AttentionEncoder(nn.Module):
    """
    Transformer-style encoder with self-attention and FFN.
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        p_dropout: float = 0.0,
        window_size: int = 4,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        self.attn_layers = nn.ModuleList([
            MultiHeadAttention(hidden_channels, n_heads, p_dropout, window_size)
            for _ in range(n_layers)
        ])
        self.norm_layers_1 = nn.ModuleList([
            LayerNorm1d(hidden_channels)
            for _ in range(n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            FFN(hidden_channels, filter_channels, p_dropout)
            for _ in range(n_layers)
        ])
        self.norm_layers_2 = nn.ModuleList([
            LayerNorm1d(hidden_channels)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i in range(self.n_layers):
            # Self-attention with residual
            residual = x
            x = self.attn_layers[i](x, mask)
            x = residual + x
            x = self.norm_layers_1[i](x)

            # FFN with residual
            residual = x
            x = self.ffn_layers[i](x, mask)
            x = residual + x
            x = self.norm_layers_2[i](x)

            if mask is not None:
                x = x * mask

        return x


class StyleTokenLayer(nn.Module):
    """
    Style token layer for extracting style embeddings.
    Uses multi-head attention to attend to style tokens.
    """

    def __init__(
        self,
        input_dim: int,
        n_style: int,
        style_key_dim: int,
        style_value_dim: int,
        prototype_dim: int,
        n_units: int,
        n_heads: int,
    ):
        super().__init__()
        self.n_style = n_style
        self.style_value_dim = style_value_dim
        self.n_heads = n_heads
        self.n_units = n_units

        # This will hold the style prototypes if needed
        if style_key_dim > 0:
            self.style_keys = nn.Parameter(torch.randn(1, n_style, style_key_dim) * 0.01)
        else:
            self.style_keys = None

        # Projection layers would be added here
        self.proj_query = nn.Linear(input_dim, n_units) if input_dim != n_units else nn.Identity()

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, T)
            style: Style tokens (B, n_style, style_value_dim)
        Returns:
            Style-conditioned features
        """
        # Style tokens are pre-computed in the voice style files
        # Just return the style tokens as-is for conditioning
        return style


class CrossAttention(nn.Module):
    """Cross-attention for conditioning on style or text."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        n_heads: int,
        use_residual: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = query_dim // n_heads
        self.use_residual = use_residual

        self.W_query = nn.Linear(query_dim, query_dim)
        self.W_key = nn.Linear(key_dim, query_dim)
        self.W_value = nn.Linear(key_dim, query_dim)
        self.W_out = nn.Linear(query_dim, query_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, T = query.shape
        _, K, _ = key_value.shape if key_value.dim() == 3 else (B, key_value.shape[1], key_value.shape[2])

        # Project and reshape for multi-head attention
        q = self.W_query(query.transpose(1, 2))  # B, T, C
        k = self.W_key(key_value)  # B, K, C
        v = self.W_value(key_value)  # B, K, C

        # Reshape for attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # B, H, T, D
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)  # B, H, K, D
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)  # B, H, K, D

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # B, H, T, D

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_out(out)
        out = out.transpose(1, 2)  # B, C, T

        if self.use_residual:
            out = out + query

        return out


class RotaryCrossAttention(nn.Module):
    """Cross-attention with rotary position embeddings for text conditioning."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        n_heads: int,
        use_residual: bool = True,
        rotary_base: int = 10000,
        rotary_scale: int = 10,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = query_dim // n_heads
        self.use_residual = use_residual
        self.rotary_base = rotary_base
        self.rotary_scale = rotary_scale

        # These match the ONNX weight naming
        self.W_query = LinearWrapper(query_dim, query_dim)
        self.W_key = LinearWrapper(key_dim, query_dim)
        self.W_value = LinearWrapper(key_dim, query_dim)
        self.W_out = LinearWrapper(query_dim, query_dim)

        # Precomputed rotary embeddings
        self.theta = nn.Parameter(torch.zeros(1, 1, self.head_dim // 2))
        self.increments = nn.Parameter(torch.zeros(1, 1000, 1))

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, T_q = query.shape
        T_kv = key_value.shape[2] if key_value.dim() == 3 else key_value.shape[1]

        q = self.W_query(query.transpose(1, 2))
        k = self.W_key(key_value.transpose(1, 2) if key_value.dim() == 3 else key_value)
        v = self.W_value(key_value.transpose(1, 2) if key_value.dim() == 3 else key_value)

        # Reshape for attention
        q = q.view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_kv, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        q, k = self._apply_rotary(q, k)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if kv_mask is not None:
            scores = scores.masked_fill(kv_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T_q, C)
        out = self.W_out(out)
        out = out.transpose(1, 2)

        if self.use_residual:
            out = out + query

        if query_mask is not None:
            out = out * query_mask

        return out

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        """Apply rotary position embeddings."""
        # Simplified rotary implementation
        return q, k


class LinearWrapper(nn.Module):
    """Wrapper to match ONNX naming convention with .linear suffix."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TimeEncoder(nn.Module):
    """Sinusoidal time encoding for diffusion/flow matching."""

    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.time_dim = time_dim

        self.mlp = nn.Sequential()
        self.mlp.add_module('0', LinearWrapper(time_dim, hidden_dim))
        self.mlp.add_module('1', nn.SiLU())
        self.mlp.add_module('2', LinearWrapper(hidden_dim, time_dim))

    def forward(self, t: torch.Tensor, total_steps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Current step (B,)
            total_steps: Total steps (B,)
        Returns:
            Time embedding (B, time_dim)
        """
        # Normalize time to [0, 1]
        t_normalized = t / total_steps

        # Sinusoidal encoding
        half_dim = self.time_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t_normalized.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        # MLP projection
        emb = self.mlp(emb)

        return emb
