"""
Vector Estimator (Flow Matching) module for Supertonic TTS.
Estimates the vector field for denoising latents.

ONNX Inputs:
  - noisy_latent: [batch_size, 144, latent_length]
  - text_emb: [batch_size, 256, text_length]
  - style_ttl: [batch_size, 50, 256]
  - latent_mask: [batch_size, 1, latent_length]
  - text_mask: [batch_size, 1, text_length]
  - current_step: [batch_size]
  - total_step: [batch_size]

ONNX Outputs:
  - denoised_latent: [batch_size, 144, latent_length]

ONNX main_blocks structure (repeated 4 times):
  - Block i*6+0: convnext (4 layers)
  - Block i*6+1: linear (time conditioning)
  - Block i*6+2: convnext (1 layer)
  - Block i*6+3: attn + norm (text cross-attention with rotary PE)
  - Block i*6+4: convnext (1 layer)
  - Block i*6+5: attention + norm (style cross-attention)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .layers import ConvNextStack, LayerNorm1d


class LinearWithBias(nn.Module):
    """Linear layer wrapper matching ONNX naming."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TimeConditioningLinear(nn.Module):
    """
    Time conditioning layer.
    ONNX: main_blocks.{1,7,13,19}.linear.linear.bias
    Only has bias, weight is in onnx::MatMul.
    """

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.linear = LinearWithBias(time_dim, channels)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        scale = self.linear(time_emb).unsqueeze(2)
        return x + scale


class TextCrossAttention(nn.Module):
    """
    Text cross-attention with rotary position embeddings.
    ONNX: main_blocks.{3,9,15,21}.attn.*

    Has: W_key, W_value, W_query, out_fc, theta, increments
    Note: theta and increments are SHARED across all text attention blocks.
    Only block 3 (first) has its own theta/increments in ONNX, others share them.

    ONNX shapes:
    - W_query: (channels, text_dim) -> bias [text_dim]
    - W_key: (text_dim, text_dim) -> bias [text_dim]
    - W_value: (text_dim, text_dim) -> bias [text_dim]
    - out_fc: (text_dim, channels) -> bias [channels]
    """

    def __init__(
        self,
        channels: int = 512,
        text_dim: int = 256,
        n_heads: int = 4,
        shared_theta: Optional[nn.Parameter] = None,
        shared_increments: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.channels = channels
        self.text_dim = text_dim
        self.n_heads = n_heads
        self.head_dim = text_dim // n_heads  # Attention in text_dim space
        # ONNX uses 1/sqrt(text_dim) not 1/sqrt(head_dim)
        self.scale = text_dim ** -0.5

        # All Q/K/V project to text_dim, out_fc projects back to channels
        self.W_query = LinearWithBias(channels, text_dim)
        self.W_key = LinearWithBias(text_dim, text_dim)
        self.W_value = LinearWithBias(text_dim, text_dim)
        self.out_fc = LinearWithBias(text_dim, channels)

        # Rotary position embedding params - uses head_dim/2 for theta
        # If shared params provided, use them; otherwise create new ones
        if shared_theta is not None:
            self.theta = shared_theta
        else:
            self.theta = nn.Parameter(torch.zeros(1, 1, self.head_dim // 2))
            self._init_theta()

        if shared_increments is not None:
            self.increments = shared_increments
        else:
            self.increments = nn.Parameter(torch.zeros(1, 1000, 1))
            self._init_increments()

    def _init_theta(self, base: int = 10000, scale: int = 10):
        half_dim = self.head_dim // 2
        theta = 1.0 / (base ** (torch.arange(0, half_dim).float() / half_dim))
        self.theta.data = theta.view(1, 1, half_dim) * scale

    def _init_increments(self):
        self.increments.data = torch.arange(1000).float().view(1, 1000, 1)

    def _apply_rotary(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        _, _, T, D = x.shape
        half_dim = D // 2

        # ONNX normalizes positions by dividing by sequence length
        positions = self.increments[:, :seq_len, :]  # (1, T, 1)
        normalized_positions = positions / seq_len  # Normalize to [0, 1)
        angles = normalized_positions * self.theta  # (1, T, half_dim)
        angles = angles.unsqueeze(1)  # (1, 1, T, half_dim)

        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        cos, sin = torch.cos(angles), torch.sin(angles)
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot

    def forward(
        self,
        x: torch.Tensor,
        text_emb: torch.Tensor,
        latent_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, _, T_lat = x.shape
        T_txt = text_emb.shape[2]

        q = self.W_query(x.transpose(1, 2))  # (B, T_lat, text_dim)
        k = self.W_key(text_emb.transpose(1, 2))  # (B, T_txt, text_dim)
        v = self.W_value(text_emb.transpose(1, 2))  # (B, T_txt, text_dim)

        q = q.view(B, T_lat, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_txt, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_txt, self.n_heads, self.head_dim).transpose(1, 2)

        q = self._apply_rotary(q, T_lat)
        k = self._apply_rotary(k, T_txt)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if text_mask is not None:
            attn = attn.masked_fill(text_mask.unsqueeze(1) == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, n_heads, T_lat, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T_lat, self.text_dim)
        out = self.out_fc(out).transpose(1, 2)  # (B, channels, T_lat)

        out = x + out
        if latent_mask is not None:
            out = out * latent_mask
        return out


class StyleCrossAttention(nn.Module):
    """
    Style cross-attention (no rotary PE).
    ONNX: main_blocks.{5,11,17,23}.attention.*

    Has: W_key, W_value, W_query, out_fc

    IMPORTANT: ONNX uses a static learned key embedding for K, NOT the style input!
    The static embedding is stored in /Expand_output_0.
    - Q: Projects from x (latent)
    - K: Projects from STATIC LEARNED EMBEDDING
    - V: Projects from style input

    ONNX shapes (all biases are [256]):
    - W_query: (channels, style_dim) -> bias [style_dim]
    - W_key: (style_dim, style_dim) -> bias [style_dim]
    - W_value: (style_dim, style_dim) -> bias [style_dim]
    - out_fc: (style_dim, channels) -> bias [channels]
    """

    def __init__(
        self,
        channels: int = 512,
        style_dim: int = 256,
        n_heads: int = 2,  # ONNX uses 2 heads (split [128, 128]), not 4
        n_style_tokens: int = 50,
        shared_key_emb: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.channels = channels
        self.style_dim = style_dim
        self.n_heads = n_heads
        self.head_dim = style_dim // n_heads  # 256 / 2 = 128
        # ONNX uses 1/sqrt(style_dim) not 1/sqrt(head_dim)
        self.scale = style_dim ** -0.5

        # All Q/K/V project to style_dim, out_fc projects back to channels
        self.W_query = LinearWithBias(channels, style_dim)
        self.W_key = LinearWithBias(style_dim, style_dim)
        self.W_value = LinearWithBias(style_dim, style_dim)
        self.out_fc = LinearWithBias(style_dim, channels)

        # Static learned key embedding - shared across all style attention blocks
        # ONNX stores this as /Expand_output_0
        if shared_key_emb is not None:
            self.key_emb = shared_key_emb
        else:
            self.key_emb = nn.Parameter(torch.randn(1, n_style_tokens, style_dim) * 0.3)

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, _, T = x.shape
        N = style.shape[1]

        q = self.W_query(x.transpose(1, 2))  # (B, T, style_dim)
        # K uses static learned embedding, expanded to batch size
        key_emb = self.key_emb.expand(B, -1, -1)  # (B, N, style_dim)
        k = self.W_key(key_emb)  # (B, N, style_dim)
        v = self.W_value(style)  # (B, N, style_dim)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # ONNX applies tanh to K^T (TanhEmbedding pattern)
        k_t = k.transpose(-2, -1)  # (B, heads, head_dim, N)
        k_t = torch.tanh(k_t)
        attn = torch.matmul(q, k_t) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, n_heads, T, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T, self.style_dim)
        out = self.out_fc(out).transpose(1, 2)  # (B, channels, T)

        out = x + out
        if mask is not None:
            out = out * mask
        return out


class TextAttentionBlock(nn.Module):
    """Container for text cross-attention + norm to match ONNX naming."""

    def __init__(
        self,
        channels: int,
        text_dim: int,
        n_heads: int,
        shared_theta: Optional[nn.Parameter] = None,
        shared_increments: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.attn = TextCrossAttention(
            channels=channels,
            text_dim=text_dim,
            n_heads=n_heads,
            shared_theta=shared_theta,
            shared_increments=shared_increments,
        )
        self.norm = LayerNorm1d(channels)

    def forward(
        self,
        x: torch.Tensor,
        text_emb: torch.Tensor,
        latent_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.attn(x, text_emb, latent_mask, text_mask)
        x = self.norm(x)
        if latent_mask is not None:
            x = x * latent_mask
        return x


class StyleAttentionBlock(nn.Module):
    """Container for style cross-attention + norm to match ONNX naming."""

    def __init__(
        self,
        channels: int,
        style_dim: int,
        n_style_tokens: int = 50,
        shared_key_emb: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.attention = StyleCrossAttention(
            channels=channels,
            style_dim=style_dim,
            n_style_tokens=n_style_tokens,
            shared_key_emb=shared_key_emb,
        )
        self.norm = LayerNorm1d(channels)

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.attention(x, style, mask)
        x = self.norm(x)
        if mask is not None:
            x = x * mask
        return x


class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class TimeEncoder(nn.Module):
    """
    Sinusoidal time encoding with MLP.

    ONNX structure:
    - Sinusoidal: t * 1000 * freq_factors -> sin/cos -> concat
    - MLP: Linear -> Mish -> Linear
    """

    def __init__(self, time_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.time_dim = time_dim

        # MLP with Mish activation (ONNX uses Mish, not SiLU)
        self.mlp = nn.Sequential()
        self.mlp.add_module('0', LinearWithBias(time_dim, hidden_dim))
        self.mlp.add_module('1', Mish())
        self.mlp.add_module('2', LinearWithBias(hidden_dim, time_dim))

    def forward(self, current_step: torch.Tensor, total_step: torch.Tensor) -> torch.Tensor:
        t = current_step / total_step
        half_dim = self.time_dim // 2

        # ONNX: freq[i] = 10000^(-i/31) for i=0..31
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)

        # ONNX multiplies time by 1000 before applying frequencies
        emb = (t * 1000).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.mlp(emb)


class ProjectionIn(nn.Module):
    """Input projection - no bias in ONNX."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Conv1d(in_dim, out_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionOut(nn.Module):
    """Output projection - no bias in ONNX."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Conv1d(in_dim, out_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VectorField(nn.Module):
    """
    Complete Vector Field network matching ONNX structure.

    main_blocks is a ModuleList where:
    - Blocks 0,6,12,18: ConvNext stack (4 layers)
    - Blocks 1,7,13,19: TimeConditioningLinear
    - Blocks 2,8,14,20: ConvNext stack (1 layer)
    - Blocks 3,9,15,21: TextCrossAttention (named 'attn')
    - Blocks 4,10,16,22: ConvNext stack (1 layer)
    - Blocks 5,11,17,23: StyleCrossAttention (named 'attention')

    Each attention block also has a .norm sub-module.
    """

    def __init__(self, config: dict):
        super().__init__()
        vf_cfg = config["ttl"]["vector_field"]
        ttl_cfg = config["ttl"]

        latent_dim = ttl_cfg["latent_dim"]
        chunk_factor = ttl_cfg["chunk_compress_factor"]

        self.proj_in = ProjectionIn(latent_dim * chunk_factor, vf_cfg["proj_in"]["odim"])

        self.time_encoder = TimeEncoder(
            time_dim=vf_cfg["time_encoder"]["time_dim"],
            hidden_dim=vf_cfg["time_encoder"]["hdim"],
        )

        # Build main_blocks as flat ModuleList to match ONNX indexing
        self.main_blocks = nn.ModuleList()
        mb_cfg = vf_cfg["main_blocks"]
        n_blocks = mb_cfg["n_blocks"]  # 4

        # Shared theta/increments for text attention blocks
        # Only block 3 has named params in ONNX, blocks 9,15,21 share them
        shared_theta = None
        shared_increments = None

        # Shared key embedding for style attention blocks
        # ONNX stores this as /Expand_output_0 and all style attention blocks share it
        shared_style_key_emb = None

        for i in range(n_blocks):
            # Block i*6+0: ConvNext (4 layers, dilation 1,2,4,8)
            self.main_blocks.append(ConvNextStack(
                dim=mb_cfg["convnext_0"]["idim"],
                kernel_size=mb_cfg["convnext_0"]["ksz"],
                intermediate_dim=mb_cfg["convnext_0"]["intermediate_dim"],
                num_layers=mb_cfg["convnext_0"]["num_layers"],
                dilation_list=mb_cfg["convnext_0"]["dilation_lst"],
            ))

            # Block i*6+1: Time conditioning
            self.main_blocks.append(TimeConditioningLinear(
                channels=mb_cfg["time_cond_layer"]["idim"],
                time_dim=mb_cfg["time_cond_layer"]["time_dim"],
            ))

            # Block i*6+2: ConvNext (1 layer)
            self.main_blocks.append(ConvNextStack(
                dim=mb_cfg["convnext_1"]["idim"],
                kernel_size=mb_cfg["convnext_1"]["ksz"],
                intermediate_dim=mb_cfg["convnext_1"]["intermediate_dim"],
                num_layers=mb_cfg["convnext_1"]["num_layers"],
                dilation_list=mb_cfg["convnext_1"]["dilation_lst"],
            ))

            # Block i*6+3: Text attention ('attn') - container with attn and norm
            # First block (i=0) creates theta/increments; others share from block 3
            text_attn_block = TextAttentionBlock(
                channels=mb_cfg["text_cond_layer"]["idim"],
                text_dim=mb_cfg["text_cond_layer"]["text_dim"],
                n_heads=mb_cfg["text_cond_layer"]["n_heads"],
                shared_theta=shared_theta,
                shared_increments=shared_increments,
            )
            self.main_blocks.append(text_attn_block)

            # After first text attention block, capture params for sharing
            if i == 0:
                shared_theta = text_attn_block.attn.theta
                shared_increments = text_attn_block.attn.increments

            # Block i*6+4: ConvNext (1 layer)
            self.main_blocks.append(ConvNextStack(
                dim=mb_cfg["convnext_2"]["idim"],
                kernel_size=mb_cfg["convnext_2"]["ksz"],
                intermediate_dim=mb_cfg["convnext_2"]["intermediate_dim"],
                num_layers=mb_cfg["convnext_2"]["num_layers"],
                dilation_list=mb_cfg["convnext_2"]["dilation_lst"],
            ))

            # Block i*6+5: Style attention ('attention') - container with attention and norm
            style_attn_block = StyleAttentionBlock(
                channels=mb_cfg["style_cond_layer"]["idim"],
                style_dim=mb_cfg["style_cond_layer"]["style_dim"],
                n_style_tokens=mb_cfg["style_cond_layer"].get("n_style_tokens", 50),
                shared_key_emb=shared_style_key_emb,
            )
            self.main_blocks.append(style_attn_block)

            # After first style attention block, capture key_emb for sharing
            if i == 0:
                shared_style_key_emb = style_attn_block.attention.key_emb

        # Final ConvNext
        lc_cfg = vf_cfg["last_convnext"]
        self.last_convnext = ConvNextStack(
            dim=lc_cfg["idim"],
            kernel_size=lc_cfg["ksz"],
            intermediate_dim=lc_cfg["intermediate_dim"],
            num_layers=lc_cfg["num_layers"],
            dilation_list=lc_cfg["dilation_lst"],
        )

        self.proj_out = ProjectionOut(
            vf_cfg["proj_out"]["idim"],
            vf_cfg["proj_out"]["ldim"] * vf_cfg["proj_out"]["chunk_compress_factor"],
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        text_emb: torch.Tensor,
        style_ttl: torch.Tensor,
        latent_mask: torch.Tensor,
        text_mask: torch.Tensor,
        current_step: torch.Tensor,
        total_step: torch.Tensor,
    ) -> torch.Tensor:
        x = self.proj_in(noisy_latent)
        time_emb = self.time_encoder(current_step, total_step)

        # Process through main_blocks
        for i in range(4):  # 4 iterations
            base = i * 6

            # ConvNext_0
            x = self.main_blocks[base + 0](x, latent_mask)

            # Time conditioning
            x = self.main_blocks[base + 1](x, time_emb)

            # ConvNext_1
            x = self.main_blocks[base + 2](x, latent_mask)

            # Text attention (attn + norm) - TextAttentionBlock
            x = self.main_blocks[base + 3](x, text_emb, latent_mask, text_mask)

            # ConvNext_2
            x = self.main_blocks[base + 4](x, latent_mask)

            # Style attention (attention + norm) - StyleAttentionBlock
            x = self.main_blocks[base + 5](x, style_ttl, latent_mask)

        x = self.last_convnext(x, latent_mask)
        x = self.proj_out(x)

        if latent_mask is not None:
            x = x * latent_mask

        return x


class FullVectorEstimator(nn.Module):
    """Complete Vector Estimator matching ONNX vector_estimator.onnx.

    IMPORTANT: The ONNX model doesn't just output the vector field - it also applies
    an Euler integration step to produce the denoised latent:
        denoised = (noisy_latent + vector_field / total_step) * latent_mask
    """

    def __init__(self, config: dict):
        super().__init__()
        self.vector_field = VectorField(config)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        text_emb: torch.Tensor,
        style_ttl: torch.Tensor,
        latent_mask: torch.Tensor,
        text_mask: torch.Tensor,
        current_step: torch.Tensor,
        total_step: torch.Tensor,
    ) -> torch.Tensor:
        # Get vector field output
        vector_field_out = self.vector_field(
            noisy_latent, text_emb, style_ttl,
            latent_mask, text_mask,
            current_step, total_step
        )

        # ONNX applies Euler integration step
        # denoised = noisy_latent + vector_field / total_step
        scaled_vector = vector_field_out / total_step.view(-1, 1, 1)
        denoised = noisy_latent + scaled_vector

        # Apply mask
        denoised = denoised * latent_mask

        return denoised
