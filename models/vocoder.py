"""
Vocoder (Decoder) module for Supertonic TTS.
Converts latent representations to audio waveforms.

ONNX Inputs:
  - latent: [batch_size, 144, latent_length]

ONNX Outputs:
  - wav_tts: [batch_size, wav_length]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .layers import ConvNextStack, LayerNorm1d


class ConvNextBlockVocoder(nn.Module):
    """
    ConvNext block variant for vocoder.
    Uses .net wrapper for depthwise conv to match ONNX naming.
    Uses causal edge padding (left-only) to match ONNX.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        intermediate_dim: int = 2048,
        dilation: int = 1,
    ):
        super().__init__()
        # Causal padding: (kernel_size - 1) * dilation on left side only
        self.causal_pad = (kernel_size - 1) * dilation

        # Depthwise conv with .net wrapper - NO padding in conv, we do it manually
        self.dwconv = nn.Sequential()
        self.dwconv.net = nn.Conv1d(
            dim, dim, kernel_size,
            padding=0, dilation=dilation, groups=dim
        )

        self.norm = LayerNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, intermediate_dim, 1)
        self.pwconv2 = nn.Conv1d(intermediate_dim, dim, 1)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Apply causal edge padding (left only)
        x = F.pad(x, (self.causal_pad, 0), mode='replicate')
        x = self.dwconv.net(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        x = self.gamma * x

        return x + residual


class ConvNextStackVocoder(nn.Module):
    """Stack of ConvNext blocks for vocoder."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        intermediate_dim: int = 2048,
        num_layers: int = 10,
        dilation_list: list = None,
    ):
        super().__init__()
        dilation_list = dilation_list or [1] * num_layers

        self.blocks = nn.ModuleList([
            ConvNextBlockVocoder(
                dim=dim,
                kernel_size=kernel_size,
                intermediate_dim=intermediate_dim,
                dilation=dilation_list[i],
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class FinalNorm(nn.Module):
    """Final normalization layer. ONNX path: decoder.final_norm.norm.*"""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class Layer1Wrapper(nn.Module):
    """Wrapper for layer1 to match ONNX naming: layer1.net.weight/bias
    Uses causal edge padding (left only) to match ONNX."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int):
        super().__init__()
        # Causal padding: (kernel_size - 1) on left side only
        self.causal_pad = kernel_size - 1
        self.net = nn.Conv1d(in_dim, out_dim, kernel_size, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply causal edge padding (left only)
        x = F.pad(x, (self.causal_pad, 0), mode='replicate')
        return self.net(x)


class DecoderHead(nn.Module):
    """
    Decoder head that converts features to waveform.
    Uses overlap-add for sample generation.

    ONNX structure:
    - layer1.net.weight/bias (Conv1d, kernel_size=3)
    - act.weight (PReLU with per-channel weights)
    - layer2.weight (Conv1d, kernel_size=1, no bias)
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 2048,
        out_dim: int = 512,  # chunk_size
        kernel_size: int = 3,
    ):
        super().__init__()

        # layer1 has .net wrapper (uses provided kernel_size)
        self.layer1 = Layer1Wrapper(in_dim, hidden_dim, kernel_size)
        # PReLU activation with single shared slope (not per-channel)
        self.act = nn.PReLU(num_parameters=1)
        # layer2 is direct Conv1d with kernel_size=1 and no bias (per ONNX)
        self.layer2 = nn.Conv1d(hidden_dim, out_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) decoder features
        Returns:
            samples: (B, out_dim, L)
        """
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        return x


class Decoder(nn.Module):
    """
    Vocoder decoder network.

    Structure from ONNX:
    - proj_in: expand latent dim (no bias)
    - convnext: 10 ConvNext blocks
    - final_norm: BatchNorm1d
    - head: output conv layers
    """

    def __init__(self, config: dict):
        super().__init__()
        ae_cfg = config["ae"]
        dec_cfg = ae_cfg["decoder"]

        self.chunk_size = ae_cfg["base_chunk_size"]

        # Input projection (latent dim expansion) - has bias (anonymous in ONNX)
        # ONNX naming: decoder.embed.net.weight/bias
        in_dim = ae_cfg["ldim"]  # 24
        hidden_dim = dec_cfg["hdim"]  # 512

        # Causal padding for embed: (kernel_size - 1) on left side only
        self.embed_causal_pad = dec_cfg["ksz_init"] - 1  # 6 for kernel_size=7

        # Create embed wrapper to match ONNX naming - NO padding in conv
        self.embed = nn.Sequential()
        self.embed.net = nn.Conv1d(in_dim, hidden_dim, dec_cfg["ksz_init"], padding=0, bias=True)

        # ConvNext stack - create as indexed list to match ONNX naming
        self.convnext = nn.ModuleList([
            ConvNextBlockVocoder(
                dim=hidden_dim,
                kernel_size=dec_cfg["ksz"],
                intermediate_dim=dec_cfg["intermediate_dim"],
                dilation=dec_cfg["dilation_lst"][i],
            )
            for i in range(dec_cfg["num_layers"])
        ])

        # Final normalization before head
        self.final_norm = FinalNorm(hidden_dim)

        # Output head
        head_cfg = dec_cfg["head"]
        self.head = DecoderHead(
            in_dim=head_cfg["idim"],
            hidden_dim=head_cfg["hdim"],
            out_dim=head_cfg["odim"],
            kernel_size=head_cfg["ksz"],
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_dim, L) latent representation
        Returns:
            waveform: (B, L * chunk_size)
        """
        B, C, L = latent.shape

        # Apply causal edge padding (left only) then embed
        x = F.pad(latent, (self.embed_causal_pad, 0), mode='replicate')
        x = self.embed.net(x)

        # ConvNext blocks
        for block in self.convnext:
            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        # Head
        x = self.head(x)  # (B, chunk_size, L)

        # Reshape to waveform
        waveform = x.transpose(1, 2).reshape(B, -1)  # (B, L * chunk_size)

        return waveform


class Normalizer(nn.Module):
    """Latent normalizer."""

    def __init__(self, scale: float = 0.25):
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale


class LatentNormalizer(nn.Module):
    """Normalize latent with learned mean and std."""

    def __init__(self, latent_dim: int = 24):
        super().__init__()
        self.register_buffer('latent_mean', torch.zeros(1, latent_dim, 1))
        self.register_buffer('latent_std', torch.ones(1, latent_dim, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.latent_mean) / self.latent_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.latent_std + self.latent_mean


class FullVocoder(nn.Module):
    """
    Complete Vocoder matching ONNX vocoder.onnx.

    This includes:
    - Latent denormalization
    - Chunk decompression (144 -> 24 * 6)
    - Decoder network
    - Waveform assembly
    """

    def __init__(self, config: dict):
        super().__init__()
        ae_cfg = config["ae"]
        ttl_cfg = config["ttl"]

        self.sample_rate = ae_cfg["sample_rate"]
        self.chunk_size = ae_cfg["base_chunk_size"]
        self.latent_dim = ae_cfg["ldim"]
        self.chunk_compress_factor = ttl_cfg["chunk_compress_factor"]

        # Normalizer scale from TTL config
        self.normalizer = Normalizer(ttl_cfg["normalizer"]["scale"])

        # Latent statistics
        self.register_buffer('latent_mean', torch.zeros(1, self.latent_dim, 1))
        self.register_buffer('latent_std', torch.ones(1, self.latent_dim, 1))

        # Decoder
        self.decoder = Decoder(config)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, 144, L) compressed latent from vector estimator
        Returns:
            waveform: (B, wav_length)
        """
        B, C, L = latent.shape

        # Denormalize with TTL normalizer
        latent = self.normalizer.inverse(latent)

        # Decompress chunks: (B, 144, L) -> (B, 24, L * 6)
        latent = latent.view(B, self.latent_dim, self.chunk_compress_factor, L)
        latent = latent.permute(0, 1, 3, 2).reshape(B, self.latent_dim, L * self.chunk_compress_factor)

        # Denormalize with latent stats
        latent = latent * self.latent_std + self.latent_mean

        # Decode to waveform
        waveform = self.decoder(latent)

        return waveform
