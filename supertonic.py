"""
Supertonic TTS - Complete PyTorch Implementation

This module provides the full TTS pipeline reconstructed from ONNX models.
"""

import json
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from models.text_encoder import FullTextEncoder
from models.duration_predictor import FullDurationPredictor
from models.vector_estimator import FullVectorEstimator
from models.vocoder import FullVocoder


class Supertonic(nn.Module):
    """
    Complete Supertonic TTS model.

    This combines all components:
    - Text Encoder: text -> text embeddings
    - Duration Predictor: text + style -> duration
    - Vector Estimator: flow matching denoiser
    - Vocoder: latent -> waveform
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.sample_rate = config["ae"]["sample_rate"]

        # Model components
        self.text_encoder = FullTextEncoder(config)
        self.duration_predictor = FullDurationPredictor(config)
        self.vector_estimator = FullVectorEstimator(config)
        self.vocoder = FullVocoder(config)

        # Flow matching parameters
        self.latent_dim = config["ttl"]["latent_dim"]
        self.chunk_compress_factor = config["ttl"]["chunk_compress_factor"]
        self.base_chunk_size = config["ae"]["base_chunk_size"]

    @classmethod
    def from_pretrained(cls, model_dir: str, device: str = 'cpu'):
        """Load model from directory containing config and weights."""
        # Load config
        config_path = os.path.join(model_dir, 'onnx', 'tts.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Create model
        model = cls(config)

        # Load weights if available
        weights_dir = os.path.join(model_dir, 'pytorch_weights')
        if os.path.exists(weights_dir):
            model.load_weights(weights_dir)

        return model.to(device)

    def load_weights(self, weights_dir: str):
        """Load PyTorch weights from directory."""
        weight_files = {
            'text_encoder': 'text_encoder.pt',
            'duration_predictor': 'duration_predictor.pt',
            'vector_estimator': 'vector_estimator.pt',
            'vocoder': 'vocoder.pt',
        }

        for component, filename in weight_files.items():
            path = os.path.join(weights_dir, filename)
            if os.path.exists(path):
                state_dict = torch.load(path, map_location='cpu')
                getattr(self, component).load_state_dict(state_dict, strict=False)
                print(f"Loaded {component} weights from {path}")

    def sample_noisy_latent(
        self,
        duration: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample initial noisy latent for flow matching."""
        batch_size = duration.shape[0]

        # Calculate latent dimensions
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).long()
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = ((wav_len_max + chunk_size - 1) // chunk_size).long()
        latent_dim = self.latent_dim * self.chunk_compress_factor  # 144

        # Convert to int for tensor creation (CUDA graph compatible)
        latent_len_int = int(latent_len)

        # Sample noise
        noisy_latent = torch.randn(batch_size, latent_dim, latent_len_int, device=device)

        # Create mask
        latent_lengths = (wav_lengths + chunk_size - 1) // chunk_size
        latent_mask = self._length_to_mask(latent_lengths, latent_len_int)

        # Apply mask
        noisy_latent = noisy_latent * latent_mask

        return noisy_latent, latent_mask

    def _length_to_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Convert lengths to binary mask."""
        batch_size = lengths.shape[0]
        ids = torch.arange(max_len, device=lengths.device)
        mask = (ids.unsqueeze(0) < lengths.unsqueeze(1)).float()
        return mask.unsqueeze(1)  # (B, 1, max_len)

    @torch.no_grad()
    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        style_ttl: torch.Tensor,
        style_dp: torch.Tensor,
        total_steps: int = 5,
        speed: float = 1.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate speech from text.

        Args:
            text_ids: (B, T) token indices
            text_mask: (B, 1, T) attention mask
            style_ttl: (B, 50, 256) style tokens for TTL
            style_dp: (B, 8, 16) style tokens for duration predictor
            total_steps: number of denoising steps
            speed: speech speed multiplier (higher = faster)

        Returns:
            waveform: (B, wav_len) generated audio
            duration: (B,) predicted duration in seconds
        """
        device = text_ids.device
        batch_size = text_ids.shape[0]

        # 1. Predict duration
        duration = self.duration_predictor(text_ids, style_dp, text_mask)
        duration = duration / speed

        # 2. Encode text
        text_emb = self.text_encoder(text_ids, style_ttl, text_mask)

        # 3. Sample initial noise
        noisy_latent, latent_mask = self.sample_noisy_latent(duration, device)

        # 4. Flow matching denoising
        total_step_tensor = torch.full((batch_size,), total_steps, device=device, dtype=torch.float32)

        for step in range(total_steps):
            current_step = torch.full((batch_size,), step, device=device, dtype=torch.float32)
            noisy_latent = self.vector_estimator(
                noisy_latent,
                text_emb,
                style_ttl,
                latent_mask,
                text_mask,
                current_step,
                total_step_tensor,
            )

        # 5. Decode to waveform
        waveform = self.vocoder(noisy_latent)

        # 6. Trim to actual duration
        wav_lengths = (duration * self.sample_rate).long()
        # Keep full waveform, user can trim based on duration

        return waveform, duration


class TextProcessor:
    """Process text to token IDs using unicode indexer."""

    AVAILABLE_LANGS = ["en", "ko", "es", "pt", "fr"]

    def __init__(self, indexer_path: str):
        with open(indexer_path, 'r') as f:
            self.indexer = json.load(f)
        # indexer is a list where indexer[unicode_value] = token_id
        # -1 means character not in vocabulary

    def __call__(
        self,
        text: str,
        lang: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process text to token IDs.

        Args:
            text: Input text
            lang: Language code (en, ko, es, pt, fr)

        Returns:
            text_ids: (1, T) token indices
            text_mask: (1, 1, T) attention mask
        """
        # Preprocess
        text = self._preprocess_text(text, lang)

        # Convert to unicode indices
        # indexer is a list: indexer[unicode_value] = token_id (-1 if not in vocab)
        unicode_values = [ord(char) for char in text]
        text_ids = [self.indexer[val] if val < len(self.indexer) and self.indexer[val] >= 0 else 0
                    for val in unicode_values]

        # Convert to tensors
        text_ids = torch.tensor([text_ids], dtype=torch.long)
        text_mask = torch.ones(1, 1, len(text_ids[0]))

        return text_ids, text_mask

    def _preprocess_text(self, text: str, lang: str) -> str:
        """Preprocess text for TTS."""
        import re
        from unicodedata import normalize

        text = normalize("NFKD", text)

        # Remove emojis
        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff"
            "\u2600-\u26ff"
            "\u2700-\u27bf]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)

        # Clean up
        text = re.sub(r"\s+", " ", text).strip()

        # Add period if needed
        if not re.search(r"[.!?;:,'\"')}\]…。」』】〉》›»]$", text):
            text += "."

        # Add language tags
        if lang not in self.AVAILABLE_LANGS:
            raise ValueError(f"Invalid language: {lang}")
        text = f"<{lang}>" + text + f"</{lang}>"

        return text


def load_voice_style(style_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load voice style from JSON file.

    Returns:
        style_ttl: (1, 50, 256) style tokens for TTL
        style_dp: (1, 8, 16) style tokens for duration predictor
    """
    with open(style_path, 'r') as f:
        style = json.load(f)

    # Extract TTL style
    ttl_dims = style["style_ttl"]["dims"]
    ttl_data = np.array(style["style_ttl"]["data"], dtype=np.float32)
    style_ttl = torch.from_numpy(ttl_data.reshape(ttl_dims))

    # Extract DP style
    dp_dims = style["style_dp"]["dims"]
    dp_data = np.array(style["style_dp"]["data"], dtype=np.float32)
    style_dp = torch.from_numpy(dp_data.reshape(dp_dims))

    return style_ttl, style_dp


# Example usage
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Supertonic TTS')
    parser.add_argument('--model-dir', type=str, default='../supertonic-2',
                        help='Path to model directory')
    parser.add_argument('--text', type=str, default='Hello, this is a test.',
                        help='Text to synthesize')
    parser.add_argument('--lang', type=str, default='en',
                        help='Language code')
    parser.add_argument('--voice', type=str, default='M1',
                        help='Voice style name')
    parser.add_argument('--output', type=str, default='output.wav',
                        help='Output file path')

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    config_path = os.path.join(args.model_dir, 'onnx', 'tts.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = Supertonic(config)
    model.eval()

    # Load text processor
    indexer_path = os.path.join(args.model_dir, 'onnx', 'unicode_indexer.json')
    text_processor = TextProcessor(indexer_path)

    # Load voice style
    style_path = os.path.join(args.model_dir, 'voice_styles', f'{args.voice}.json')
    style_ttl, style_dp = load_voice_style(style_path)

    # Process text
    text_ids, text_mask = text_processor(args.text, args.lang)

    print(f"Text: {args.text}")
    print(f"Text IDs shape: {text_ids.shape}")

    # Note: Full inference requires loading weights
    print("\nNote: To run inference, you need to:")
    print("1. Run convert_weights.py to extract weights from ONNX")
    print("2. Load the weights into the model")
    print("3. Then call model.forward() for synthesis")
