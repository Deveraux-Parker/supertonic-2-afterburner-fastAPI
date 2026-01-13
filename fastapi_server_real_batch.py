#!/usr/bin/env python3
"""
FastAPI server for Supertonic PyTorch with REAL GPU batching.

This actually batches multiple requests into a single model forward pass.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import base64
import io
import time
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from collections import deque
import statistics

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import scipy.io.wavfile as wavfile
import uvicorn

# Configure PyTorch
torch.set_float32_matmul_precision('high')
torch._dynamo.config.capture_scalar_outputs = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safetensors.torch import load_file
from supertonic import Supertonic, TextProcessor, load_voice_style


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BatchConfig:
    max_batch_size: int = 16
    max_wait_time_ms: float = 10.0  # Much shorter - we're fast now
    queue_timeout_ms: float = 30000
    enable_batching: bool = True
    # Bucket boundaries (token lengths)
    short_threshold: int = 50    # ~200 chars -> fast processing
    medium_threshold: int = 120  # ~500 chars -> medium processing
    # > medium_threshold = long bucket


# =============================================================================
# Smart Text Chunking
# =============================================================================

import re

def smart_chunk_text(text: str, target_chars: int = 100, max_chars: int = 150) -> List[str]:
    """
    Split text at natural boundaries to normalize to SHORT size (~100 chars).

    Strategy: Chunk EVERYTHING to ~100 chars so all batch items are short.
    Short items = fast processing, minimal padding, maximum throughput.

    - Prefers splitting at sentence boundaries (. ! ?)
    - Falls back to clause boundaries (, ; :) if sentences are too long
    - Force-splits at word boundaries only if absolutely necessary
    """
    text = text.strip()
    if not text:
        return []

    # Short text - return as-is
    if len(text) <= target_chars:
        return [text]

    chunks = []

    # First try to split at sentence boundaries
    # This regex keeps the punctuation with the sentence
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)

    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If sentence fits in current chunk, add it
        if len(current) + len(sentence) + 1 <= target_chars:
            current = f"{current} {sentence}".strip() if current else sentence
        else:
            # Current chunk is ready to emit (if it has content)
            if current:
                chunks.append(current)
                current = ""

            # Now handle this sentence
            if len(sentence) <= target_chars:
                # Sentence fits as a new chunk
                current = sentence
            elif len(sentence) <= max_chars:
                # Sentence is medium-long, try to split at clause boundaries
                sub_chunks = _split_at_clauses(sentence, target_chars, max_chars)
                chunks.extend(sub_chunks[:-1])
                current = sub_chunks[-1] if sub_chunks else ""
            else:
                # Very long sentence - must split at clauses or words
                sub_chunks = _split_at_clauses(sentence, target_chars, max_chars)
                chunks.extend(sub_chunks[:-1])
                current = sub_chunks[-1] if sub_chunks else ""

    # Don't forget the last chunk
    if current:
        chunks.append(current)

    return chunks if chunks else [text]


def _split_at_clauses(text: str, target: int, max_chars: int) -> List[str]:
    """Split at clause boundaries (, ; :) or force-split if needed."""
    if len(text) <= max_chars:
        return [text]

    # Try splitting at clause boundaries
    clause_pattern = r'(?<=[,;:])\s+'
    parts = re.split(clause_pattern, text)

    chunks = []
    current = ""

    for part in parts:
        if len(current) + len(part) + 1 <= target:
            current = (current + " " + part).strip() if current else part
        else:
            if current:
                chunks.append(current)
            if len(part) > max_chars:
                # Force split at word boundaries
                words = part.split()
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 <= max_chars:
                        current = (current + " " + word).strip() if current else word
                    else:
                        if current:
                            chunks.append(current)
                        current = word
            else:
                current = part

    if current:
        chunks.append(current)

    return chunks


# =============================================================================
# Request/Response Models
# =============================================================================

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    voice: str = Field(default="M1")
    language: str = Field(default="en", pattern="^(en|ko|es|pt|fr)$")
    steps: int = Field(default=2, ge=1, le=10)
    speed: float = Field(default=1.05, gt=0.1, lt=3.0)
    output_format: str = Field(default="wav", pattern="^(wav|base64)$")


class TTSResponse(BaseModel):
    audio_base64: Optional[str] = None
    duration_seconds: float
    sample_rate: int
    inference_time_ms: float
    characters_per_second: float
    real_time_factor: float
    batch_size: int = 1
    queue_time_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    compiled: bool
    device: str
    available_voices: List[str]
    batching_enabled: bool
    batch_config: Dict[str, Any]


class MetricsResponse(BaseModel):
    total_requests: int
    total_batches: int
    avg_batch_size: float
    avg_queue_time_ms: float
    avg_inference_time_ms: float
    current_queue_size: int
    vram_used_mb: float
    vram_total_mb: float


class LongFormRequest(BaseModel):
    """Request for long-form text synthesis with automatic chunking."""
    text: str = Field(..., min_length=1, max_length=50000)  # Allow much longer text
    voice: str = Field(default="M1")
    language: str = Field(default="en", pattern="^(en|ko|es|pt|fr)$")
    steps: int = Field(default=2, ge=1, le=10)
    speed: float = Field(default=1.05, gt=0.1, lt=3.0)
    # Chunking options
    target_chunk_chars: int = Field(default=300, ge=100, le=800)
    max_chunk_chars: int = Field(default=900, ge=200, le=1000)


class LongFormResponse(BaseModel):
    """Response for long-form synthesis."""
    audio_base64: str
    duration_seconds: float
    sample_rate: int
    total_inference_time_ms: float
    chunks_processed: int
    characters_per_second: float
    real_time_factor: float


# =============================================================================
# Batching Infrastructure
# =============================================================================

@dataclass
class PendingRequest:
    request: TTSRequest
    future: asyncio.Future
    enqueue_time: float
    # Pre-processed tensors
    text_ids: Optional[torch.Tensor] = None
    text_length: int = 0
    # Priority: True = first chunk (HIGH priority), False = subsequent chunk (NORMAL)
    is_first_chunk: bool = True


@dataclass
class BatchResult:
    audio: np.ndarray
    sample_rate: int
    inference_time: float
    duration: float
    batch_size: int
    queue_time: float


class RealBatchProcessor:
    """
    REAL batching with PRIORITY QUEUES for streaming.

    Two queues:
    - HIGH: First chunks (index 0) - ensures fast time-to-first-audio
    - NORMAL: Subsequent chunks - processed after HIGH queue is empty

    This allows new users to "jump the line" for their first chunk while
    subsequent chunks are generated just-in-time during playback.
    """

    def __init__(self, model_manager: 'ModelManager', config: BatchConfig):
        self.model_manager = model_manager
        self.config = config
        # Priority queues: HIGH for first chunks, NORMAL for subsequent
        self.high_queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self.normal_queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False

        # Metrics
        self.total_requests = 0
        self.total_batches = 0
        self.batch_sizes: deque = deque(maxlen=1000)
        self.queue_times: deque = deque(maxlen=1000)
        self.inference_times: deque = deque(maxlen=1000)
        self.first_chunk_times: deque = deque(maxlen=1000)  # Track first chunk latency

    async def start(self):
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_loop())

    async def stop(self):
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

    async def submit(self, request: TTSRequest, is_first_chunk: bool = True) -> BatchResult:
        """
        Submit a request. First chunks get HIGH priority for fast time-to-first-audio.
        """
        if not self.config.enable_batching:
            return await self._process_single(request)

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        # Pre-process text on submit to minimize batch processing time
        text_ids, _ = self.model_manager.text_processor(request.text, request.language)

        pending = PendingRequest(
            request=request,
            future=future,
            enqueue_time=time.perf_counter(),
            text_ids=text_ids,
            text_length=text_ids.shape[1],
            is_first_chunk=is_first_chunk,
        )

        # Route to appropriate queue
        if is_first_chunk:
            await self.high_queue.put(pending)
        else:
            await self.normal_queue.put(pending)

        self.total_requests += 1

        try:
            result = await asyncio.wait_for(
                future,
                timeout=self.config.queue_timeout_ms / 1000
            )
            # Track first chunk latency
            if is_first_chunk:
                self.first_chunk_times.append(result.queue_time)
            return result
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out")

    async def _process_single(self, request: TTSRequest) -> BatchResult:
        """Single request without batching."""
        start = time.perf_counter()

        audio, sample_rate, inference_time, duration = self.model_manager.generate_single(
            text=request.text,
            voice=request.voice,
            lang=request.language,
            steps=request.steps,
            speed=request.speed
        )

        return BatchResult(
            audio=audio,
            sample_rate=sample_rate,
            inference_time=inference_time,
            duration=duration,
            batch_size=1,
            queue_time=0.0
        )

    async def _process_loop(self):
        while self.is_running:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch_real(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Batch error: {e}")
                import traceback
                traceback.print_exc()

    async def _collect_batch(self) -> List[PendingRequest]:
        """
        Collect batch with PRIORITY: HIGH queue first, then NORMAL.

        Strategy:
        1. Always drain HIGH queue first (first chunks = fast time-to-first-audio)
        2. Fill remaining batch slots from NORMAL queue
        3. This ensures new users get their first chunk ASAP
        """
        batch: List[PendingRequest] = []

        # First, try to get from HIGH queue (first chunks - priority)
        try:
            first = await asyncio.wait_for(self.high_queue.get(), timeout=0.005)
            batch.append(first)
        except asyncio.TimeoutError:
            # No high priority items, try normal queue
            try:
                first = await asyncio.wait_for(self.normal_queue.get(), timeout=0.1)
                batch.append(first)
            except asyncio.TimeoutError:
                return []

        deadline = time.perf_counter() + (self.config.max_wait_time_ms / 1000)

        # Fill batch: prioritize HIGH queue, then NORMAL
        while len(batch) < self.config.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break

            # Always check HIGH queue first (non-blocking)
            try:
                req = self.high_queue.get_nowait()
                batch.append(req)
                continue
            except asyncio.QueueEmpty:
                pass

            # Then check NORMAL queue with timeout
            try:
                req = await asyncio.wait_for(self.normal_queue.get(), timeout=min(remaining, 0.002))
                batch.append(req)
            except asyncio.TimeoutError:
                # Check HIGH one more time before giving up
                try:
                    req = self.high_queue.get_nowait()
                    batch.append(req)
                except asyncio.QueueEmpty:
                    break

        return batch

    async def _process_batch_real(self, batch: List[PendingRequest]):
        """
        REAL BATCHING: Pad inputs, run ONE forward pass, split outputs.
        """
        if not batch:
            return

        batch_size = len(batch)
        process_start = time.perf_counter()
        queue_times = [process_start - req.enqueue_time for req in batch]

        try:
            # Run batched inference synchronously (it's fast anyway)
            # Don't use run_in_executor - cuda graphs don't work across threads
            results = self._batch_inference_sync(batch)

            inference_time = time.perf_counter() - process_start

            # Distribute results
            for i, pending in enumerate(batch):
                if not pending.future.done():
                    result = BatchResult(
                        audio=results[i]['audio'],
                        sample_rate=results[i]['sample_rate'],
                        inference_time=inference_time / batch_size,  # Per-request share
                        duration=results[i]['duration'],
                        batch_size=batch_size,
                        queue_time=queue_times[i] * 1000,
                    )
                    pending.future.set_result(result)

            # Metrics
            self.total_batches += 1
            self.batch_sizes.append(batch_size)
            self.queue_times.extend(queue_times)
            self.inference_times.append(inference_time)

        except Exception as e:
            import traceback
            print(f"BATCH ERROR: {e}")
            traceback.print_exc()
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(e)

    def _batch_inference_sync(self, batch: List[PendingRequest]) -> List[Dict]:
        """
        Synchronous batched inference - runs on GPU.
        """
        mm = self.model_manager
        if mm is None:
            raise RuntimeError("Model manager is None")
        device = mm.device
        batch_size = len(batch)

        # 1. Pad text_ids to same length
        text_ids_list = [req.text_ids.squeeze(0) for req in batch if req.text_ids is not None]  # Remove batch dim
        if len(text_ids_list) != batch_size:
            raise RuntimeError("Some requests have None text_ids")
        text_lengths = [t.shape[0] for t in text_ids_list]
        max_len = max(text_lengths)

        # Pad sequences
        padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, ids in enumerate(text_ids_list):
            padded_ids[i, :len(ids)] = ids.to(device)

        # Create mask (B, 1, T)
        text_mask = torch.zeros(batch_size, 1, max_len, device=device)
        for i, length in enumerate(text_lengths):
            text_mask[i, 0, :length] = 1.0

        # 2. Get voice styles for each request (may be different voices)
        # Group by voice to minimize tensor creation
        voice_groups: Dict[str, List[int]] = {}
        for i, req in enumerate(batch):
            voice = req.request.voice
            if voice not in voice_groups:
                voice_groups[voice] = []
            voice_groups[voice].append(i)

        # Stack styles
        style_ttl = torch.zeros(batch_size, 50, 256, device=device)
        style_dp = torch.zeros(batch_size, 8, 16, device=device)

        for voice, indices in voice_groups.items():
            ttl, dp = mm.voices[voice]
            for idx in indices:
                style_ttl[idx] = ttl.squeeze(0)
                style_dp[idx] = dp.squeeze(0)

        # 3. Get steps/speed (use first request's settings for simplicity)
        # TODO: Could group by settings for more flexibility
        steps = batch[0].request.steps
        speed = batch[0].request.speed

        # 4. Run SINGLE forward pass for entire batch
        inference_start = time.perf_counter()

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                waveforms, durations = mm.model(
                    padded_ids, text_mask, style_ttl, style_dp,
                    total_steps=steps, speed=speed
                )

        torch.cuda.synchronize()
        inference_time = time.perf_counter() - inference_start

        # 5. GPU-side normalization for speed
        sample_rate = mm.sample_rate

        # Normalize on GPU (faster than CPU)
        max_vals = waveforms.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        waveforms_norm = waveforms / max_vals * 0.95

        # Transfer all to CPU at once
        durations_cpu = durations.cpu().numpy()
        wav_lengths = (durations_cpu * sample_rate).astype(np.int32)
        waveforms_cpu = waveforms_norm.cpu().numpy()

        # Build results
        results = []
        for i in range(batch_size):
            wav_length = wav_lengths[i]
            audio = waveforms_cpu[i, :wav_length].copy()  # Copy to avoid memory issues

            results.append({
                'audio': audio,
                'sample_rate': sample_rate,
                'duration': float(durations_cpu[i]),
                'inference_time': inference_time,
            })

        return results

    def get_metrics(self) -> Dict[str, Any]:
        vram_used = 0
        vram_total = 0
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024 / 1024
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "avg_batch_size": statistics.mean(self.batch_sizes) if self.batch_sizes else 0,
            "avg_queue_time_ms": statistics.mean(self.queue_times) * 1000 if self.queue_times else 0,
            "avg_first_chunk_ms": statistics.mean(self.first_chunk_times) if self.first_chunk_times else 0,
            "avg_inference_time_ms": statistics.mean(self.inference_times) * 1000 if self.inference_times else 0,
            "high_queue_size": self.high_queue.qsize(),
            "normal_queue_size": self.normal_queue.qsize(),
            "current_queue_size": self.high_queue.qsize() + self.normal_queue.qsize(),
            "vram_used_mb": vram_used,
            "vram_total_mb": vram_total,
        }


# =============================================================================
# Model Manager
# =============================================================================

class ModelManager:
    def __init__(self):
        self.model = None
        self.text_processor = None
        self.voices = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = None
        self.is_compiled = False

    def load_base_model(self):
        print("Loading base model...")

        script_dir = Path(__file__).parent
        config_path = script_dir / '../supertonic-2/onnx/tts.json'
        weights_path = script_dir / 'supertonic.safetensors'
        indexer_path = script_dir / '../supertonic-2/onnx/unicode_indexer.json'
        voice_dir = script_dir / '../supertonic-2/voice_styles'

        with open(config_path, 'r') as f:
            config = json.load(f)

        self.sample_rate = config['ae']['sample_rate']

        self.model = Supertonic(config)
        self.model.load_state_dict(load_file(weights_path))
        self.model.eval()
        self.model.to(self.device)

        self.text_processor = TextProcessor(str(indexer_path))

        print("Loading voices...")
        for voice_file in voice_dir.glob('*.json'):
            voice_name = voice_file.stem
            style_ttl, style_dp = load_voice_style(str(voice_file))
            self.voices[voice_name] = (style_ttl.to(self.device), style_dp.to(self.device))

        print(f"Loaded voices: {list(self.voices.keys())}")

    def compile_model(self):
        print("Compiling model with torch.compile...")
        self.model = torch.compile(
            self.model,
            mode='reduce-overhead',
            dynamic=True,  # Important for variable batch sizes
            fullgraph=False,
            backend='inductor'
        )

        # Warmup with various batch sizes
        print("Warming up with various batch sizes...")
        for batch_size in [1, 2, 4, 8]:
            self._warmup_batch(batch_size)

        self.is_compiled = True
        print("Model compiled!")

    def _warmup_batch(self, batch_size: int):
        """Warmup with a specific batch size."""
        text = "Hello, this is a warmup text for compilation."
        text_ids, _ = self.text_processor(text, 'en')

        # Create batch
        text_ids = text_ids.repeat(batch_size, 1).to(self.device)
        text_mask = torch.ones(batch_size, 1, text_ids.shape[1], device=self.device)

        style_ttl, style_dp = self.voices['M1']
        style_ttl = style_ttl.repeat(batch_size, 1, 1)
        style_dp = style_dp.repeat(batch_size, 1, 1)

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                _ = self.model(text_ids, text_mask, style_ttl, style_dp, total_steps=2)

        torch.cuda.synchronize()
        print(f"  Warmed up batch_size={batch_size}")

    def generate_single(self, text: str, voice: str = 'M1', lang: str = 'en',
                        steps: int = 2, speed: float = 1.05) -> Tuple[np.ndarray, int, float, float]:
        """Generate audio for a single request."""
        if voice not in self.voices:
            raise ValueError(f"Unknown voice: {voice}")

        start = time.perf_counter()

        style_ttl, style_dp = self.voices[voice]
        text_ids, text_mask = self.text_processor(text, lang)
        text_ids = text_ids.to(self.device)
        text_mask = text_mask.to(self.device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                waveform, duration = self.model(
                    text_ids, text_mask, style_ttl, style_dp,
                    total_steps=steps, speed=speed
                )

        torch.cuda.synchronize()

        audio = waveform[0].cpu().numpy()
        dur = duration[0].item()
        wav_length = int(dur * self.sample_rate)
        audio = audio[:wav_length]

        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95

        inference_time = time.perf_counter() - start

        return audio, self.sample_rate, inference_time, dur


# =============================================================================
# FastAPI App
# =============================================================================

model_manager: Optional[ModelManager] = None
batcher: Optional[RealBatchProcessor] = None
batch_config = BatchConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, batcher

    model_manager = ModelManager()
    model_manager.load_base_model()
    model_manager.compile_model()

    batcher = RealBatchProcessor(model_manager, batch_config)
    await batcher.start()

    # Auto-warmup: run inference requests to warm up the full pipeline
    print("\nWarming up inference pipeline...")
    warmup_texts = [
        "Hello, this is a warmup request.",
        "Testing the text to speech system.",
        "One more warmup to ensure everything is ready.",
    ]
    for i, text in enumerate(warmup_texts):
        warmup_request = TTSRequest(text=text, voice="M1", output_format="wav")
        try:
            await batcher.submit(warmup_request, is_first_chunk=True)
            print(f"  Warmup {i+1}/{len(warmup_texts)} complete")
        except Exception as e:
            print(f"  Warmup {i+1} failed: {e}")

    # Reset metrics after warmup
    batcher.total_requests = 0
    batcher.total_batches = 0
    batcher.batch_sizes.clear()
    batcher.queue_times.clear()
    batcher.inference_times.clear()
    batcher.first_chunk_times.clear()

    print(f"\nServer ready with PRIORITY BATCHING")
    print(f"  Max batch size: {batch_config.max_batch_size}")
    print(f"  Max wait time: {batch_config.max_wait_time_ms}ms")
    print(f"  Priority queues: HIGH (first chunks) + NORMAL (subsequent)")

    yield

    if batcher:
        await batcher.stop()


app = FastAPI(
    title="Supertonic TTS API (Real Batching)",
    description="Ultra-fast TTS with real GPU batching",
    version="3.0.0",
    lifespan=lifespan
)

# Mount static files for the web UI
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/ui")
async def serve_ui():
    """Serve the web UI for testing TTS"""
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="UI not found")


@app.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model_manager else "initializing",
        model_loaded=model_manager is not None,
        compiled=model_manager.is_compiled if model_manager else False,
        device=str(model_manager.device) if model_manager else "unknown",
        available_voices=list(model_manager.voices.keys()) if model_manager else [],
        batching_enabled=batch_config.enable_batching,
        batch_config={
            "max_batch_size": batch_config.max_batch_size,
            "max_wait_time_ms": batch_config.max_wait_time_ms,
            "queue_timeout_ms": batch_config.queue_timeout_ms,
        }
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    if not batcher:
        raise HTTPException(status_code=503, detail="Server not ready")
    return MetricsResponse(**batcher.get_metrics())


@app.post("/config/batch")
async def configure_batching(
    enable: Optional[bool] = None,
    max_batch_size: Optional[int] = None,
    max_wait_time_ms: Optional[float] = None,
):
    global batch_config

    if enable is not None:
        batch_config.enable_batching = enable
    if max_batch_size is not None:
        batch_config.max_batch_size = max_batch_size
    if max_wait_time_ms is not None:
        batch_config.max_wait_time_ms = max_wait_time_ms

    if batcher:
        batcher.config = batch_config

    return {"message": "Updated", "config": batch_config}


@app.post("/synthesize", response_model=TTSResponse)
async def synthesize(request: TTSRequest):
    """
    Synthesize text to speech.

    All text is automatically chunked to ~200 char mid-range segments at
    punctuation boundaries for optimal batching efficiency. Chunks are
    processed together and audio is seamlessly concatenated.
    """
    if not model_manager or not batcher:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Auto-chunk ALL text to SHORT size for maximum throughput
        # Target ~100 chars, max 150, cut at punctuation
        chunks = smart_chunk_text(request.text, target_chars=100, max_chars=150)

        if not chunks:
            raise HTTPException(status_code=400, detail="Empty text")

        # Single chunk - process directly
        if len(chunks) == 1:
            result = await batcher.submit(request)

            chars_per_sec = len(request.text) / result.inference_time if result.inference_time > 0 else 0
            rtf = result.inference_time / result.duration if result.duration > 0 else 0

            response = TTSResponse(
                duration_seconds=result.duration,
                sample_rate=result.sample_rate,
                inference_time_ms=result.inference_time * 1000,
                characters_per_second=chars_per_sec,
                real_time_factor=rtf,
                batch_size=result.batch_size,
                queue_time_ms=result.queue_time,
            )

            if request.output_format == "base64":
                audio_int16 = (result.audio * 32767).astype(np.int16)
                buffer = io.BytesIO()
                wavfile.write(buffer, result.sample_rate, audio_int16)
                response.audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return response

        # Multiple chunks - process all and concatenate
        chunk_requests = [
            TTSRequest(
                text=chunk,
                voice=request.voice,
                language=request.language,
                steps=request.steps,
                speed=request.speed,
                output_format="wav"  # Always wav for internal processing
            )
            for chunk in chunks
        ]

        # Submit all chunks - they'll batch together nicely (all similar size!)
        results = await asyncio.gather(*[
            batcher.submit(req) for req in chunk_requests
        ])

        # Concatenate audio from all chunks
        combined_audio = np.concatenate([r.audio for r in results])
        total_duration = sum(r.duration for r in results)
        total_inference = sum(r.inference_time for r in results)
        avg_batch_size = sum(r.batch_size for r in results) / len(results)
        avg_queue_time = sum(r.queue_time for r in results) / len(results)

        chars_per_sec = len(request.text) / total_inference if total_inference > 0 else 0
        rtf = total_inference / total_duration if total_duration > 0 else 0

        response = TTSResponse(
            duration_seconds=total_duration,
            sample_rate=results[0].sample_rate,
            inference_time_ms=total_inference * 1000,
            characters_per_second=chars_per_sec,
            real_time_factor=rtf,
            batch_size=int(avg_batch_size),
            queue_time_ms=avg_queue_time,
        )

        if request.output_format == "base64":
            audio_int16 = (combined_audio * 32767).astype(np.int16)
            buffer = io.BytesIO()
            wavfile.write(buffer, results[0].sample_rate, audio_int16)
            response.audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize/stream")
async def synthesize_stream(request: TTSRequest):
    """
    Stream audio chunks as Server-Sent Events.

    Text is chunked to ~100 chars. Each chunk is generated and streamed immediately
    as a base64-encoded WAV. While the client plays chunk N, we generate chunk N+1.

    Returns: text/event-stream with JSON events containing:
      - event: "chunk" with {index, total, audio_base64, duration_seconds, text}
      - event: "done" when complete
    """
    if not model_manager or not batcher:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Chunk the text
    chunks = smart_chunk_text(request.text, target_chars=100, max_chars=150)
    if not chunks:
        raise HTTPException(status_code=400, detail="Empty text")

    sample_rate = model_manager.sample_rate

    async def generate_events():
        """Generator that yields SSE events as each chunk is ready."""
        total_chunks = len(chunks)

        for i, chunk_text in enumerate(chunks):
            # Generate this chunk
            # FIRST chunk (i=0) gets HIGH priority for instant playback
            # Subsequent chunks go to NORMAL queue (generated while user listens)
            chunk_request = TTSRequest(
                text=chunk_text,
                voice=request.voice,
                language=request.language,
                steps=request.steps,
                speed=request.speed,
                output_format="wav"
            )
            is_first = (i == 0)
            result = await batcher.submit(chunk_request, is_first_chunk=is_first)

            # Convert to WAV base64
            audio_int16 = (result.audio * 32767).astype(np.int16)
            buffer = io.BytesIO()
            wavfile.write(buffer, sample_rate, audio_int16)
            audio_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Send chunk event
            event_data = json.dumps({
                "index": i,
                "total": total_chunks,
                "audio_base64": audio_b64,
                "duration_seconds": result.duration,
                "text": chunk_text,
            })
            yield f"event: chunk\ndata: {event_data}\n\n"

        # Send done event
        yield f"event: done\ndata: {json.dumps({'total_chunks': total_chunks})}\n\n"

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Chunk-Count": str(len(chunks)),
        }
    )


@app.post("/synthesize/long", response_model=LongFormResponse)
async def synthesize_long(request: LongFormRequest):
    """
    Synthesize long-form text with automatic chunking.

    - Splits text at natural sentence/clause boundaries
    - Processes chunks in parallel through the batching system
    - Concatenates audio seamlessly
    - Returns single combined audio file
    """
    if not model_manager or not batcher:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.perf_counter()

        # Smart chunk the text
        chunks = smart_chunk_text(
            request.text,
            target_chars=request.target_chunk_chars,
            max_chars=request.max_chunk_chars
        )

        # Process all chunks (they'll be batched automatically by bucket)
        chunk_requests = [
            TTSRequest(
                text=chunk,
                voice=request.voice,
                language=request.language,
                steps=request.steps,
                speed=request.speed,
                output_format="wav"
            )
            for chunk in chunks
        ]

        # Submit all chunks concurrently - they'll batch based on length
        results = await asyncio.gather(*[
            batcher.submit(req) for req in chunk_requests
        ])

        total_inference_time = sum(r.inference_time for r in results)

        # Concatenate audio
        combined_audio = np.concatenate([r.audio for r in results])
        total_duration = sum(r.duration for r in results)

        # Convert to WAV and base64
        audio_int16 = (combined_audio * 32767).astype(np.int16)
        buffer = io.BytesIO()
        wavfile.write(buffer, results[0].sample_rate, audio_int16)

        total_time = time.perf_counter() - start_time
        chars_per_sec = len(request.text) / total_time if total_time > 0 else 0
        rtf = total_time / total_duration if total_duration > 0 else 0

        return LongFormResponse(
            audio_base64=base64.b64encode(buffer.getvalue()).decode('utf-8'),
            duration_seconds=total_duration,
            sample_rate=results[0].sample_rate,
            total_inference_time_ms=total_inference_time * 1000,
            chunks_processed=len(chunks),
            characters_per_second=chars_per_sec,
            real_time_factor=rtf,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voices")
async def list_voices():
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "voices": list(model_manager.voices.keys()),
        "languages": ["en", "ko", "es", "pt", "fr"],
        "default_voice": "M1",
        "default_language": "en"
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-wait", type=float, default=10.0)
    parser.add_argument("--no-batch", action="store_true")
    args = parser.parse_args()

    batch_config.max_batch_size = args.batch_size
    batch_config.max_wait_time_ms = args.max_wait
    batch_config.enable_batching = not args.no_batch

    uvicorn.run(
        "fastapi_server_real_batch:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )
