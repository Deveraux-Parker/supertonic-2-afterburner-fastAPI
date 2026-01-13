# Supertonic TTS - PyTorch Implementation

High-performance Text-to-Speech server using PyTorch with GPU batching and real-time streaming.

**IMPORTANT: You will need to download the safetensors version of this Supertonic 2 model here:**
https://huggingface.co/DevParker/Supertonic-2-Saftetensor/tree/main

This is a GPU based implementation of Supertonic 2 and does not use the ONNX model.

## Key Features

- **Real-time Streaming** - First audio chunk in ~100-300ms, gapless playback
- **100 Concurrent Users** - Perfect streaming for all users simultaneously
- **Priority Queues** - First chunks prioritized for instant response
- **Auto-chunking** - Text automatically split at punctuation for optimal batching
- **245 req/s throughput** - 935x realtime audio generation

See [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) for detailed benchmarks.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Start the Server

```bash
python fastapi_server_real_batch.py --batch-size 32 --max-wait 5 --port 8002
```

The server automatically:
- Compiles the model with torch.compile
- Warms up the inference pipeline
- Initializes priority queues for streaming

### 3. Web UI

Open http://localhost:8002/ui for an interactive demo with:
- Voice selection (M1-M5, F1-F5)
- Real-time audio playback
- API documentation

### 4. Make Requests

```bash
# Health check
curl http://localhost:8002/

# Generate speech (returns base64 audio)
curl -X POST http://localhost:8002/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "voice": "M1", "output_format": "base64"}'

# Stream audio chunks (Server-Sent Events)
curl -X POST http://localhost:8002/synthesize/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a longer text that will be chunked and streamed.", "voice": "F1"}'
```

## Streaming Architecture

The server uses a **priority queue system** optimized for real-time streaming:

```
User Request ("Hello world. How are you today?")
         │
         ▼
    ┌─────────────────┐
    │  Auto-Chunking  │  Split at ~100 chars, punctuation boundaries
    │  "Hello world." │
    │  "How are you?" │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌─────────┐     ┌──────────┐
│  HIGH   │     │  NORMAL  │
│  QUEUE  │     │  QUEUE   │
│(chunk 0)│     │(chunk 1+)│
└────┬────┘     └────┬─────┘
     │               │
     └───────┬───────┘
             ▼
    ┌─────────────────┐
    │  Batch Process  │  GPU batching with padding
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  SSE Stream     │  Each chunk sent as ready
    └─────────────────┘
```

### How It Works

1. **Auto-chunking**: All text is split into ~100 character chunks at punctuation boundaries
2. **Priority routing**: First chunks go to HIGH queue, subsequent chunks to NORMAL queue
3. **Instant playback**: User hears audio within ~100-300ms
4. **Just-in-time generation**: While chunk N plays (~3 seconds), chunks N+1, N+2... are generated
5. **Gapless streaming**: Subsequent chunks always ready before previous finishes

### Performance (100 Concurrent Users)

| Metric | Value |
|--------|-------|
| First chunk latency (p50) | 284ms |
| First chunk latency (p95) | 359ms |
| First chunk latency (max) | 433ms |
| Gapless streams | 100% |

## API Reference

### POST /synthesize

Generate speech from text. Long text is automatically chunked and concatenated.

**Request:**
```json
{
  "text": "Text to synthesize (any length)",
  "voice": "M1",           // M1-M5, F1-F5
  "language": "en",        // en, ko, es, pt, fr
  "steps": 2,              // 1-10 (quality vs speed)
  "speed": 1.05,           // 0.1-3.0
  "output_format": "base64" // wav or base64
}
```

**Response:**
```json
{
  "audio_base64": "UklGR...",
  "duration_seconds": 2.5,
  "sample_rate": 24000,
  "inference_time_ms": 45.2,
  "batch_size": 8,
  "queue_time_ms": 12.3
}
```

### POST /synthesize/stream

Stream audio chunks as Server-Sent Events. Optimal for real-time playback.

**Request:** Same as `/synthesize`

**Response:** `text/event-stream` with events:

```
event: chunk
data: {"index": 0, "total": 3, "audio_base64": "...", "duration_seconds": 2.1, "text": "First chunk."}

event: chunk
data: {"index": 1, "total": 3, "audio_base64": "...", "duration_seconds": 1.8, "text": "Second chunk."}

event: chunk
data: {"index": 2, "total": 3, "audio_base64": "...", "duration_seconds": 2.3, "text": "Third chunk."}

event: done
data: {"total_chunks": 3}
```

### POST /synthesize/long

Synthesize long-form text (up to 50,000 characters) with custom chunking options.

**Request:**
```json
{
  "text": "Very long text...",
  "voice": "M1",
  "target_chunk_chars": 300,
  "max_chunk_chars": 900
}
```

### GET /voices

List available voices and languages.

### GET /metrics

Server performance metrics including queue sizes and latency stats.

### GET /

Health check and server status.

## Client Examples

### Python (Streaming)

```python
import requests
import json
import base64
import io
from pydub import AudioSegment
from pydub.playback import play

def stream_tts(text, voice="M1"):
    response = requests.post(
        "http://localhost:8002/synthesize/stream",
        json={"text": text, "voice": voice},
        stream=True
    )

    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data:') and 'audio_base64' in line:
                data = json.loads(line[5:])
                audio_bytes = base64.b64decode(data['audio_base64'])
                audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
                play(audio)  # Play each chunk as it arrives

stream_tts("Hello! This text will be streamed chunk by chunk.")
```

### JavaScript (Browser)

```javascript
async function streamTTS(text, voice = "M1") {
    const response = await fetch("http://localhost:8002/synthesize/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const audioContext = new AudioContext();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value);
        const events = buffer.split("\n\n");
        buffer = events.pop();

        for (const event of events) {
            if (event.includes("audio_base64")) {
                const data = JSON.parse(event.split("data: ")[1]);
                // Decode and play audio chunk
                const audioData = atob(data.audio_base64);
                // ... play with Web Audio API
            }
        }
    }
}
```

### cURL

```bash
# Non-streaming (wait for complete audio)
curl -X POST http://localhost:8002/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "voice": "M1", "output_format": "base64"}' \
  | jq -r '.audio_base64' | base64 -d > speech.wav

# Streaming (Server-Sent Events)
curl -X POST http://localhost:8002/synthesize/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "This will stream chunk by chunk.", "voice": "F1"}'
```

## Benchmarking

### Streaming Benchmark

Test concurrent streaming performance:

```python
import asyncio
import aiohttp

async def test_streaming(num_users=100):
    url = "http://localhost:8002"
    text = "Your test text here..."

    async with aiohttp.ClientSession() as session:
        tasks = [stream_request(session, url, text, i) for i in range(num_users)]
        results = await asyncio.gather(*tasks)

    # Analyze first chunk latencies and gap detection
```

### Load Test

```bash
python realistic_load_test.py --duration 20 --users "100,200,300"
```

## Server Configuration

### Command Line Options

```bash
python fastapi_server_real_batch.py \
  --batch-size 32 \      # Max requests per GPU batch
  --max-wait 5 \         # Max wait time (ms) to form batch
  --host 0.0.0.0 \
  --port 8002
```

### Runtime Configuration

```bash
curl -X POST "http://localhost:8002/config/batch?max_batch_size=32&max_wait_time_ms=5"
```

## File Structure

```
supertonic-pytorch/
├── fastapi_server_real_batch.py  # Production server
├── supertonic.py                 # Core TTS model
├── realistic_load_test.py        # Load testing script
├── static/
│   └── index.html                # Web UI
├── BENCHMARK_REPORT.md           # Performance documentation
├── README.md                     # This file
├── supertonic.safetensors        # Model weights
└── requirements_api.txt          # Python dependencies
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- NVIDIA GPU with 8GB+ VRAM (24GB recommended)

## License

- **Code**: [MIT License](LICENSE) - Copyright (c) 2025 Supertone Inc.
- **Model Weights**: [BigScience Open RAIL-M License](MODEL_LICENSE) - See use restrictions in Attachment A
