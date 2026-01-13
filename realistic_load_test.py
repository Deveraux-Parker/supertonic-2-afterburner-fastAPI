#!/usr/bin/env python3
"""
Realistic TTS Load Test for Supertonic (Bucketed Version)
==========================================================

This benchmark simulates realistic TTS usage patterns with LENGTH-AWARE testing:
- 50% SHORT requests (~50-150 chars) - UI prompts, confirmations
- 35% MEDIUM requests (~150-400 chars) - Sentences, short responses
- 15% LONG requests (~400-900 chars) - Paragraphs, detailed explanations

Users:
1. Request audio generation for text (weighted by length)
2. Listen to the FULL generated audio (wait for audio duration)
3. Request the next audio after listening
4. Repeat throughout the test

This tests the length-aware bucketing system to verify:
- Short requests maintain low latency even under load
- Long requests don't block short ones
- Overall throughput is optimized

Usage:
------
    python realistic_load_test.py --url http://localhost:8002 --duration 20 --users "100,200,300"

Author: Supertonic Team
"""

import asyncio
import aiohttp
import time
import random
import statistics
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import argparse
import sys

# =============================================================================
# Test Texts by Length Category
# =============================================================================

# SHORT texts (~50-150 chars) - 50% of requests
# These should hit the SHORT bucket and get fastest processing
SHORT_TEXTS = [
    "Hello!",
    "Thank you.",
    "How can I help you today?",
    "Your request has been received.",
    "Please wait a moment.",
    "The operation completed successfully.",
    "Would you like to continue?",
    "I understand. Let me help with that.",
    "Great choice! Processing now.",
    "One moment please.",
    "Sure, I can do that for you.",
    "Here's what I found.",
    "Is there anything else?",
    "Perfect, that's confirmed.",
    "Let me check that for you.",
]

# MEDIUM texts (~150-400 chars) - 35% of requests
# These should hit the MEDIUM bucket
MEDIUM_TEXTS = [
    "Welcome to our service. I'm here to help you with any questions or concerns you might have today.",
    "Thank you for your patience while I look into this matter. I should have an answer for you shortly.",
    "I've found several options that match your requirements. Let me walk you through each one.",
    "Your account has been updated with the new information. The changes will take effect immediately.",
    "I understand your concern and I want to make sure we resolve this to your satisfaction.",
    "Based on your preferences, I would recommend the following solution for your needs.",
    "The system is currently processing your request. This typically takes just a few moments.",
    "I've reviewed your history and I can see what happened. Let me explain the situation.",
]

# LONG texts (~400-900 chars) - 15% of requests
# These should hit the LONG bucket
LONG_TEXTS = [
    "Thank you for contacting our support team today. I've thoroughly reviewed your account and the issue you've described. Based on my analysis, I can see that the problem originated from a system update that occurred last week. I've already initiated the correction process, and you should see the changes reflected in your account within the next twenty-four hours. Is there anything else I can help you with?",
    "I'd like to provide you with a comprehensive overview of our service options. We offer three main tiers: Basic, Professional, and Enterprise. The Basic tier includes all essential features and is perfect for individuals or small teams. The Professional tier adds advanced analytics and priority support. The Enterprise tier provides custom solutions, dedicated account management, and unlimited usage. Each tier can be billed monthly or annually with a discount for annual subscriptions.",
    "Let me explain how our new feature works in detail. First, you'll need to navigate to your settings panel and enable the advanced options. Once enabled, you'll see a new menu item appear in your dashboard. Click on it to access the configuration wizard, which will guide you through the setup process step by step. The wizard includes helpful tips and examples to ensure you get the most out of this feature. After completing the setup, you can customize the behavior to match your specific needs.",
]

# Weighted distribution
TEXT_WEIGHTS = {
    'short': 0.50,   # 50% short
    'medium': 0.35,  # 35% medium
    'long': 0.15,    # 15% long
}

VOICES = ["M1", "M2", "M3", "F1", "F2", "F3"]


def get_random_text() -> Tuple[str, str]:
    """Get a random text with weighted distribution. Returns (text, category)."""
    r = random.random()
    if r < TEXT_WEIGHTS['short']:
        return random.choice(SHORT_TEXTS), 'short'
    elif r < TEXT_WEIGHTS['short'] + TEXT_WEIGHTS['medium']:
        return random.choice(MEDIUM_TEXTS), 'medium'
    else:
        return random.choice(LONG_TEXTS), 'long'


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UserMetrics:
    """Metrics collected for a single simulated user."""
    user_id: int
    requests_made: int = 0
    requests_failed: int = 0
    total_audio_seconds: float = 0.0
    # Track latencies by category
    latencies_short: List[float] = field(default_factory=list)
    latencies_medium: List[float] = field(default_factory=list)
    latencies_long: List[float] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)


@dataclass
class TestResults:
    """Aggregated results for a single concurrency level test."""
    concurrent_users: int
    test_duration: float
    total_requests: int
    failed_requests: int
    requests_per_second: float
    audio_seconds_generated: float
    # Overall latencies
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    # Latencies by category
    short_p50_ms: float
    short_p95_ms: float
    short_count: int
    medium_p50_ms: float
    medium_p95_ms: float
    medium_count: int
    long_p50_ms: float
    long_p95_ms: float
    long_count: int
    avg_batch_size: float

    def to_dict(self) -> dict:
        return {
            "concurrent_users": self.concurrent_users,
            "test_duration_seconds": round(self.test_duration, 2),
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "requests_per_second": round(self.requests_per_second, 2),
            "audio_seconds_generated": round(self.audio_seconds_generated, 2),
            "latency_ms": {
                "avg": round(self.avg_latency_ms, 1),
                "p50": round(self.p50_latency_ms, 1),
                "p95": round(self.p95_latency_ms, 1),
                "p99": round(self.p99_latency_ms, 1),
            },
            "latency_by_category": {
                "short": {"p50": round(self.short_p50_ms, 1), "p95": round(self.short_p95_ms, 1), "count": self.short_count},
                "medium": {"p50": round(self.medium_p50_ms, 1), "p95": round(self.medium_p95_ms, 1), "count": self.medium_count},
                "long": {"p50": round(self.long_p50_ms, 1), "p95": round(self.long_p95_ms, 1), "count": self.long_count},
            },
            "avg_batch_size": round(self.avg_batch_size, 1),
            "success_rate_percent": round(self.total_requests / max(1, self.total_requests + self.failed_requests) * 100, 2),
        }


# =============================================================================
# User Simulation
# =============================================================================

async def simulate_user(
    session: aiohttp.ClientSession,
    user_id: int,
    base_url: str,
    test_duration: float,
    listen_ratio: float,
    metrics: UserMetrics,
    start_event: asyncio.Event,
):
    """
    Simulate a realistic user making TTS requests with varied text lengths.

    Waits for the FULL audio duration before making next request (realistic usage).
    """
    await start_event.wait()

    # Stagger start
    await asyncio.sleep(random.uniform(0, 0.5))

    end_time = time.perf_counter() + test_duration

    while time.perf_counter() < end_time:
        text, category = get_random_text()
        voice = random.choice(VOICES)

        request_start = time.perf_counter()

        try:
            async with session.post(
                f"{base_url}/synthesize",
                json={"text": text, "voice": voice, "output_format": "base64"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    latency = (time.perf_counter() - request_start) * 1000

                    metrics.requests_made += 1
                    metrics.total_audio_seconds += data["duration_seconds"]

                    # Track latency by category
                    if category == 'short':
                        metrics.latencies_short.append(latency)
                    elif category == 'medium':
                        metrics.latencies_medium.append(latency)
                    else:
                        metrics.latencies_long.append(latency)

                    if "batch_size" in data:
                        metrics.batch_sizes.append(data["batch_size"])

                    # Wait for FULL audio duration (realistic - user listens)
                    listen_time = data["duration_seconds"] * listen_ratio
                    if time.perf_counter() + listen_time < end_time:
                        await asyncio.sleep(listen_time)
                else:
                    metrics.requests_failed += 1
                    await asyncio.sleep(0.1)

        except asyncio.TimeoutError:
            metrics.requests_failed += 1
            await asyncio.sleep(0.1)
        except Exception:
            metrics.requests_failed += 1
            await asyncio.sleep(0.1)


def calc_percentiles(latencies: List[float]) -> Tuple[float, float, float]:
    """Calculate p50, p95, p99 for a list of latencies."""
    if not latencies:
        return 0, 0, 0
    latencies = sorted(latencies)
    n = len(latencies)
    return latencies[n // 2], latencies[int(n * 0.95)], latencies[min(int(n * 0.99), n - 1)]


# =============================================================================
# Load Test Runner
# =============================================================================

async def run_load_test(
    base_url: str,
    concurrent_users: int,
    test_duration: float,
    listen_ratio: float,
) -> TestResults:
    """Run a load test with the specified number of concurrent users."""

    print(f"\n{'='*70}")
    print(f"Testing with {concurrent_users} concurrent users for {test_duration}s")
    print(f"Text distribution: {TEXT_WEIGHTS['short']*100:.0f}% short, "
          f"{TEXT_WEIGHTS['medium']*100:.0f}% medium, {TEXT_WEIGHTS['long']*100:.0f}% long")
    print(f"{'='*70}")

    user_metrics = [UserMetrics(user_id=i) for i in range(concurrent_users)]
    start_event = asyncio.Event()
    connector = aiohttp.TCPConnector(limit=concurrent_users + 50)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            simulate_user(session, i, base_url, test_duration, listen_ratio,
                         user_metrics[i], start_event)
            for i in range(concurrent_users)
        ]

        test_start = time.perf_counter()
        start_event.set()
        await asyncio.gather(*tasks)
        actual_duration = time.perf_counter() - test_start

    # Aggregate results
    total_requests = sum(m.requests_made for m in user_metrics)
    failed_requests = sum(m.requests_failed for m in user_metrics)
    total_audio = sum(m.total_audio_seconds for m in user_metrics)

    # Aggregate latencies by category
    all_short = [lat for m in user_metrics for lat in m.latencies_short]
    all_medium = [lat for m in user_metrics for lat in m.latencies_medium]
    all_long = [lat for m in user_metrics for lat in m.latencies_long]
    all_latencies = all_short + all_medium + all_long
    all_batch_sizes = [bs for m in user_metrics for bs in m.batch_sizes]

    # Calculate percentiles
    p50, p95, p99 = calc_percentiles(all_latencies)
    short_p50, short_p95, _ = calc_percentiles(all_short)
    medium_p50, medium_p95, _ = calc_percentiles(all_medium)
    long_p50, long_p95, _ = calc_percentiles(all_long)

    results = TestResults(
        concurrent_users=concurrent_users,
        test_duration=actual_duration,
        total_requests=total_requests,
        failed_requests=failed_requests,
        requests_per_second=total_requests / actual_duration if actual_duration > 0 else 0,
        audio_seconds_generated=total_audio,
        avg_latency_ms=statistics.mean(all_latencies) if all_latencies else 0,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        short_p50_ms=short_p50,
        short_p95_ms=short_p95,
        short_count=len(all_short),
        medium_p50_ms=medium_p50,
        medium_p95_ms=medium_p95,
        medium_count=len(all_medium),
        long_p50_ms=long_p50,
        long_p95_ms=long_p95,
        long_count=len(all_long),
        avg_batch_size=statistics.mean(all_batch_sizes) if all_batch_sizes else 1.0,
    )

    # Print results
    success_rate = total_requests / max(1, total_requests + failed_requests) * 100

    print(f"\nResults for {concurrent_users} users:")
    print(f"  Duration: {results.test_duration:.1f}s")
    print(f"  Total requests: {results.total_requests:,} ({results.failed_requests} failed)")
    print(f"  Throughput: {results.requests_per_second:.1f} req/s")
    print(f"  Audio generated: {results.audio_seconds_generated:,.1f}s")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"\n  Overall latency: p50={results.p50_latency_ms:.0f}ms, p95={results.p95_latency_ms:.0f}ms")
    print(f"\n  By text length:")
    print(f"    SHORT  ({results.short_count:>4} reqs): p50={results.short_p50_ms:>6.0f}ms, p95={results.short_p95_ms:>6.0f}ms")
    print(f"    MEDIUM ({results.medium_count:>4} reqs): p50={results.medium_p50_ms:>6.0f}ms, p95={results.medium_p95_ms:>6.0f}ms")
    print(f"    LONG   ({results.long_count:>4} reqs): p50={results.long_p50_ms:>6.0f}ms, p95={results.long_p95_ms:>6.0f}ms")

    return results


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Realistic TTS Load Test with Bucketing Analysis")
    parser.add_argument("--url", default="http://localhost:8002", help="TTS server URL")
    parser.add_argument("--duration", type=float, default=20, help="Test duration per level (seconds)")
    parser.add_argument("--users", type=str, default="100,200,300,400,500", help="Comma-separated user counts")
    parser.add_argument("--output", type=str, default=None, help="JSON output file")
    parser.add_argument("--listen-ratio", type=float, default=1.0, help="Listen ratio (1.0 = full audio duration)")

    args = parser.parse_args()
    user_counts = [int(x.strip()) for x in args.users.split(",")]

    print("=" * 70)
    print("REALISTIC TTS LOAD TEST (Bucketed)")
    print("=" * 70)
    print(f"Server: {args.url}")
    print(f"Duration per level: {args.duration}s")
    print(f"User counts: {user_counts}")
    print(f"Listen ratio: {args.listen_ratio} (wait {args.listen_ratio*100:.0f}% of audio)")
    print(f"\nText distribution:")
    print(f"  - SHORT  (50-150 chars):  {TEXT_WEIGHTS['short']*100:.0f}%")
    print(f"  - MEDIUM (150-400 chars): {TEXT_WEIGHTS['medium']*100:.0f}%")
    print(f"  - LONG   (400-900 chars): {TEXT_WEIGHTS['long']*100:.0f}%")

    # Verify server
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{args.url}/", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    print(f"\nERROR: Server returned status {resp.status}")
                    sys.exit(1)
                data = await resp.json()
                print(f"\nServer: {data.get('status', 'unknown')} on {data.get('device', 'unknown')}")
                print(f"Batching: {data.get('batching_enabled', 'N/A')}")
        except aiohttp.ClientError as e:
            print(f"\nERROR: Cannot connect to {args.url}: {e}")
            sys.exit(1)

    # Run tests
    all_results = []
    for user_count in user_counts:
        results = await run_load_test(args.url, user_count, args.duration, args.listen_ratio)
        all_results.append(results)

        if user_count != user_counts[-1]:
            print("\nPausing 3s before next test...")
            await asyncio.sleep(3)

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY - Latency by Text Length (demonstrates bucketing effectiveness)")
    print("=" * 100)
    print(f"{'Users':>6} | {'Req/s':>7} | {'Success':>7} | "
          f"{'SHORT p95':>10} | {'MEDIUM p95':>11} | {'LONG p95':>10} | {'Overall p95':>11}")
    print("-" * 100)

    for r in all_results:
        success_pct = r.total_requests / max(1, r.total_requests + r.failed_requests) * 100
        print(f"{r.concurrent_users:>6} | {r.requests_per_second:>7.1f} | {success_pct:>6.1f}% | "
              f"{r.short_p95_ms:>9.0f}ms | {r.medium_p95_ms:>10.0f}ms | {r.long_p95_ms:>9.0f}ms | "
              f"{r.p95_latency_ms:>10.0f}ms")

    # Analysis
    print("\n" + "=" * 100)
    print("BUCKETING ANALYSIS")
    print("=" * 100)

    if all_results:
        # Check if short requests maintain low latency
        last = all_results[-1]
        if last.short_p95_ms < last.long_p95_ms * 0.5:
            print(f"\n[OK] Bucketing effective: SHORT p95 ({last.short_p95_ms:.0f}ms) << LONG p95 ({last.long_p95_ms:.0f}ms)")
            print("     Short requests are not blocked by long requests!")
        else:
            print(f"\n[!] SHORT and LONG latencies are similar - bucketing may need tuning")

        # Find recommended capacity
        viable = [r for r in all_results if r.failed_requests == 0 and r.p95_latency_ms < 2000]
        if viable:
            best = max(viable, key=lambda r: r.concurrent_users)
            audio_rate = best.audio_seconds_generated / best.test_duration
            print(f"\nRECOMMENDED: {best.concurrent_users} concurrent users")
            print(f"  Throughput: {best.requests_per_second:.1f} req/s")
            print(f"  Audio rate: {audio_rate:.0f}x realtime")
            print(f"  SHORT p95: {best.short_p95_ms:.0f}ms")

    # Save results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "server_url": args.url,
                "duration_per_test": args.duration,
                "listen_ratio": args.listen_ratio,
                "user_counts": user_counts,
                "text_weights": TEXT_WEIGHTS,
            },
            "results": [r.to_dict() for r in all_results],
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
