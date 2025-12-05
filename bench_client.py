#!/usr/bin/env python3
"""
Benchmark client for CS 5416 final project.

Measures:
  - Per-request latency (mean, median, p95, p99)
  - Overall throughput (requests/sec and requests/min)

Usage examples:

  # Simple run with defaults (server from NODE_0_IP)
  NODE_0_IP=127.0.0.1:5000 python3 bench_client.py

  # Explicit server, 100 requests, concurrency 8
  python3 bench_client.py --server 127.0.0.1:5000 --num-requests 100 --concurrency 8

  # Label this run as batch_size=4 (for your report)
  BATCH_SIZE=4 NODE_0_IP=127.0.0.1:5000 \
  python3 bench_client.py --label "batch_size=4"
"""

import argparse
import os
import time
import uuid
import statistics
import concurrent.futures
from typing import List, Tuple

import requests


# Some representative queries so the server does non-trivial work
DEFAULT_QUERIES = [
    "What is your refund policy?",
    "My order hasn't arrived yet, tracking number is ABC123.",
    "How do I update my billing information?",
    "Is there a warranty on electronic items?",
    "Can I change the shipping address after placing an order?",
    "What payment methods do you accept?",
    "How do I cancel my subscription?",
    "I received a damaged item, what should I do?",
]


def send_request(session: requests.Session, url: str, query: str) -> Tuple[float, bool]:
    """
    Send a single request and measure latency.

    Returns:
        (latency_seconds, success_bool)
    """
    request_id = f"req_{uuid.uuid4()}"
    payload = {
        "request_id": request_id,
        "query": query,
    }

    t0 = time.time()
    try:
        resp = session.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        _ = resp.json()  # we don't actually care about content, just that it's OK
        t1 = time.time()
        return (t1 - t0, True)
    except Exception as e:
        t1 = time.time()
        print(f"[ERROR] Request {request_id} failed: {e}")
        return (t1 - t0, False)


def run_benchmark(
    server: str,
    num_requests: int,
    concurrency: int,
    label: str = "",
) -> None:
    """
    Run the benchmark against Node 0's /query endpoint.
    """
    url = f"http://{server}/query"
    print("=" * 70)
    print(f"Benchmarking server: {url}")
    print(f"Total requests   : {num_requests}")
    print(f"Concurrency      : {concurrency}")
    if label:
        print(f"Run label        : {label}")
    batch_size_env = os.environ.get("BATCH_SIZE")
    if batch_size_env:
        print(f"BATCH_SIZE (env) : {batch_size_env}")
    print("=" * 70)

    # Warm-up request (helps avoid first-request cold start skew)
    with requests.Session() as warm_sess:
        print("\n[Warm-up] Sending single warm-up request...")
        _, ok = send_request(warm_sess, url, DEFAULT_QUERIES[0])
        if not ok:
            print("[Warm-up] WARNING: warm-up request failed; benchmark may be meaningless")

    # Actual benchmark
    print("\nRunning benchmark...")
    latencies: List[float] = []
    successes = 0

    start = time.time()
    with requests.Session() as session:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Prepare futures
            futures = []
            for i in range(num_requests):
                query = DEFAULT_QUERIES[i % len(DEFAULT_QUERIES)]
                fut = executor.submit(send_request, session, url, query)
                futures.append(fut)

            # Collect results
            for fut in concurrent.futures.as_completed(futures):
                latency, ok = fut.result()
                latencies.append(latency)
                if ok:
                    successes += 1
    end = time.time()

    elapsed = end - start
    total = len(latencies)

    print("\n=== Results ===")
    print(f"Elapsed wall time : {elapsed:.3f} s")
    print(f"Total requests    : {total}")
    print(f"Successful        : {successes}")
    print(f"Failed            : {total - successes}")

    if total == 0:
        print("No requests completed; cannot compute stats.")
        return

    # Throughput
    throughput_rps = successes / elapsed if elapsed > 0 else 0.0
    throughput_rpm = throughput_rps * 60.0

    print("\nThroughput:")
    print(f"  {throughput_rps:.2f} requests/second")
    print(f"  {throughput_rpm:.2f} requests/minute")

    # Latency stats (only successes)
    success_latencies = [l for l in latencies if l is not None]
    success_latencies.sort()
    if not success_latencies:
        print("\nNo successful requests; skipping latency stats.")
        return

    def percentile(data: List[float], p: float) -> float:
        """Compute percentile p (0-100) of a sorted list."""
        if not data:
            return float("nan")
        k = (len(data) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(data) - 1)
        if f == c:
            return data[int(k)]
        d0 = data[f] * (c - k)
        d1 = data[c] * (k - f)
        return d0 + d1

    mean_lat = statistics.mean(success_latencies)
    median_lat = statistics.median(success_latencies)
    p95_lat = percentile(success_latencies, 95)
    p99_lat = percentile(success_latencies, 99)

    print("\nLatency (successful requests only):")
    print(f"  Mean      : {mean_lat:.3f} s")
    print(f"  Median    : {median_lat:.3f} s")
    print(f"  95th pct  : {p95_lat:.3f} s")
    print(f"  99th pct  : {p99_lat:.3f} s")

    print("\nDone.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS5416 RAG pipeline benchmark client")
    parser.add_argument(
        "--server",
        type=str,
        default=os.environ.get("NODE_0_IP", "127.0.0.1:5000"),
        help="Node 0 server host:port (default: NODE_0_IP env or 127.0.0.1:5000)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="Total number of requests to send (default: 50)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent client threads (default: 4)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional label for this run (e.g., 'batch_size=4')",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        server=args.server,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        label=args.label,
    )