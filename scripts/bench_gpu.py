"""CLI: Benchmark GPU vs CPU CA evolution.

Verifies numerical correctness and measures throughput at various batch sizes.
"""

import argparse
import time
import numpy as np

from wheeler_memory import (
    evolve_and_interpret,
    hash_to_frame,
    gpu_available,
    gpu_evolve_batch,
    gpu_evolve_single,
)


def verify_correctness(n=100):
    """Run n inputs through both CPU and GPU, assert matching results."""
    print(f"\n{'='*55}")
    print(f"  CORRECTNESS VERIFICATION  (n={n})")
    print(f"{'='*55}")

    if not gpu_available():
        print("  GPU not available! Build with: cd wheeler_memory/gpu && make")
        return False

    texts = [f"correctness test input {i} with unique content {i**2}" for i in range(n)]
    frames = [hash_to_frame(t) for t in texts]

    mismatches = 0
    tick_errors = 0

    for i, frame in enumerate(frames):
        cpu_result = evolve_and_interpret(frame.copy())
        gpu_result = gpu_evolve_single(frame.copy())

        cpu_att = cpu_result["attractor"]
        gpu_att = gpu_result["attractor"]

        if not np.allclose(cpu_att, gpu_att, atol=1e-4):
            max_diff = np.max(np.abs(cpu_att - gpu_att))
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH #{i}: max_diff={max_diff:.6f}")

        if cpu_result["convergence_ticks"] != gpu_result["convergence_ticks"]:
            tick_errors += 1
            if tick_errors <= 3:
                print(f"  TICK MISMATCH #{i}: CPU={cpu_result['convergence_ticks']} GPU={gpu_result['convergence_ticks']}")

    if mismatches == 0 and tick_errors == 0:
        print(f"  ✓ All {n} inputs match exactly (atol=1e-4)")
        return True
    else:
        print(f"  ✗ {mismatches} attractor mismatches, {tick_errors} tick mismatches out of {n}")
        return False


def benchmark(batch_sizes=None):
    """Benchmark CPU vs GPU at various batch sizes."""
    if batch_sizes is None:
        batch_sizes = [1, 10, 50, 100, 500, 1000, 5000]

    print(f"\n{'='*55}")
    print(f"  PERFORMANCE BENCHMARK")
    print(f"{'='*55}")
    print(f"  {'Batch':>6}  {'CPU (s)':>9}  {'GPU (s)':>9}  {'Speedup':>8}  {'GPU samp/s':>10}")
    print(f"  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*10}")

    for n in batch_sizes:
        texts = [f"benchmark input {i} batch {n}" for i in range(n)]
        frames = [hash_to_frame(t) for t in texts]

        # CPU timing
        cpu_start = time.time()
        for f in frames:
            evolve_and_interpret(f.copy())
        cpu_time = time.time() - cpu_start

        # GPU timing (batch)
        gpu_start = time.time()
        gpu_evolve_batch([f.copy() for f in frames])
        gpu_time = time.time() - gpu_start

        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        gpu_rate = n / gpu_time if gpu_time > 0 else float('inf')

        print(f"  {n:>6}  {cpu_time:>9.4f}  {gpu_time:>9.4f}  {speedup:>7.1f}×  {gpu_rate:>10.0f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Wheeler Memory GPU benchmark")
    parser.add_argument("--verify-only", action="store_true", help="Only run correctness check")
    parser.add_argument("--skip-verify", action="store_true", help="Skip correctness check")
    parser.add_argument("--batch-sizes", type=str, default=None,
                        help="Comma-separated batch sizes (default: 1,10,50,100,500,1000,5000)")
    args = parser.parse_args()

    print(f"GPU available: {gpu_available()}")
    if not gpu_available():
        print("Build the GPU kernel first: cd wheeler_memory/gpu && make")
        return

    if not args.skip_verify:
        ok = verify_correctness()
        if not ok:
            print("\nCorrectness check failed — aborting benchmark.")
            return

    if args.verify_only:
        return

    batch_sizes = None
    if args.batch_sizes:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    benchmark(batch_sizes)


if __name__ == "__main__":
    main()
