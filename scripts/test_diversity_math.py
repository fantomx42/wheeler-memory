"""CLI: Large-scale diversity report using UltraData-Math (HuggingFace).

Fetches up to 10,000 samples from openbmb/UltraData-Math L3-QA-Synthetic
via the HuggingFace Datasets Server Row API, evolves each through the
Wheeler CA, and generates a diversity report with vectorised correlation.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from wheeler_memory import evolve_and_interpret, get_cell_roles, hash_to_frame
from wheeler_memory import gpu_available, gpu_evolve_batch

# Discrete 3-color map for cell roles
ROLE_CMAP = mcolors.ListedColormap(["#3B82F6", "#9CA3AF", "#EF4444"])
ROLE_NORM = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], ROLE_CMAP.N)

HF_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100  # HF API max per request
DATASET_NAME = "openbmb/UltraData-Math"
DATASET_CONFIG = "UltraData-Math-L3-QA-Synthetic"


def fetch_math_samples(n=10000, offset=0):
    """Fetch n samples via paginated HF Datasets Server API calls.

    Includes exponential backoff on HTTP 429 (rate limit) and a small
    inter-request delay to stay under HuggingFace rate limits.
    """
    samples = []
    pages = (n + PAGE_SIZE - 1) // PAGE_SIZE
    print(f"Fetching {n} samples ({pages} pages) from L3-QA-Synthetic...")

    for page in range(pages):
        page_offset = offset + page * PAGE_SIZE
        remaining = n - len(samples)
        length = min(PAGE_SIZE, remaining)
        url = (
            f"{HF_API}?dataset=openbmb/UltraData-Math"
            f"&config=UltraData-Math-L3-QA-Synthetic"
            f"&split=train"
            f"&offset={page_offset}"
            f"&length={length}"
        )

        # Retry with exponential backoff on rate limit
        max_retries = 4
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                for row_obj in data.get("rows", []):
                    text = row_obj.get("row", {}).get("content", "")
                    samples.append(text[:500])
                break  # success
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)  # 2, 4, 8, 16s
                    print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1})...")
                    time.sleep(wait)
                else:
                    print(f"  Page {page + 1}/{pages} failed: {e}")
                    break
            except Exception as e:
                print(f"  Page {page + 1}/{pages} failed: {e}")
                break

        # Small delay between requests to avoid triggering 429s
        time.sleep(0.2)

        # Progress every 10 pages
        if (page + 1) % 10 == 0 or page == pages - 1:
            print(f"  [{page + 1}/{pages}] fetched {len(samples)} samples")

    print(f"  Total: {len(samples)} samples\n")
    return samples


def fetch_math_samples_local(n=10000, seed=42):
    """Load samples from locally cached HuggingFace dataset.

    Uses random sampling across the full 27M rows instead of sequential
    offset, giving better coverage of the dataset.
    """
    from datasets import load_dataset

    print(f"Loading {n} samples from local cache ({DATASET_NAME})...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    total = len(ds)
    print(f"  Dataset loaded: {total:,} rows")

    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=min(n, total), replace=False)
    indices.sort()  # sequential access is faster on Arrow

    samples = []
    for i, idx in enumerate(indices):
        text = ds[int(idx)]["content"]
        samples.append(text[:500])
        if (i + 1) % 5000 == 0 or i == len(indices) - 1:
            print(f"  [{i + 1:>6}/{n}] sampled")

    print(f"  Total: {len(samples)} samples (random seed={seed})\n")
    return samples


def vectorised_corrmatrix(attractors):
    """Compute full NxN Pearson correlation matrix via numpy BLAS."""
    X = np.array(attractors, dtype=np.float32)  # N x 4096
    # Standardise each row
    X -= X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0  # avoid div-by-zero
    X /= std
    # Pearson r = dot(Xi, Xj) / D
    corr = (X @ X.T) / X.shape[1]
    return corr.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(
        description="Wheeler Memory diversity test on UltraData-Math"
    )
    parser.add_argument(
        "--output", default="diversity_report_math.png", help="Output image path"
    )
    parser.add_argument(
        "--n", type=int, default=10000, help="Number of math samples to test"
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Row offset into the dataset"
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU batch evolution (requires libwheeler_ca.so)"
    )
    parser.add_argument(
        "--local", action="store_true", help="Load from local HF cache (faster, no rate limits)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for local sampling"
    )
    args = parser.parse_args()

    use_gpu = args.gpu and gpu_available()
    if args.gpu and not gpu_available():
        print("Warning: --gpu requested but GPU not available, falling back to CPU")

    n = args.n

    # ── Phase 1: Fetch ────────────────────────────────────────────────
    fetch_start = time.time()
    if args.local:
        math_texts = fetch_math_samples_local(n=n, seed=args.seed)
    else:
        math_texts = fetch_math_samples(n=n, offset=args.offset)
    fetch_time = time.time() - fetch_start
    if len(math_texts) < n:
        print(f"Warning: only got {len(math_texts)} samples (requested {n})")
    n = len(math_texts)

    # ── Phase 2: Evolve through CA ────────────────────────────────────
    attractors = []
    states = []
    ticks_list = []
    labels = []

    backend = "GPU" if use_gpu else "CPU"
    print(f"Evolving {n} inputs through Wheeler CA ({backend})...")
    evolve_start = time.time()

    if use_gpu:
        # Batch GPU evolution
        frames = [hash_to_frame(t) for t in math_texts]
        print(f"  Hashed {len(frames)} frames, launching GPU batch...")
        results = gpu_evolve_batch(frames)
        for i, result in enumerate(results):
            attractors.append(result["attractor"].flatten())
            states.append(result["state"])
            ticks_list.append(result["convergence_ticks"])
            labels.append(math_texts[i][:60].replace("\n", " "))
        evolve_time = time.time() - evolve_start
        print(f"  GPU batch done in {evolve_time:.2f}s ({n / evolve_time:.0f} samples/s)")
    else:
        # Sequential CPU evolution
        for i, text in enumerate(math_texts):
            frame = hash_to_frame(text)
            result = evolve_and_interpret(frame)

            attractors.append(result["attractor"].flatten())
            states.append(result["state"])
            ticks_list.append(result["convergence_ticks"])
            labels.append(text[:60].replace("\n", " "))

            if (i + 1) % 1000 == 0 or i == n - 1:
                elapsed = time.time() - evolve_start
                rate = (i + 1) / elapsed
                eta = (n - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i + 1:>6}/{n}]  {elapsed:>6.1f}s elapsed  "
                    f"{rate:>7.0f} samples/s  ETA {eta:.0f}s"
                )
        evolve_time = time.time() - evolve_start

    # ── Phase 3: Correlation matrix (vectorised) ──────────────────────
    print(f"\nComputing {n}×{n} correlation matrix...")
    corr_start = time.time()
    corr_matrix = vectorised_corrmatrix(attractors)
    corr_time = time.time() - corr_start
    print(f"  Done in {corr_time:.1f}s")

    # Stats (upper triangle only, exclude diagonal)
    off_diag = corr_matrix[np.triu_indices(n, k=1)]
    avg_corr = float(np.mean(np.abs(off_diag)))
    max_corr = float(np.max(np.abs(off_diag)))
    min_corr = float(np.min(np.abs(off_diag)))
    median_corr = float(np.median(np.abs(off_diag)))
    p95_corr = float(np.percentile(np.abs(off_diag), 95))
    p99_corr = float(np.percentile(np.abs(off_diag), 99))

    n_converged = states.count("CONVERGED")
    n_oscillating = states.count("OSCILLATING")
    n_chaotic = states.count("CHAOTIC")
    total_time = fetch_time + evolve_time + corr_time

    print(f"\n{'=' * 65}")
    print(f"  ULTRADATA-MATH DIVERSITY REPORT  (n={n:,})")
    print(f"{'=' * 65}")
    print(f"  Source:        openbmb/UltraData-Math L3-QA-Synthetic")
    print(f"  Samples:       {n:,}")
    print(f"  Fetch time:    {fetch_time:.1f}s")
    print(f"  Evolve time:   {evolve_time:.1f}s  ({n / evolve_time:.0f} samples/s)")
    print(f"  Corr time:     {corr_time:.1f}s")
    print(f"  Total time:    {total_time:.1f}s")
    print(f"  ─────────────────────────────────")
    print(f"  Converged:     {n_converged:,}/{n:,}")
    print(f"  Oscillating:   {n_oscillating:,}/{n:,}")
    print(f"  Chaotic:       {n_chaotic:,}/{n:,}")
    print(f"  Avg ticks:     {np.mean(ticks_list):.1f}")
    print(f"  ─────────────────────────────────")
    print(f"  Avg |r|:       {avg_corr:.6f}")
    print(f"  Median |r|:    {median_corr:.6f}")
    print(f"  P95 |r|:       {p95_corr:.6f}")
    print(f"  P99 |r|:       {p99_corr:.6f}")
    print(f"  Max |r|:       {max_corr:.6f}")
    print(f"  Min |r|:       {min_corr:.6f}")
    print(f"  Pairs:         {len(off_diag):,}")

    pass_avg = avg_corr < 0.5
    pass_max = max_corr < 0.85
    overall = pass_avg and pass_max
    print(f"\n  Avg < 0.5:     {'PASS ✓' if pass_avg else 'FAIL ✗'} ({avg_corr:.6f})")
    print(f"  Max < 0.85:    {'PASS ✓' if pass_max else 'FAIL ✗'} ({max_corr:.6f})")
    print(f"  Overall:       {'PASS ✓' if overall else 'FAIL ✗'}")
    print(f"{'=' * 65}\n")

    # ── Phase 4: Visual report ────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))

    # Correlation matrix (downsampled for display if n > 200)
    ax1 = fig.add_subplot(3, 2, 1)
    if n > 200:
        # Downsample for visual clarity
        step = n // 200
        display_corr = corr_matrix[::step, ::step]
        ax1.set_title(f"Correlation Matrix (every {step}th sample)")
    else:
        display_corr = corr_matrix
        ax1.set_title("Attractor Correlation Matrix")
    im = ax1.imshow(display_corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax1.set_xlabel("Memory Index")
    ax1.set_ylabel("Memory Index")
    fig.colorbar(im, ax=ax1, shrink=0.8)

    # Histogram
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.hist(off_diag, bins=100, edgecolor="none", alpha=0.8, color="#3B82F6")
    ax2.axvline(0.85, color="red", linestyle="--", linewidth=1.5, label="Max threshold (0.85)")
    ax2.axvline(-0.85, color="red", linestyle="--", linewidth=1.5)
    ax2.axvline(avg_corr, color="orange", linestyle="--", label=f"Avg |r| = {avg_corr:.4f}")
    ax2.set_title(f"Correlation Distribution ({len(off_diag):,} pairs)")
    ax2.set_xlim(-1, 1)
    ax2.set_xlabel("Pearson Correlation")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=9)

    # Sample attractors (first 8 converged)
    converged_indices = [i for i, s in enumerate(states) if s == "CONVERGED"]
    # Pick 8 evenly spaced from converged samples
    if len(converged_indices) >= 8:
        step = len(converged_indices) // 8
        show_indices = [converged_indices[i * step] for i in range(8)]
    else:
        show_indices = converged_indices[:8]

    for plot_idx, data_idx in enumerate(show_indices):
        ax = fig.add_subplot(3, 8, 9 + plot_idx)
        att = np.array(attractors[data_idx]).reshape(64, 64)
        roles = get_cell_roles(att)
        ax.imshow(roles, cmap=ROLE_CMAP, norm=ROLE_NORM, interpolation="nearest")
        label = labels[data_idx][:18]
        ax.set_title(f"#{data_idx}", fontsize=7)
        ax.axis("off")

    # State bar chart
    ax_bar = fig.add_subplot(3, 2, 5)
    state_names = ["CONVERGED", "OSCILLATING", "CHAOTIC"]
    state_counts = [n_converged, n_oscillating, n_chaotic]
    state_colors = ["#22C55E", "#F59E0B", "#EF4444"]
    bars = ax_bar.bar(state_names, state_counts, color=state_colors, edgecolor="black")
    ax_bar.set_title("State Distribution")
    ax_bar.set_ylabel("Count")
    for bar, count in zip(bars, state_counts):
        if count > 0:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(1, n * 0.005),
                f"{count:,}",
                ha="center", va="bottom", fontweight="bold", fontsize=9,
            )

    # Info panel
    ax_info = fig.add_subplot(3, 2, 6)
    ax_info.axis("off")
    info_text = (
        f"Dataset:     openbmb/UltraData-Math\n"
        f"Config:      L3-QA-Synthetic\n"
        f"Samples:     {n:,}\n"
        f"Offset:      {args.offset:,}\n"
        f"Pairs:       {len(off_diag):,}\n"
        f"\n"
        f"Avg |r|:     {avg_corr:.6f}\n"
        f"Median |r|:  {median_corr:.6f}\n"
        f"P95 |r|:     {p95_corr:.6f}\n"
        f"P99 |r|:     {p99_corr:.6f}\n"
        f"Max |r|:     {max_corr:.6f}\n"
        f"\n"
        f"Avg ticks:   {np.mean(ticks_list):.1f}\n"
        f"Fetch:       {fetch_time:.1f}s\n"
        f"Evolve:      {evolve_time:.1f}s\n"
        f"Correlation: {corr_time:.1f}s\n"
        f"Total:       {total_time:.1f}s"
    )
    ax_info.text(
        0.05, 0.95, info_text, transform=ax_info.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3F4F6", edgecolor="#D1D5DB"),
    )

    # Role legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#EF4444", edgecolor="black", label="Local Max (+1)"),
        Patch(facecolor="#9CA3AF", edgecolor="black", label="Slope (0)"),
        Patch(facecolor="#3B82F6", edgecolor="black", label="Local Min (−1)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=3,
        fontsize=10, frameon=True, fancybox=True, shadow=True,
    )

    plt.suptitle(
        f"Wheeler Memory × UltraData-Math  —  n={n:,}\n"
        f"Avg |r|={avg_corr:.4f},  Max |r|={max_corr:.4f}  —  "
        f"{'PASS ✓' if overall else 'FAIL ✗'}  |  "
        f"States: {n_converged:,}C / {n_oscillating:,}O / {n_chaotic:,}X",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(args.output, dpi=150)
    print(f"Saved visual report to {args.output}")


if __name__ == "__main__":
    main()
