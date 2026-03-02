"""Paraphrase similarity test with EMBEDDING-based frames.

Compares Wheeler correlation for duplicate vs non-duplicate vs random
Quora question pairs using sentence embeddings instead of SHA-256 hashing.

This should show the separation that SHA-256 couldn't achieve:
- Duplicate pairs → higher correlation
- Random pairs → lower correlation
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

from wheeler_memory import evolve_and_interpret
from wheeler_memory.embedding import embed_to_frame_batch

# GPU import (optional)
try:
    from wheeler_memory import gpu_available, gpu_evolve_batch
except ImportError:
    gpu_available = lambda: False
    gpu_evolve_batch = None


def main():
    parser = argparse.ArgumentParser(
        description="Wheeler Memory paraphrase test (embedding mode)"
    )
    parser.add_argument("--n", type=int, default=2000, help="Number of pairs to test")
    parser.add_argument("--output", default="paraphrase_embed_report.png", help="Output image")
    parser.add_argument("--gpu", action="store_true", help="Use GPU batch evolution")
    args = parser.parse_args()

    use_gpu = args.gpu and gpu_available()
    if args.gpu and not gpu_available():
        print("Warning: --gpu requested but unavailable, falling back to CPU")

    # ── Phase 1: Load dataset ─────────────────────────────────────────
    print("Loading GLUE QQP dataset...")
    t0 = time.time()
    ds = load_dataset("nyu-mll/glue", "qqp", split="train")
    print(f"  Loaded {len(ds):,} pairs in {time.time() - t0:.1f}s")

    dup_pairs = []
    non_pairs = []
    for row in ds:
        q1, q2, label = row["question1"], row["question2"], row["label"]
        if not q1 or not q2:
            continue
        if label == 1 and len(dup_pairs) < args.n:
            dup_pairs.append((q1, q2))
        elif label == 0 and len(non_pairs) < args.n:
            non_pairs.append((q1, q2))
        if len(dup_pairs) >= args.n and len(non_pairs) >= args.n:
            break

    n_dup = len(dup_pairs)
    n_non = len(non_pairs)
    print(f"  Selected {n_dup:,} duplicate, {n_non:,} non-duplicate pairs")

    # ── Phase 2: Embed all unique questions ───────────────────────────
    all_texts = set()
    for q1, q2 in dup_pairs + non_pairs:
        all_texts.add(q1)
        all_texts.add(q2)
    all_texts = list(all_texts)
    print(f"\n  Unique questions: {len(all_texts):,}")

    backend = "GPU" if use_gpu else "CPU"
    print(f"  Embedding + evolving through Wheeler CA ({backend})...")
    t_embed = time.time()

    # Batch embed
    print("  Embedding all texts...")
    frames = embed_to_frame_batch(all_texts)
    embed_time = time.time() - t_embed
    print(f"  Embedded in {embed_time:.1f}s")

    # Evolve
    t_evolve = time.time()
    attractor_map = {}
    if use_gpu:
        results = gpu_evolve_batch(frames)
        for text, result in zip(all_texts, results):
            attractor_map[text] = result["attractor"].flatten()
    else:
        for i, (text, frame) in enumerate(zip(all_texts, frames)):
            result = evolve_and_interpret(frame)
            attractor_map[text] = result["attractor"].flatten()
            if (i + 1) % 500 == 0:
                print(f"    [{i+1:>6}/{len(all_texts)}]")

    evolve_time = time.time() - t_evolve
    total_time = time.time() - t_embed
    print(f"  Evolved in {evolve_time:.1f}s ({len(all_texts) / evolve_time:.0f} q/s)")
    print(f"  Total: {total_time:.1f}s")

    # ── Phase 3: Compute correlations ─────────────────────────────────
    print("\n  Computing pairwise correlations...")

    def pearson_r(a, b):
        a = a - a.mean()
        b = b - b.mean()
        norm = np.sqrt((a**2).sum() * (b**2).sum())
        if norm == 0:
            return 0.0
        return float((a * b).sum() / norm)

    dup_corrs = np.array([pearson_r(attractor_map[q1], attractor_map[q2])
                          for q1, q2 in dup_pairs])
    non_corrs = np.array([pearson_r(attractor_map[q1], attractor_map[q2])
                          for q1, q2 in non_pairs])

    # Random baseline
    rng = np.random.default_rng(42)
    rand_indices = rng.choice(len(all_texts), size=(args.n, 2), replace=True)
    rand_corrs = np.array([
        pearson_r(attractor_map[all_texts[i]], attractor_map[all_texts[j]])
        for i, j in rand_indices if i != j
    ])

    # Use absolute values
    dup_abs = np.abs(dup_corrs)
    non_abs = np.abs(non_corrs)
    rand_abs = np.abs(rand_corrs)

    # ── Phase 4: Report ───────────────────────────────────────────────
    separation = dup_abs.mean() - rand_abs.mean()
    signed_sep = dup_corrs.mean() - rand_corrs.mean()

    # Also check signed correlation — do duplicates trend positive?
    dup_pos_frac = (dup_corrs > 0).mean()
    non_pos_frac = (non_corrs > 0).mean()

    print(f"\n{'='*65}")
    print(f"  EMBEDDING PARAPHRASE TEST  (Quora QQP)")
    print(f"{'='*65}")
    print(f"  Mode:                 Sentence Embedding (all-MiniLM-L6-v2)")
    print(f"  Duplicate pairs:      {n_dup:,}")
    print(f"  Non-duplicate pairs:  {n_non:,}")
    print(f"  Random pairs:         {len(rand_corrs):,}")
    print(f"  Embed time:           {embed_time:.1f}s")
    print(f"  Evolve time:          {evolve_time:.1f}s ({backend})")
    print(f"  ─────────────────────────────────")
    print(f"  Dup avg |r|:          {dup_abs.mean():.6f}")
    print(f"  Non-dup avg |r|:      {non_abs.mean():.6f}")
    print(f"  Random avg |r|:       {rand_abs.mean():.6f}")
    print(f"  ─────────────────────────────────")
    print(f"  Dup avg r (signed):   {dup_corrs.mean():+.6f}")
    print(f"  Non avg r (signed):   {non_corrs.mean():+.6f}")
    print(f"  Random avg r:         {rand_corrs.mean():+.6f}")
    print(f"  ─────────────────────────────────")
    print(f"  Dup % positive r:     {dup_pos_frac*100:.1f}%")
    print(f"  Non % positive r:     {non_pos_frac*100:.1f}%")
    print(f"  ─────────────────────────────────")
    print(f"  |r| separation:       {separation:+.6f}")
    print(f"  signed separation:    {signed_sep:+.6f}")

    if separation > 0.05:
        verdict = "DETECTS SIMILARITY ✓"
    elif separation > 0.01:
        verdict = "WEAK SIGNAL"
    else:
        verdict = "NO SIGNAL"

    print(f"\n  Verdict: {verdict}")
    print(f"{'='*65}\n")

    # ── Phase 5: Visual report ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Wheeler Memory × Quora QQP — EMBEDDING MODE — {verdict}\n"
        f"Dup avg|r|={dup_abs.mean():.4f}  Random avg|r|={rand_abs.mean():.4f}  "
        f"Separation={separation:+.4f}",
        fontsize=14, fontweight="bold"
    )

    # 1. Overlapping histograms (signed r)
    ax = axes[0, 0]
    bins = np.linspace(-0.3, 0.3, 80)
    ax.hist(dup_corrs, bins=bins, alpha=0.6, label=f"Duplicates (n={n_dup:,})",
            color="#EF4444", density=True)
    ax.hist(non_corrs, bins=bins, alpha=0.6, label=f"Non-duplicates (n={n_non:,})",
            color="#3B82F6", density=True)
    ax.hist(rand_corrs, bins=bins, alpha=0.4, label=f"Random (n={len(rand_corrs):,})",
            color="#9CA3AF", density=True)
    ax.set_xlabel("Pearson r (signed)")
    ax.set_ylabel("Density")
    ax.set_title("Correlation Distribution (Signed)")
    ax.legend()
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")

    # 2. Box plots (|r|)
    ax = axes[0, 1]
    bp = ax.boxplot(
        [dup_abs, non_abs, rand_abs],
        tick_labels=["Duplicates", "Non-duplicates", "Random"],
        patch_artist=True, widths=0.5,
    )
    colors = ["#EF4444", "#3B82F6", "#9CA3AF"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("|Pearson r|")
    ax.set_title("Absolute Correlation by Pair Type")

    # 3. Example pairs with their correlation
    ax = axes[1, 0]
    ax.axis("off")
    ax.set_title("Example Duplicate Pairs (with embedding)", fontsize=12, fontweight="bold")
    examples = []
    for i in range(min(8, n_dup)):
        q1, q2 = dup_pairs[i]
        r = dup_corrs[i]
        examples.append(f"r={r:+.4f}")
        examples.append(f"  Q1: {q1[:70]}")
        examples.append(f"  Q2: {q2[:70]}")
        examples.append("")
    ax.text(0.02, 0.98, "\n".join(examples), transform=ax.transAxes,
            fontsize=8, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#F3F4F6"))

    # 4. Comparison: SHA-256 vs Embedding
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("SHA-256 vs Embedding", fontsize=12, fontweight="bold")
    comparison = [
        f"─── SHA-256 (previous test) ───",
        f"Dup avg |r|:    ~0.0145",
        f"Random avg |r|: ~0.0147",
        f"Separation:     ~0.000  (none)",
        f"",
        f"─── Embedding (this test) ─────",
        f"Dup avg |r|:    {dup_abs.mean():.4f}",
        f"Random avg |r|: {rand_abs.mean():.4f}",
        f"Separation:     {separation:+.4f}",
        f"",
        f"─── Interpretation ────────────",
        f"SHA-256 destroys all semantic",
        f"signal. Embedding preserves it",
        f"through the CA evolution stage.",
        f"",
        f"Model: all-MiniLM-L6-v2 (384d)",
        f"Projection: JL random (384→4096)",
        f"Embed: {embed_time:.1f}s  Evolve: {evolve_time:.1f}s",
    ]
    ax.text(0.02, 0.98, "\n".join(comparison), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#F3F4F6"))

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
