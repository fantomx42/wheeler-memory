"""Paraphrase similarity test: do duplicate questions get higher Wheeler correlation?

Tests the OPPOSITE of the diversity test — whether Wheeler Memory can detect
that semantically similar inputs are related. Spoiler: SHA-256 hashing destroys
semantic similarity, so we expect both duplicate and random pairs to have
equally low correlation.

This is an honest test of the system's limitations.
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

from wheeler_memory import hash_to_frame, evolve_and_interpret
from wheeler_memory import gpu_available, gpu_evolve_batch


def main():
    parser = argparse.ArgumentParser(
        description="Wheeler Memory paraphrase similarity test (Quora QQP)"
    )
    parser.add_argument(
        "--n", type=int, default=5000, help="Number of pairs to test"
    )
    parser.add_argument(
        "--output", default="paraphrase_report.png", help="Output image path"
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU batch evolution"
    )
    args = parser.parse_args()

    use_gpu = args.gpu and gpu_available()
    if args.gpu and not gpu_available():
        print("Warning: --gpu requested but unavailable, falling back to CPU")

    # ── Phase 1: Load dataset ─────────────────────────────────────────
    print("Loading GLUE QQP dataset...")
    t0 = time.time()
    ds = load_dataset("nyu-mll/glue", "qqp", split="train")
    print(f"  Loaded {len(ds):,} pairs in {time.time() - t0:.1f}s")

    # Separate duplicates and non-duplicates
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
    print(f"  Selected {n_dup:,} duplicate pairs, {n_non:,} non-duplicate pairs")

    # ── Phase 2: Evolve all unique questions ──────────────────────────
    # Collect all unique questions
    all_texts = set()
    for q1, q2 in dup_pairs + non_pairs:
        all_texts.add(q1)
        all_texts.add(q2)
    all_texts = list(all_texts)
    print(f"\n  Unique questions: {len(all_texts):,}")

    backend = "GPU" if use_gpu else "CPU"
    print(f"  Evolving through Wheeler CA ({backend})...")
    t0 = time.time()

    # Build attractor lookup
    attractor_map = {}

    if use_gpu:
        frames = [hash_to_frame(t) for t in all_texts]
        results = gpu_evolve_batch(frames)
        for text, result in zip(all_texts, results):
            attractor_map[text] = result["attractor"].flatten()
    else:
        for i, text in enumerate(all_texts):
            frame = hash_to_frame(text)
            result = evolve_and_interpret(frame)
            attractor_map[text] = result["attractor"].flatten()
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print(f"    [{i+1:>6}/{len(all_texts)}] {elapsed:.1f}s")

    evolve_time = time.time() - t0
    print(f"  Done in {evolve_time:.1f}s ({len(all_texts) / evolve_time:.0f} q/s)")

    # ── Phase 3: Compute pairwise correlations ────────────────────────
    print("\n  Computing pairwise correlations...")

    def pearson_r(a, b):
        a = a - a.mean()
        b = b - b.mean()
        norm = np.sqrt((a**2).sum() * (b**2).sum())
        if norm == 0:
            return 0.0
        return float((a * b).sum() / norm)

    dup_corrs = []
    for q1, q2 in dup_pairs:
        r = pearson_r(attractor_map[q1], attractor_map[q2])
        dup_corrs.append(abs(r))

    non_corrs = []
    for q1, q2 in non_pairs:
        r = pearson_r(attractor_map[q1], attractor_map[q2])
        non_corrs.append(abs(r))

    # Also compute random pairs for baseline
    rng = np.random.default_rng(42)
    rand_indices = rng.choice(len(all_texts), size=(args.n, 2), replace=True)
    rand_corrs = []
    for i, j in rand_indices:
        if i == j:
            continue
        r = pearson_r(
            attractor_map[all_texts[i]],
            attractor_map[all_texts[j]],
        )
        rand_corrs.append(abs(r))

    dup_corrs = np.array(dup_corrs)
    non_corrs = np.array(non_corrs)
    rand_corrs = np.array(rand_corrs)

    # ── Phase 4: Report ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  PARAPHRASE SIMILARITY TEST  (Quora QQP)")
    print(f"{'='*65}")
    print(f"  Duplicate pairs:      {n_dup:,}")
    print(f"  Non-duplicate pairs:  {n_non:,}")
    print(f"  Random pairs:         {len(rand_corrs):,}")
    print(f"  Unique questions:     {len(all_texts):,}")
    print(f"  Evolve time:          {evolve_time:.1f}s ({backend})")
    print(f"  ─────────────────────────────────")
    print(f"  Duplicate avg |r|:    {dup_corrs.mean():.6f}")
    print(f"  Non-dup avg |r|:      {non_corrs.mean():.6f}")
    print(f"  Random avg |r|:       {rand_corrs.mean():.6f}")
    print(f"  ─────────────────────────────────")
    print(f"  Dup median |r|:       {np.median(dup_corrs):.6f}")
    print(f"  Non-dup median |r|:   {np.median(non_corrs):.6f}")
    print(f"  Random median |r|:    {np.median(rand_corrs):.6f}")
    print(f"  ─────────────────────────────────")
    print(f"  Dup max |r|:          {dup_corrs.max():.6f}")
    print(f"  Non-dup max |r|:      {non_corrs.max():.6f}")
    print(f"  Random max |r|:       {rand_corrs.max():.6f}")

    # The key question: can Wheeler tell duplicates apart?
    separation = dup_corrs.mean() - rand_corrs.mean()
    print(f"\n  Separation (dup - random): {separation:+.6f}")

    if separation > 0.05:
        verdict = "DETECTS SIMILARITY"
        print(f"  Verdict: {verdict} — duplicates have meaningfully higher correlation")
    elif separation > 0.01:
        verdict = "WEAK SIGNAL"
        print(f"  Verdict: {verdict} — slight difference, not reliable")
    else:
        verdict = "NO SEMANTIC SIGNAL"
        print(f"  Verdict: {verdict} — SHA-256 hash destroys meaning, as expected")

    print(f"{'='*65}\n")

    # ── Phase 5: Visual report ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Wheeler Memory × Quora Question Pairs — {verdict}\n"
        f"Dup avg|r|={dup_corrs.mean():.4f}  Random avg|r|={rand_corrs.mean():.4f}  "
        f"Separation={separation:+.4f}",
        fontsize=14, fontweight="bold"
    )

    # 1. Overlapping histograms
    ax = axes[0, 0]
    bins = np.linspace(0, 0.15, 60)
    ax.hist(dup_corrs, bins=bins, alpha=0.6, label=f"Duplicates (n={n_dup:,})", color="#EF4444", density=True)
    ax.hist(non_corrs, bins=bins, alpha=0.6, label=f"Non-duplicates (n={n_non:,})", color="#3B82F6", density=True)
    ax.hist(rand_corrs, bins=bins, alpha=0.4, label=f"Random (n={len(rand_corrs):,})", color="#9CA3AF", density=True)
    ax.set_xlabel("|Pearson r|")
    ax.set_ylabel("Density")
    ax.set_title("Correlation Distribution by Pair Type")
    ax.legend()

    # 2. Box plots
    ax = axes[0, 1]
    bp = ax.boxplot(
        [dup_corrs, non_corrs, rand_corrs],
        labels=["Duplicates", "Non-duplicates", "Random"],
        patch_artist=True,
        widths=0.5,
    )
    colors = ["#EF4444", "#3B82F6", "#9CA3AF"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("|Pearson r|")
    ax.set_title("Correlation by Pair Type")

    # 3. Example pairs
    ax = axes[1, 0]
    ax.axis("off")
    ax.set_title("Example Duplicate Pairs", fontsize=12, fontweight="bold")
    examples = []
    for i in range(min(8, n_dup)):
        q1, q2 = dup_pairs[i]
        r = dup_corrs[i]
        examples.append(f"|r|={r:.4f}")
        examples.append(f"  Q1: {q1[:70]}")
        examples.append(f"  Q2: {q2[:70]}")
        examples.append("")
    ax.text(0.02, 0.98, "\n".join(examples), transform=ax.transAxes,
            fontsize=8, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#F3F4F6"))

    # 4. Stats panel
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("Analysis", fontsize=12, fontweight="bold")
    analysis = [
        f"Dataset:      GLUE QQP (Quora Question Pairs)",
        f"Total pairs:  {n_dup + n_non:,}",
        f"Backend:      {backend}",
        f"Evolve time:  {evolve_time:.1f}s",
        f"",
        f"── Means ──────────────────────",
        f"Dup avg |r|:    {dup_corrs.mean():.6f}",
        f"Non-dup avg:    {non_corrs.mean():.6f}",
        f"Random avg:     {rand_corrs.mean():.6f}",
        f"",
        f"── What this means ────────────",
        f"Wheeler Memory uses SHA-256 to",
        f"convert text → CA frame. This",
        f"is a one-way hash: paraphrases",
        f"get completely different hashes,",
        f"making semantic similarity",
        f"detection IMPOSSIBLE by design.",
        f"",
        f"This is NOT a bug — it's the",
        f"tradeoff for perfect diversity.",
        f"To fix: add an embedding layer",
        f"before the CA stage.",
    ]
    ax.text(0.02, 0.98, "\n".join(analysis), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#F3F4F6"))

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved visual report to {args.output}")


if __name__ == "__main__":
    main()
