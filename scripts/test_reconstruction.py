"""Reconstructive recall demo — same memory, different contexts, different reconstructions.

Demonstrates the Darman architecture: stored memories are reconstructed differently
depending on the current query context.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.reconstruction import reconstruct


def plot_attractor(ax, attractor, title, subtitle=""):
    """Plot a 64x64 attractor with tri-color cell roles."""
    grid = attractor.reshape(64, 64)

    # Classify cells by role
    display = np.zeros_like(grid)
    for r in range(64):
        for c in range(64):
            val = grid[r, c]
            neighbors = [
                grid[(r - 1) % 64, c],
                grid[(r + 1) % 64, c],
                grid[r, (c - 1) % 64],
                grid[r, (c + 1) % 64],
            ]
            if val >= max(neighbors):
                display[r, c] = 1.0   # local max → red
            elif val <= min(neighbors):
                display[r, c] = -1.0  # local min → blue
            else:
                display[r, c] = 0.0   # slope → gray

    cmap = ListedColormap(["#3B82F6", "#9CA3AF", "#EF4444"])
    ax.imshow(display, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=10, fontweight="bold")
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def pearson_r(a, b):
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    norm = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / norm) if norm > 0 else 0.0


def main():
    # ── Define the experiment ─────────────────────────────────────
    stored_text = "Python is a versatile programming language used for web development data science and automation"
    contexts = [
        ("machine learning neural networks deep learning", "ML context"),
        ("web server flask django http api", "Web context"),
        ("raspberry pi gpio hardware automation", "Hardware context"),
    ]
    alphas = [0.0, 0.15, 0.3, 0.5, 0.7]

    # ── Evolve base memory ────────────────────────────────────────
    print("Evolving base memory...")
    stored_frame = hash_to_frame(stored_text)
    stored_result = evolve_and_interpret(stored_frame)
    stored_att = stored_result["attractor"]
    print(f"  Stored: {stored_result['state']} in {stored_result['convergence_ticks']} ticks")

    # ── Evolve context queries ────────────────────────────────────
    context_results = []
    for ctx_text, ctx_name in contexts:
        frame = hash_to_frame(ctx_text)
        result = evolve_and_interpret(frame)
        context_results.append((ctx_text, ctx_name, result))
        print(f"  {ctx_name}: {result['state']} in {result['convergence_ticks']} ticks")

    # ── Reconstruct at different alphas ───────────────────────────
    print("\nReconstructing...")
    reconstructions = {}  # (ctx_name, alpha) → recon_result
    for ctx_text, ctx_name, ctx_result in context_results:
        for alpha in alphas:
            recon = reconstruct(stored_att, ctx_result["attractor"], alpha=alpha)
            reconstructions[(ctx_name, alpha)] = recon
            print(f"  {ctx_name} α={alpha:.2f}: "
                  f"r→stored={recon['correlation_with_stored']:.4f}, "
                  f"r→query={recon['correlation_with_query']:.4f}, "
                  f"{recon['state']} in {recon['convergence_ticks']} ticks")

    # ── Cross-context comparison ──────────────────────────────────
    print("\nCross-context divergence at α=0.3:")
    alpha_test = 0.3
    for i, (_, name_a, _) in enumerate(context_results):
        for j, (_, name_b, _) in enumerate(context_results):
            if j <= i:
                continue
            att_a = reconstructions[(name_a, alpha_test)]["attractor"]
            att_b = reconstructions[(name_b, alpha_test)]["attractor"]
            r = pearson_r(att_a, att_b)
            print(f"  {name_a} vs {name_b}: r={r:.4f}")

    # ── Visual report ─────────────────────────────────────────────
    n_ctx = len(contexts)
    n_alpha = len(alphas)
    fig, axes = plt.subplots(
        n_ctx + 1, n_alpha + 1,
        figsize=(3 * (n_alpha + 1), 3 * (n_ctx + 1)),
    )

    fig.suptitle(
        "Reconstructive Recall — Darman Architecture\n"
        f"Same memory, different contexts, different reconstructions",
        fontsize=14, fontweight="bold",
    )

    # Row 0: stored memory repeated + alphas as headers
    plot_attractor(axes[0, 0], stored_att, "STORED MEMORY",
                   f"'{stored_text[:40]}...'")
    for j, alpha in enumerate(alphas):
        axes[0, j + 1].axis("off")
        axes[0, j + 1].text(
            0.5, 0.5, f"α = {alpha}",
            ha="center", va="center", fontsize=14, fontweight="bold",
            transform=axes[0, j + 1].transAxes,
        )
        if alpha == 0.0:
            axes[0, j + 1].text(0.5, 0.3, "(pure memory)",
                                ha="center", va="center", fontsize=9,
                                color="#666", transform=axes[0, j + 1].transAxes)
        elif alpha == 0.3:
            axes[0, j + 1].text(0.5, 0.3, "(default)",
                                ha="center", va="center", fontsize=9,
                                color="#22C55E", fontweight="bold",
                                transform=axes[0, j + 1].transAxes)

    # Rows 1+: each context × each alpha
    for i, (ctx_text, ctx_name, ctx_result) in enumerate(context_results):
        row = i + 1
        # Column 0: context attractor
        plot_attractor(axes[row, 0], ctx_result["attractor"],
                       f"CONTEXT: {ctx_name}",
                       f"'{ctx_text[:35]}...'")

        # Columns 1+: reconstructions
        for j, alpha in enumerate(alphas):
            recon = reconstructions[(ctx_name, alpha)]
            r_stored = recon["correlation_with_stored"]
            r_query = recon["correlation_with_query"]
            plot_attractor(
                axes[row, j + 1], recon["attractor"],
                f"r→S={r_stored:.3f}  r→Q={r_query:.3f}",
                f"{recon['state']} ({recon['convergence_ticks']}t)",
            )

    plt.tight_layout()
    output = "docs/assets/reconstruction_demo.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
