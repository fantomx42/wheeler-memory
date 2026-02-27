"""Lichtenberg Visualization — visualize Wheeler's attractor topology.

Existing frames as terminal nodes sized by basin width, brightness by hit count.
Query seed as the ground point. CA propagation paths as branching lines.
Synthesized candidate frames as dotted nodes.

Uses PCA via numpy SVD for 2D projection — no external dependency beyond
matplotlib + numpy.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Project N-dimensional vectors to 2D via PCA (SVD).

    Args:
        vectors: (n_samples, n_features) array

    Returns:
        (n_samples, 2) projected coordinates
    """
    centered = vectors - vectors.mean(axis=0)
    if centered.shape[0] < 2:
        return np.zeros((centered.shape[0], 2))
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:2].T


def plot_topology(
    attractors: dict[str, np.ndarray],
    basin_widths: dict[str, float],
    hit_counts: dict[str, int],
    query_attractor: np.ndarray = None,
    candidates: list[np.ndarray] = None,
    output_path: Path = None,
) -> Figure:
    """Static Lichtenberg topology figure.

    - PCA projects attractors to 2D
    - Nodes sized by basin_width, brightness by hit_count
    - Query seed as highlighted ground point
    - Lines from query to recalled attractors
    - Dotted nodes for synthesized candidates
    """
    keys = list(attractors.keys())
    if not keys:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No attractors to plot", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    # Flatten all attractors for PCA
    flat_vectors = np.array([attractors[k].flatten() for k in keys])

    # Add query and candidates to projection if present
    extra_vectors = []
    extra_labels = []
    if query_attractor is not None:
        extra_vectors.append(query_attractor.flatten())
        extra_labels.append("query")
    if candidates:
        for i, c in enumerate(candidates):
            extra_vectors.append(c.flatten())
            extra_labels.append(f"candidate_{i}")

    if extra_vectors:
        all_vectors = np.vstack([flat_vectors, np.array(extra_vectors)])
    else:
        all_vectors = flat_vectors

    coords_2d = _pca_2d(all_vectors)
    attractor_coords = coords_2d[:len(keys)]

    # Normalize sizes and colors
    widths = np.array([basin_widths.get(k, 0.1) for k in keys])
    sizes = 100 + widths / (widths.max() + 1e-8) * 500

    counts = np.array([hit_counts.get(k, 0) for k in keys], dtype=float)
    max_count = counts.max() if counts.max() > 0 else 1
    alphas = 0.3 + 0.7 * (counts / max_count)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    # Attractor nodes
    for i, k in enumerate(keys):
        ax.scatter(
            attractor_coords[i, 0], attractor_coords[i, 1],
            s=sizes[i], c=[[0.4, 0.7, 1.0, alphas[i]]],
            edgecolors="white", linewidths=0.5, zorder=3,
        )
        label = k[:8]
        ax.annotate(
            label, (attractor_coords[i, 0], attractor_coords[i, 1]),
            textcoords="offset points", xytext=(5, 5),
            fontsize=7, color="white", alpha=0.6,
        )

    # Query ground point
    idx = len(keys)
    if query_attractor is not None:
        qx, qy = coords_2d[idx, 0], coords_2d[idx, 1]
        ax.scatter(qx, qy, s=200, c="yellow", marker="*", zorder=5, label="Query")
        # Draw lines from query to attractors
        for i in range(len(keys)):
            ax.plot(
                [qx, attractor_coords[i, 0]], [qy, attractor_coords[i, 1]],
                color="yellow", alpha=0.15, linewidth=0.5, zorder=1,
            )
        idx += 1

    # Candidate nodes (dotted)
    if candidates:
        for i in range(len(candidates)):
            cx, cy = coords_2d[idx + i, 0], coords_2d[idx + i, 1]
            ax.scatter(
                cx, cy, s=150, facecolors="none", edgecolors="lime",
                linewidths=1.5, linestyles="dashed", zorder=4, label="Candidate" if i == 0 else None,
            )

    ax.set_title("Wheeler Attractor Topology (Lichtenberg)", color="white", fontsize=14)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    if query_attractor is not None or candidates:
        ax.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white", fontsize=9)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return fig


def animate_apple_test(
    test_result: dict,
    output_path: Path,
) -> None:
    """Animate the apple test progression.

    Frames:
    1. Existing attractors as nodes
    2. Gap detection highlighted
    3. Synthesized candidate appears (dotted)
    4. Holdout text introduced
    5+. Convergence/dissolution/drift trajectory

    Saves as .gif.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory = test_result.get("trajectory", [])
    candidate = test_result.get("candidate")
    holdout_attractor = test_result.get("holdout_attractor")
    verdict = test_result.get("verdict", "unknown")

    # We need at least the trajectory
    n_frames = max(5, min(len(trajectory), 30))
    step = max(1, len(trajectory) // n_frames) if trajectory else 1

    fig, (ax_main, ax_traj) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0a0a0a")
    ax_main.set_facecolor("#0a0a0a")
    ax_traj.set_facecolor("#0a0a0a")

    # Trajectory plot (right panel)
    ax_traj.set_title("Correlation with Candidate", color="white", fontsize=11)
    ax_traj.set_xlabel("CA Step", color="white", fontsize=9)
    ax_traj.set_ylabel("Pearson r", color="white", fontsize=9)
    ax_traj.tick_params(colors="white", labelsize=8)
    for spine in ax_traj.spines.values():
        spine.set_color("#333333")
    ax_traj.set_xlim(0, len(trajectory) or 1)
    if trajectory:
        ax_traj.set_ylim(min(trajectory) - 0.05, max(trajectory) + 0.05)

    line, = ax_traj.plot([], [], color="cyan", linewidth=1.5)

    # Main panel: simple state indicator
    ax_main.set_xlim(-1, 1)
    ax_main.set_ylim(-1, 1)
    ax_main.set_aspect("equal")
    for spine in ax_main.spines.values():
        spine.set_color("#333333")
    ax_main.tick_params(colors="white", labelsize=8)

    status_text = ax_main.text(
        0, 0.8, "", ha="center", va="center", color="white", fontsize=14,
        transform=ax_main.transAxes,
    )
    verdict_text = ax_main.text(
        0.5, 0.2, "", ha="center", va="center", fontsize=18, fontweight="bold",
        transform=ax_main.transAxes,
    )

    def init():
        line.set_data([], [])
        status_text.set_text("")
        verdict_text.set_text("")
        return line, status_text, verdict_text

    def animate(frame_idx):
        actual_step = frame_idx * step

        if frame_idx == 0:
            status_text.set_text("Phase 1: Training attractors stored")
            verdict_text.set_text("")
        elif frame_idx == 1:
            status_text.set_text("Phase 2: Gap detection")
            verdict_text.set_text("")
        elif frame_idx == 2:
            status_text.set_text("Phase 3: Candidate synthesized")
            verdict_text.set_text("")
        elif frame_idx == 3:
            status_text.set_text("Phase 4: Holdout exposed")
            verdict_text.set_text("")
        else:
            status_text.set_text(f"Phase 5: Evolution (step {actual_step})")
            # Update trajectory
            end = min(actual_step, len(trajectory))
            if end > 0:
                line.set_data(range(end), trajectory[:end])

            if frame_idx >= n_frames - 1:
                colors = {"convergence": "lime", "dissolution": "orange", "hallucination": "red"}
                verdict_text.set_text(f"Verdict: {verdict.upper()}")
                verdict_text.set_color(colors.get(verdict, "white"))

        return line, status_text, verdict_text

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=500, blit=True)
    anim.save(str(output_path), writer=PillowWriter(fps=2))
    plt.close(fig)
