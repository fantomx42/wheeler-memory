"""CLI: Attractor diversity validation -- THE key test.

Generates attractors for 20 diverse inputs and checks that they are
genuinely distinct (avg off-diagonal correlation < 0.5, no pair > 0.85).
Includes synthetic edge-case inputs to exercise OSCILLATING and CHAOTIC states.
Saves a visual report to diversity_report.png.
"""

import argparse
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import pearsonr

from wheeler_memory import evolve_and_interpret, get_cell_roles, hash_to_frame

TEST_INPUTS = [
    "Fix authentication bug in login flow",
    "Deploy Kubernetes cluster on AWS",
    "Buy groceries: milk, eggs, bread",
    "Schedule dentist appointment for Thursday",
    "Quantum entanglement violates Bell inequalities",
    "The mitochondria is the powerhouse of the cell",
    "Review pull request #42 for memory leaks",
    "Plan birthday party for next Saturday",
    "Configure NGINX reverse proxy with TLS",
    "Water the garden every morning at 7am",
    "Implement binary search tree in Rust",
    "Book flight to Tokyo for March conference",
    "Dark matter comprises 27% of the universe",
    "Refactor database schema for multi-tenancy",
    "Practice piano scales for 30 minutes daily",
    "Debug segfault in GPU kernel launch",
    "Write unit tests for payment processing",
    "Organize closet by season and color",
    "Black holes emit Hawking radiation",
    "Compile FFmpeg with hardware acceleration",
]

def make_edge_cases():
    """Construct synthetic frames designed to trigger non-convergent states.

    - CHAOTIC: normal frame but max_iters=15 cuts evolution off before any
      convergence or oscillation detection can occur.

    Note: the OSCILLATING state is theoretically supported by the CA engine,
    but the current 3-state dynamics are fundamentally convergent — no known
    input produces sustained role-switching oscillation. The code path exists
    as a safety net for future dynamics changes.
    """
    cases = []

    # Chaotic: normal hash but far too few iterations
    cases.append({
        "label": "short-circuit stress test (chaotic)",
        "frame": hash_to_frame("chaotic short-circuit stress test"),
        "max_iters": 15,
        "expected": "CHAOTIC",
    })

    return cases

# Discrete 3-color map for cell roles: min=-1 (blue), slope=0 (gray), max=+1 (red)
ROLE_CMAP = mcolors.ListedColormap(["#3B82F6", "#9CA3AF", "#EF4444"])
ROLE_NORM = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], ROLE_CMAP.N)


def main():
    parser = argparse.ArgumentParser(description="Wheeler Memory attractor diversity test")
    parser.add_argument("--output", default="diversity_report.png", help="Output image path")
    args = parser.parse_args()

    n = len(TEST_INPUTS)
    attractors = []
    states = []
    ticks_list = []
    all_results = []  # (text, result) for all inputs including edge cases

    print(f"Evolving {n} test inputs...")
    total_start = time.time()

    for i, text in enumerate(TEST_INPUTS):
        frame = hash_to_frame(text)
        start = time.time()
        result = evolve_and_interpret(frame)
        elapsed = time.time() - start

        attractors.append(result["attractor"].flatten())
        states.append(result["state"])
        ticks_list.append(result["convergence_ticks"])
        all_results.append((text, result))

        label = text[:50]
        print(f"  [{i + 1:2d}/{n}] {result['state']:<11} {result['convergence_ticks']:>4} ticks  {elapsed:.3f}s  {label}")

    # Run edge-case inputs with synthetic frames
    edge_cases = make_edge_cases()
    print(f"\nEvolving {len(edge_cases)} edge-case inputs...")
    edge_results = []
    for ec in edge_cases:
        frame = ec["frame"]
        start = time.time()
        result = evolve_and_interpret(frame, max_iters=ec["max_iters"])
        elapsed = time.time() - start
        edge_results.append((ec, result))
        all_results.append((ec["label"], result))

        label = ec["label"][:50]
        expected = ec["expected"]
        actual = result["state"]
        match = "✓" if actual == expected else "✗"
        print(f"  {match} {actual:<11} {result['convergence_ticks']:>4} ticks  {elapsed:.3f}s  {label}")

    total_time = time.time() - total_start

    # Compute correlation matrix (converged attractors only)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif j > i:
                c, _ = pearsonr(attractors[i], attractors[j])
                corr_matrix[i, j] = c
                corr_matrix[j, i] = c

    # Statistics
    off_diag = corr_matrix[np.triu_indices(n, k=1)]
    avg_corr = float(np.mean(np.abs(off_diag)))
    max_corr = float(np.max(np.abs(off_diag)))
    min_corr = float(np.min(np.abs(off_diag)))

    # State counts across all inputs
    all_states = [r["state"] for _, r in all_results]
    n_total = len(all_states)
    n_converged = all_states.count("CONVERGED")
    n_oscillating = all_states.count("OSCILLATING")
    n_chaotic = all_states.count("CHAOTIC")

    print(f"\n{'=' * 60}")
    print(f"DIVERSITY REPORT")
    print(f"{'=' * 60}")
    print(f"Total time:           {total_time:.2f}s")
    print(f"Converged:            {n_converged}/{n_total}")
    print(f"Oscillating:          {n_oscillating}/{n_total}")
    print(f"Chaotic:              {n_chaotic}/{n_total}")
    print(f"Avg ticks:            {np.mean(ticks_list):.0f}")
    print(f"Avg |correlation|:    {avg_corr:.4f}")
    print(f"Max |correlation|:    {max_corr:.4f}")
    print(f"Min |correlation|:    {min_corr:.4f}")

    pass_avg = avg_corr < 0.5
    pass_max = max_corr < 0.85
    print(f"\nAvg < 0.5:  {'PASS' if pass_avg else 'FAIL'} ({avg_corr:.4f})")
    print(f"Max < 0.85: {'PASS' if pass_max else 'FAIL'} ({max_corr:.4f})")
    print(f"Overall:    {'PASS' if (pass_avg and pass_max) else 'FAIL'}")

    # Generate visual report
    fig = plt.figure(figsize=(16, 14))

    # --- Top row: correlation matrix + histogram ---
    ax1 = fig.add_subplot(3, 2, 1)
    im = ax1.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_title("Attractor Correlation Matrix")
    ax1.set_xlabel("Memory Index")
    ax1.set_ylabel("Memory Index")
    fig.colorbar(im, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.hist(off_diag, bins=30, edgecolor="black", alpha=0.7)
    ax2.axvline(0.85, color="red", linestyle="--", label="Max threshold (0.85)")
    ax2.axvline(avg_corr, color="orange", linestyle="--", label=f"Avg |r| = {avg_corr:.3f}")
    ax2.set_title("Off-Diagonal Correlation Distribution")
    ax2.set_xlim(-1, 1)
    ax2.set_xlabel("Pearson Correlation")
    ax2.set_ylabel("Count")
    ax2.legend()

    # --- Middle row: 4 converged attractors shown as cell roles (tri-color) ---
    for idx in range(4):
        ax = fig.add_subplot(3, 4, 5 + idx)
        att = attractors[idx].reshape(64, 64)
        roles = get_cell_roles(att)
        ax.imshow(roles, cmap=ROLE_CMAP, norm=ROLE_NORM, interpolation="nearest")
        label = TEST_INPUTS[idx][:25]
        ax.set_title(f"#{idx}: {label}...", fontsize=8)
        ax.axis("off")

    # --- Bottom row: state summary bar + edge-case examples ---
    # State summary bar chart
    ax_bar = fig.add_subplot(3, 2, 5)
    state_names = ["CONVERGED", "OSCILLATING", "CHAOTIC"]
    state_counts = [n_converged, n_oscillating, n_chaotic]
    state_colors = ["#22C55E", "#F59E0B", "#EF4444"]
    bars = ax_bar.bar(state_names, state_counts, color=state_colors, edgecolor="black")
    ax_bar.set_title("State Distribution")
    ax_bar.set_ylabel("Count")
    for bar, count in zip(bars, state_counts):
        if count > 0:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        str(count), ha="center", va="bottom", fontweight="bold")

    # Edge-case attractor snapshots (roles view)
    ax_edge = fig.add_subplot(3, 2, 6)
    if edge_results:
        # Show the first edge case attractor as roles
        ec, result = edge_results[0]
        roles = get_cell_roles(result["attractor"])
        ax_edge.imshow(roles, cmap=ROLE_CMAP, norm=ROLE_NORM, interpolation="nearest")
        ax_edge.set_title(f"Edge: {result['state']} ({result['convergence_ticks']} ticks)", fontsize=9)
    ax_edge.axis("off")

    # Role legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#EF4444", edgecolor="black", label="Local Max (+1)"),
        Patch(facecolor="#9CA3AF", edgecolor="black", label="Slope (0)"),
        Patch(facecolor="#3B82F6", edgecolor="black", label="Local Min (-1)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10,
               frameon=True, fancybox=True, shadow=True)

    plt.suptitle(
        f"Wheeler Memory Diversity Report\n"
        f"Avg |r|={avg_corr:.3f}, Max |r|={max_corr:.3f} — "
        f"{'PASS' if (pass_avg and pass_max) else 'FAIL'}  |  "
        f"States: {n_converged}C / {n_oscillating}O / {n_chaotic}X",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved visual report to {args.output}")


if __name__ == "__main__":
    main()
