"""MemoryBrick: temporal history of a memory's formation.

A brick captures the full evolution timeline of a CA memory --
every tick from initial seed to final attractor. This enables
visual debugging, failure analysis, and audit trails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class MemoryBrick:
    """Complete temporal record of memory formation."""

    evolution_history: list[np.ndarray]
    final_attractor: np.ndarray
    convergence_ticks: int
    state: str  # 'CONVERGED' | 'OSCILLATING' | 'CHAOTIC'
    metadata: dict = field(default_factory=dict)

    def get_frame_at_tick(self, n: int) -> np.ndarray:
        """Retrieve the frame at tick n."""
        return self.evolution_history[n]

    def find_divergence_point(self) -> int | None:
        """Find the tick where oscillation began (for oscillating bricks).

        Scans backwards from the end to find where the pattern started
        repeating. Returns None if the brick converged normally.
        """
        if self.state != "OSCILLATING":
            return None

        period = self.metadata.get("cycle_period", 2)
        n = len(self.evolution_history)

        # Walk backwards from the end to find where periodicity started
        from .oscillation import get_cell_roles

        for t in range(n - period - 1, 0, -1):
            roles_t = get_cell_roles(self.evolution_history[t])
            roles_tp = get_cell_roles(self.evolution_history[t + period])
            if not np.array_equal(roles_t, roles_tp):
                return t + 1
        return 0

    def save(self, filepath: str | Path) -> None:
        """Save brick to a single .npz file.

        Stores all frames as a stacked array plus metadata as JSON string.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        stacked = np.stack(self.evolution_history)
        np.savez_compressed(
            filepath,
            history=stacked,
            attractor=self.final_attractor,
            convergence_ticks=np.array(self.convergence_ticks),
            state=np.array(self.state),
            metadata_json=np.array(json.dumps(self.metadata)),
        )

    @classmethod
    def load(cls, filepath: str | Path) -> MemoryBrick:
        """Load a brick from a .npz file."""
        data = np.load(filepath, allow_pickle=False)
        history = [data["history"][i] for i in range(data["history"].shape[0])]
        metadata = json.loads(str(data["metadata_json"]))
        return cls(
            evolution_history=history,
            final_attractor=data["attractor"],
            convergence_ticks=int(data["convergence_ticks"]),
            state=str(data["state"]),
            metadata=metadata,
        )

    @classmethod
    def from_evolution_result(cls, result: dict, extra_metadata: dict | None = None) -> MemoryBrick:
        """Construct a brick from an evolve_and_interpret result dict."""
        metadata = dict(result.get("metadata", {}))
        if extra_metadata:
            metadata.update(extra_metadata)
        return cls(
            evolution_history=result["history"],
            final_attractor=result["attractor"],
            convergence_ticks=result["convergence_ticks"],
            state=result["state"],
            metadata=metadata,
        )
