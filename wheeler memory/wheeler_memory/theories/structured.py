"""Structured Theory Output — hand the LLM a structured description of what Wheeler knows.

Builds a Theory object from recalled memories, basin widths, and temperature,
then converts it to an LLM prompt that instructs the model to express the theory
rather than reason independently.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..dynamics import evolve_and_interpret
from ..hashing import hash_to_frame
from ..storage import DEFAULT_DATA_DIR, list_memories, recall_memory
from ..temperature import effective_temperature
from .basin import measure_basin_width


@dataclass
class TheoryFrame:
    hex_key: str
    text: str
    temperature: float
    basin_width: float
    confidence: float        # temperature * basin_width (normalized)
    relationships: list[str] = field(default_factory=list)  # hex_keys of associated frames


@dataclass
class TheoryHypothesis:
    synthesized_from: list[str]
    description: str
    confidence: float
    status: str = "untested"  # "untested" | "converged" | "dissolved"


@dataclass
class Theory:
    active_frames: list[TheoryFrame] = field(default_factory=list)
    hypotheses: list[TheoryHypothesis] = field(default_factory=list)
    context_budget: dict[str, float] = field(default_factory=dict)  # hex_key → fraction


def build_theory(
    query: str,
    top_k: int = 5,
    data_dir: Path = None,
) -> Theory:
    """Recall memories, compute basin widths, build structured theory.

    Context budget = (hit_count * basin_width) / sum(all), normalized to 1.0.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    # Recall relevant memories
    recalled = recall_memory(query, top_k=top_k, data_dir=data_dir)
    if not recalled:
        return Theory()

    # Build theory frames with basin widths
    frames = []
    raw_weights = {}

    for mem in recalled:
        hex_key = mem["hex_key"]
        chunk = mem["chunk"]

        # Load attractor for basin width measurement
        att_path = data_dir / "chunks" / chunk / "attractors" / f"{hex_key}.npy"
        if not att_path.exists():
            continue

        attractor = np.load(att_path)
        # Use lightweight basin measurement (few probes)
        bw_result = measure_basin_width(attractor, n_probes=10, steps=5)
        bw = bw_result["width"]

        temp = mem.get("temperature", 0.0)
        confidence = temp * bw if bw > 0 else 0.0

        frame = TheoryFrame(
            hex_key=hex_key,
            text=mem["text"],
            temperature=temp,
            basin_width=bw,
            confidence=confidence,
        )
        frames.append(frame)

        # Weight for context budget: hit_count proxy via temperature * basin_width
        hit_count = max(1, int(temp * 10))  # Approximate from temperature
        raw_weights[hex_key] = hit_count * bw

    # Normalize context budget
    total_weight = sum(raw_weights.values())
    context_budget = {}
    if total_weight > 0:
        context_budget = {k: v / total_weight for k, v in raw_weights.items()}
    elif frames:
        # Equal distribution if all weights are zero
        equal = 1.0 / len(frames)
        context_budget = {f.hex_key: equal for f in frames}

    return Theory(
        active_frames=frames,
        hypotheses=[],
        context_budget=context_budget,
    )


def theory_to_prompt(theory: Theory) -> str:
    """Convert Theory to an LLM prompt string.

    Instructs the LLM to express the theory derived from memory,
    not to reason independently.
    """
    lines = [
        "You are expressing a theory derived from memory, not reasoning independently.",
        "The following concepts are active in memory with varying levels of confidence.",
        "",
    ]

    if theory.active_frames:
        lines.append("Active concepts:")
        for frame in theory.active_frames:
            budget = theory.context_budget.get(frame.hex_key, 0.0)
            lines.append(
                f"  - \"{frame.text}\" "
                f"(confidence: {frame.confidence:.3f}, "
                f"temperature: {frame.temperature:.3f}, "
                f"basin width: {frame.basin_width:.3f}, "
                f"context share: {budget:.1%})"
            )
        lines.append("")

    if theory.hypotheses:
        lines.append("Hypotheses (predicted but not directly observed):")
        for hyp in theory.hypotheses:
            lines.append(
                f"  - {hyp.description} "
                f"(confidence: {hyp.confidence:.3f}, status: {hyp.status}, "
                f"synthesized from: {', '.join(hyp.synthesized_from)})"
            )
        lines.append("")

    lines.append(
        "Allocate your response proportionally to the context budget above. "
        "Do not introduce concepts not present in this theory."
    )

    return "\n".join(lines)
