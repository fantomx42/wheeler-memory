"""Wheeler Memory: cellular automata-based associative memory system."""

from .attention import (
    AttentionBudget,
    compute_attention_budget,
    salience_from_label,
    salience_from_temperature,
)
from .brick import MemoryBrick
from .chunking import (
    CHUNK_KEYWORDS,
    DEFAULT_CHUNK,
    find_brick_across_chunks,
    list_existing_chunks,
    select_chunk,
    select_recall_chunks,
)
from .dynamics import apply_ca_dynamics, evolve_and_interpret
from .hashing import hash_to_frame, text_to_hex
from .oscillation import detect_oscillation, get_cell_roles
from .rotation import store_with_rotation_retry
from .storage import list_memories, recall_memory, store_memory
from .temperature import (
    HALF_LIFE_DAYS,
    HIT_SATURATION,
    MAX_ATTRACTORS,
    MAX_WARMTH,
    TIER_DEAD,
    TIER_FADING,
    TIER_HOT,
    TIER_WARM,
    WARMTH_HALF_LIFE_DAYS,
    WARMTH_HOP1,
    WARMTH_HOP2,
    compute_temperature,
    compute_warmth,
    effective_temperature,
    temperature_tier,
)
from .eviction import (
    EvictionResult,
    forget_by_text,
    forget_memory,
    score_memories,
    sweep_and_evict,
)
from .consolidation import (
    ConsolidationResult,
    consolidate_brick,
    consolidation_stats,
    select_keyframes,
    sleep_consolidate,
)
from .warming import get_neighbors, load_associations

# GPU backend (optional — available only when a HIP .so is built)
try:
    from .gpu_dynamics import (
        gpu_available,
        gpu_evolve_batch,
        gpu_evolve_single,
        gpu_version,
        gpu_query_vram,
    )
except ImportError:
    gpu_available = lambda: False
    gpu_evolve_single = None
    gpu_evolve_batch = None
    gpu_version = lambda: None
    gpu_query_vram = lambda *_: None

# Embedding backend (optional — requires sentence-transformers)
try:
    from .embedding import embed_available, embed_to_frame, embed_to_frame_batch
except ImportError:
    embed_available = lambda: False
    embed_to_frame = None
    embed_to_frame_batch = None

# Reconstructive recall
from .reconstruction import reconstruct, reconstruct_batch

# LLM agent (optional — requires Ollama running locally)
from .agent import WheelerAgent

__all__ = [
    "hash_to_frame",
    "text_to_hex",
    "apply_ca_dynamics",
    "evolve_and_interpret",
    "get_cell_roles",
    "detect_oscillation",
    "MemoryBrick",
    "store_memory",
    "recall_memory",
    "list_memories",
    "store_with_rotation_retry",
    "CHUNK_KEYWORDS",
    "DEFAULT_CHUNK",
    "select_chunk",
    "select_recall_chunks",
    "find_brick_across_chunks",
    "list_existing_chunks",
    "compute_temperature",
    "temperature_tier",
    "HALF_LIFE_DAYS",
    "HIT_SATURATION",
    "MAX_ATTRACTORS",
    "MAX_WARMTH",
    "TIER_DEAD",
    "TIER_FADING",
    "TIER_HOT",
    "TIER_WARM",
    "WARMTH_HALF_LIFE_DAYS",
    "WARMTH_HOP1",
    "WARMTH_HOP2",
    "compute_warmth",
    "effective_temperature",
    "get_neighbors",
    "load_associations",
    # Eviction
    "sweep_and_evict",
    "forget_memory",
    "forget_by_text",
    "score_memories",
    "EvictionResult",
    # Consolidation
    "sleep_consolidate",
    "consolidate_brick",
    "select_keyframes",
    "consolidation_stats",
    "ConsolidationResult",
    # GPU (optional)
    "gpu_available",
    "gpu_evolve_single",
    "gpu_evolve_batch",
    "gpu_version",
    "gpu_query_vram",
    # Embedding (optional)
    "embed_available",
    "embed_to_frame",
    "embed_to_frame_batch",
    # Reconstructive recall
    "reconstruct",
    "reconstruct_batch",
    # Attention model
    "AttentionBudget",
    "compute_attention_budget",
    "salience_from_label",
    "salience_from_temperature",
    # LLM agent
    "WheelerAgent",
]

