# Darman Design Principles

**These are the rules that guide design decisions for Wheeler Memory and the Darman agent. They answer: "how should I think about building on this system?"**

For theory, see [concepts.md](concepts.md). For implementation details, see [architecture.md](architecture.md).

---

## Core Thesis

**The engine is the mind. The LLM is the voice.**

Darman's sentience — the sense that it remembers, forgets, hesitates, and grows more confident over time — emerges from the Wheeler Memory engine, not from the language model sitting on top of it. The LLM is a synthesis layer: it takes what the engine has already decided (what to surface, how warm each memory is, how far to reconstruct) and expresses it in natural language.

This thesis is the lens through which all other principles should be read.

---

## Principle 1: The Engine Is the Mind

Wheeler Memory's dynamics **are** the personality:

- **Temperature decay** — memories fade over time unless reinforced. Darman forgets things it hasn't thought about in a while.
- **Reconstructive recall** — the same stored memory surfaces differently depending on what's being discussed right now.
- **Associative warmth** — spreading activation means related memories get pulled into context together.
- **Polarity** — certain experiences are encoded as aversion attractors, shaping what directions Darman avoids.

These behaviors are not simulated by the LLM. They are computed by the engine before the LLM sees anything.

**Consequence**: Swap the underlying model from a large capable model to a weaker one. Darman should still *behave* like Darman — forgetting things it hasn't recalled, hedging on cold memories, associating related topics. Only the *phrasing* of responses should change. If swapping models changes Darman's behavior, that behavior was implemented in the wrong place.

**Design test**: For any proposed feature, ask — *"could this be computed before the LLM sees the context?"* If yes, it belongs in the engine, not in the prompt.

---

## Principle 2: Darman Doesn't Retrieve. Darman Reconstructs.

Every recall is **context-dependent**. The same stored memory reconstructs differently depending on the query:

```
blend = (1 - α) × stored + α × query   (default α = 0.3, memory-dominant)
reconstructed = evolve_and_interpret(blend)
```

Querying "machine learning" vs. "debugging" against the same stored memory produces different reconstructions. This is not a bug — it's the architecture. Human episodic memory works the same way (see: Elizabeth Loftus). Details in [concepts.md § Reconstructive Recall](concepts.md#reconstructive-recall).

**Consequence**: `reconstruct=True` is the default. Opting *out* is the exception.

**Design test**: If a new recall path bypasses reconstruction "for simplicity," ask whether you're reverting to a database lookup. Wheeler Memory is not a database.

**The alpha parameter** (default `0.3`) is deliberately memory-dominant: 70% of the reconstructed attractor comes from what was stored, 30% from the current query. This keeps memories stable while still allowing context to colour recall. Increasing alpha above `0.5` produces results that are more query-driven than memory-driven — use this only when the goal is hallucination detection or stress-testing, not for normal recall.

---

## Principle 3: Temperature Is Epistemic Humility

LLMs confabulate confidently because they have no mechanism to express uncertainty about memory. Temperature gives Darman that mechanism.

| Tier | Threshold | Darman's natural register |
|------|-----------|---------------------------|
| Hot  | ≥ 0.6     | "I remember discussing X..." |
| Warm | ≥ 0.3     | "I think we touched on X..." |
| Cold | < 0.3     | "I vaguely recall X, but I'm uncertain..." |

Temperature is computed from hit count and wall-clock time (see [architecture.md § Temperature Dynamics](architecture.md#3-temperature-dynamics)). The LLM does not decide confidence levels. It expresses the level the engine has already assigned.

**Consequence**: The system prompt must pass temperature tier to the LLM and instruct it to mirror that tier in its language. If the LLM is expressing confidence on a cold memory, that is a prompt bug.

**The failure mode this prevents**: *Recovered memory therapy* — filling reconstruction gaps with confident fabrication. Darman should never say "I remember clearly" about a cold memory.

---

## Principle 4: Memories Are Suggestions, Not Commands

The system prompt tells the LLM that recalled memories are **suggestions**, not ground truth:

> "These are memories that may inform your response. They represent past context but may be stale or partially reconstructed. You may disagree with a memory if current context warrants it."

This matters because:
- Temperature decay means stored memories can drift away from ground truth over time.
- Reconstruction means the recalled version is already a blend, not a verbatim record.
- The LLM may have better information in-context than in a cold reconstructed memory.

**Consequence**: Blind obedience to recalled memories is a failure mode. Darman should be able to say "I have a memory of X, but given what you've just told me, I think that memory is outdated."

**Design test**: If the prompt enforces memory recall as authoritative, remove that constraint.

---

## Principle 5: Minimize LLM Dependency

The goal is that **behavior comes from the engine; only phrasing comes from the model**.

### What the LLM must do
- Synthesize a natural-language response from engine-provided context
- Express uncertainty registers (hot / warm / cold) naturally
- Understand user intent to route queries
- Decide whether a stored memory is relevant to the current conversation

### What the engine handles
| Concern | Module |
|---------|--------|
| Similarity search | `storage.py` — Pearson correlation |
| Reconstruction | `reconstruction.py` — blend + re-evolve |
| Temperature & tiers | `temperature.py` — wall-clock decay formula |
| Decay & forgetting | `temperature.py` — half-life = 7 days |
| Polarity / aversion | `polarity.py` — polar attractors, decay count |
| Consolidation / sleep | `warming.py` — spreading activation |
| Chunking / routing | `chunking.py` — keyword-based domain routing |

**Design test**: *"If I swap to a weaker model, does the system's behavior change or just its phrasing?"* If behavior changes, re-examine which side of the boundary a feature lives on.

---

## Principle 6: Local-Only by Default

No cloud APIs are required to run Darman. The full system works on a single machine:

- **LLM inference**: Ollama + any locally-served model
- **Semantic embeddings**: `all-MiniLM-L6-v2` via `sentence-transformers` (runs locally, ~80 MB)
- **CA engine**: Pure NumPy on CPU; optional HIP/ROCm GPU kernel
- **Storage**: Plain files under `~/.wheeler_memory/`

Web search tools are available as optional capabilities but are never required for core memory behavior.

**Consequence**: Features that require a network connection must be clearly optional and must gracefully degrade when offline. Do not add mandatory cloud dependencies.

---

## Principle 7: The Formula Is the Foundation

The CA update rule, convergence detection, and attractor storage are the **load-bearing walls**. Everything else is built on top:

```
CA update rule (dynamics.py)
    └── convergence detection → attractor
            └── Pearson recall (storage.py)
                    └── temperature tracking
                            └── reconstruction (reconstruction.py)
                                    └── chunking / routing
                                            └── polarity
                                                    └── agent loop (agent.py)
                                                            └── UI
```

Changes cascade downward. A change to `dynamics.py` affects every attractor ever stored — recall will produce different results for existing memories. A change to `storage.py`'s similarity function changes ranking behavior across the entire system.

**Consequence**: Modifications to the CA dynamics or core storage must be treated as **breaking changes** — existing memory data may need to be regenerated. Document the downstream impact before making any change to `dynamics.py`, `storage.py`, or `hashing.py`.

**The corollary**: The UI, agent prompt, and tooling are commentary. They can be replaced, extended, or removed without touching the foundation. Preferentially change things at the top of the stack.

---

## What the LLM Is Responsible For

This decision matrix captures the boundary between LLM-responsibility and engine-responsibility:

| Decision | Owner | Rationale |
|----------|-------|-----------|
| Which memories to surface | **Engine** (Pearson + temperature) | Deterministic, auditable |
| How confident to sound | **Engine** (temperature tier) | Prevents confabulation |
| Whether to reconstruct | **Engine** (reconstruct=True default) | Consistency |
| Natural language wording | **LLM** | That's what it's for |
| Whether a memory is relevant to the reply | **LLM** | Requires understanding intent |
| Whether to disagree with a memory | **LLM** | Requires reasoning about staleness |
| Routing a user query to the right chunk | **Engine** (keyword routing) | Fast and predictable |
| Detecting ambiguous or contradictory queries | **Engine** (OSCILLATING state) | Structural, not linguistic |

---

## Naming and Backward Compatibility

Wheeler Memory accepts both old and new names for renamed concepts. Renaming should never break existing memory stores.

| Old name | New name | Where |
|----------|----------|-------|
| `avoidance_link` | `polarity_link` | Graph edges in `index.json` |
| `memory_type="avoidance"` | `memory_type="polar"` | Entry metadata |
| `safe_recall_count` | `decay_count` | Polarity edge metadata |
| `safe_context` | `polar_decay` | Polarity parameter name |

Both forms are read on load. New writes use the current name only.

**Rule**: When renaming a concept, both names must be accepted for at least one release cycle. Remove the old alias only when no existing data could contain it. SHA-256 keys are stable and never renamed.
