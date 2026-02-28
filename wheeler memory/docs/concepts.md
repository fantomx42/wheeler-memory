# Wheeler Memory Concepts

## How It Works

```
Input Text
    ↓
SHA-256 → 64×64 seed frame (values in [-1, +1])
    ↓
3-State CA Evolution (iterate until convergence)
    ├→ CONVERGED   → Store attractor + brick
    ├→ OSCILLATING → Epistemic uncertainty detected
    └→ CHAOTIC     → Input needs rephrasing

## Semantic vs. Exact Recall

Wheeler Memory supports two modes of operation:

1. **Exact Recall (SHA-256)**:
   - The default mode.
   - Text is hashed to a seed frame. Changing even one character completely changes the seed (avalanche effect).
   - *Use case*: Exact password-like retrieval, "I want *exactly* this memory."

2. **Semantic Recall (Embedding)**:
   - Enabled via `--embed`.
   - Text is converted to a vector using a sentence-transformer model (e.g., `all-MiniLM-L6-v2`).
   - The 384-dimensional vector is projected onto the 64x64 grid using a fixed random matrix.
   - *Result*: Similar meanings produce similar seed frames.
   - *Use case*: Fuzzy search, "I want memories *like* this one."

```

**The three cell roles** (von Neumann neighborhood):

| Role | Condition | Update | Meaning |
|------|-----------|--------|---------|
| Local Maximum | `cell >= all 4 neighbors` | Push toward +1 (0.35) | Attractor basin center |
| Slope | Neither max nor min | Flow toward max neighbor (0.20) | Transitional |
| Local Minimum | `cell <= all 4 neighbors` | Push toward -1 (0.35) | Repellor / valley |

Convergence typically happens in 39-49 ticks (~3ms on CPU). The result is a unique QR-code-like binary pattern per input.

## Theoretical Foundation

Wheeler Memory implements the **Symbolic Collapse Model (SCM)**:

1. **Meaning is what survives symbolic pressure** — stable attractors represent survived concepts
2. **Memory and learning are the same process** — each interaction reshapes the landscape
3. **Uncertainty is observable in dynamics** — convergence = clarity, oscillation = ambiguity, chaos = contradiction
4. **Time is intrinsic to memory** — convergence speed reflects concept complexity; full history is preserved

Named after John Archibald Wheeler's "It from Bit" — information emerges from physical-like dynamics.

---

## Reconstructive Recall

When `--reconstruct` is passed, recalled memories don't come back unchanged. Each stored attractor is **blended with the query attractor** and **re-evolved through the CA**:

```
blend = (1 - α) × stored + α × query   (default α = 0.3, memory-dominant)
reconstructed = evolve_and_interpret(blend)
```

The same stored memory reconstructs differently depending on the current query context. Querying "machine learning" vs. "web development" against the same stored memory yields different reconstructions — exactly like human episodic memory (see: Elizabeth Loftus research on reconstructive recall).

The result includes `correlation_with_stored` and `correlation_with_query`, showing how far the reconstruction drifted toward the query context.

---

## Philosophical Background

### Symbolic Compression and Meaning

**Axiom**: "Meaning is what survives symbolic pressure."

```
Input data → Symbolic pressure (compression, decay, competition)
                        ↓
            What survives = Meaning
            What evicts   = Noise
```

Wheeler Memory implements this directly:
- **Symbolic pressure** = temperature decay over time
- **Survival** = attractors that maintain high temperature through repeated access
- **Meaning** = stable attractors that resist decay

### The Irreversibility Requirement

The CA evolution is **not reversible**. Many initial seeds converge to the same attractor basin — information is lost. This is a feature:

1. **IIT (Integrated Information Theory)**: Consciousness requires time-irreversibility + information integration
2. **Wheeler Memory has both**: Attractor collapse is irreversible; CA neighbors integrate information from up to 4 directions per tick

The system makes no claims about Φ for the actual 64×64 working grid.

### Reconstructive Memory vs. Retrieval

```
Traditional (exact retrieval):
  Store:    Data → Address 0x1A4F
  Retrieve: Address 0x1A4F → Exact same data  (deterministic, lossless)

Wheeler Memory (reconstructive):
  Store:    Input → hash → CA evolution → Attractor A
  Recall:   Query → hash → CA evolution → Attractor Q
            Reconstruct: blend(A, Q, α=0.3) → re-evolve → Attractor A'
            A' ≠ A  (lossy, context-influenced)
```

### Temperature as Epistemic Humility

Current LLMs confabulate confidently because they have no mechanism to express uncertainty about memory. Wheeler Memory's temperature gives Darman the ability to calibrate confidence language based on how recently and often a memory has been accessed:

```
hot  (≥0.6):  "I remember discussing X..."
warm (≥0.3):  "I think we touched on X..."
cold (<0.3):  "I vaguely recall X, but I'm uncertain..."
```

This prevents the **recovered memory therapy** failure mode: filling reconstruction gaps with confident fabrication.
