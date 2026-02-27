# CLI Tools Reference

## Store a memory

```bash
wheeler-store "fix the python debug error"
# Chunk:    code (auto)
# State:    CONVERGED
# Ticks:    43
# Rotation: 0° (attempt 1)
# Time:     0.003s
# Memory stored successfully.

wheeler-store --chunk hardware "solder the GPIO header"   # explicit chunk
echo "piped input" | wheeler-store -                       # stdin
wheeler-store --embed "fuzzy memory"                       # store with semantic embedding
wheeler-store --salience high "critical insight"           # deep attractor (3000 iters, 1e-6 threshold)
wheeler-store --salience low "background note"             # fast store (200 iters, 5e-4 threshold)
```

## Recall memories

```bash
wheeler-recall "python bug"
# Rank  Similarity  Chunk        State        Ticks  Text
# ----------------------------------------------------------------------------------
# 1        0.0145  code         CONVERGED       43  fix the python debug error
# ...

wheeler-recall --chunk code "debug error"   # search specific chunk
wheeler-recall --top-k 10 "something"       # more results
wheeler-recall --embed "debugging issues"   # fuzzy semantic search
wheeler-recall --salience high "important query"  # more CA patience for query evolution
```

### Salience levels

The `--salience` flag controls how much computational attention a store or recall operation receives:

| Level | max_iters | threshold | Use case |
|-------|-----------|-----------|----------|
| `low` | 200 | 5e-4 | Bulk ingestion, background notes |
| `medium` | 1000 | 1e-4 | Default (omitting `--salience` is the same) |
| `high` | 3000 | 1e-6 | Important memories, deep attractors |

When omitted, salience defaults to `medium` (backwards compatible). For reconstruction, if no explicit salience is given, hot memories automatically get more attention based on their temperature.

### Temperature-boosted recall

```bash
wheeler-recall --temperature-boost 0.1 "python bug"
```

When `--temperature-boost` is nonzero, ranking uses `similarity + boost × temperature` — hotter memories get a slight ranking bonus. Default boost is 0.0 (pure similarity ranking, identical to previous behavior).

## Scrub a brick timeline

```bash
wheeler-scrub --text "fix the python debug error"           # find by text
wheeler-scrub --text "solder header" --chunk hardware       # in specific chunk
wheeler-scrub path/to/brick.npz                              # direct path
```

Opens an interactive matplotlib viewer with a tick slider.

## Diversity report

```bash
wheeler-diversity
# Evolves 20 diverse test inputs, computes pairwise correlations.
# PASS when avg correlation < 0.5 and max < 0.85.
```

## GPU benchmark

```bash
wheeler-bench-gpu                                # CPU vs GPU comparison
wheeler-bench-gpu --verify-only                  # correctness check only
wheeler-bench-gpu --batch-sizes 100,500,2000     # custom sizes
```

## Large-scale diversity (UltraData-Math)

```bash
wheeler-diversity-math --n 1000                  # 1K samples (CPU)
wheeler-diversity-math --n 1000 --gpu            # 1K samples (GPU batch)
```

## Inspect temperatures

```bash
wheeler-temps                     # all memories
wheeler-temps --chunk code        # specific chunk
wheeler-temps --tier hot          # filter by tier
wheeler-temps --sort hits         # sort by hit count
```

## Forget / evict memories

```bash
wheeler-forget                      # full sweep (fade + evict + capacity)
wheeler-forget --dry-run            # show what would happen
wheeler-forget --text "some memory" # forget a specific memory
wheeler-forget --hex abc123...      # forget by hex key
wheeler-forget --coldest 10         # diagnostic: show 10 coldest memories
```

The sweep runs three phases:

1. **Fade** — memories below temperature 0.05 have their brick (`.npz` evolution history) deleted. The attractor and index entry remain, so the memory is still recallable but its formation history is lost.
2. **Evict** — memories below temperature 0.01 are fully removed (attractor, index entry, association edges, warmth).
3. **Capacity** — if the total memory count exceeds 10,000, the bottom 10% by temperature are evicted (never warm or hot memories).

Memories younger than 1 day are never evicted regardless of temperature.

## Trauma encoding

```bash
# Store a trauma pair (experience + avoidance)
wheeler-trauma store "the car crash" --avoidance "driving on highways"
# Trauma pair stored.
#   Pair ID:     a1b2c3d4e5f6
#   Experience:  abc123...  (chunk: general)
#   Avoidance:   def456...  (chunk: general)

# List all trauma pairs
wheeler-trauma list

# Show therapy status for a pair
wheeler-trauma status a1b2c3d4e5f6

# Record a safe exposure (therapy)
wheeler-trauma expose a1b2c3d4e5f6

# Unlink a pair (keeps memories)
wheeler-trauma remove a1b2c3d4e5f6
```

### Trauma-aware recall

```bash
# Default: avoidance co-fires automatically
wheeler-recall "car crash"

# Safe exposure mode (therapy): no avoidance injection, records safe exposure
wheeler-recall --trauma-mode safe "car crash"

# Skip trauma processing entirely
wheeler-recall --trauma-mode suppress "car crash"
```

When a recalled memory is the experience half of a trauma pair, the avoidance memory is automatically injected into results (marked `[AVOIDANCE]`). The `--trauma-mode` flag controls this:

| Mode | Avoidance injection | Counters updated |
|------|---------------------|------------------|
| `auto` (default) | Yes, if suppression > floor | activation_count incremented |
| `safe` | No | safe_exposure_count incremented, suppression decays |
| `suppress` | No | No changes |

Therapy mechanics: after 3 safe exposures, each additional safe exposure decays suppression by 15% multiplicatively. Suppression never reaches zero (floor = 0.05).

## Sleep consolidation

```bash
wheeler-sleep                      # consolidate all eligible memories
wheeler-sleep --dry-run            # show what would be consolidated
wheeler-sleep --chunk code         # consolidate specific chunk
wheeler-sleep --stats              # show per-memory frame counts + potential savings
```

Consolidation prunes redundant intermediate frames *within* each brick, keeping only salient keyframes where the pattern changed significantly. Frame 0 (seed) and the final frame (attractor) are always preserved.

Thresholds are temperature-tiered:

- **Hot (>=0.6)** — skipped entirely (actively used memories)
- **Warm (>=0.3)** — light pruning (~40-60% frames retained)
- **Cold (<0.3)** — aggressive pruning (~15-25% frames retained)

Already-consolidated bricks are skipped (idempotent). This is distinct from eviction — consolidation reduces storage while preserving the formation story at key transition points.
