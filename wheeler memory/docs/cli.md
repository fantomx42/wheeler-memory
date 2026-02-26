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
```

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
