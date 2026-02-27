"""Wheeler Theories — compartmentalized experiments on CA attractor topology.

Each module is an isolated experiment that wraps around core Wheeler Memory
without modifying it. Import failures are non-fatal.
"""

try:
    from .basin import find_basin_gaps, map_all_basins, measure_basin_width
except ImportError:
    measure_basin_width = None
    find_basin_gaps = None
    map_all_basins = None

try:
    from .synthesis import apple_test, run_apple_battery, synthesize_from_gap
except ImportError:
    synthesize_from_gap = None
    apple_test = None
    run_apple_battery = None

try:
    from .structured import Theory, build_theory, theory_to_prompt
except ImportError:
    Theory = None
    build_theory = None
    theory_to_prompt = None

try:
    from .resonance import ResonanceResult, query_corpus
except ImportError:
    ResonanceResult = None
    query_corpus = None

try:
    from .metrics import (
        basin_width,
        context_weight,
        energy,
        hallucination_score,
        topology_consistency,
    )
except ImportError:
    energy = None
    basin_width = None
    context_weight = None
    hallucination_score = None
    topology_consistency = None

try:
    from .lichtenberg import animate_apple_test, plot_topology
except ImportError:
    plot_topology = None
    animate_apple_test = None
