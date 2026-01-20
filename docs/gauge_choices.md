# Gauge choices (admissible)

These choices are permitted gauges; anything else requires explicit justification.

- **Baseline expectation**: degree-corrected configuration by default; distance-aware variant allowed if documented.
- **Deadzone**: small epsilon (default `1e-9`); alternative per-degree deadzone allowed if recorded alongside diagnostics.
- **Edge inclusion**: hemibrain export via neuPrint with `min_total_weight=1` is the default gauge for Sprint 01/02; higher thresholds (e.g., 3 or 5) are allowed only with a recorded rationale and side-by-side diagnostics.
- **Neighborhood**: k-hop adjacency expansion with non-negative weights; kNN or radius neighborhoods allowed if deterministic and documented.
- **Channel coupling**: explicit coupling matrix provided by config; no learned coupling.
- **Symmetries**: relabeling invariance of nodes; channel permutations only if coupling respects the symmetry.
- **Randomness**: avoid; if used (e.g., partition seeds), seed must be fixed and recorded.
