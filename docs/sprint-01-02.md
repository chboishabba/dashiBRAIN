# Sprint 1–2 (solo prototype)

Path: **Sprint 1–2 solo prototype** — build ingestion, baseline residuals, initial ternary field, and single-scale kernel flow.

## Objectives
- Load hemibrain edge data into a sparse carrier with optional metadata (coords, labels).
- Compute baseline expectations (degree- or distance-aware) and residual adjacency.
- Ternarise residuals into an initial valuation field `s0` with deadzone diagnostics.
- Implement kernel operator and flow (single scale), including defect tracking.

## Deliverables
- `io/hemibrain_loader.py`: edge list → CSR matrix (+ optional metadata hooks).
- `baseline/residuals.py`: baseline expectation + residual computation.
- `valuation/init_field.py`: ternary init from residuals with deadzone reports.
- `kernel/operator.py`: neighborhood construction + kernel step.
- `kernel/flow.py`: iteration, defect computation, convergence/cycle detection.
- `tests/` smoke checks for determinism/involution on toy graphs.
- Brief report (in this file) on initial defect curves for toy data (TBD once data wired).

## Exit criteria
- Changing baseline parameters predictably changes `s0` (documented).
- Kernel flow returns `(s_t, history)` with reproducible defect curve on toy graphs.
- Involution/locality invariants hold in tests for the toy carrier.
- Coarse-graining remains out-of-scope for this sprint.

## Data input (hemibrain edges)
- Export once via neuPrint (`neuprint-python fetch_adjacencies`) using dataset `hemibrain:v1.2.1`, `min_total_weight=1`, and write to `data/hemibrain/edges.csv`.
- Validate before loading: `python scripts/validate_edges.py data/hemibrain/edges.csv` (checks headers, non-negative weights).
- Optional partitions: align metadata to IDs and expose labels with `--partition-field cell_type` on `scripts/defect_curve.py`.
- Record gauges when you run: dataset version, `min_total_weight`, deadzone (default `1e-9`), node/edge counts, initial sign balance, and defect curve behavior (converged/cycle/flat).

## Data snapshot (2026-XX-XX)
- neuPrint fetch completed: dataset `hemibrain:v1.2.1`, `min_total_weight=1`.
- Files: `data/hemibrain/edges.csv` (8,034,510 rows; CSR collapses to 7,084,254 unique weighted edges over 179,907 nodes), `data/hemibrain/neurons.csv` (metadata from neuPrint).
- Validation: `python scripts/validate_edges.py data/hemibrain/edges.csv` → zero negative/zero/missing weights; comma delimiter.
- Baseline options: `--baseline sparse_dc` (edge-local degree-corrected, default), `--baseline dense_dc` (toy graphs only; disabled >50k nodes), `--baseline raw` (diagnostic gauge).
- Note: k-hop expansion beyond `--hops 1` will densify; keep `--hops 1` for hemibrain-scale runs until a non-densifying neighborhood op is added.
- Full run (sparse baseline, hops=1, steps=5, deadzone=1e-9): N=179,907, edges=7,084,254; residual sign balance {+1: 7,066,918, 0: 0, -1: 17,336}; kernel converged in 4 steps with defect curve 633 → 42 → 2 → 0. Outputs: `outputs/defect_curve_*.csv/.png`.
- Closed-state artifacts: `outputs/defect_curve_*_state.csv` (node index, body_id, s_final); `outputs/defect_curve_*_defect_nodes.csv` (per-step changed nodes); optional partition summaries `outputs/defect_curve_*_partition_<field>.csv`. CLI prints flips count, final sign balance, and component counts for +1/-1 subgraphs (undirected).

## Constraints and gauges
- Kernel semantics must follow definitions in `docs/formal_axioms.md`.
- No learned weights; channel coupling is explicit matrix input.
- Reproducibility favored over performance; sparse operations preferred but clarity wins.

## Open questions (track actively)
- Preferred baseline: degree-corrected vs distance-aware (no decision yet).
- Deadzone size: default small epsilon vs adaptive per-degree.
- Neighborhood definition: k-hop expansion vs precomputed kNN (current default: simple k-hop multiply).
- Metadata availability: coordinates/cell-type fields to include in loader schema.

## Checklist
- [ ] Document baseline choice for hemibrain once data is available (default: degree-corrected; distance-aware optional).
- [ ] Document deadzone default after first residual diagnostics (current working default: `1e-9`).
- [x] Add minimal CLI or notebook for plotting defect curves (`scripts/defect_curve.py`).

## Current decisions
- Baseline: edge-local degree-corrected expectation (`baseline/sparse_degree_corrected.py`); distance-aware hook to be added once coordinate schema is locked. Dense baseline retained only for small toy graphs.
- Deadzone: `1e-9` symmetric ternary deadzone; adjust after residual diagnostics on hemibrain.

## Notebook
- `notebooks/sprint-01-02.ipynb` runs the same pipeline as the CLI and saves timestamped CSV/PNG defect curves.

## Sprint 2.5 — closed-state inspection
Goal: characterise the kernel-closed valuation `s*` (hops=1) before altering gauges or neighborhoods.

### Objectives
- Persist terminal state from `scripts/defect_curve.py` runs (`s_final` per node + per-step defect nodes).
- Quantify what changed: flips from `s0`→`s*`, final sign balance, component counts for +1/-1 subgraphs.
- Join `s*` with `data/hemibrain/neurons.csv` to tabulate distribution/entropy per metadata field (cell type, region, etc.).
- Identify defect-resolving boundaries: nodes in defect at each step of the run.

### Deliverables
- CLI emits CSV artifacts: `outputs/defect_curve_*_state.csv` (node index, body_id, `s_final`), `outputs/defect_curve_*_defect_nodes.csv` (per-step defect membership), and optional `outputs/defect_curve_*_partition_<field>.csv` (metadata counts/purity).
- CLI stdout reports flips count, final sign balance, and undirected component counts for +1/-1 subgraphs for reproducibility.

### Exit criteria
- Running `python scripts/defect_curve.py` on hemibrain with `--hops 1` saves the above artifacts and prints the diagnostics without extra notebooks.
- Defect boundary nodes per step are captured for inspection; no new hops>1/densification work begins until this is done.

## Sprint 2.5 results (hemibrain, sparse DC, hops=1, deadzone=1e-9)
- Run: `outputs/defect_curve_20260120-142843*` (CLI with `--metadata data/hemibrain/neurons.csv`, partitions `type`, `instance`).
- Convergence: 4 steps; defect curve 633 → 42 → 2 → 0; status converged.
- Flips: 677 nodes changed between `s0` and `s_final`.
- Final sign balance: +1 = 137,901; 0 = 42,006; -1 = 0.
- Components (induced +1 subgraph): 8 components, largest = 137,879; no surviving -1 components.
- Defect-resolving nodes: 677 total; steps {1: 633, 2: 42, 3: 2, 4: 0}; very sparse metadata overlap (only two labeled types: OCG09, LC14).
- Neutral layer (value = 0): 42,006 nodes; sparse labeling (top counts: LC14=10, DNES1=4, OCG09=3; most neutral nodes unlabeled).
- Metadata coverage: `type` labels present for ~22.7k of 180k nodes; `instance` labels similar; purity extremes reflect small labeled subsets, not bulk structure.
