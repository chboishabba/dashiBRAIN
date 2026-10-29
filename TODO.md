# TODO (Sprint 1–2 prototype)

## Sprint 1 — ingestion & baseline
- [x] Implement hemibrain edge loader to CSR with id mapping and optional metadata (coords, cell type).
- [x] Decide baseline expectation (start with degree-corrected); expose hook for distance-aware baseline.
- [x] Compute residual adjacency and sanity-check distribution (sign balance report).
- [x] Define default deadzone value and document rationale.

## Sprint 2 — kernel core (single scale)
- [x] Implement k-hop neighborhood expansion (sparse) and ensure non-negative weights.
- [x] Implement kernel operator with channel coupling hook and ternary deadzone.
- [x] Implement kernel flow with defect trace, convergence/cycle detection, deterministic behavior.
- [x] Add toy-graph tests for locality, involution symmetry (where applicable), and determinism.

## Reports / diagnostics
- [x] Add quick script/notebook to plot defect curve for toy data (see `scripts/defect_curve.py`; outputs defect curve as text/CSV).
- [x] Record baseline and deadzone choices once hemibrain data is wired (sparse DC, deadzone=1e-9, hops=1).
- [ ] Run notebook `notebooks/sprint-01-02.ipynb` with real data and paste summary into `docs/sprint-01-02.md` (optional; CLI run already captured).

## Sprint 2.5 — closed-state inspection (completed)
- [x] Persist `s_final` and per-step defect nodes from hemibrain run.
- [x] Emit partition summaries (entropy/purity) for available metadata (`type`, `instance`).
- [x] Record convergence/flip/component metrics in `docs/sprint-01-02.md`.

## Sprint 3
- [x] Analyze defect/neutral geometry: distance of neutral + defect nodes to +1 components; identify boundary structure.
- [x] Test gauge robustness: rerun hemibrain with varied deadzone (1e-6, 1e-4) and baseline (`raw`) to see if ~8 components persist.

## Sprint 4 — coarse-grain / renormalization
- [x] Implement random block aggregation to produce a coarse hemibrain edge list (deterministic seed, configurable block size).
- [x] Run kernel defect curve on the coarse graph (same gauges) and record component counts/neutral fractions vs baseline.
- [x] Sweep random block sizes/seeds and log coarse defect curves/component counts; note scale at which neutral/defect structure disappears.
- [x] Add degree-binned aggregation variant to test degree-preserving coarse-grain.
- [x] Add ROI aggregation (metadata-driven) and show defect collapse under biologically meaningful coarse blocks.
- [x] Add hop-radius aggregation to test minimal locality preservation; log collapse thresholds (r=1,2 tested).
- [ ] Add voxel aggregation using soma/synapse coordinates once exported; rerun defect curves (blocked: missing coords in current metadata export).


## AO fleet discriminator — 2026-10-29

- [x] Reject hexadecimal literals and memory-address noise before topic admission.
- [x] Reject raw UI and DOM tokens before topic admission.
- [x] Accept only K=50 or K=100 spectral truncation receipts.
- [x] Keep eigenspace, primary topic, subtopic, fibre, and goal as separate runtime levels.
- [x] Require lambda_i > lambda_plus for selected spectral basis rows.
- [x] Build a deterministic full W=20 schedule from top Pareto-ranked unblocked goals.
- [x] Require pairwise disjoint typed support for all concurrent goals.
- [x] Fail loudly when twenty compatible unblocked goals do not exist.
- [x] Add deterministic finite Max Cut diagnostics.
- [x] Check x^T L x = 4 * cut-edge count on undirected unweighted fixtures.
- [x] Keep optional SDP upper-bound receipts non-promoting.
- [x] Preserve existing collision/retention and vertex-map renderer behavior.
- [ ] Run focused tests and append the changelog only after implementation.

- [x] Construct primary topics from the declared primary-distance threshold graph.
- [x] Construct subtopics at the tighter local threshold inside primary topics.
- [x] Construct deterministic complete-link fibres at the tightest threshold.
