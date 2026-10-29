# Changelog

## Unreleased
- Added the AO high-resolution discriminator on the existing consumer-retention runtime: K=50/K=100 spectral admission, five distinct eigenspace/topic/subtopic/fibre/goal levels, implementation-noise token rejection, above-MP-bulk-edge index selection, primary/subtopic threshold components, and deterministic complete-link fibres.
- Added deterministic full W=20 scheduling over unblocked Pareto-eligible goals with pairwise disjoint typed support; the scheduler fails closed when twenty independent goals are unavailable.
- Added a literal finite Max Cut audit with boundary vertices and exact ±1 Laplacian identity `x^T L x = 4 * cut-edge count`; optional SDP upper bounds are validation receipts only and do not create physics authority.
- Added focused tests and CI for the discriminator while preserving the existing collision/retention and vertex-map orbit renderer.
- Local isolated replay of the new discriminator and finite hierarchy logic passed. Exact GitHub-branch pytest status is delegated to the new focused workflow because this runtime cannot resolve github.com for a clone.
- Added Brain-IT cross-modal consumer-retention and exact finite commutation audits (toy diagnostics only), tests, and an evidence-gated roadmap; no real NSD or learned-model result asserted.
- Added DASHI sprint documentation (`README`, formal axioms, dataset and gauge specs, sprint plan) and TODOs for sprint 1–2.
- Scaffolded Python package structure (`dashi/`) with loaders, baseline residuals, valuation init, kernel operator/flow modules.
- Added lightweight unit tests for kernel determinism and k-hop expansion.
- Added `scripts/defect_curve.py` CLI to run baseline → ternary init → kernel flow and emit timestamped defect curves (CSV/PNG).
- Added `notebooks/sprint-01-02.ipynb` for interactive runs, aligned with CLI outputs.
- Documented Sprint 2.5 hemibrain run (sparse DC, hops=1, deadzone=1e-9) with saved `s_final`, defect nodes, and partition summaries; CLI now supports metadata ID override and reports entropy/purity per partition.
- Documented Sprint 3 geometry analysis and gauge sweeps: neutral/defect distance to +1 components, confirmation that the “8 components” reduce to one dominant domain plus tiny satellites, and invariance under deadzone/baseline changes.
- Added random-block coarse-grain utility (`scripts/renormalize_random_blocks.py`) and sweeps (k=50/100/200, seeds 0/1) showing coarse graphs collapse to single +1 components with zero or one-step defect resolution.
- Added degree-binned coarse-grain utility (`scripts/renormalize_degree_binned.py`) and initial run (k=100) also collapsing to a single +1 component, reinforcing that locality drives the original neutral/defect geometry.
- Added ROI coarse-grain utility (`scripts/renormalize_roi.py`) and run showing biological ROI aggregation collapses defect geometry to a single +1 domain (defect=0 in one step); noted that sparsity/geometric structure is sub-ROI.
- Added hop-radius coarse-grain utility (`scripts/renormalize_hop_radius.py`) with r=1,2 runs collapsing defect to zero in one step, further localizing constraints to finer-than-radius neighborhoods.
- Added voxel coarse-grain utility (`scripts/renormalize_voxel.py`); current metadata lacks soma coords so voxel run is degenerate until a coordinate-rich export is produced.
- Clarified formal regimes (atomic → affine → nonlinear → exploded) in `docs/formal_axioms.md`, noting that exploded structure only persists when locality is preserved.
