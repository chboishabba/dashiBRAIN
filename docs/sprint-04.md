# Sprint 4 — coarse-grain / renormalization plan

Objective: test whether the dominant +1 domain, neutral shell, and defect pockets observed in Sprint 3 persist under admissible coarse-graining. Start with a non-anatomical null: random block aggregation.

## Method (intended)
- Load hemibrain edge list (`data/hemibrain/edges.csv`).
- Assign each node to a random block of size `k` (deterministic RNG seed; default `k=100` so ~1.8k blocks).
- Aggregate edges: for each source block `B_i` and target block `B_j`, sum weights of all edges crossing the two blocks (CSR-compatible, keep self-loop weights).
- Emit a coarse edge list CSV (block IDs as integers) suitable for `scripts/defect_curve.py` (`source_id`, `target_id`, `weight`).
- Run the existing defect-curve CLI on the coarse graph with the same gauges (hops=1, deadzone sweep as needed).
- Sweeps: vary block size and seed to find the scale where neutral/defect structure disappears; record defect curves and component counts per run.
- Variant (planned): degree-binned aggregation to test whether preserving degree classes retains neutral/defect structure.

## Degree-binned variant (planned)
- Compute node degree (in + out) on the original graph.
- Sort nodes deterministically by degree (tie-break by index) and group into consecutive bins of size `k` (target block size).
- Aggregate edges between bins as in the random-block method.
- Goal: preserve degree heterogeneity while discarding spatial locality; check whether neutral/defect geometry survives degree-aware coarse-graining.

## Metrics to record
- Component counts and largest component size on the coarse +1 subgraph.
- Defect curve shape and convergence step.
- Neutral fraction and distribution (distance-to-+1 in coarse graph if needed).
- Compare against baseline hemibrain: does a single dominant +1 domain persist? Do tiny satellites/neutral-only pockets collapse or proliferate?

## Scope and non-goals
- Do not introduce new kernel gauges; reuse existing CLI.
- Do not interpret components biologically; this is a structural null test.
- Keep block assignment purely random for the first pass; degree-binned or metadata-aware grouping is deferred.

## Run 1 (random blocks, k=100, seed=0)
- Coarse graph: N=1,800 blocks, edges=2,767,255; avg block size ≈ 100.
- Kernel on coarse graph (sparse_dc, deadzone=1e-9, hops=1): converged in 1 step with defect curve 0; final state all +1 (no neutrals/defects).
- Components (coarse +1 subgraph): 1 component (size 1,800).
- Artifacts: `outputs/coarse_random_blocks_20260120-150343_edges.csv`, `outputs/coarse_random_blocks_20260120-150343_blocks.csv`, `outputs/defect_curve_coarse_random_blocks_20260120-150400*`.
- Interpretation: random-block coarse-grain collapses the neutral shell/defect pockets into a single coherent +1 domain at this scale; need additional coarse-grain variants (different block sizes, degree-binned) to test stability.

## Random-block sweep (k ∈ {50, 100, 200}, seeds {0, 1})
- k=50 (seeds 0/1, ~3,599 blocks, ~5.18M edges): converged in 1 step, defect=0, final all +1, +1 components=1.
- k=100 (seed 0; above): converged in 1 step, defect=0, all +1.
- k=200 (seeds 0/1, ~900 blocks, ~0.81M edges): converged in 2 steps with small initial defect (263 / 255) that vanishes at step 2; final all +1, +1 components=1.
- Conclusion: across block sizes 50–200 and multiple seeds, random-block coarse-graining consistently destroys the neutral/defect structure; the kernel projects to a trivial +1 fixed point at coarse scales.

## Degree-binned coarse-grain (k=100)
- Blocks: 1,800 (target size 100), coarse edges: 846,301.
- Kernel (sparse_dc, deadzone=1e-9, hops=1): converged in 1 step, defect=0, final all +1; +1 components=1.
- Artifacts: `outputs/coarse_degree_binned_20260120-151527_edges.csv`, `outputs/coarse_degree_binned_20260120-151527_blocks.csv`, `outputs/defect_curve_coarse_degree_binned_20260120-151536*`.
- Interpretation: preserving degree heterogeneity (without locality) still collapses the neutral/defect structure; locality drives the original geometry.
