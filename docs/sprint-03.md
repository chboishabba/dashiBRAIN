# Sprint 3 — geometry & gauges

Hemibrain defect/neutral geometry analysis plus gauge robustness sweeps.

## Baseline run (geometry source)
- Dataset: `data/hemibrain/edges.csv` (no metadata), hops=1, deadzone=1e-9 (converged at step 4).
- Baseline: `sparse_dc`; defect curve 633 → 42 → 2 → 0 (converged at step 4).
- Artifacts: `outputs/defect_curve_20260120-142843_state.csv`, `outputs/defect_curve_20260120-142843_defect_nodes.csv`, summary JSON `outputs/sprint3_default_geometry_summary.json`.

## Geometry observations (sparse_dc, deadzone=1e-9)
- +1 components: 8 total; sizes = 137,879 then tiny satellites (6, 6, 2 ×5).
- Neutral layer: 42,006 nodes total; 41,690 are exactly 1 hop from a +1 component, 210 at 2 hops, 8 at 3 hops, 1 at 4 hops; 97 neutrals sit in neutral-only islands unreachable from +1 components.
- Defect nodes: 677 changed nodes; 441 at 1 hop from +1, 193 at 2 hops, 4 at 3 hops; 39 defect nodes reside in neutral-only islands (no +1 reachability).
- Component affiliation: 41,899 neutrals and 634 defect nodes attach to the largest +1 component; tiny satellites attract ≤4 neutrals and ≤2 defects each; no neutral/defect node has neighbors from multiple +1 components, so the neutral shell is single-sourced rather than straddling component boundaries.

## Gauge robustness sweeps (outputs in `outputs/`)
- sparse_dc, deadzone=1e-6 → `outputs/sweep_sparse_dc_dz1e-6_20260120-145029*`: defect curve unchanged (633 → 42 → 2 → 0), +1 comps = 8 (largest 137,879), final balance {+1: 137,901, 0: 42,006, -1: 0}.
- sparse_dc, deadzone=1e-4 → `outputs/sweep_sparse_dc_dz1e-4_20260120-145050*`: identical curve and component counts to the baseline.
- raw baseline, deadzone=1e-9 → `outputs/sweep_raw_dz1e-9_20260120-145116*`: residual sign balance all positive, but kernel converges to the same final state and 8-component structure as sparse_dc.

Result: component count (~8) and defect curve are invariant across the tested deadzones and the raw baseline gauge.
