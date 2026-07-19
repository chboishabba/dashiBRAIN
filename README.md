# DASHI kernel prototype (hemibrain)

This repo implements the DASHI kernel formalism for the Drosophila hemibrain. The objective is to detect kernel-closed (or low-defect) ternary fields and characterize which parts of their geometry survive admissible coarse-graining, not to fit black-box ML models.

## Guiding principles
- Kernel-first: all downstream work respects kernel closure/defect semantics.
- Reproducibility over speed: correctness and invariance before performance.
- Formal traceability: functions map to definitions (carrier, valuation, kernel, defect, renormalisation, latent structure).
- Non-ML bias: no learned parameters unless explicitly marked as admissible gauges.

## Current focus
- Full hemibrain sparse-baseline kernel runs and closed-state capture.
- Defect, neutral-shell, and signed-component geometry.
- Random, degree-binned, ROI, hop-radius, and voxel coarse-graining probes.
- Formal separation of atomic, affine, nonlinear, and exploded regimes.
- Weighted threshold-CSP interpretation of kernel closure and defect.

## Formal notes
- `docs/formal_axioms.md`: core carrier, kernel, defect, and renormalisation definitions.
- `docs/nonlinear-sparsity.md`: nonlinear sparsity theorems, ℓ1/ReLU contrast, and exact weighted-CSP correspondence.
- `docs/sprint-01-02.md` through `docs/sprint-04.md`: implementation and experimental record.

## Layout
- `docs/`: formal axioms, theorem notes, dataset/gauge specs, and sprint records.
- `dashi/`: Python package with `io`, `baseline`, `valuation`, `kernel`, `analysis`, and `viz` modules.
- `scripts/`: hemibrain runs, diagnostics, and coarse-graining tools.
- `tests/`: determinism, invariance, flow, and nonlinear-sparsity checks.

## Quick start

Defect curve CLI for an edge list with columns `source_id,target_id,weight`:

```bash
PYTHONPATH=. python scripts/defect_curve.py path/to/edges.csv \
  --baseline sparse_dc --steps 10 --hops 1 --deadzone 1e-9
```

Outputs are timestamped CSV/PNG artifacts, including the final state and per-step defecting nodes. Notebook variant: `notebooks/sprint-01-02.ipynb`.

Run tests with:

```bash
PYTHONPATH=. python -m unittest discover tests
```
