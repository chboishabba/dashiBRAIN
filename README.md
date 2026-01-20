# DASHI kernel prototype (hemibrain)

This repo implements the DASHI kernel formalism for the Drosophila hemibrain. The objective is to detect kernel-closed (or low-defect) ternary fields that persist across coarse-grainings, not to fit ML models.

## Guiding principles
- Kernel-first: all downstream work respects kernel closure/defect semantics.
- Reproducibility over speed: correctness and invariance before performance.
- Formal traceability: functions map to definitions (carrier, valuation, kernel, defect, renormalisation, latent structure).
- Non-ML bias: no learned parameters unless explicitly marked as admissible gauges.

## Current focus (Sprint 1–2 solo prototype)
- Sprint 1: load hemibrain edges/metadata, compute baseline expectations, and ternarise residuals.
- Sprint 2: implement kernel operator and flow (single scale) with defect reporting.
- Out-of-scope: coarse-graining, fibers, performance tuning, learned weights.

## Layout (planned)
- `docs/`: formal axioms, dataset spec, gauge choices, sprint plan.
- `dashi/`: Python package with `io`, `baseline`, `valuation`, `kernel`, `analysis` (defect/latent structure later), `viz` stubs.
- `tests/`: lightweight determinism and invariance checks.

## Quick start (after deps)
```bash
python -m pip install -e .  # (planned once packaging is added)
```

Defect curve CLI (edge list CSV/TSV with columns `source_id,target_id,weight`):
```bash
python scripts/defect_curve.py path/to/edges.csv --steps 10 --hops 1 --deadzone 1e-9
```

Outputs are timestamped CSV/PNG (default prefix `outputs/defect_curve_YYYYMMDD-HHMMSS.*`). Notebook variant: `notebooks/sprint-01-02.ipynb`.

For now, see `docs/sprint-01-02.md` for the active checklist and module stubs.
