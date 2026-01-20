# Repository Guidelines

## Project Structure & Module Organization
- `dashi/`: core Python package. Key areas: `baseline/` (residuals, ternary init), `kernel/` (operator + flow), `io/hemibrain_loader.py` (edge/metadata ingest), `valuation/` (field init), `analysis/` and `viz/` stubs.
- `scripts/defect_curve.py`: CLI to run kernel iterations over an edge list and emit CSV/PNG defect curves into `outputs/`.
- `docs/`: dataset spec, gauge choices, formal axioms, and sprint plan; use these when changing assumptions or gauges.
- `tests/`: unit tests for kernel flow determinism and hop expansion. Add new cases near similar patterns.
- `notebooks/`: ad hoc exploration (e.g., `notebooks/sprint-01-02.ipynb`). Data files are expected under `data/hemibrain/` but are not versioned.

## Build, Test, and Development Commands
- Create a virtual env and install runtime deps (numpy, scipy, matplotlib): `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (or `pip install numpy scipy matplotlib` until packaging lands).
- Run defect curve CLI: `python scripts/defect_curve.py data/hemibrain/edges.csv --steps 10 --hops 1 --deadzone 1e-9`.
- Run tests: `python -m unittest` (or `python -m unittest tests/test_kernel_flow.py` for the focused suite).
- Outputs are timestamped; keep `outputs/` in `.gitignore` if expanded.

## Coding Style & Naming Conventions
- Python 3 with type hints; prefer small pure functions and dataclasses (see `HemibrainGraph`).
- 4-space indentation, `snake_case` for functions/variables, `CapWords` for classes. Keep arrays `np.int8` for ternary states and `np.float32`/CSR for adjacency to match current usage.
- Validate inputs (non-negative weights, expected columns) and raise explicit errors, mirroring `io/hemibrain_loader.py`.
- When adding CLIs, use `argparse` with clear defaults; echo key metrics (N, edges, defect status) to stdout for reproducibility.

## Testing Guidelines
- Extend `tests/` with small deterministic graphs; assert convergence/cycle detection, hop expansion, and invariance under repeated runs.
- Keep test data inline (small matrices) and avoid large fixtures. Seed any randomness explicitly.
- Aim for coverage of new kernel parameters or ternary projections when modifying them.

## Commit & Pull Request Guidelines
- Use concise, imperative commit subjects (e.g., “Add kernel cycle detection guard”) with optional body noting rationale and gauges touched (deadzone, hops, dataset).
- PRs should summarize behavior changes, link relevant docs sections (e.g., `docs/gauge_choices.md`), list commands/tests run, and note data inputs/outputs affected. Include screenshots/plots only when they clarify defect curve shifts.
