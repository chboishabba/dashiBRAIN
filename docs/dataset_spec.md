# Hemibrain dataset spec (expected)

## Edge list
- File: CSV/TSV with columns: `source_id`, `target_id`, `weight` (synapse count or probability).
- Optional columns: `distance`, `type` (chemical/electrical), `dataset_split`.
- Weight domain: non-negative; zero-weight edges may be omitted.

## Neuron metadata (optional)
- File: CSV/TSV keyed by `neuron_id`.
- Suggested fields: `x`, `y`, `z` (microns), `hemisphere`, `cell_type`, `region`.

## Channel metadata (optional)
- If multiple channels exist, provide mapping `channel_id` -> description (e.g., modality or synapse type).

## Loader expectations
- Edge list sorted is not required; duplicates allowed (aggregated on load).
- Missing metadata is tolerated; loader stores `None` for absent tables.
- All IDs treated as strings externally; internally mapped to dense indices `[0..N-1]`.

## Baseline hooks
- Degree-corrected baseline uses in/out degree from loaded graph.
- Distance-aware baseline (if distances provided) must document formula and units.
