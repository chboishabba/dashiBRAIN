"""Synapse-derived MaleCNS neuron-to-neuropil membership.

MaleCNS ``somaNeuromere`` is a soma/segment annotation and is not a central-
brain neuropil carrier.  The official v1.0 synaptic partner table instead
contains ``body_pre``, ``body_post`` and ``primary_post`` for each synaptic
partner pair.  This module uses those released synapse locations to derive a
fractional neuron->neuropil membership without forcing each neuron into one
region.

Scientific source:
Berg et al., "Sexual dimorphism in the complete Drosophila male central
nervous system connectome", Cell (2026), DOI 10.1016/j.cell.2026.08.015.
MaleCNS v1.0 download documentation describes
``syn-partners-male-cns-v1.0-minconf-0.5.feather`` as partner pairs with body
IDs and primary neuropil.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp


_BILATERAL_SUFFIX = re.compile(r"\((?:L|R|M)\)$")


def canonicalize_malecns_neuropil(label: object) -> str:
    """Collapse MaleCNS laterality suffixes onto atlas-level neuropil names.

    Examples: ``AL(L) -> AL``, ``WED(R) -> WED``.  Unspecified/background
    labels remain explicit rather than being silently mapped to a named ROI.
    """
    if label is None:
        return ""
    text = str(label).strip()
    if not text or text.lower() in {"none", "nan"}:
        return ""
    return _BILATERAL_SUFFIX.sub("", text)


@dataclass(frozen=True)
class NeuropilMembership:
    regions: tuple[str, ...]
    membership: sp.csr_matrix  # region x neuron; columns sum to 1 when observed
    observed_synapse_incidence: np.ndarray  # per-neuron incidence used for normalization
    source_rows: int

    @property
    def observed_neuron_count(self) -> int:
        return int(np.count_nonzero(self.observed_synapse_incidence))


def _resolve(names: Sequence[str], candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in names:
            return name
    raise KeyError(f"missing required MaleCNS synapse column; tried {tuple(candidates)!r}; available={tuple(names)!r}")


def _iter_feather_batches(path: Path, columns: Sequence[str]):
    """Yield selected Arrow columns one record batch at a time.

    Opening the IPC file with a memory map avoids materializing the 6.8 GB
    partner table as Python objects and keeps residency tied to batch size plus
    the final sparse membership accumulator.
    """
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise ImportError("pyarrow is required for MaleCNS synapse partner ingestion") from exc

    source = pa.memory_map(str(path), "r")
    reader = ipc.open_file(source)
    schema_names = reader.schema.names
    indices = [schema_names.index(c) for c in columns]
    try:
        for batch_i in range(reader.num_record_batches):
            batch = reader.get_batch(batch_i)
            yield [batch.column(i) for i in indices]
    finally:
        source.close()


def load_synapse_neuropil_membership(
    path: str | Path,
    node_ids: Sequence[str],
    *,
    allowed_regions: Iterable[str] | None = None,
) -> NeuropilMembership:
    """Build a fractional region x neuron membership from real synapse rows.

    Each partner row contributes one incidence to both participating neurons in
    the row's ``primary_post`` neuropil.  A neuron's incidences are normalized
    across retained neuropils, so a neuron spanning multiple neuropils remains
    fractionally represented instead of receiving a fabricated single label.

    ``allowed_regions`` is normally the functional atlas vocabulary.  Applying
    it during ingestion is both semantically appropriate and substantially
    lowers memory for this large table.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"MaleCNS synapse partner table not found: {p}")

    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyarrow is required for MaleCNS synapse partner ingestion") from exc

    source = pa.memory_map(str(p), "r")
    reader = ipc.open_file(source)
    names = reader.schema.names
    pre_col = _resolve(names, ("body_pre", "bodyId_pre", "pre"))
    post_col = _resolve(names, ("body_post", "bodyId_post", "post"))
    roi_col = _resolve(names, ("primary_post", "primary_roi", "neuropil", "roi"))
    source.close()

    node_lookup = {str(node_id): i for i, node_id in enumerate(node_ids)}
    allowed = None if allowed_regions is None else {str(r) for r in allowed_regions}

    # Sparse dictionary over only observed (region, neuron) pairs.  The number
    # of such pairs is bounded by biological neuropil occupancy, not synapse-row
    # count, which is the key residency improvement over retaining all rows.
    counts: dict[tuple[str, int], int] = {}
    incidence = np.zeros(len(node_ids), dtype=np.float64)
    source_rows = 0

    for pre_arr, post_arr, roi_arr in _iter_feather_batches(p, (pre_col, post_col, roi_col)):
        pres = pre_arr.to_numpy(zero_copy_only=False)
        posts = post_arr.to_numpy(zero_copy_only=False)
        rois = roi_arr.to_pylist()
        source_rows += len(rois)
        for pre, post, raw_roi in zip(pres, posts, rois):
            roi = canonicalize_malecns_neuropil(raw_roi)
            if not roi or (allowed is not None and roi not in allowed):
                continue
            for body in (pre, post):
                node_i = node_lookup.get(str(body))
                if node_i is None:
                    continue
                key = (roi, node_i)
                counts[key] = counts.get(key, 0) + 1
                incidence[node_i] += 1.0

    regions = tuple(sorted({roi for roi, _ in counts}))
    if not regions:
        raise ValueError("no synapse-derived MaleCNS neuropils overlap the requested vocabulary")
    region_index = {r: i for i, r in enumerate(regions)}

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for (roi, node_i), count in counts.items():
        denom = incidence[node_i]
        if denom <= 0:
            continue
        rows.append(region_index[roi])
        cols.append(node_i)
        vals.append(float(count) / float(denom))

    membership = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(len(regions), len(node_ids)),
        dtype=np.float32,
    ).tocsr()
    membership.sum_duplicates()
    return NeuropilMembership(regions, membership, incidence, source_rows)
