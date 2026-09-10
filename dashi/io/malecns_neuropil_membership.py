"""Synapse-derived MaleCNS neuron-to-neuropil membership.

MaleCNS ``somaNeuromere`` is a soma/segment annotation and is not a central-
brain neuropil carrier. The official v1.0 synaptic partner table instead
contains ``body_pre``, ``body_post`` and ``primary_post`` for each synaptic
partner pair. This module uses those released synapse locations to derive a
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
    """Collapse MaleCNS laterality suffixes onto atlas-level neuropil names."""
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
    observed_synapse_incidence: np.ndarray
    source_rows: int

    @property
    def observed_neuron_count(self) -> int:
        return int(np.count_nonzero(self.observed_synapse_incidence))


def _resolve(names: Sequence[str], candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in names:
            return name
    raise KeyError(
        f"missing required MaleCNS synapse column; tried {tuple(candidates)!r}; "
        f"available={tuple(names)!r}"
    )


def _iter_feather_batches(path: Path, columns: Sequence[str]):
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:  # pragma: no cover
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


def _graph_integer_ids(node_ids: Sequence[str]) -> np.ndarray:
    try:
        ids = np.asarray([int(x) for x in node_ids], dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("MaleCNS synapse membership requires integer body IDs") from exc
    if ids.size and np.any(ids[1:] < ids[:-1]):
        raise ValueError("MaleCNS graph body IDs must be sorted for vectorized lookup")
    return ids


def _located_indices(sorted_ids: np.ndarray, bodies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return graph indices and mask for body IDs present in the graph."""
    bodies = np.asarray(bodies, dtype=np.int64)
    pos = np.searchsorted(sorted_ids, bodies)
    valid = pos < sorted_ids.size
    safe = np.minimum(pos, max(sorted_ids.size - 1, 0))
    if sorted_ids.size:
        valid &= sorted_ids[safe] == bodies
    else:
        valid[:] = False
    return pos, valid


def load_synapse_neuropil_membership(
    path: str | Path,
    node_ids: Sequence[str],
    *,
    allowed_regions: Iterable[str],
) -> NeuropilMembership:
    """Build bounded fractional region x neuron membership from real synapses.

    Only the requested functional-region vocabulary is accumulated. Each
    retained partner row contributes one incidence to both participating
    neurons in the row's ``primary_post`` neuropil. Counts are accumulated
    batchwise into a bounded ``regions x neurons`` matrix and normalized per
    neuron afterwards. No individual synapse row survives the batch boundary.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"MaleCNS synapse partner table not found: {p}")

    try:
        import pyarrow as pa
        import pyarrow.compute as pc
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

    graph_ids = _graph_integer_ids(node_ids)
    requested = tuple(sorted({str(r).strip() for r in allowed_regions if str(r).strip()}))
    if not requested:
        raise ValueError("allowed_regions must contain at least one declared functional region")
    requested_array = pa.array(requested)

    n_regions = len(requested)
    n_nodes = len(node_ids)
    counts = np.zeros((n_regions, n_nodes), dtype=np.int64)
    source_rows = 0

    for pre_arr, post_arr, roi_arr in _iter_feather_batches(p, (pre_col, post_col, roi_col)):
        source_rows += len(roi_arr)

        # MaleCNS/neuPrint names bilateral ROIs like AL(L), AMMC(R), etc. The
        # functional carrier is bilateral at the present resolution, so only
        # this explicit laterality suffix is collapsed.
        roi_base = pc.replace_substring_regex(
            pc.cast(roi_arr, pa.string()),
            pattern=r"\((?:L|R|M)\)$",
            replacement="",
        )
        roi_index = pc.index_in(roi_base, value_set=requested_array).to_numpy(
            zero_copy_only=False
        )
        keep = roi_index >= 0
        if not np.any(keep):
            continue

        region_i = roi_index[keep].astype(np.int64, copy=False)
        pres = pre_arr.to_numpy(zero_copy_only=False)[keep]
        posts = post_arr.to_numpy(zero_copy_only=False)[keep]

        pre_i, pre_valid = _located_indices(graph_ids, pres)
        post_i, post_valid = _located_indices(graph_ids, posts)

        if np.any(pre_valid):
            np.add.at(counts, (region_i[pre_valid], pre_i[pre_valid]), 1)
        if np.any(post_valid):
            np.add.at(counts, (region_i[post_valid], post_i[post_valid]), 1)

    region_mass = counts.sum(axis=1)
    keep_regions = region_mass > 0
    if not np.any(keep_regions):
        raise ValueError("no synapse-derived MaleCNS neuropils overlap the requested vocabulary")

    counts = counts[keep_regions]
    regions = tuple(r for r, keep in zip(requested, keep_regions) if keep)
    incidence = counts.sum(axis=0).astype(np.float64, copy=False)

    membership_dense = counts.astype(np.float32)
    observed = incidence > 0
    membership_dense[:, observed] /= incidence[observed]
    membership = sp.csr_matrix(membership_dense)
    membership.eliminate_zeros()

    return NeuropilMembership(regions, membership, incidence, source_rows)
