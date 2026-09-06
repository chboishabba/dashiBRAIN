"""Load registered Drosophila functional traces and mapping tables.

The loaders are schema-adaptive but conservative: ambiguous archive indices,
coordinates or regions are never upgraded to direct neuron identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from dashi.io.malecns_loader import _read_table


@dataclass(frozen=True)
class FunctionalTraceTable:
    unit_ids: tuple[str, ...]
    traces: np.ndarray  # time x unit
    time: np.ndarray | None = None
    identity_kind: str = "declared_unit_id"


@dataclass(frozen=True)
class RegistrationMap:
    functional_to_region: Mapping[str, str]
    functional_to_connectome: Mapping[str, str]
    evidence_kind: Mapping[str, str]
    residual: Mapping[str, float]

    def direct_identity_fraction(self) -> float:
        if not self.evidence_kind:
            return 0.0
        direct = sum(v == "direct_identity" for v in self.evidence_kind.values())
        return direct / len(self.evidence_kind)


def _first_present(columns: Sequence[str], aliases: Sequence[str]) -> str | None:
    for a in aliases:
        if a in columns:
            return a
    return None


def load_functional_traces(path: str | Path) -> FunctionalTraceTable:
    p = Path(path)
    if p.suffix.lower() in {".pkl", ".pickle"}:
        from dashi.io.gauthey_compact import load_audio_correlated_pickle
        matrix = load_audio_correlated_pickle(p)
        return FunctionalTraceTable(
            matrix.unit_ids,
            matrix.traces,
            None,
            identity_kind=matrix.identity_kind,
        )

    table = _read_table(p)
    cols = list(table)
    if not cols:
        raise ValueError("functional table is empty")

    time_col = _first_present(cols, ("time", "t", "timestamp", "seconds", "frame_time"))
    unit_col = _first_present(cols, ("unit_id", "roi_id", "region", "neuropil", "cell_id"))
    value_col = _first_present(cols, ("dff", "df_f", "delta_f_over_f", "value", "activity"))

    if time_col and unit_col and value_col:
        times = sorted({float(x) for x in table[time_col]})
        units = sorted({str(x) for x in table[unit_col]})
        ti = {t: i for i, t in enumerate(times)}
        ui = {u: i for i, u in enumerate(units)}
        traces = np.full((len(times), len(units)), np.nan, dtype=float)
        for t, u, v in zip(table[time_col], table[unit_col], table[value_col]):
            traces[ti[float(t)], ui[str(u)]] = float(v)
        col_mean = np.nanmean(traces, axis=0)
        inds = np.where(np.isnan(traces))
        traces[inds] = np.take(np.nan_to_num(col_mean, nan=0.0), inds[1])
        return FunctionalTraceTable(tuple(units), traces, np.asarray(times, dtype=float))

    value_columns = [c for c in cols if c != time_col]
    if len(value_columns) < 2:
        raise ValueError("functional table must contain at least two units")
    traces = np.column_stack([[float(x) for x in table[c]] for c in value_columns])
    times = np.asarray([float(x) for x in table[time_col]], dtype=float) if time_col else None
    return FunctionalTraceTable(tuple(value_columns), traces, times)


def load_registration_map(path: str | Path) -> RegistrationMap:
    table = _read_table(path)
    cols = list(table)
    if not cols:
        raise ValueError("registration table is empty")

    fcol = _first_present(cols, ("functional_id", "roi_id", "source_id", "unit_id"))
    rcol = _first_present(cols, ("region", "neuropil", "atlas_region", "jrc2018_region"))
    ccol = _first_present(cols, ("connectome_id", "body_id", "body", "target_id"))
    ecol = _first_present(cols, ("evidence_kind", "evidence", "match_type"))
    dcol = _first_present(cols, ("residual", "registration_error", "distance_um", "l2_error"))
    if fcol is None:
        raise ValueError("registration map lacks a functional unit identifier")

    to_region: dict[str, str] = {}
    to_connectome: dict[str, str] = {}
    evidence: dict[str, str] = {}
    residual: dict[str, float] = {}
    for i, f in enumerate(table[fcol]):
        fid = str(f)
        if rcol:
            to_region[fid] = str(table[rcol][i])
        if ccol and str(table[ccol][i]).strip():
            to_connectome[fid] = str(table[ccol][i])
        if ecol:
            evidence[fid] = str(table[ecol][i])
        else:
            evidence[fid] = "atlas_region_overlap" if rcol else "coordinate_proximity"
        if dcol and str(table[dcol][i]).strip():
            residual[fid] = float(table[dcol][i])

    return RegistrationMap(to_region, to_connectome, evidence, residual)


def aggregate_functional_traces_by_region(
    functional: FunctionalTraceTable,
    registration: RegistrationMap,
) -> FunctionalTraceTable:
    groups: dict[str, list[int]] = {}
    for i, uid in enumerate(functional.unit_ids):
        region = registration.functional_to_region.get(uid)
        if region:
            groups.setdefault(region, []).append(i)
    if not groups:
        raise ValueError("registration supplies no functional-to-region mappings")
    regions = tuple(sorted(groups))
    traces = np.column_stack([np.mean(functional.traces[:, groups[r]], axis=1) for r in regions])
    return FunctionalTraceTable(regions, traces, functional.time, identity_kind="registered_region")
