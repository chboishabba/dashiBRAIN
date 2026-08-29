"""Hemibrain experiment receipts aligned with the dashi_agda BIDI consumer.

This module does not alter the kernel or the coarse-graining algorithms.  It
turns existing fine/coarse artifacts into a typed empirical receipt that keeps
three claims separate:

1. fine-state kernel closure / low defect;
2. persistence of a declared fine consumer through a coarse quotient;
3. prediction-envelope closure by a later measurement.

The current Sprint-3/Sprint-4 artifacts are especially informative for the
neutral/defect consumer: the fine state contains a large neutral/defect
geometry, while several tested coarse quotients collapse to an all-+1 state.
That is evidence *against* persistence for those quotients, not a failed proof
that should be promoted by relabelling the quotient.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FineGeometryReceipt:
    source: str
    nodes: int
    positive_nodes: int
    neutral_nodes: int
    negative_nodes: int
    component_count: int
    defect_associated_nodes: int

    @property
    def carries_neutral_or_defect_structure(self) -> bool:
        return self.neutral_nodes > 0 or self.defect_associated_nodes > 0


@dataclass(frozen=True)
class CoarseGeometryReceipt:
    quotient_name: str
    state_source: str
    defect_curve_source: str | None
    nodes: int
    positive_nodes: int
    neutral_nodes: int
    negative_nodes: int
    terminal_defect: int | None

    @property
    def all_positive(self) -> bool:
        return (
            self.nodes > 0
            and self.positive_nodes == self.nodes
            and self.neutral_nodes == 0
            and self.negative_nodes == 0
        )


@dataclass(frozen=True)
class HemibrainPersistenceReceipt:
    fine: FineGeometryReceipt
    coarse: CoarseGeometryReceipt
    consumer: str
    coarse_persistence_observed: bool
    quotient_erases_declared_consumer: bool
    empirical_scope_only: bool = True
    measurement_closes_prediction_claimed: bool = False
    biological_interpretation_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def load_fine_geometry_summary(path: str | Path) -> FineGeometryReceipt:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    graph = payload["graph"]
    state = payload["state"]
    components = payload["components"]
    distances = payload.get("distances", {})
    defect = distances.get("defect", {})
    unreached_defect = int(distances.get("unreached_defect", 0))

    # `distances.defect.count` counts reached defect nodes.  Add the explicitly
    # reported unreached defect islands to recover the full defect-associated
    # population used by the Sprint-3 geometry discussion.
    defect_associated = int(defect.get("count", 0)) + unreached_defect

    return FineGeometryReceipt(
        source=str(source),
        nodes=int(graph["nodes"]),
        positive_nodes=int(state.get("+1", 0)),
        neutral_nodes=int(state.get("0", 0)),
        negative_nodes=int(state.get("-1", 0)),
        component_count=int(components["count"]),
        defect_associated_nodes=defect_associated,
    )



def _terminal_defect_from_curve(path: str | Path | None) -> int | None:
    if path is None:
        return None
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return int(rows[-1]["defect"])



def load_coarse_geometry_state(
    state_path: str | Path,
    *,
    quotient_name: str,
    defect_curve_path: str | Path | None = None,
) -> CoarseGeometryReceipt:
    state_source = Path(state_path)
    counts = {-1: 0, 0: 0, 1: 0}
    nodes = 0
    with state_source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "value" not in (reader.fieldnames or []):
            raise ValueError("coarse state artifact must contain a 'value' column")
        for row in reader:
            value = int(row["value"])
            if value not in counts:
                raise ValueError(f"unexpected ternary state value: {value}")
            counts[value] += 1
            nodes += 1

    return CoarseGeometryReceipt(
        quotient_name=quotient_name,
        state_source=str(state_source),
        defect_curve_source=None if defect_curve_path is None else str(defect_curve_path),
        nodes=nodes,
        positive_nodes=counts[1],
        neutral_nodes=counts[0],
        negative_nodes=counts[-1],
        terminal_defect=_terminal_defect_from_curve(defect_curve_path),
    )



def compare_neutral_defect_persistence(
    fine: FineGeometryReceipt,
    coarse: CoarseGeometryReceipt,
) -> HemibrainPersistenceReceipt:
    """Compare the literal neutral/defect consumer across one quotient.

    This intentionally uses a weak consumer: whether any neutral/defect
    structure survives at all.  If the fine artifact has such structure and the
    coarse artifact is all +1, the quotient has erased the consumer before any
    stronger component/topology claim is considered.
    """

    fine_signal = fine.carries_neutral_or_defect_structure
    coarse_signal = coarse.neutral_nodes > 0 or coarse.negative_nodes > 0
    erases = fine_signal and not coarse_signal
    persists = (not fine_signal) or coarse_signal

    return HemibrainPersistenceReceipt(
        fine=fine,
        coarse=coarse,
        consumer="neutral_or_defect_presence",
        coarse_persistence_observed=persists,
        quotient_erases_declared_consumer=erases,
    )



def build_recorded_sprint3_sprint4_receipt(
    repo_root: str | Path,
    *,
    quotient: str = "degree_binned_k100",
) -> HemibrainPersistenceReceipt:
    """Build a receipt from checked-in artifacts, without hard-coded counts."""

    root = Path(repo_root)
    fine = load_fine_geometry_summary(root / "outputs/sprint3_default_geometry_summary.json")

    quotient_files = {
        "degree_binned_k100": (
            "outputs/defect_curve_coarse_degree_binned_20260120-151536_state.csv",
            "outputs/defect_curve_coarse_degree_binned_20260120-151536.csv",
        ),
        "random_blocks_k100_seed0": (
            "outputs/defect_curve_coarse_random_blocks_20260120-150400_state.csv",
            "outputs/defect_curve_coarse_random_blocks_20260120-150400.csv",
        ),
        "hop_radius_r1": (
            "outputs/defect_radius_r1_20260120-153948_state.csv",
            "outputs/defect_radius_r1_20260120-153948.csv",
        ),
        "hop_radius_r2": (
            "outputs/defect_radius_r2_20260120-154023_state.csv",
            "outputs/defect_radius_r2_20260120-154023.csv",
        ),
        "roi": (
            "outputs/defect_roi_20260120-153259_state.csv",
            "outputs/defect_roi_20260120-153259.csv",
        ),
    }
    try:
        state_rel, curve_rel = quotient_files[quotient]
    except KeyError as exc:
        raise ValueError(f"unknown recorded quotient: {quotient}") from exc

    coarse = load_coarse_geometry_state(
        root / state_rel,
        quotient_name=quotient,
        defect_curve_path=root / curve_rel,
    )
    return compare_neutral_defect_persistence(fine, coarse)
