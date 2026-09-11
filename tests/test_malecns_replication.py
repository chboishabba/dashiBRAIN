import numpy as np
import pytest

from dashi.analysis.malecns_replication import (
    FrozenStructuralCarrier,
    ReplicateFunctionalInput,
    evaluate_replication_set,
    frozen_candidate_matrices,
)
from dashi.io.functional_imaging_loader import FunctionalTraceTable
from dashi.io.region_functional_adapter import RegionFunctionalProducer


def _carrier():
    regions = ("A", "B", "C", "D")
    d = np.array(
        [
            [0.0, 2.0, 1.0, 0.5],
            [1.0, 0.0, 3.0, 1.0],
            [2.0, 0.5, 0.0, 2.0],
            [1.5, 1.0, 0.5, 0.0],
        ],
        dtype=float,
    )
    s = d * np.array(
        [
            [0.0, 0.8, 0.8, 0.8],
            [0.4, 0.0, 0.4, 0.4],
            [0.7, 0.7, 0.0, 0.7],
            [0.2, 0.2, 0.2, 0.0],
        ]
    )
    return FrozenStructuralCarrier(regions, d, s)


def _producer(source: str):
    # Long enough to use the published regressor while keeping the fixture simple.
    t = np.arange(120, dtype=float)
    traces = np.column_stack(
        [
            np.sin(t / 11.0),
            np.cos(t / 13.0),
            np.sin(t / 17.0 + 0.2),
            np.cos(t / 19.0 - 0.3),
        ]
    )
    return RegionFunctionalProducer(
        atlas_identifier="fixture-atlas",
        traces=FunctionalTraceTable(
            unit_ids=("A", "B", "C", "D"),
            traces=traces,
            time=t,
            identity_kind="declared_region_identity",
        ),
        source_identifier=source,
    )


def _replicate(name: str, source: str, *, independent=True):
    return ReplicateFunctionalInput(
        replicate_id=name,
        producer=_producer(source),
        overlap_kernel=np.eye(4, dtype=float),
        independent_recording=independent,
        independence_receipt=f"{name} is a distinct recording",
    )


def test_frozen_candidate_family_contains_only_predeclared_representations():
    matrices = frozen_candidate_matrices(_carrier())
    assert tuple(matrices) == (
        "absolute_reverse",
        "relative_shape_reverse",
        "signed_reverse",
        "magnitude_shape_reverse",
    )
    assert all(matrix.shape == (4, 4) for matrix in matrices.values())


def test_reused_or_pooled_recording_is_not_accepted_as_independent_replication():
    with pytest.raises(ValueError, match="not marked as an independent recording"):
        evaluate_replication_set(_carrier(), [_replicate("pooled", "same-source", independent=False)])


def test_duplicate_replicate_ids_are_rejected():
    reps = [_replicate("r1", "s1"), _replicate("r1", "s2")]
    with pytest.raises(ValueError, match="must be unique"):
        evaluate_replication_set(_carrier(), reps)


def test_region_order_must_match_frozen_structural_carrier():
    bad = _replicate("r1", "s1")
    producer = bad.producer
    swapped = RegionFunctionalProducer(
        atlas_identifier=producer.atlas_identifier,
        traces=FunctionalTraceTable(
            unit_ids=("B", "A", "C", "D"),
            traces=producer.traces.traces,
            time=producer.traces.time,
            identity_kind=producer.traces.identity_kind,
        ),
        source_identifier=producer.source_identifier,
    )
    bad = ReplicateFunctionalInput(
        replicate_id="r1",
        producer=swapped,
        overlap_kernel=np.eye(4),
        independent_recording=True,
        independence_receipt="distinct recording",
    )
    with pytest.raises(ValueError, match="region order must exactly match"):
        evaluate_replication_set(_carrier(), [bad])


def test_two_independent_recordings_are_scored_without_reselection():
    summary = evaluate_replication_set(
        _carrier(),
        [_replicate("r1", "source-1"), _replicate("r2", "source-2")],
    )
    assert summary.replicate_count == 2
    assert [score.replicate_id for score in summary.scores] == ["r1", "r2"]
    assert all(np.isfinite(score.residual_absolute_reverse) for score in summary.scores)
    assert all(np.isfinite(score.residual_magnitude_shape_reverse) for score in summary.scores)
