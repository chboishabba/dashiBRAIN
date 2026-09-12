"""Independent-recording replication for the MaleCNS structure/function carrier.

The representation search is frozen before this module is entered.  A replicate
supplies only a new region-resolved functional recording (plus its overlap
geometry); the MaleCNS structural carrier and candidate representation family
are held fixed.

This module deliberately distinguishes:

* representation replication: the same named structural candidates are scored;
* coefficient fitting inside each LORO training fold: still permitted;
* representation reselection/compression: forbidden here;
* pooled or re-used functional recordings: not independent replication.

The canonical candidate family after the single-session search is

    D^T   absolute/density-like unsigned reverse
    P^T   row-relative unsigned wiring shape
    S^T   full coarse signed reverse
    (m P)^T  polarity-free, scale-free sender-gain candidate

where m_i = |sum_j S_ij| / sum_j |D_ij|.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from dashi.analysis.gauthey_lbm_experiment import published_lbm_stimulus_regressor
from dashi.analysis.network_scale_shape_discriminator import decompose_network_scale_shape
from dashi.analysis.sender_magnitude_shape_composition import compose_sender_magnitude_shape
from dashi.analysis.signed_fibre_discriminator import singleton_family
from dashi.analysis.overlap_controlled_structure_function import (
    evaluate_overlap_controlled_leave_one_region_out,
)
from dashi.analysis.stimulus_controlled_functional import residualize_region_traces_against_stimulus
from dashi.analysis.structure_function_real import functional_correlation
from dashi.io.region_functional_adapter import RegionFunctionalProducer


@dataclass(frozen=True)
class FrozenStructuralCarrier:
    regions: tuple[str, ...]
    unsigned_direct: np.ndarray
    signed_direct: np.ndarray

    def validate(self) -> None:
        n = len(self.regions)
        d = np.asarray(self.unsigned_direct, dtype=float)
        s = np.asarray(self.signed_direct, dtype=float)
        if d.shape != (n, n) or s.shape != (n, n):
            raise ValueError("structural matrices must align with regions")
        if len(set(self.regions)) != n:
            raise ValueError("regions must be unique")


@dataclass(frozen=True)
class ReplicateFunctionalInput:
    replicate_id: str
    producer: RegionFunctionalProducer
    overlap_kernel: np.ndarray
    independent_recording: bool
    independence_receipt: str

    def validate(self, structural_regions: tuple[str, ...]) -> None:
        self.producer.validate()
        if not self.replicate_id.strip():
            raise ValueError("replicate_id must be non-empty")
        if not self.independence_receipt.strip():
            raise ValueError("independence_receipt must be non-empty")
        if not self.independent_recording:
            raise ValueError(
                f"replicate {self.replicate_id!r} is not marked as an independent recording"
            )
        if tuple(self.producer.traces.unit_ids) != tuple(structural_regions):
            raise ValueError(
                "replicate region order must exactly match frozen structural carrier"
            )
        kernel = np.asarray(self.overlap_kernel, dtype=float)
        n = len(structural_regions)
        if kernel.shape != (n, n):
            raise ValueError("overlap kernel must align with frozen structural carrier")


@dataclass(frozen=True)
class ReplicateScore:
    replicate_id: str
    source_identifier: str
    atlas_identifier: str
    timepoint_count: int
    residual_absolute_reverse: float
    residual_relative_shape_reverse: float
    residual_signed_reverse: float
    residual_magnitude_shape_reverse: float

    @property
    def magnitude_shape_gain_over_unsigned(self) -> float:
        return float(self.residual_absolute_reverse - self.residual_magnitude_shape_reverse)

    @property
    def magnitude_shape_gain_over_relative_shape(self) -> float:
        return float(self.residual_relative_shape_reverse - self.residual_magnitude_shape_reverse)


@dataclass(frozen=True)
class ReplicationSummary:
    scores: tuple[ReplicateScore, ...]

    @property
    def replicate_count(self) -> int:
        return len(self.scores)

    @property
    def magnitude_shape_better_than_absolute_count(self) -> int:
        return sum(
            score.residual_magnitude_shape_reverse < score.residual_absolute_reverse
            for score in self.scores
        )

    @property
    def magnitude_shape_better_than_relative_shape_count(self) -> int:
        return sum(
            score.residual_magnitude_shape_reverse < score.residual_relative_shape_reverse
            for score in self.scores
        )

    @property
    def median_magnitude_shape_gain_over_unsigned(self) -> float:
        if not self.scores:
            return float("nan")
        return float(np.median([s.magnitude_shape_gain_over_unsigned for s in self.scores]))


def frozen_candidate_matrices(carrier: FrozenStructuralCarrier) -> Mapping[str, np.ndarray]:
    """Build the already-chosen candidate family without any outcome information."""
    carrier.validate()
    d = np.asarray(carrier.unsigned_direct, dtype=float)
    s = np.asarray(carrier.signed_direct, dtype=float)
    dec = decompose_network_scale_shape(d, s)
    comp = compose_sender_magnitude_shape(d, s)
    return {
        "absolute_reverse": d.T,
        "relative_shape_reverse": dec.row_shape.T,
        "signed_reverse": s.T,
        "magnitude_shape_reverse": comp.magnitude_sender_shape.T,
    }


def evaluate_independent_replicate(
    carrier: FrozenStructuralCarrier,
    replicate: ReplicateFunctionalInput,
    *,
    correlation_threshold: float = 0.98,
) -> ReplicateScore:
    """Score one independent recording with a frozen representation family."""
    carrier.validate()
    replicate.validate(carrier.regions)

    traces = np.asarray(replicate.producer.traces.traces, dtype=float)
    stimulus = published_lbm_stimulus_regressor(traces.shape[0])
    stim = residualize_region_traces_against_stimulus(traces, stimulus)
    observed = functional_correlation(stim.residual_traces, carrier.regions).matrix
    kernel = np.asarray(replicate.overlap_kernel, dtype=float)
    matrices = frozen_candidate_matrices(carrier)

    def score(name: str) -> float:
        result = evaluate_overlap_controlled_leave_one_region_out(
            singleton_family(carrier.regions, name, matrices[name]),
            observed,
            kernel,
            correlation_threshold=correlation_threshold,
        )
        return float(result.weighted_mean_residual)

    return ReplicateScore(
        replicate_id=replicate.replicate_id,
        source_identifier=replicate.producer.source_identifier,
        atlas_identifier=replicate.producer.atlas_identifier,
        timepoint_count=int(traces.shape[0]),
        residual_absolute_reverse=score("absolute_reverse"),
        residual_relative_shape_reverse=score("relative_shape_reverse"),
        residual_signed_reverse=score("signed_reverse"),
        residual_magnitude_shape_reverse=score("magnitude_shape_reverse"),
    )


def evaluate_replication_set(
    carrier: FrozenStructuralCarrier,
    replicates: Sequence[ReplicateFunctionalInput],
    *,
    correlation_threshold: float = 0.98,
) -> ReplicationSummary:
    """Evaluate multiple independent recordings without representation reselection."""
    ids = [rep.replicate_id for rep in replicates]
    if len(set(ids)) != len(ids):
        raise ValueError("replicate_id values must be unique")
    if not replicates:
        raise ValueError("at least one independent replicate is required")
    return ReplicationSummary(
        tuple(
            evaluate_independent_replicate(
                carrier,
                replicate,
                correlation_threshold=correlation_threshold,
            )
            for replicate in replicates
        )
    )
