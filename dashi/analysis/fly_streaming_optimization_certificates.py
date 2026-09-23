"""Fly-specific local optimization certificates.

These are thin semantic instances of :mod:`local_optimization_compiler`; they do
not replace the production JRC or MaleCNS implementations.  Their purpose is to
pin the smallest local facts from which the large computation is compiled:

* JRC painted-domain streaming: one domain contributes a label-count vector.
* MaleCNS partner streaming: one partner row contributes total incidence for
  both bodies and, when its neuropil is retained, one regional incidence for
  both bodies.

The implementation tactic (``np.bincount``, Arrow batching, ``np.add.at``) is
not itself a proof obligation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from dashi.analysis.local_optimization_compiler import (
    LocalResidencyBound,
    LocalSemanticCertificate,
)


# ---------------------------------------------------------------------------
# JRC painted-domain local counting law
# ---------------------------------------------------------------------------


def jrc_domain_count_spec(selected_labels: np.ndarray, domain_mask: np.ndarray, max_label: int) -> np.ndarray:
    """Reference mathematical count: one explicit equality test per label."""
    labels = np.asarray(selected_labels, dtype=np.int64)
    mask = np.asarray(domain_mask, dtype=bool)
    if labels.shape != mask.shape:
        raise ValueError("selected_labels and domain_mask must have identical shapes")
    out = np.zeros(max_label + 1, dtype=np.int64)
    for label in range(max_label + 1):
        out[label] = int(np.count_nonzero(mask & (labels == label)))
    return out


def jrc_domain_count_optimized(selected_labels: np.ndarray, domain_mask: np.ndarray, max_label: int) -> np.ndarray:
    """Production-shaped local count using one vectorized bincount."""
    labels = np.asarray(selected_labels, dtype=np.int64)
    mask = np.asarray(domain_mask, dtype=bool)
    if labels.shape != mask.shape:
        raise ValueError("selected_labels and domain_mask must have identical shapes")
    return np.bincount(labels[mask], minlength=max_label + 1)[: max_label + 1]


def jrc_domain_local_count_law(selected_labels: np.ndarray, domain_mask: np.ndarray, max_label: int) -> bool:
    return np.array_equal(
        jrc_domain_count_spec(selected_labels, domain_mask, max_label),
        jrc_domain_count_optimized(selected_labels, domain_mask, max_label),
    )


def jrc_stream_residency_bound(
    *,
    selected_label_bytes: int,
    one_domain_bytes: int,
    score_state_bytes: int,
    scratch_bytes: int = 0,
    fixed_runtime_overhead_bytes: int = 0,
) -> LocalResidencyBound:
    """Bound independent of the number of painted domains in the source set."""
    return LocalResidencyBound(
        persistent_state_bytes=selected_label_bytes + score_state_bytes,
        stream_item_bytes=one_domain_bytes,
        scratch_bytes=scratch_bytes,
        fixed_runtime_overhead_bytes=fixed_runtime_overhead_bytes,
    )


# ---------------------------------------------------------------------------
# MaleCNS synapse-partner local contribution law
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartnerContribution:
    pre_index: int
    post_index: int
    region_index: int | None


@dataclass
class PartnerAccumulator:
    counts: np.ndarray  # region x neuron
    total_incidence: np.ndarray  # neuron

    def copy(self) -> "PartnerAccumulator":
        return PartnerAccumulator(self.counts.copy(), self.total_incidence.copy())

    def observation(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (
            tuple(int(x) for x in self.counts.ravel()),
            tuple(int(x) for x in self.total_incidence.ravel()),
        )


def _validate_partner(acc: PartnerAccumulator, item: PartnerContribution) -> None:
    n = acc.total_incidence.size
    if not (0 <= item.pre_index < n and 0 <= item.post_index < n):
        raise IndexError("partner body index outside accumulator")
    if item.region_index is not None and not (0 <= item.region_index < acc.counts.shape[0]):
        raise IndexError("region index outside accumulator")


def partner_row_spec_update(acc: PartnerAccumulator, item: PartnerContribution) -> PartnerAccumulator:
    """Literal local row contribution law."""
    _validate_partner(acc, item)
    out = acc.copy()
    out.total_incidence[item.pre_index] += 1
    out.total_incidence[item.post_index] += 1
    if item.region_index is not None:
        out.counts[item.region_index, item.pre_index] += 1
        out.counts[item.region_index, item.post_index] += 1
    return out


def partner_row_optimized_update(acc: PartnerAccumulator, item: PartnerContribution) -> PartnerAccumulator:
    """Vectorized producer-shaped update with the same consumer observation."""
    _validate_partner(acc, item)
    out = acc.copy()
    np.add.at(out.total_incidence, np.asarray([item.pre_index, item.post_index]), 1)
    if item.region_index is not None:
        np.add.at(
            out.counts,
            (
                np.asarray([item.region_index, item.region_index]),
                np.asarray([item.pre_index, item.post_index]),
            ),
            1,
        )
    return out


def malecns_partner_local_certificate() -> LocalSemanticCertificate[
    PartnerAccumulator, PartnerContribution, tuple[tuple[int, ...], tuple[int, ...]]
]:
    return LocalSemanticCertificate(
        spec_update=partner_row_spec_update,
        optimized_update=partner_row_optimized_update,
        observe=lambda state: state.observation(),
    )


def malecns_membership_from_counts(counts: np.ndarray, total_incidence: np.ndarray) -> np.ndarray:
    """Compile fractional membership after the row fold.

    The denominator is total incidence across all neuropils.  Retained regional
    membership mass is therefore allowed to be below one.
    """
    c = np.asarray(counts, dtype=float)
    total = np.asarray(total_incidence, dtype=float)
    if c.ndim != 2 or total.ndim != 1 or c.shape[1] != total.size:
        raise ValueError("counts must be [region,neuron] aligned with total_incidence")
    out = np.zeros_like(c, dtype=float)
    observed = total > 0
    out[:, observed] = c[:, observed] / total[observed]
    return out


def malecns_stream_residency_bound(
    *,
    region_count: int,
    neuron_count: int,
    counter_itemsize: int = 8,
    incidence_itemsize: int = 8,
    one_batch_bytes: int,
    scratch_bytes: int = 0,
    fixed_runtime_overhead_bytes: int = 0,
) -> LocalResidencyBound:
    if region_count < 0 or neuron_count < 0:
        raise ValueError("region_count and neuron_count must be non-negative")
    persistent = (
        region_count * neuron_count * counter_itemsize
        + neuron_count * incidence_itemsize
    )
    return LocalResidencyBound(
        persistent_state_bytes=persistent,
        stream_item_bytes=one_batch_bytes,
        scratch_bytes=scratch_bytes,
        fixed_runtime_overhead_bytes=fixed_runtime_overhead_bytes,
    )
