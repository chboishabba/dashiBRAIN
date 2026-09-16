"""Exact polarity-assignment test on the compressed MaleCNS sender-scalar carrier.

The signed-fibre compression result showed that the full presynaptic-transmitter-
signed region matrix S can be compressed, for the current held-out consumer, to
one signed tendency a_i per sender region together with the fixed unsigned
regional carrier D:

    S_hat[i,j] = a_i D[i,j].

This module isolates *polarity assignment* from magnitude, topology, and
pair-specific S/D detail.  Let m_i = |a_i| and z_i = sign(a_i).  The observed
carrier is

    S_scalar[i,j] = m_i z_i D[i,j].

A polarity null keeps D and every sender magnitude m_i fixed, and reassigns only
the sign labels z_i across sender regions while preserving the number of
positive, negative, and zero labels.  When the number of distinct assignments is
small, all assignments are enumerated exactly; otherwise a Monte Carlo fallback
is used.

This is a benchmark randomization test on the coarse presynaptic-transmitter
sign convention.  It is not receptor-resolved excitation/inhibition physiology.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np

from dashi.analysis.overlap_controlled_structure_function import (
    evaluate_overlap_controlled_leave_one_region_out,
)
from dashi.analysis.signed_fibre_compression import sender_scalar_signed_direct
from dashi.analysis.signed_fibre_discriminator import singleton_family


@dataclass(frozen=True)
class SenderScalarPolarityNullResult:
    observed_residual: float
    magnitude_only_residual: float
    null_residuals: np.ndarray
    empirical_p_value: float
    exact_enumeration: bool
    assignment_count: int
    negative_sender_count: int
    zero_sender_count: int
    observed_signs: np.ndarray
    sender_magnitudes: np.ndarray


def _score_reverse(
    regions: tuple[str, ...],
    matrix: np.ndarray,
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    *,
    name: str,
    correlation_threshold: float,
) -> float:
    return evaluate_overlap_controlled_leave_one_region_out(
        singleton_family(regions, name, np.asarray(matrix, dtype=float).T),
        observed,
        overlap_kernel,
        correlation_threshold=correlation_threshold,
    ).weighted_mean_residual


def _sign_labels(values: np.ndarray, *, zero_tolerance: float = 1e-12) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    signs = np.sign(x)
    signs[np.abs(x) <= zero_tolerance] = 0.0
    return signs


def distinct_sign_assignment_count(signs: np.ndarray) -> int:
    """Number of distinct assignments preserving +, -, and zero label counts."""
    z = np.asarray(signs, dtype=float)
    n = z.size
    n_neg = int(np.count_nonzero(z < 0))
    n_zero = int(np.count_nonzero(z == 0))
    # choose negative locations, then zero locations among the remainder
    return int(comb(n, n_neg) * comb(n - n_neg, n_zero))


def _enumerate_sign_assignments(signs: np.ndarray):
    z = np.asarray(signs, dtype=float)
    n = z.size
    n_neg = int(np.count_nonzero(z < 0))
    n_zero = int(np.count_nonzero(z == 0))
    indices = tuple(range(n))
    for neg_idx in combinations(indices, n_neg):
        neg = set(neg_idx)
        remaining = tuple(i for i in indices if i not in neg)
        for zero_idx in combinations(remaining, n_zero):
            out = np.ones(n, dtype=float)
            if neg_idx:
                out[np.asarray(neg_idx, dtype=int)] = -1.0
            if zero_idx:
                out[np.asarray(zero_idx, dtype=int)] = 0.0
            yield out


def evaluate_sender_scalar_polarity_null(
    regions: tuple[str, ...],
    unsigned_direct: np.ndarray,
    signed_direct: np.ndarray,
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    *,
    max_exact_assignments: int = 100_000,
    n_null: int = 1_000,
    seed: int = 0,
    correlation_threshold: float = 0.98,
) -> SenderScalarPolarityNullResult:
    """Test whether the anatomical assignment of sender-scalar polarity matters."""
    d = np.asarray(unsigned_direct, dtype=float)
    s = np.asarray(signed_direct, dtype=float)
    obs = np.asarray(observed, dtype=float)
    kernel = np.asarray(overlap_kernel, dtype=float)
    n = len(regions)
    if any(x.shape != (n, n) for x in (d, s, obs, kernel)):
        raise ValueError("all matrices must align with regions")
    if n_null < 1:
        raise ValueError("n_null must be >= 1")

    scalar_matrix, tendency = sender_scalar_signed_direct(d, s)
    magnitudes = np.abs(tendency)
    signs = _sign_labels(tendency)
    observed_residual = _score_reverse(
        regions,
        scalar_matrix,
        obs,
        kernel,
        name="observed_sender_scalar_signed_reverse",
        correlation_threshold=correlation_threshold,
    )
    magnitude_matrix = d * magnitudes[:, None]
    magnitude_only_residual = _score_reverse(
        regions,
        magnitude_matrix,
        obs,
        kernel,
        name="sender_scalar_magnitude_only_reverse",
        correlation_threshold=correlation_threshold,
    )

    assignment_count = distinct_sign_assignment_count(signs)
    exact = assignment_count <= max_exact_assignments
    residuals: list[float] = []
    if exact:
        assignments = _enumerate_sign_assignments(signs)
    else:
        rng = np.random.default_rng(seed)
        assignments = (signs[rng.permutation(n)] for _ in range(n_null))

    for draw_i, reassigned_signs in enumerate(assignments):
        null_tendency = magnitudes * np.asarray(reassigned_signs, dtype=float)
        null_matrix = d * null_tendency[:, None]
        residuals.append(
            _score_reverse(
                regions,
                null_matrix,
                obs,
                kernel,
                name=f"sender_scalar_polarity_assignment_{draw_i}",
                correlation_threshold=correlation_threshold,
            )
        )

    nulls = np.asarray(residuals, dtype=float)
    if exact:
        # The exact permutation distribution includes the observed assignment.
        p_value = float(np.count_nonzero(nulls <= observed_residual + 1e-15) / nulls.size)
    else:
        p_value = float((1 + np.count_nonzero(nulls <= observed_residual)) / (nulls.size + 1))

    return SenderScalarPolarityNullResult(
        observed_residual=float(observed_residual),
        magnitude_only_residual=float(magnitude_only_residual),
        null_residuals=nulls,
        empirical_p_value=p_value,
        exact_enumeration=exact,
        assignment_count=assignment_count,
        negative_sender_count=int(np.count_nonzero(signs < 0)),
        zero_sender_count=int(np.count_nonzero(signs == 0)),
        observed_signs=signs,
        sender_magnitudes=magnitudes,
    )
