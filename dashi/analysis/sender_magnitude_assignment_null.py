"""Test sender-indexed magnitude on the scale-free MaleCNS carrier.

After the scale/shape and polarity experiments, the remaining compressed
candidate is a sender-indexed magnitude field modulating relative wiring shape.
This module separates two non-equivalent magnitude definitions:

    net_magnitude[i] = |sum_j S[i,j]| / sum_j |D[i,j]|
    absolute_ratio_magnitude[i] = sum_j |S[i,j]| / sum_j |D[i,j]|

The first is |a_i| from the validated sender-scalar compression.  The second is
the D-weighted mean absolute signed ratio and must not be silently identified
with |a_i| when a sender row contains mixed signs.

For a scale-free test, D is decomposed to row-relative shape P and candidate
carriers are P[i,j] * magnitude[i].  A sender-assignment null keeps P and the
entire multiset of magnitudes fixed while permuting only which sender receives
which magnitude.  This tests sender-indexed gain assignment rather than global
magnitude distribution, absolute coupling scale, polarity, or pair-specific R.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dashi.analysis.network_scale_shape_discriminator import decompose_network_scale_shape
from dashi.analysis.overlap_controlled_structure_function import (
    evaluate_overlap_controlled_leave_one_region_out,
)
from dashi.analysis.signed_fibre_compression import sender_scalar_signed_direct
from dashi.analysis.signed_fibre_discriminator import singleton_family


@dataclass(frozen=True)
class SenderMagnitudeAssignmentNullResult:
    net_magnitude_residual: float
    absolute_ratio_magnitude_residual: float
    net_assignment_null_residuals: np.ndarray
    net_assignment_empirical_p_value: float
    net_magnitudes: np.ndarray
    absolute_ratio_magnitudes: np.ndarray


def sender_magnitude_fields(
    unsigned_direct: np.ndarray,
    signed_direct: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return |net signed tendency| and mean absolute signed-ratio magnitude."""
    d = np.asarray(unsigned_direct, dtype=float)
    s = np.asarray(signed_direct, dtype=float)
    if d.shape != s.shape or d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("unsigned_direct and signed_direct must be aligned square matrices")

    _scalar_matrix, tendency = sender_scalar_signed_direct(d, s)
    net = np.abs(tendency)

    denom = np.sum(np.abs(d), axis=1)
    absolute_ratio = np.zeros(d.shape[0], dtype=float)
    nz = denom > 1e-15
    absolute_ratio[nz] = np.sum(np.abs(s[nz]), axis=1) / denom[nz]
    return net, absolute_ratio


def scale_free_sender_gain_matrix(
    unsigned_direct: np.ndarray,
    magnitude: np.ndarray,
) -> np.ndarray:
    """Return P * m where P is exact L1 row-relative unsigned wiring shape."""
    d = np.asarray(unsigned_direct, dtype=float)
    m = np.asarray(magnitude, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("unsigned_direct must be square")
    if m.shape != (d.shape[0],):
        raise ValueError("magnitude must align with sender regions")
    p = decompose_network_scale_shape(d, d).row_shape
    return p * m[:, None]


def evaluate_sender_magnitude_assignment_null(
    regions: tuple[str, ...],
    unsigned_direct: np.ndarray,
    signed_direct: np.ndarray,
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    *,
    n_null: int = 999,
    seed: int = 0,
    correlation_threshold: float = 0.98,
) -> SenderMagnitudeAssignmentNullResult:
    """Test whether the anatomical assignment of |a_i| matters with P fixed."""
    if n_null < 1:
        raise ValueError("n_null must be >= 1")
    d = np.asarray(unsigned_direct, dtype=float)
    s = np.asarray(signed_direct, dtype=float)
    obs = np.asarray(observed, dtype=float)
    kernel = np.asarray(overlap_kernel, dtype=float)
    n = len(regions)
    if any(x.shape != (n, n) for x in (d, s, obs, kernel)):
        raise ValueError("all matrices must align with regions")

    net, absolute_ratio = sender_magnitude_fields(d, s)

    def score(name: str, matrix: np.ndarray) -> float:
        return evaluate_overlap_controlled_leave_one_region_out(
            singleton_family(regions, name, np.asarray(matrix, dtype=float).T),
            obs,
            kernel,
            correlation_threshold=correlation_threshold,
        ).weighted_mean_residual

    net_matrix = scale_free_sender_gain_matrix(d, net)
    abs_matrix = scale_free_sender_gain_matrix(d, absolute_ratio)
    net_score = score("relative_shape_times_net_sender_magnitude", net_matrix)
    abs_score = score("relative_shape_times_absolute_ratio_magnitude", abs_matrix)

    rng = np.random.default_rng(seed)
    nulls = np.empty(n_null, dtype=float)
    for draw in range(n_null):
        permuted = net[rng.permutation(n)]
        nulls[draw] = score(
            f"net_sender_magnitude_assignment_{draw}",
            scale_free_sender_gain_matrix(d, permuted),
        )

    p_value = float((1 + np.count_nonzero(nulls <= net_score)) / (n_null + 1))
    return SenderMagnitudeAssignmentNullResult(
        net_magnitude_residual=float(net_score),
        absolute_ratio_magnitude_residual=float(abs_score),
        net_assignment_null_residuals=nulls,
        net_assignment_empirical_p_value=p_value,
        net_magnitudes=net,
        absolute_ratio_magnitudes=absolute_ratio,
    )
