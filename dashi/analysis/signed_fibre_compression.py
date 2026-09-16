"""Compress the real MaleCNS signed fibre below one full region-pair matrix.

The nested fibre ladder selected ``signed_reverse`` almost universally.  This
module asks how much information inside that single fibre is actually needed by
the held-out structure/function consumer.

Let D be unsigned region connectivity and S the coarse presynaptic-transmitter-
signed region matrix.  Define R = S / D on non-zero unsigned support.

Two smaller carriers are tested:

* sender-scalar: retain one weighted signed tendency a_i per source region and
  reconstruct S_hat[i,j] = a_i D[i,j];
* ratio-low-rank: approximate R by a truncated SVD and reconstruct
  S_hat = D * R_k.

These are representation/compression tests only.  They do not turn the coarse
transmitter sign benchmark into receptor-resolved physiology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dashi.analysis.signed_fibre_discriminator import (
    signed_ratio_to_unsigned,
    singleton_family,
)
from dashi.analysis.overlap_controlled_structure_function import (
    evaluate_overlap_controlled_leave_one_region_out,
)


@dataclass(frozen=True)
class SignedFibreCompressionResult:
    true_signed_reverse_residual: float
    sender_scalar_residual: float
    rank_residuals: dict[int, float]
    sender_scalar_tendencies: np.ndarray


def sender_scalar_signed_direct(
    unsigned_direct: np.ndarray,
    signed_direct: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compress signed structure to one D-weighted signed tendency per sender."""
    d = np.asarray(unsigned_direct, dtype=float)
    s = np.asarray(signed_direct, dtype=float)
    if d.shape != s.shape or d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("unsigned_direct and signed_direct must be aligned square matrices")
    denom = np.sum(np.abs(d), axis=1)
    tendency = np.zeros(d.shape[0], dtype=float)
    nz = denom > 1e-15
    # This is the D-weighted average of R=S/D over each sender row.
    tendency[nz] = np.sum(s[nz], axis=1) / denom[nz]
    return d * tendency[:, None], tendency


def low_rank_ratio_signed_direct(
    unsigned_direct: np.ndarray,
    signed_direct: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Reconstruct D * R_k where R_k is the rank-k SVD approximation to S/D."""
    d = np.asarray(unsigned_direct, dtype=float)
    s = np.asarray(signed_direct, dtype=float)
    if d.shape != s.shape or d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("unsigned_direct and signed_direct must be aligned square matrices")
    n = d.shape[0]
    if not (1 <= rank <= n):
        raise ValueError("rank must be in [1, n_regions]")
    ratio = signed_ratio_to_unsigned(d, s)
    u, singular, vt = np.linalg.svd(ratio, full_matrices=False)
    ratio_k = (u[:, :rank] * singular[:rank]) @ vt[:rank, :]
    # Zero-support edges stay zero because D is held fixed.
    return d * ratio_k


def evaluate_signed_fibre_compression(
    regions: tuple[str, ...],
    unsigned_direct: np.ndarray,
    signed_direct: np.ndarray,
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    *,
    max_rank: int = 8,
    correlation_threshold: float = 0.98,
) -> SignedFibreCompressionResult:
    """Compare true S^T with scalar-sender and low-rank-ratio compressions."""
    d = np.asarray(unsigned_direct, dtype=float)
    s = np.asarray(signed_direct, dtype=float)
    obs = np.asarray(observed, dtype=float)
    kernel = np.asarray(overlap_kernel, dtype=float)
    n = len(regions)
    if any(x.shape != (n, n) for x in (d, s, obs, kernel)):
        raise ValueError("all matrices must align with regions")
    if max_rank < 1:
        raise ValueError("max_rank must be >= 1")
    max_rank = min(int(max_rank), n)

    def score(name: str, matrix: np.ndarray) -> float:
        return evaluate_overlap_controlled_leave_one_region_out(
            singleton_family(regions, name, matrix.T),
            obs,
            kernel,
            correlation_threshold=correlation_threshold,
        ).weighted_mean_residual

    true_score = score("signed_reverse", s)
    scalar_matrix, tendency = sender_scalar_signed_direct(d, s)
    scalar_score = score("sender_scalar_signed_reverse", scalar_matrix)
    rank_scores: dict[int, float] = {}
    for rank in range(1, max_rank + 1):
        rank_scores[rank] = score(
            f"ratio_rank_{rank}_signed_reverse",
            low_rank_ratio_signed_direct(d, s, rank),
        )

    return SignedFibreCompressionResult(
        true_signed_reverse_residual=float(true_score),
        sender_scalar_residual=float(scalar_score),
        rank_residuals=rank_scores,
        sender_scalar_tendencies=tendency,
    )
