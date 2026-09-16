"""Discriminate the compressed MaleCNS signed fibre from magnitude-only controls.

The nested compression experiment selected ``signed_reverse`` in 25/26 outer
folds.  This module asks a narrower question than the earlier whole-graph nulls:
does the selected carrier use transmitter-sign information, or merely the
magnitude/direction geometry already present in the signed matrix?

Let D be unsigned region connectivity and S the presynaptic-transmitter-signed
region matrix.  Controls keep D fixed and alter only the signed layer:

* ``abs(S)`` removes polarity while retaining exact pairwise signed magnitude;
* sender-pattern permutation computes R = S / D where D > 0 and permutes whole
  rows of R before reconstructing S_null = D * R_perm.  Thus unsigned pairwise
  connectivity is identical in every draw while coarse sender-specific signed
  tendencies are reassigned across source regions.

These are benchmark controls, not receptor-resolved physiology.  The upstream
MaleCNS sign carrier is itself a coarse presynaptic-transmitter annotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dashi.analysis.ndim_structure_function import StructuralFibreFamily
from dashi.analysis.overlap_controlled_structure_function import (
    evaluate_overlap_controlled_leave_one_region_out,
)


@dataclass(frozen=True)
class SignedFibreDiscriminatorResult:
    true_residual: float
    unsigned_reverse_residual: float
    magnitude_only_residual: float
    sender_pattern_null_residuals: np.ndarray
    sender_pattern_empirical_p_value: float


def singleton_family(
    regions: tuple[str, ...],
    name: str,
    matrix: np.ndarray,
) -> StructuralFibreFamily:
    x = np.asarray(matrix, dtype=float)
    n = len(regions)
    if x.shape != (n, n):
        raise ValueError("singleton fibre matrix must align with regions")
    return StructuralFibreFamily(regions, {name: x})


def signed_ratio_to_unsigned(unsigned_direct: np.ndarray, signed_direct: np.ndarray) -> np.ndarray:
    """Return S/D where defined, with zero ratio on zero unsigned support."""
    d = np.asarray(unsigned_direct, dtype=float)
    s = np.asarray(signed_direct, dtype=float)
    if d.shape != s.shape or d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("unsigned_direct and signed_direct must be aligned square matrices")
    ratio = np.zeros_like(s, dtype=float)
    nz = np.abs(d) > 1e-15
    ratio[nz] = s[nz] / d[nz]
    return ratio


def sender_pattern_permuted_signed_direct(
    unsigned_direct: np.ndarray,
    signed_direct: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    """Keep unsigned D fixed while assigning each sender another sender's S/D row."""
    d = np.asarray(unsigned_direct, dtype=float)
    ratio = signed_ratio_to_unsigned(d, signed_direct)
    p = np.asarray(permutation, dtype=int)
    n = d.shape[0]
    if p.shape != (n,) or set(p.tolist()) != set(range(n)):
        raise ValueError("permutation must contain every region index exactly once")
    return d * ratio[p, :]


def evaluate_signed_reverse_discriminator(
    regions: tuple[str, ...],
    unsigned_direct: np.ndarray,
    signed_direct: np.ndarray,
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    *,
    n_null: int = 100,
    seed: int = 0,
    correlation_threshold: float = 0.98,
) -> SignedFibreDiscriminatorResult:
    """Compare true signed_reverse with unsigned/magnitude/sign-assignment controls."""
    if n_null < 1:
        raise ValueError("n_null must be >= 1")
    d = np.asarray(unsigned_direct, dtype=float)
    s = np.asarray(signed_direct, dtype=float)
    obs = np.asarray(observed, dtype=float)
    k = np.asarray(overlap_kernel, dtype=float)
    n = len(regions)
    if d.shape != (n, n) or s.shape != (n, n) or obs.shape != (n, n) or k.shape != (n, n):
        raise ValueError("all matrices must align with regions")

    def score(name: str, matrix: np.ndarray) -> float:
        return evaluate_overlap_controlled_leave_one_region_out(
            singleton_family(regions, name, matrix),
            obs,
            k,
            correlation_threshold=correlation_threshold,
        ).weighted_mean_residual

    true_residual = score("signed_reverse", s.T)
    unsigned_reverse_residual = score("direct_reverse", d.T)
    magnitude_only_residual = score("abs_signed_reverse", np.abs(s).T)

    rng = np.random.default_rng(seed)
    nulls = np.empty(n_null, dtype=float)
    for draw in range(n_null):
        s_null = sender_pattern_permuted_signed_direct(d, s, rng.permutation(n))
        nulls[draw] = score("sender_pattern_permuted_signed_reverse", s_null.T)

    p_value = float((1 + np.count_nonzero(nulls <= true_residual)) / (n_null + 1))
    return SignedFibreDiscriminatorResult(
        true_residual=float(true_residual),
        unsigned_reverse_residual=float(unsigned_reverse_residual),
        magnitude_only_residual=float(magnitude_only_residual),
        sender_pattern_null_residuals=nulls,
        sender_pattern_empirical_p_value=p_value,
    )
