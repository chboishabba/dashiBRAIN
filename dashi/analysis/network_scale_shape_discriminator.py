"""Decompose MaleCNS regional structure into scale, relative shape, and sign ratio.

The region-level direct carrier D is already membership-mass normalized by the
real-data aggregation pipeline, so it is a density-like coupling rather than raw
synapse count.  For each sender region i, decompose

    D[i,j] = out_strength[i] * row_shape[i,j]

where out_strength[i] = sum_j |D[i,j]| and row_shape is L1 row-normalized.
For the coarse transmitter-signed carrier S, additionally define

    sign_ratio[i,j] = S[i,j] / D[i,j]

on non-zero unsigned support, giving

    S = out_strength[:,None] * row_shape * sign_ratio.

This lets the held-out consumer distinguish questions about absolute/density-like
regional strength from relative wiring shape and relative signed tendency.
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
class NetworkScaleShapeDecomposition:
    out_strength: np.ndarray
    row_shape: np.ndarray
    sign_ratio: np.ndarray


@dataclass(frozen=True)
class NetworkScaleShapeResult:
    absolute_reverse_residual: float
    relative_shape_reverse_residual: float
    strength_only_reverse_residual: float
    signed_reverse_residual: float
    relative_signed_shape_reverse_residual: float
    reconstruction_error_unsigned: float
    reconstruction_error_signed: float


def decompose_network_scale_shape(
    unsigned_direct: np.ndarray,
    signed_direct: np.ndarray,
) -> NetworkScaleShapeDecomposition:
    d = np.asarray(unsigned_direct, dtype=float)
    s = np.asarray(signed_direct, dtype=float)
    if d.shape != s.shape or d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("unsigned_direct and signed_direct must be aligned square matrices")
    strength = np.sum(np.abs(d), axis=1)
    shape = np.zeros_like(d, dtype=float)
    nz = strength > 1e-15
    shape[nz] = d[nz] / strength[nz, None]
    ratio = signed_ratio_to_unsigned(d, s)
    return NetworkScaleShapeDecomposition(strength, shape, ratio)


def reconstruct_unsigned_from_scale_shape(
    decomposition: NetworkScaleShapeDecomposition,
) -> np.ndarray:
    return decomposition.out_strength[:, None] * decomposition.row_shape


def reconstruct_signed_from_scale_shape(
    decomposition: NetworkScaleShapeDecomposition,
) -> np.ndarray:
    return (
        decomposition.out_strength[:, None]
        * decomposition.row_shape
        * decomposition.sign_ratio
    )


def sender_strength_only_matrix(unsigned_direct: np.ndarray) -> np.ndarray:
    """Retain sender strength but erase target-specific relative wiring shape.

    Each sender spreads its total absolute regional coupling uniformly across all
    non-self targets.  This is deliberately a scale-only control, not a plausible
    biological network model.
    """
    d = np.asarray(unsigned_direct, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("unsigned_direct must be square")
    n = d.shape[0]
    strength = np.sum(np.abs(d), axis=1)
    out = np.zeros_like(d, dtype=float)
    if n <= 1:
        return out
    for i in range(n):
        targets = np.ones(n, dtype=bool)
        targets[i] = False
        out[i, targets] = strength[i] / np.count_nonzero(targets)
    return out


def evaluate_network_scale_shape(
    regions: tuple[str, ...],
    unsigned_direct: np.ndarray,
    signed_direct: np.ndarray,
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    *,
    correlation_threshold: float = 0.98,
) -> NetworkScaleShapeResult:
    d = np.asarray(unsigned_direct, dtype=float)
    s = np.asarray(signed_direct, dtype=float)
    obs = np.asarray(observed, dtype=float)
    kernel = np.asarray(overlap_kernel, dtype=float)
    n = len(regions)
    if any(x.shape != (n, n) for x in (d, s, obs, kernel)):
        raise ValueError("all matrices must align with regions")

    dec = decompose_network_scale_shape(d, s)
    d_recon = reconstruct_unsigned_from_scale_shape(dec)
    s_recon = reconstruct_signed_from_scale_shape(dec)

    def score(name: str, matrix: np.ndarray) -> float:
        return evaluate_overlap_controlled_leave_one_region_out(
            singleton_family(regions, name, np.asarray(matrix).T),
            obs,
            kernel,
            correlation_threshold=correlation_threshold,
        ).weighted_mean_residual

    # Remove row scale while retaining exact target-relative wiring shape.
    relative_shape = dec.row_shape
    # Retain scale alone while erasing all target-specific shape.
    strength_only = sender_strength_only_matrix(d)
    # Retain relative unsigned shape and sign ratio but remove absolute sender scale.
    relative_signed_shape = dec.row_shape * dec.sign_ratio

    return NetworkScaleShapeResult(
        absolute_reverse_residual=score("absolute_reverse", d),
        relative_shape_reverse_residual=score("relative_shape_reverse", relative_shape),
        strength_only_reverse_residual=score("strength_only_reverse", strength_only),
        signed_reverse_residual=score("signed_reverse", s),
        relative_signed_shape_reverse_residual=score(
            "relative_signed_shape_reverse", relative_signed_shape
        ),
        reconstruction_error_unsigned=float(np.max(np.abs(d_recon - d))),
        reconstruction_error_signed=float(np.max(np.abs(s_recon - s))),
    )
