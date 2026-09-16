"""Directly test the jointly compressed MaleCNS carrier m_i P_ij.

Earlier real-data experiments separately showed that (i) sender scale can be
removed from the signed carrier with little loss and (ii) binary polarity
placement is not distinguished once the signed carrier is compressed to one
sender scalar.  Those statements do not logically imply that both coordinates
can be dropped simultaneously.

This module evaluates the composed quotient directly.

Let

    D[i,j] = s_out[i] * P[i,j]
    a[i]   = sum_j S[i,j] / sum_j |D[i,j]|
    m[i]   = |a[i]|.

We compare singleton carriers built from

    S[i,j]             full signed regional coupling
    a[i] D[i,j]        signed sender-scalar on density-like coupling
    m[i] D[i,j]        magnitude-only sender-scalar on density-like coupling
    a[i] P[i,j]        signed sender tendency on relative wiring shape
    m[i] P[i,j]        magnitude-only sender tendency on relative wiring shape
    P[i,j]             relative unsigned wiring shape.

The last-but-one object is the actual joint quotient whose adequacy cannot be
inferred from the two one-axis experiments separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dashi.analysis.network_scale_shape_discriminator import (
    decompose_network_scale_shape,
)
from dashi.analysis.signed_fibre_compression import sender_scalar_signed_direct
from dashi.analysis.signed_fibre_discriminator import singleton_family
from dashi.analysis.overlap_controlled_structure_function import (
    evaluate_overlap_controlled_leave_one_region_out,
)


@dataclass(frozen=True)
class SenderMagnitudeShapeComposition:
    sender_tendency: np.ndarray
    sender_magnitude: np.ndarray
    relative_shape: np.ndarray
    signed_sender_density: np.ndarray
    magnitude_sender_density: np.ndarray
    signed_sender_shape: np.ndarray
    magnitude_sender_shape: np.ndarray


@dataclass(frozen=True)
class SenderMagnitudeShapeCompositionResult:
    full_signed_reverse_residual: float
    signed_sender_density_residual: float
    magnitude_sender_density_residual: float
    signed_sender_shape_residual: float
    magnitude_sender_shape_residual: float
    relative_shape_residual: float
    composition: SenderMagnitudeShapeComposition


def compose_sender_magnitude_shape(
    unsigned_direct: np.ndarray,
    signed_direct: np.ndarray,
) -> SenderMagnitudeShapeComposition:
    d = np.asarray(unsigned_direct, dtype=float)
    s = np.asarray(signed_direct, dtype=float)
    if d.shape != s.shape or d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("unsigned_direct and signed_direct must be aligned square matrices")

    signed_sender_density, tendency = sender_scalar_signed_direct(d, s)
    magnitude = np.abs(tendency)
    dec = decompose_network_scale_shape(d, s)
    p = dec.row_shape

    return SenderMagnitudeShapeComposition(
        sender_tendency=tendency,
        sender_magnitude=magnitude,
        relative_shape=p,
        signed_sender_density=signed_sender_density,
        magnitude_sender_density=d * magnitude[:, None],
        signed_sender_shape=p * tendency[:, None],
        magnitude_sender_shape=p * magnitude[:, None],
    )


def evaluate_sender_magnitude_shape_composition(
    regions: tuple[str, ...],
    unsigned_direct: np.ndarray,
    signed_direct: np.ndarray,
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    *,
    correlation_threshold: float = 0.98,
) -> SenderMagnitudeShapeCompositionResult:
    d = np.asarray(unsigned_direct, dtype=float)
    s = np.asarray(signed_direct, dtype=float)
    obs = np.asarray(observed, dtype=float)
    kernel = np.asarray(overlap_kernel, dtype=float)
    n = len(regions)
    if any(x.shape != (n, n) for x in (d, s, obs, kernel)):
        raise ValueError("all matrices must align with regions")

    comp = compose_sender_magnitude_shape(d, s)

    def score(name: str, matrix: np.ndarray) -> float:
        return evaluate_overlap_controlled_leave_one_region_out(
            singleton_family(regions, name, np.asarray(matrix, dtype=float).T),
            obs,
            kernel,
            correlation_threshold=correlation_threshold,
        ).weighted_mean_residual

    return SenderMagnitudeShapeCompositionResult(
        full_signed_reverse_residual=float(score("full_signed_reverse", s)),
        signed_sender_density_residual=float(
            score("signed_sender_density", comp.signed_sender_density)
        ),
        magnitude_sender_density_residual=float(
            score("magnitude_sender_density", comp.magnitude_sender_density)
        ),
        signed_sender_shape_residual=float(
            score("signed_sender_shape", comp.signed_sender_shape)
        ),
        magnitude_sender_shape_residual=float(
            score("magnitude_sender_shape", comp.magnitude_sender_shape)
        ),
        relative_shape_residual=float(score("relative_shape", comp.relative_shape)),
        composition=comp,
    )
