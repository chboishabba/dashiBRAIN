"""Derive the sender-gain/relative-shape candidate from the local hyperfabric.

This is a provenance/representation adapter, not a new predictive model.  The
candidate m_i P_ij is computed only after projecting the retained
``direct_forward`` and ``signed_forward`` coordinates from the richer local
fibre hyperfabric.  It therefore tests whether the current compressed candidate
is genuinely a function of that richer chart rather than an unrelated static
feature table.

The adapter does not claim that m_i P_ij is consumer-sufficient.  Adequacy is
still decided by the held-out consumer and replication/null programme.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dashi.analysis.local_fibre_hyperfabric import LocalFibreHyperfabric
from dashi.analysis.sender_magnitude_shape_composition import (
    SenderMagnitudeShapeComposition,
    compose_sender_magnitude_shape,
)


@dataclass(frozen=True)
class HyperfabricSenderGainProjection:
    regions: tuple[str, ...]
    unsigned_direct: np.ndarray
    signed_direct: np.ndarray
    composition: SenderMagnitudeShapeComposition

    @property
    def magnitude_sender_shape(self) -> np.ndarray:
        return self.composition.magnitude_sender_shape


def project_sender_gain_from_hyperfabric(
    fabric: LocalFibreHyperfabric,
    *,
    unsigned_coordinate: str = "direct_forward",
    signed_coordinate: str = "signed_forward",
) -> HyperfabricSenderGainProjection:
    """Project D and S from the fabric, then derive P, a_i, |a_i| and m_i P_ij."""
    chart = fabric.project_chart((unsigned_coordinate, signed_coordinate))
    d = np.asarray(chart.fibres[unsigned_coordinate], dtype=float)
    s = np.asarray(chart.fibres[signed_coordinate], dtype=float)
    composition = compose_sender_magnitude_shape(d, s)
    return HyperfabricSenderGainProjection(tuple(chart.regions), d, s, composition)


def sender_gain_projection_matches_direct_composition(
    fabric: LocalFibreHyperfabric,
    unsigned_direct: np.ndarray,
    signed_direct: np.ndarray,
    *,
    atol: float = 0.0,
) -> bool:
    """Check that hyperfabric-derived mP equals direct composition on the same chart."""
    projected = project_sender_gain_from_hyperfabric(fabric)
    direct = compose_sender_magnitude_shape(unsigned_direct, signed_direct)
    return bool(
        np.allclose(projected.unsigned_direct, np.asarray(unsigned_direct), rtol=0.0, atol=atol)
        and np.allclose(projected.signed_direct, np.asarray(signed_direct), rtol=0.0, atol=atol)
        and np.allclose(
            projected.composition.relative_shape,
            direct.relative_shape,
            rtol=0.0,
            atol=atol,
        )
        and np.allclose(
            projected.composition.sender_magnitude,
            direct.sender_magnitude,
            rtol=0.0,
            atol=atol,
        )
        and np.allclose(
            projected.magnitude_sender_shape,
            direct.magnitude_sender_shape,
            rtol=0.0,
            atol=atol,
        )
    )
