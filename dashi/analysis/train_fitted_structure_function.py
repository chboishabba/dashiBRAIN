"""Train-only linear structure/function comparator for MaleCNS.

This module is deliberately separate from the fixed DASHI mixture.  It fits
coefficients only on the declared training region pairs and evaluates the frozen
model on held-out pairs.  No hyperparameter search, held-out tuning, or
post-hoc weight adjustment is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from dashi.analysis.structure_function_real import FunctionalAssociation, RegionStructuralFeatures


@dataclass(frozen=True)
class TrainFittedMixtureResult:
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    fit_pair_count: int
    held_out_pair_count: int
    held_out_residuals: np.ndarray
    held_out_prediction: np.ndarray
    held_out_observed: np.ndarray

    @property
    def mean_residual(self) -> float:
        return float(np.mean(self.held_out_residuals))


def _validate_masks(n: int, fit_mask: np.ndarray, held_out_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    fit = upper & np.asarray(fit_mask, dtype=bool)
    held = upper & np.asarray(held_out_mask, dtype=bool)
    if fit.shape != (n, n) or held.shape != (n, n):
        raise ValueError("fit and held-out masks must match region matrix shape")
    if np.any(fit & held):
        raise ValueError("fit and held-out masks must be disjoint")
    if not np.any(fit) or not np.any(held):
        raise ValueError("fit and held-out masks must both select at least one pair")
    return fit, held


def fit_train_only_structure_mixture(
    structural: RegionStructuralFeatures,
    functional: FunctionalAssociation,
    *,
    fit_mask: np.ndarray,
    held_out_mask: np.ndarray,
    include_signed: bool = True,
) -> TrainFittedMixtureResult:
    """Fit a least-squares structural mixture on training pairs only.

    Features are direct connectivity and two-hop connectivity, with signed
    direct connectivity included only when available and requested.  The fit has
    no intercept, matching the existing benchmark's zero-structure/zero-signal
    convention.  Held-out observations are never passed to ``lstsq``.
    """
    if structural.regions != functional.unit_ids:
        raise ValueError("functional units must be ordered exactly like structural regions")

    observed = np.asarray(functional.matrix, dtype=float)
    n = observed.shape[0]
    fit, held = _validate_masks(n, fit_mask, held_out_mask)

    features: list[np.ndarray] = [
        np.asarray(structural.direct, dtype=float),
        np.asarray(structural.two_hop, dtype=float),
    ]
    names: list[str] = ["direct", "two_hop"]
    if include_signed and structural.signed_direct is not None:
        features.append(np.asarray(structural.signed_direct, dtype=float))
        names.append("signed_direct")

    x_fit = np.column_stack([feature[fit] for feature in features])
    y_fit = observed[fit]
    coefficients, _, _, _ = np.linalg.lstsq(x_fit, y_fit, rcond=None)

    x_held = np.column_stack([feature[held] for feature in features])
    prediction = x_held @ coefficients
    y_held = observed[held]
    residuals = np.abs(prediction - y_held)

    return TrainFittedMixtureResult(
        feature_names=tuple(names),
        coefficients=np.asarray(coefficients, dtype=float),
        fit_pair_count=int(np.sum(fit)),
        held_out_pair_count=int(np.sum(held)),
        held_out_residuals=np.asarray(residuals, dtype=float),
        held_out_prediction=np.asarray(prediction, dtype=float),
        held_out_observed=np.asarray(y_held, dtype=float),
    )
