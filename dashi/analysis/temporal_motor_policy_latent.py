"""Consumer-relative temporal latent ladder for future Drosophila kinematics.

This module is deliberately downstream of a same-trial neural/behaviour binding.
It asks whether a compact code fit from *past/present neural history only* carries
held-out information about future kinematics.  Predictive success is not promoted
to literal motor-plan identity, memory content, causation, or physical latent
dimensionality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TemporalMotorSamples:
    neural_history: np.ndarray
    future_behaviour: np.ndarray
    current_behaviour: np.ndarray
    source_time_index: np.ndarray
    target_time_index: np.ndarray


@dataclass(frozen=True)
class TemporalMotorLatentScore:
    dimension: int
    future_mae: float
    future_r2: float
    persistence_mae: float
    train_mean_mae: float
    neural_improves_over_persistence: bool
    motor_policy_identity_certified: bool = False
    causal_plan_certified: bool = False


@dataclass(frozen=True)
class TemporalMotorLatentResult:
    history_bins: int
    future_lag: int
    feature_means: np.ndarray
    feature_scales: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    train_sample_count: int
    held_out_sample_count: int
    purged_sample_count: int
    train_source_time_index: np.ndarray
    held_out_source_time_index: np.ndarray
    scores: tuple[TemporalMotorLatentScore, ...]
    target_semantics: str
    motor_policy_identity_certified: bool = False
    memory_content_certified: bool = False
    physical_latent_dimension_identified: bool = False


def _as_time_by_feature(values: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim == 1:
        out = out[:, None]
    if out.ndim != 2:
        raise ValueError(f"{name} must be [time,feature]")
    if out.shape[0] < 1 or out.shape[1] < 1:
        raise ValueError(f"{name} must be non-empty")
    return out


def build_temporal_motor_samples(
    neural_time_by_feature: np.ndarray,
    behaviour_time_by_feature: np.ndarray,
    *,
    history_bins: int,
    future_lag: int,
) -> TemporalMotorSamples:
    """Build causal neural-history -> future-behaviour samples.

    A sample anchored at time ``t`` uses neural observations only from
    ``t-history_bins+1 .. t`` and targets behaviour at ``t+future_lag``.
    """

    neural = _as_time_by_feature(neural_time_by_feature, name="neural")
    behaviour = _as_time_by_feature(behaviour_time_by_feature, name="behaviour")
    if neural.shape[0] != behaviour.shape[0]:
        raise ValueError("neural and behaviour must share one synchronized time axis")
    if history_bins < 1:
        raise ValueError("history_bins must be >= 1")
    if future_lag < 1:
        raise ValueError("future_lag must be >= 1 for a predictive motor-latent test")

    first_source = history_bins - 1
    last_source = neural.shape[0] - future_lag - 1
    if last_source < first_source:
        raise ValueError("recording is too short for requested history and future lag")

    source = np.arange(first_source, last_source + 1, dtype=np.int64)
    target = source + future_lag
    histories = np.stack(
        [
            neural[t - history_bins + 1 : t + 1].reshape(-1)
            for t in source
        ],
        axis=0,
    )
    return TemporalMotorSamples(
        neural_history=np.asarray(histories, dtype=float),
        future_behaviour=np.asarray(behaviour[target], dtype=float),
        current_behaviour=np.asarray(behaviour[source], dtype=float),
        source_time_index=source,
        target_time_index=target,
    )


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    ss_res = float(np.sum(np.square(true - pred)))
    centered = true - float(np.mean(true))
    ss_tot = float(np.sum(np.square(centered)))
    if ss_tot <= 1e-15:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def _fit_linear(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    design_train = np.column_stack((np.ones(train_x.shape[0], dtype=float), train_x))
    beta, *_ = np.linalg.lstsq(design_train, train_y, rcond=None)
    design_test = np.column_stack((np.ones(test_x.shape[0], dtype=float), test_x))
    return np.asarray(design_test @ beta, dtype=float)


def evaluate_temporal_motor_latent_ladder(
    neural_time_by_feature: np.ndarray,
    behaviour_time_by_feature: np.ndarray,
    *,
    dimensions: Sequence[int],
    history_bins: int,
    future_lag: int,
    holdout_fraction: float = 0.2,
) -> TemporalMotorLatentResult:
    """Fit a training-neural-only PCA ladder and predict future held-out behavior.

    The final temporal block is held out.  A purge interval of
    ``history_bins + future_lag - 1`` samples separates fitting windows from the
    held-out windows so overlapping neural histories/future targets cannot leak
    through the split boundary.
    """

    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must lie strictly between 0 and 0.5")
    requested = tuple(int(d) for d in dimensions)
    if not requested or any(d < 1 for d in requested):
        raise ValueError("dimensions must contain positive integers")
    if len(set(requested)) != len(requested):
        raise ValueError("dimensions must be unique")

    samples = build_temporal_motor_samples(
        neural_time_by_feature,
        behaviour_time_by_feature,
        history_bins=history_bins,
        future_lag=future_lag,
    )
    n_samples = samples.neural_history.shape[0]
    holdout_count = max(1, int(np.ceil(n_samples * holdout_fraction)))
    held_start = n_samples - holdout_count
    purge = history_bins + future_lag - 1
    train_stop = held_start - purge
    if train_stop < 3:
        raise ValueError("too few training samples after temporal purge")

    train_x = samples.neural_history[:train_stop]
    held_x = samples.neural_history[held_start:]
    train_y = samples.future_behaviour[:train_stop]
    held_y = samples.future_behaviour[held_start:]
    held_current = samples.current_behaviour[held_start:]

    means = np.mean(train_x, axis=0)
    scales = np.std(train_x, axis=0)
    scales = np.asarray(scales, dtype=float)
    scales[scales <= 1e-12] = 1.0
    z_train = (train_x - means) / scales
    z_held = (held_x - means) / scales
    _u, singular_values, components = np.linalg.svd(z_train, full_matrices=False)

    max_dimension = int(components.shape[0])
    if any(d > max_dimension for d in requested):
        raise ValueError(
            f"requested dimension exceeds training latent rank {max_dimension}"
        )
    train_latent = z_train @ components.T
    held_latent = z_held @ components.T

    train_mean = np.mean(train_y, axis=0, keepdims=True)
    mean_prediction = np.repeat(train_mean, held_y.shape[0], axis=0)
    persistence_mae = _mae(held_y, held_current)
    train_mean_mae = _mae(held_y, mean_prediction)

    scores: list[TemporalMotorLatentScore] = []
    for dimension in requested:
        prediction = _fit_linear(
            train_latent[:, :dimension],
            train_y,
            held_latent[:, :dimension],
        )
        future_mae = _mae(held_y, prediction)
        scores.append(
            TemporalMotorLatentScore(
                dimension=dimension,
                future_mae=future_mae,
                future_r2=_r2(held_y, prediction),
                persistence_mae=persistence_mae,
                train_mean_mae=train_mean_mae,
                neural_improves_over_persistence=bool(future_mae < persistence_mae),
            )
        )

    return TemporalMotorLatentResult(
        history_bins=history_bins,
        future_lag=future_lag,
        feature_means=np.asarray(means, dtype=float),
        feature_scales=scales,
        components=np.asarray(components, dtype=float),
        singular_values=np.asarray(singular_values, dtype=float),
        train_sample_count=int(train_stop),
        held_out_sample_count=int(holdout_count),
        purged_sample_count=int(purge),
        train_source_time_index=np.asarray(samples.source_time_index[:train_stop]),
        held_out_source_time_index=np.asarray(samples.source_time_index[held_start:]),
        scores=tuple(scores),
        target_semantics=(
            "future synchronized kinematics from a PCA code fit on neural-history geometry only; "
            "predictive adequacy is consumer-relative and does not certify motor-plan identity"
        ),
    )
