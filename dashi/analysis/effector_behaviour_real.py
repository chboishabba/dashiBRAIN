"""Real-data staging for motor/effector/behaviour evaluation.

The present MaleCNS programme can support coarse motor-module and behavioural
prediction before it can justify same-animal neuron->muscle identity.  This
module therefore works at explicitly declared module resolution and never treats
cross-animal motor atlases as literal same-animal identity maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MotorModuleMap:
    neuron_to_module: Mapping[str, str]
    module_to_effector: Mapping[str, str]
    same_animal_identity: bool = False


@dataclass(frozen=True)
class BehaviourPredictionResult:
    predicted: np.ndarray
    observed: np.ndarray
    residuals: np.ndarray
    module_ids: tuple[str, ...]

    @property
    def mean_residual(self) -> float:
        return float(np.mean(self.residuals))


def aggregate_motor_drive(
    neuron_ids: Sequence[str],
    drive: np.ndarray,
    mapping: MotorModuleMap,
) -> tuple[tuple[str, ...], np.ndarray]:
    x = np.asarray(drive, dtype=float)
    if x.ndim != 2:
        raise ValueError("drive must be [time, neuron]")
    if x.shape[1] != len(neuron_ids):
        raise ValueError("neuron_ids must align with drive columns")

    groups: dict[str, list[int]] = {}
    for i, nid in enumerate(neuron_ids):
        module = mapping.neuron_to_module.get(nid)
        if module:
            groups.setdefault(module, []).append(i)
    if not groups:
        raise ValueError("no motor neurons map to declared modules")

    modules = tuple(sorted(groups))
    agg = np.column_stack([np.mean(x[:, groups[m]], axis=1) for m in modules])
    return modules, agg


def fit_linear_effector_to_behaviour(
    train_effector: np.ndarray,
    train_behaviour: np.ndarray,
    test_effector: np.ndarray,
    test_behaviour: np.ndarray,
    *,
    module_ids: Sequence[str],
    ridge: float = 1e-6,
) -> BehaviourPredictionResult:
    x = np.asarray(train_effector, dtype=float)
    y = np.asarray(train_behaviour, dtype=float)
    xt = np.asarray(test_effector, dtype=float)
    yt = np.asarray(test_behaviour, dtype=float)
    if x.ndim != 2 or xt.ndim != 2:
        raise ValueError("effector matrices must be 2-D")
    if x.shape[1] != xt.shape[1]:
        raise ValueError("train/test effector dimensions differ")
    if y.ndim == 1:
        y = y[:, None]
    if yt.ndim == 1:
        yt = yt[:, None]
    gram = x.T @ x + ridge * np.eye(x.shape[1])
    beta = np.linalg.solve(gram, x.T @ y)
    pred = xt @ beta
    residuals = np.linalg.norm(pred - yt, axis=1)
    return BehaviourPredictionResult(pred, yt, residuals, tuple(module_ids))
