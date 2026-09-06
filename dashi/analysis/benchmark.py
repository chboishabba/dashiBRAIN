"""Drosophila structure -> function -> effector benchmark surfaces.

The benchmark compares structural predictors against held-out registered
functional observations, then permits a separately receipted continuation from
registered functional state through descending/VNC drive into effector/body
prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, Iterable, Mapping, Sequence, TypeVar

from dashi.analysis.embodied import (
    DROSOPHILA_CALCIUM_SOURCE,
    MALE_CNS_SOURCE,
    ScientificSource,
)
from dashi.analysis.functional_registration import TURNER_STRUCTURE_FUNCTION_SOURCE


class PredictorKind(str, Enum):
    DIRECT_EDGE = "direct_edge_baseline"
    PATH_AWARE = "path_aware_baseline"
    DASHI = "dashi_predictor"


StructuralUnitT = TypeVar("StructuralUnitT")
FunctionalUnitT = TypeVar("FunctionalUnitT")
FeatureT = TypeVar("FeatureT")
PredictionT = TypeVar("PredictionT")
ObservedT = TypeVar("ObservedT")
ResidualT = TypeVar("ResidualT")
FoldT = TypeVar("FoldT")
ConfidenceT = TypeVar("ConfidenceT")


@dataclass(frozen=True)
class StructureFunctionBenchmark(Generic[
    StructuralUnitT,
    FunctionalUnitT,
    FeatureT,
    PredictionT,
    ObservedT,
    ResidualT,
    FoldT,
    ConfidenceT,
]):
    direct_feature: Callable[[StructuralUnitT, StructuralUnitT], FeatureT]
    path_feature: Callable[[StructuralUnitT, StructuralUnitT], FeatureT]
    predict: Callable[[PredictorKind, FeatureT], PredictionT]
    observed: Callable[[FunctionalUnitT, FunctionalUnitT], ObservedT]
    residual: Callable[[PredictionT, ObservedT], ResidualT]
    registration_confidence: Callable[[FunctionalUnitT], ConfidenceT]
    training_fold: FoldT
    held_out_fold: FoldT
    folds_disjoint: Callable[[FoldT, FoldT], bool]
    source: ScientificSource = TURNER_STRUCTURE_FUNCTION_SOURCE

    def __post_init__(self) -> None:
        if not self.folds_disjoint(self.training_fold, self.held_out_fold):
            raise ValueError("training and held-out folds must be disjoint")

    def evaluate_pair(
        self,
        su: tuple[StructuralUnitT, StructuralUnitT],
        fu: tuple[FunctionalUnitT, FunctionalUnitT],
    ) -> Mapping[PredictorKind, ResidualT]:
        a, b = su
        fa, fb = fu
        obs = self.observed(fa, fb)
        direct = self.direct_feature(a, b)
        path = self.path_feature(a, b)
        return {
            PredictorKind.DIRECT_EDGE: self.residual(
                self.predict(PredictorKind.DIRECT_EDGE, direct), obs
            ),
            PredictorKind.PATH_AWARE: self.residual(
                self.predict(PredictorKind.PATH_AWARE, path), obs
            ),
            PredictorKind.DASHI: self.residual(
                self.predict(PredictorKind.DASHI, path), obs
            ),
        }


RegisteredFunctionalStateT = TypeVar("RegisteredFunctionalStateT")
DescendingVNCT = TypeVar("DescendingVNCT")
MotorDriveT = TypeVar("MotorDriveT")
EffectorStateT = TypeVar("EffectorStateT")
EffectorCoefficientT = TypeVar("EffectorCoefficientT")
BodyStateT = TypeVar("BodyStateT")
BehaviourT = TypeVar("BehaviourT")
BehaviourResidualT = TypeVar("BehaviourResidualT")


@dataclass(frozen=True)
class FunctionalEffectorBenchmark(Generic[
    RegisteredFunctionalStateT,
    DescendingVNCT,
    MotorDriveT,
    EffectorStateT,
    EffectorCoefficientT,
    BodyStateT,
    BehaviourT,
    BehaviourResidualT,
]):
    infer_descending_vnc: Callable[[RegisteredFunctionalStateT], DescendingVNCT]
    infer_motor_drive: Callable[[DescendingVNCT], MotorDriveT]
    infer_effector: Callable[[MotorDriveT], EffectorStateT]
    project_effector: Callable[[EffectorStateT], EffectorCoefficientT]
    body_from_effector: Callable[[EffectorStateT], BodyStateT]
    predict_behaviour: Callable[[BodyStateT], BehaviourT]
    observed_behaviour: Callable[[RegisteredFunctionalStateT], BehaviourT]
    residual: Callable[[BehaviourT, BehaviourT], BehaviourResidualT]
    connectome_source: ScientificSource = MALE_CNS_SOURCE
    functional_source: ScientificSource = DROSOPHILA_CALCIUM_SOURCE
    actuation_source: ScientificSource | None = None
    biomechanics_source: ScientificSource | None = None
    behaviour_source: ScientificSource | None = None

    @property
    def downstream_receipts_complete(self) -> bool:
        return all(
            source is not None
            for source in (
                self.actuation_source,
                self.biomechanics_source,
                self.behaviour_source,
            )
        )

    def evaluate(self, state: RegisteredFunctionalStateT) -> BehaviourResidualT:
        descending = self.infer_descending_vnc(state)
        drive = self.infer_motor_drive(descending)
        effector = self.infer_effector(drive)
        body = self.body_from_effector(effector)
        predicted = self.predict_behaviour(body)
        observed = self.observed_behaviour(state)
        return self.residual(predicted, observed)


def mean_numeric_residual(values: Iterable[float]) -> float:
    values = tuple(values)
    if not values:
        raise ValueError("at least one residual is required")
    return sum(values) / len(values)


def dashi_beats_baseline(
    dashi_residuals: Sequence[float], baseline_residuals: Sequence[float]
) -> bool:
    """Empirical comparison helper; this is a result predicate, not an assumption."""

    return mean_numeric_residual(dashi_residuals) < mean_numeric_residual(
        baseline_residuals
    )
