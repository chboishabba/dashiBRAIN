from dashi.analysis.benchmark import (
    FunctionalEffectorBenchmark,
    PredictorKind,
    StructureFunctionBenchmark,
    dashi_beats_baseline,
)
from dashi.analysis.embodied import ScientificSource


def test_structure_function_benchmark_requires_disjoint_folds():
    try:
        StructureFunctionBenchmark(
            direct_feature=lambda a, b: 1.0,
            path_feature=lambda a, b: 2.0,
            predict=lambda kind, x: x,
            observed=lambda a, b: 1.5,
            residual=lambda p, o: abs(p - o),
            registration_confidence=lambda _: 1.0,
            training_fold="same",
            held_out_fold="same",
            folds_disjoint=lambda a, b: a != b,
        )
    except ValueError as exc:
        assert "disjoint" in str(exc)
    else:
        raise AssertionError("fold leakage must be rejected")


def test_evaluate_pair_preserves_predictor_separation():
    b = StructureFunctionBenchmark(
        direct_feature=lambda a, c: 1.0,
        path_feature=lambda a, c: 3.0,
        predict=lambda kind, x: (
            x if kind is not PredictorKind.DASHI else 2.0
        ),
        observed=lambda a, c: 2.0,
        residual=lambda p, o: abs(p - o),
        registration_confidence=lambda _: 1.0,
        training_fold="train",
        held_out_fold="test",
        folds_disjoint=lambda a, c: a != c,
    )
    r = b.evaluate_pair(("n1", "n2"), ("f1", "f2"))
    assert r[PredictorKind.DIRECT_EDGE] == 1.0
    assert r[PredictorKind.PATH_AWARE] == 1.0
    assert r[PredictorKind.DASHI] == 0.0


def test_dashi_improvement_is_measured_not_assumed():
    assert dashi_beats_baseline([0.1, 0.2], [0.4, 0.5]) is True
    assert dashi_beats_baseline([0.6], [0.2]) is False


def test_effector_benchmark_reports_missing_downstream_receipts():
    b = FunctionalEffectorBenchmark(
        infer_descending_vnc=lambda x: x,
        infer_motor_drive=lambda x: x,
        infer_effector=lambda x: x,
        project_effector=lambda x: x,
        body_from_effector=lambda x: x,
        predict_behaviour=lambda x: x,
        observed_behaviour=lambda x: x,
        residual=lambda p, o: abs(p - o),
    )
    assert b.downstream_receipts_complete is False


def test_effector_benchmark_closes_only_with_separate_receipts():
    actuation = ScientificSource("A", "Actuation", "doi:10.1/a")
    biomechanics = ScientificSource("B", "Biomechanics", "doi:10.1/b")
    behaviour = ScientificSource("C", "Behaviour", "doi:10.1/c")
    b = FunctionalEffectorBenchmark(
        infer_descending_vnc=lambda x: x + 1,
        infer_motor_drive=lambda x: x + 1,
        infer_effector=lambda x: x + 1,
        project_effector=lambda x: x,
        body_from_effector=lambda x: x + 1,
        predict_behaviour=lambda x: x,
        observed_behaviour=lambda _: 4,
        residual=lambda p, o: abs(p - o),
        actuation_source=actuation,
        biomechanics_source=biomechanics,
        behaviour_source=behaviour,
    )
    assert b.downstream_receipts_complete is True
    assert b.evaluate(0) == 0
