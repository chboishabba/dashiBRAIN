import numpy as np
import pytest

from dashi.analysis.consumer_relative_scorecard import (
    build_consumer_relative_scorecard,
    collect_overlap_controlled_loro_predictions,
    score_prediction_vector,
)
from dashi.analysis.ndim_structure_function import StructuralFibreFamily
from dashi.analysis.overlap_controlled_structure_function import (
    evaluate_overlap_controlled_leave_one_region_out,
)


def test_prediction_metrics_have_standard_meaning():
    observed = np.array([-0.5, 0.0, 0.5, 1.0], dtype=float)
    perfect = score_prediction_vector(observed, observed)

    assert perfect.pair_count == 4
    assert perfect.mae == pytest.approx(0.0)
    assert perfect.r2 == pytest.approx(1.0)
    assert perfect.pearson_r == pytest.approx(1.0)
    assert perfect.spearman_rho == pytest.approx(1.0)

    reversed_metrics = score_prediction_vector(observed[::-1], observed)
    assert reversed_metrics.pearson_r == pytest.approx(-1.0)
    assert reversed_metrics.spearman_rho == pytest.approx(-1.0)


def test_scorecard_reports_baseline_normalized_gains_without_promoting_sufficiency():
    observed = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    candidate = np.array([0.0, 1.0, 1.5, 2.5], dtype=float)
    baselines = {
        "zero": np.zeros_like(observed),
        "mean": np.full_like(observed, 1.5),
    }

    scorecard = build_consumer_relative_scorecard(
        candidate,
        observed,
        baseline_predictions=baselines,
        candidate_name="finite candidate",
        target_semantics="synthetic declared consumer",
        latent_dimension=2,
        description_length=2.0,
        description_length_unit="declared coordinates",
    )

    assert scorecard.candidate_name == "finite candidate"
    assert scorecard.latent_dimension == 2
    assert scorecard.consumer_sufficiency_certified is False
    by_name = {comparison.baseline_name: comparison for comparison in scorecard.baselines}
    assert set(by_name) == {"zero", "mean"}
    assert by_name["zero"].mae_absolute_improvement > 0
    assert by_name["zero"].mae_fractional_improvement > 0
    assert by_name["mean"].mae_absolute_improvement > 0


def test_overlap_controlled_prediction_bundle_preserves_existing_weighted_loro_mae():
    regions = ("A", "B", "C", "D", "E")
    structural = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 0.0, 1.5, 2.5, 3.5],
            [2.0, 1.5, 0.0, 1.2, 2.2],
            [3.0, 2.5, 1.2, 0.0, 1.8],
            [4.0, 3.5, 2.2, 1.8, 0.0],
        ],
        dtype=float,
    )
    family = StructuralFibreFamily(regions, {"candidate": structural})
    overlap = np.array(
        [
            [1.0, 0.10, 0.20, 0.05, 0.15],
            [0.10, 1.0, 0.12, 0.18, 0.08],
            [0.20, 0.12, 1.0, 0.07, 0.11],
            [0.05, 0.18, 0.07, 1.0, 0.16],
            [0.15, 0.08, 0.11, 0.16, 1.0],
        ],
        dtype=float,
    )
    observed = 0.25 + 0.12 * structural + 0.3 * overlap
    np.fill_diagonal(observed, 1.0)

    old = evaluate_overlap_controlled_leave_one_region_out(family, observed, overlap)
    bundle = collect_overlap_controlled_loro_predictions(family, observed, overlap)
    metrics = score_prediction_vector(bundle.prediction, bundle.observed)

    assert bundle.prediction.shape == bundle.observed.shape
    assert bundle.prediction.size == len(regions) * (len(regions) - 1)
    assert metrics.mae == pytest.approx(old.weighted_mean_residual, abs=1e-12)
    assert tuple(bundle.fold_regions) == regions
    assert len(bundle.selected_fibres_by_fold) == len(regions)
