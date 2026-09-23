import numpy as np

from dashi.analysis.ndim_structure_function import build_ndim_structural_fibres, leave_one_region_out_masks
from dashi.analysis.overlap_controlled_structure_function import (
    OverlapMembership,
    evaluate_overlap_controlled_leave_one_region_out,
    fit_overlap_nuisance,
    overlap_cosine_kernel,
    residualize_overlap_geometry,
    restrict_overlap_kernel,
)
from dashi.analysis.structure_function_real import RegionStructuralFeatures


def _family():
    direct = np.array([
        [0., 1., 2., 1.],
        [2., 0., 1., 2.],
        [1., 3., 0., 1.],
        [2., 1., 2., 0.],
    ])
    p = direct / direct.sum(axis=1, keepdims=True)
    return build_ndim_structural_fibres(
        RegionStructuralFeatures(("A", "B", "C", "D"), direct, p @ p, None)
    )


def test_overlap_kernel_retains_shared_roi_geometry_without_row_renormalization():
    w = np.array([
        [1.0, 0.5, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 1.0],
    ])
    k = overlap_cosine_kernel(w)
    assert np.allclose(np.diag(k), 1.0)
    assert 0.0 < k[0, 1] < 1.0
    assert 0.0 < k[0, 2] < 1.0
    assert k[1, 2] == 0.0


def test_train_only_nuisance_removes_pure_overlap_signal():
    w = np.array([
        [1., 0., 0., 0.],
        [0.8, 0.2, 0., 0.],
        [0., 0.2, 0.8, 0.],
        [0., 0., 0.3, 0.7],
    ])
    k = overlap_cosine_kernel(w)
    observed = 0.4 + 0.6 * k
    np.fill_diagonal(observed, 1.0)
    fit, held = leave_one_region_out_masks(4, 0)
    nuisance = fit_overlap_nuisance(observed, k, fit)
    residual = residualize_overlap_geometry(observed, k, nuisance)
    assert abs(nuisance.intercept - 0.4) < 1e-10
    assert abs(nuisance.coefficient - 0.6) < 1e-10
    assert np.max(np.abs(residual[held])) < 1e-10


def test_loro_overlap_control_is_fit_separately_inside_each_fold():
    family = _family()
    w = np.array([
        [1., 0., 0., 0.],
        [0.7, 0.3, 0., 0.],
        [0., 0.2, 0.8, 0.],
        [0., 0., 0.4, 0.6],
    ])
    k = overlap_cosine_kernel(w)
    observed = 0.2 + 0.5 * k
    np.fill_diagonal(observed, 1.0)
    result = evaluate_overlap_controlled_leave_one_region_out(family, observed, k, correlation_threshold=1.0)
    assert len(result.folds) == 4
    assert all(f.train_pair_count == 3 for f in result.folds)
    assert all(f.held_out_pair_count == 3 for f in result.folds)
    assert result.weighted_mean_residual < 1e-8


def test_restrict_overlap_kernel_follows_declared_common_region_order():
    membership = OverlapMembership(
        ("C", "A", "B"),
        np.array([
            [0., 1., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
        ]),
    )
    k = restrict_overlap_kernel(membership, ("A", "B", "C"))
    assert k.shape == (3, 3)
    assert np.allclose(np.diag(k), 1.0)
    assert k[0, 2] == 0.0
