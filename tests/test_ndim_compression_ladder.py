import numpy as np

from dashi.analysis.ndim_compression_ladder import (
    choose_smallest_inner_adequate_subset,
    evaluate_nested_fibre_compression,
    evaluate_nested_fibre_compression_overlap_controlled,
)
from dashi.analysis.ndim_structure_function import StructuralFibreFamily


def _synthetic_family(n=6):
    regions = tuple(f"R{i}" for i in range(n))
    ii, jj = np.indices((n, n))
    signal = (ii + jj).astype(float)
    nuisance1 = (ii - jj).astype(float)
    nuisance2 = ((ii * 3 + jj * 5) % 7).astype(float)
    np.fill_diagonal(signal, 0.0)
    np.fill_diagonal(nuisance1, 0.0)
    np.fill_diagonal(nuisance2, 0.0)
    family = StructuralFibreFamily(
        regions,
        {
            "signal": signal,
            "nuisance1": nuisance1,
            "nuisance2": nuisance2,
        },
    )
    observed = 0.25 + 0.04 * signal
    observed = (observed + observed.T) / 2.0
    np.fill_diagonal(observed, 1.0)
    return family, observed


def test_smallest_inner_subset_can_recover_single_predictive_fibre():
    family, observed = _synthetic_family()
    subset, score, full = choose_smallest_inner_adequate_subset(
        family, observed, tolerance=1e-8
    )
    assert subset == ("signal",)
    assert score <= full + 1e-8


def test_nested_compression_never_uses_outer_region_for_subset_selection():
    family, observed = _synthetic_family()
    baseline = evaluate_nested_fibre_compression(family, observed, tolerance=1e-8)

    changed = observed.copy()
    held = 0
    changed[held, 1:] = 99.0
    changed[1:, held] = 99.0
    perturbed = evaluate_nested_fibre_compression(family, changed, tolerance=1e-8)

    base_fold = baseline.folds[held]
    changed_fold = perturbed.folds[held]
    assert changed_fold.selected_fibres == base_fold.selected_fibres
    assert np.isclose(changed_fold.inner_full_residual, base_fold.inner_full_residual)
    assert np.isclose(changed_fold.inner_selected_residual, base_fold.inner_selected_residual)
    assert not np.isclose(changed_fold.outer_residual, base_fold.outer_residual)


def test_nested_compression_reports_smaller_transferable_carrier():
    family, observed = _synthetic_family()
    result = evaluate_nested_fibre_compression(family, observed, tolerance=1e-8)
    assert len(result.folds) == len(family.regions)
    assert result.mean_selected_size == 1.0
    assert result.selection_frequency == {"signal": len(family.regions)}
    assert result.weighted_mean_residual < 1e-10


def test_overlap_controlled_nested_selection_does_not_see_outer_region():
    family, signal_only = _synthetic_family()
    n = len(family.regions)
    ii, jj = np.indices((n, n))
    overlap_kernel = np.exp(-np.abs(ii - jj) / 2.0)
    observed = signal_only + 0.6 * overlap_kernel
    np.fill_diagonal(observed, 1.0)

    baseline = evaluate_nested_fibre_compression_overlap_controlled(
        family,
        observed,
        overlap_kernel,
        tolerance=1e-8,
    )

    held = 0
    changed = observed.copy()
    changed[held, 1:] += 10.0
    changed[1:, held] += 10.0
    perturbed = evaluate_nested_fibre_compression_overlap_controlled(
        family,
        changed,
        overlap_kernel,
        tolerance=1e-8,
    )

    a = baseline.folds[held]
    b = perturbed.folds[held]
    assert b.selected_fibres == a.selected_fibres
    assert np.isclose(b.inner_full_residual, a.inner_full_residual)
    assert np.isclose(b.inner_selected_residual, a.inner_selected_residual)
    assert not np.isclose(b.outer_residual, a.outer_residual)
