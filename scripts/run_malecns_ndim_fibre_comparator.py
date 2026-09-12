#!/usr/bin/env python3
"""Run the NDim/fibre-aware MaleCNS structure-function comparator.

This is a bounded follow-up to the first real region-level benchmark. It does
not replace the historical fixed-weight DASHI result. It exposes distinct
structural fibres, selects a structurally compatible subset using training
feature geometry only, fits that subset on training pairs only, freezes it, and
then evaluates untouched held-out pairs.

The script also reports a stricter leave-one-region-out result, where every pair
containing the held-out neuropil is excluded from fitting; region-label
permutation nulls that refit the complete NDim consumer inside every null draw;
fold-matched null diagnostics; foldwise fibre/coefficient stability; and a
structural null that preserves each region's weighted in/out strength and
source-level signed tendency while scrambling pair-specific wiring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dashi.analysis.ndim_stability_nulls import (
    strength_preserving_structural_null_loro,
    summarize_loro_fibre_stability,
)
from dashi.analysis.ndim_structure_function import (
    build_ndim_structural_fibres,
    evaluate_leave_one_region_out,
    fit_ndim_fibre_consumer,
    region_label_permutation_null_leave_one_region_out,
    region_label_permutation_null_pair_holdout,
)
from dashi.analysis.structure_function_real import (
    RegionStructuralFeatures,
    aggregate_connectome_by_membership,
    evaluate_region_structure_function,
    functional_correlation,
    pairwise_train_holdout_masks,
)
from dashi.io.malecns_loader import load_malecns_graph
from dashi.io.malecns_manifest import MaleCNSManifest
from dashi.io.malecns_neuropil_membership import load_synapse_neuropil_membership
from dashi.io.malecns_signs import signed_adjacency_for_graph
from dashi.io.region_functional_adapter import load_region_functional_producer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="data/malecns")
    p.add_argument("--region-functional", required=True)
    p.add_argument("--region-functional-atlas", required=True)
    p.add_argument("--region-functional-source", required=True)
    p.add_argument("--synapse-partners", required=True)
    p.add_argument("--correlation-threshold", type=float, default=0.98)
    p.add_argument("--null-count", type=int, default=100)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    manifest = MaleCNSManifest(base_dir=args.base_dir)
    graph = load_malecns_graph(
        manifest.target_path("connectome_weights_significant"),
        annotations_path=manifest.target_path("body_annotations"),
        signed_by_transmitter=False,
    )
    signed = None
    if manifest.is_present("body_neurotransmitters"):
        signed = signed_adjacency_for_graph(
            graph, manifest.target_path("body_neurotransmitters")
        )

    producer = load_region_functional_producer(
        args.region_functional,
        atlas_identifier=args.region_functional_atlas,
        source_identifier=args.region_functional_source,
    )
    neuropil = load_synapse_neuropil_membership(
        args.synapse_partners,
        graph.idx_to_id,
        allowed_regions=producer.traces.unit_ids,
    )
    structural = aggregate_connectome_by_membership(
        graph.carrier,
        neuropil.membership,
        neuropil.regions,
        signed_adjacency=signed,
    )

    functional_set = set(producer.traces.unit_ids)
    common = tuple(r for r in structural.regions if r in functional_set)
    if len(common) < 3:
        raise RuntimeError(f"need >=3 common regions; got {common}")

    si = [structural.regions.index(r) for r in common]
    fi = [producer.traces.unit_ids.index(r) for r in common]
    structural_common = RegionStructuralFeatures(
        common,
        structural.direct[np.ix_(si, si)],
        structural.two_hop[np.ix_(si, si)],
        structural.signed_direct[np.ix_(si, si)] if structural.signed_direct is not None else None,
    )
    functional = functional_correlation(producer.traces.traces[:, fi], common)
    fit, held = pairwise_train_holdout_masks(len(common), modulus=3, held_out_residue=0)

    fixed = evaluate_region_structure_function(
        structural_common,
        functional,
        fit_mask=fit,
        held_out_mask=held,
    )
    family = build_ndim_structural_fibres(structural_common)
    ndim = fit_ndim_fibre_consumer(
        family,
        functional.matrix,
        fit,
        held,
        correlation_threshold=args.correlation_threshold,
    )
    blocked = evaluate_leave_one_region_out(
        family,
        functional.matrix,
        correlation_threshold=args.correlation_threshold,
    )
    stability = summarize_loro_fibre_stability(blocked)
    pair_null = region_label_permutation_null_pair_holdout(
        family,
        functional.matrix,
        fit,
        held,
        n_null=args.null_count,
        seed=20260911,
        correlation_threshold=args.correlation_threshold,
    )
    blocked_null = region_label_permutation_null_leave_one_region_out(
        family,
        functional.matrix,
        n_null=args.null_count,
        seed=20260912,
        correlation_threshold=args.correlation_threshold,
    )
    strength_null = strength_preserving_structural_null_loro(
        structural_common,
        functional.matrix,
        n_null=args.null_count,
        seed=20260913,
        correlation_threshold=args.correlation_threshold,
    )

    fold_null_by_region: dict[str, dict[str, float]] = {}
    if (
        blocked_null.null_fold_residuals is not None
        and blocked_null.observed_fold_residuals is not None
        and blocked_null.fold_empirical_p_values is not None
    ):
        for i, region in enumerate(blocked_null.fold_regions):
            null_col = blocked_null.null_fold_residuals[:, i]
            fold_null_by_region[region] = {
                "observed_mean_residual": float(blocked_null.observed_fold_residuals[i]),
                "permutation_p": float(blocked_null.fold_empirical_p_values[i]),
                "null_mean_residual": float(np.mean(null_col)),
                "null_median_residual": float(np.median(null_col)),
                "null_q05_residual": float(np.quantile(null_col, 0.05)),
                "null_q95_residual": float(np.quantile(null_col, 0.95)),
                "fraction_nulls_beaten": float(np.mean(blocked_null.observed_fold_residuals[i] < null_col)),
            }

    payload = {
        "status": "real_region_level_ndim_fibre_comparator",
        "regions": list(common),
        "fit_pair_count": ndim.fit_pair_count,
        "held_out_pair_count": ndim.held_out_pair_count,
        "functional_source": {
            "atlas_identifier": args.region_functional_atlas,
            "source_identifier": args.region_functional_source,
        },
        "structural_source": {
            "mode": "malecns_synapse_primary_neuropil_fractional_membership",
            "source_identifier": args.synapse_partners,
            "source_rows": neuropil.source_rows,
            "observed_neuron_count": neuropil.observed_neuron_count,
            "membership_semantics": "retained-region numerator divided by total MaleCNS synaptic incidence across all primary neuropils",
        },
        "historical_fixed_model": {
            "mean_direct_edge_residual": fixed.mean_direct,
            "mean_path_aware_residual": fixed.mean_path,
            "mean_fixed_dashi_residual": fixed.mean_dashi,
        },
        "ndim_pair_holdout": {
            "candidate_fibres": list(family.fibres),
            "selected_fibres": list(ndim.selected_fibres),
            "rejected_constant": list(ndim.compatibility.rejected_constant),
            "rejected_redundant": [
                {"fibre": a, "conflicts_with": b, "training_feature_correlation": c}
                for a, b, c in ndim.compatibility.rejected_redundant
            ],
            "correlation_threshold": ndim.compatibility.correlation_threshold,
            "coefficients_intercept_then_selected": ndim.coefficients.tolist(),
            "mean_held_out_residual": ndim.mean_residual,
            "beats_direct": ndim.mean_residual < fixed.mean_direct,
            "beats_path": ndim.mean_residual < fixed.mean_path,
            "beats_fixed_dashi": ndim.mean_residual < fixed.mean_dashi,
            "region_label_permutation_null_p": pair_null.empirical_p_value,
            "null_mean_residual": float(np.mean(pair_null.null_residuals)),
            "null_min_residual": float(np.min(pair_null.null_residuals)),
        },
        "ndim_leave_one_region_out": {
            "mean_fold_residual": blocked.mean_fold_residual,
            "weighted_mean_residual": blocked.weighted_mean_residual,
            "region_label_permutation_null_p": blocked_null.empirical_p_value,
            "null_mean_residual": float(np.mean(blocked_null.null_residuals)),
            "null_min_residual": float(np.min(blocked_null.null_residuals)),
            "fold_matched_nulls": fold_null_by_region,
            "strength_preserving_wiring_null": {
                "description": "pair-specific wiring scrambled while weighted in/out strength and source-level signed tendency are retained; NDim fibres and every LORO fit are recomputed per draw",
                "empirical_p_value": strength_null.empirical_p_value,
                "null_mean_residual": strength_null.null_mean_residual,
                "null_min_residual": strength_null.null_min_residual,
                "max_row_strength_error": strength_null.max_row_strength_error,
                "max_column_strength_error": strength_null.max_column_strength_error,
            },
            "fibre_stability": {
                "fold_count": stability.fold_count,
                "intercept_mean": stability.intercept_mean,
                "intercept_std": stability.intercept_std,
                "fibres": [
                    {
                        "fibre": s.fibre,
                        "selected_fold_count": s.selected_fold_count,
                        "selected_fold_fraction": s.selected_fold_fraction,
                        "coefficient_mean_when_selected": s.coefficient_mean_when_selected,
                        "coefficient_std_when_selected": s.coefficient_std_when_selected,
                        "coefficient_min_when_selected": s.coefficient_min_when_selected,
                        "coefficient_max_when_selected": s.coefficient_max_when_selected,
                    }
                    for s in stability.fibres
                ],
            },
            "folds": [
                {
                    "held_out_region": f.held_out_region,
                    "train_pair_count": f.train_pair_count,
                    "held_out_pair_count": f.held_out_pair_count,
                    "mean_residual": f.mean_residual,
                    "selected_fibres": list(f.selected_fibres),
                    "coefficients_intercept_then_selected": f.coefficients.tolist(),
                }
                for f in blocked.folds
            ],
        },
        "interpretation_boundaries": {
            "loro_lower_than_pair_residual_implies_stronger_absolute_performance": False,
            "loro_permutation_p_below_0_10_implies_confirmed_structure_function_coupling": False,
            "foldwise_nominal_p_values_are_multiplicity_corrected": False,
            "single_animal_region_level_result_implies_population_generalization": False,
            "strength_preserving_null_preserves_pair_specific_wiring": False,
        },
        "firewalls": {
            "compatibility_selection_uses_functional_outcomes": False,
            "pair_fit_uses_held_out_outcomes": False,
            "leave_one_region_out_fit_contains_held_region_pairs": False,
            "permutation_null_reuses_frozen_observed_coefficients": False,
            "strength_null_reuses_observed_pairwise_wiring": False,
            "strength_null_reuses_frozen_observed_coefficients": False,
            "shared_region_identity_is_neuron_identity": False,
            "more_fibres_implies_better_prediction": False,
            "pair_holdout_is_equivalent_to_region_holdout": False,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
