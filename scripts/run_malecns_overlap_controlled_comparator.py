#!/usr/bin/env python3
"""Test MaleCNS structure/function coupling after removing soft-atlas overlap covariance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dashi.analysis.ndim_structure_function import (
    build_ndim_structural_fibres,
    evaluate_leave_one_region_out,
)
from dashi.analysis.overlap_controlled_nulls import (
    overlap_controlled_strength_preserving_null_loro,
)
from dashi.analysis.overlap_controlled_structure_function import (
    evaluate_overlap_controlled_leave_one_region_out,
    load_overlap_membership_csv,
    overlap_controlled_label_permutation_null_loro,
    restrict_overlap_kernel,
)
from dashi.analysis.structure_function_real import (
    RegionStructuralFeatures,
    aggregate_connectome_by_membership,
    functional_correlation,
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
    p.add_argument("--functional-membership", required=True)
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
    functional_membership = load_overlap_membership_csv(args.functional_membership)

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
    overlap_kernel = restrict_overlap_kernel(functional_membership, common)
    family = build_ndim_structural_fibres(structural_common)

    raw = evaluate_leave_one_region_out(
        family,
        functional.matrix,
        correlation_threshold=args.correlation_threshold,
    )
    controlled = evaluate_overlap_controlled_leave_one_region_out(
        family,
        functional.matrix,
        overlap_kernel,
        correlation_threshold=args.correlation_threshold,
    )
    real_label, label_nulls, label_p = overlap_controlled_label_permutation_null_loro(
        family,
        functional.matrix,
        overlap_kernel,
        n_null=args.null_count,
        seed=20260914,
        correlation_threshold=args.correlation_threshold,
    )
    strength = overlap_controlled_strength_preserving_null_loro(
        structural_common,
        functional.matrix,
        overlap_kernel,
        n_null=args.null_count,
        seed=20260915,
        correlation_threshold=args.correlation_threshold,
    )

    payload = {
        "status": "real_region_level_overlap_controlled_ndim_comparator",
        "regions": list(common),
        "region_count": len(common),
        "functional_source": {
            "atlas_identifier": args.region_functional_atlas,
            "source_identifier": args.region_functional_source,
            "membership_source": args.functional_membership,
        },
        "overlap_nuisance": {
            "kernel": "cosine similarity of ROI->painted-domain membership rows",
            "fit": "intercept + one overlap coefficient, fit independently on training pairs inside each fold",
            "held_out_pairs_used_for_nuisance_fit": False,
        },
        "raw_soft_loro": {
            "weighted_mean_residual": raw.weighted_mean_residual,
        },
        "overlap_controlled_loro": {
            "weighted_mean_residual": controlled.weighted_mean_residual,
            "mean_fold_residual": controlled.mean_fold_residual,
            "region_label_permutation_null_p": label_p,
            "label_null_mean_residual": float(np.mean(label_nulls)),
            "label_null_min_residual": float(np.min(label_nulls)),
            "strength_preserving_wiring_null_p": strength.empirical_p_value,
            "strength_null_mean_residual": strength.null_mean_residual,
            "strength_null_min_residual": strength.null_min_residual,
            "max_row_strength_error": strength.max_row_strength_error,
            "max_column_strength_error": strength.max_column_strength_error,
            "folds": [
                {
                    "held_out_region": f.held_out_region,
                    "mean_residual": f.mean_residual,
                    "nuisance_intercept": f.nuisance.intercept,
                    "nuisance_overlap_coefficient": f.nuisance.coefficient,
                    "selected_fibres": list(f.selected_fibres),
                    "coefficients_intercept_then_selected": f.coefficients.tolist(),
                }
                for f in controlled.folds
            ],
        },
        "firewalls": {
            "atlas_overlap_kernel_is_structural_connectome_feature": False,
            "nuisance_fit_uses_held_out_functional_pairs": False,
            "label_null_permuted_function_without_overlap_geometry": False,
            "strength_null_scrambles_functional_overlap_geometry": False,
            "raw_soft_low_residual_implies_wiring_specific_signal": False,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
