#!/usr/bin/env python3
"""Run the sender-magnitude assignment test on real MaleCNS/Gauthey data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dashi.analysis.gauthey_lbm_experiment import published_lbm_stimulus_regressor
from dashi.analysis.overlap_controlled_structure_function import (
    load_overlap_membership_csv,
    restrict_overlap_kernel,
)
from dashi.analysis.sender_magnitude_assignment_null import (
    evaluate_sender_magnitude_assignment_null,
)
from dashi.analysis.stimulus_controlled_functional import (
    residualize_region_traces_against_stimulus,
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
    p.add_argument("--null-count", type=int, default=999)
    p.add_argument("--correlation-threshold", type=float, default=0.98)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    manifest = MaleCNSManifest(base_dir=args.base_dir)
    graph = load_malecns_graph(
        manifest.target_path("connectome_weights_significant"),
        annotations_path=manifest.target_path("body_annotations"),
        signed_by_transmitter=False,
    )
    if not manifest.is_present("body_neurotransmitters"):
        raise RuntimeError("body_neurotransmitters artifact is required")
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
    common = tuple(region for region in structural.regions if region in functional_set)
    if len(common) < 4:
        raise RuntimeError(f"need >=4 common regions; got {common}")

    si = [structural.regions.index(region) for region in common]
    fi = [producer.traces.unit_ids.index(region) for region in common]
    structural_common = RegionStructuralFeatures(
        common,
        structural.direct[np.ix_(si, si)],
        structural.two_hop[np.ix_(si, si)],
        structural.signed_direct[np.ix_(si, si)]
        if structural.signed_direct is not None
        else None,
    )
    if structural_common.signed_direct is None:
        raise RuntimeError("signed regional carrier was not produced")

    traces = np.asarray(producer.traces.traces[:, fi], dtype=float)
    stimulus = published_lbm_stimulus_regressor(traces.shape[0])
    stim_fit = residualize_region_traces_against_stimulus(traces, stimulus)
    functional = functional_correlation(stim_fit.residual_traces, common)
    overlap_kernel = restrict_overlap_kernel(functional_membership, common)

    result = evaluate_sender_magnitude_assignment_null(
        common,
        structural_common.direct,
        structural_common.signed_direct,
        functional.matrix,
        overlap_kernel,
        n_null=args.null_count,
        seed=20260917,
        correlation_threshold=args.correlation_threshold,
    )

    nulls = result.net_assignment_null_residuals
    payload = {
        "status": "real_region_level_sender_magnitude_assignment_test",
        "regions": list(common),
        "region_count": len(common),
        "structural_source": {
            "mode": "malecns_synapse_primary_neuropil_fractional_membership",
            "source_identifier": args.synapse_partners,
            "source_rows": neuropil.source_rows,
            "observed_neuron_count": neuropil.observed_neuron_count,
        },
        "functional_consumer": (
            "published-stimulus-residualized functional correlation with foldwise "
            "atlas-overlap nuisance removal"
        ),
        "magnitude_test": {
            "net_magnitude_definition": "abs(sum_j S_ij) / sum_j abs(D_ij)",
            "absolute_ratio_magnitude_definition": "sum_j abs(S_ij) / sum_j abs(D_ij)",
            "carrier": "row-relative unsigned shape P multiplied by sender magnitude",
            "net_magnitude_residual": result.net_magnitude_residual,
            "absolute_ratio_magnitude_residual": result.absolute_ratio_magnitude_residual,
            "net_assignment_empirical_p_value": result.net_assignment_empirical_p_value,
            "null_count": int(nulls.size),
            "null_mean_residual": float(np.mean(nulls)),
            "null_median_residual": float(np.median(nulls)),
            "null_min_residual": float(np.min(nulls)),
            "null_max_residual": float(np.max(nulls)),
            "null_q05_residual": float(np.quantile(nulls, 0.05)),
            "null_q95_residual": float(np.quantile(nulls, 0.95)),
            "net_magnitude_by_region": {
                region: float(value)
                for region, value in zip(common, result.net_magnitudes.tolist())
            },
            "absolute_ratio_magnitude_by_region": {
                region: float(value)
                for region, value in zip(common, result.absolute_ratio_magnitudes.tolist())
            },
        },
        "firewalls": {
            "absolute_sender_scale_used_by_test": False,
            "row_relative_wiring_shape_changed_by_null": False,
            "magnitude_multiset_changed_by_null": False,
            "polarity_used_by_net_magnitude_carrier": False,
            "pair_specific_sign_ratio_used_by_net_magnitude_carrier": False,
            "sender_magnitude_assignment_result_implies_mechanism": False,
            "single_session_result_implies_population_generalization": False,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
