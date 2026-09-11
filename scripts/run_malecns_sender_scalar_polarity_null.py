#!/usr/bin/env python3
"""Run the exact sender-scalar polarity assignment test on real MaleCNS data."""

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
from dashi.analysis.sender_scalar_polarity_null import (
    evaluate_sender_scalar_polarity_null,
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
    p.add_argument("--max-exact-assignments", type=int, default=100000)
    p.add_argument("--null-count", type=int, default=1000)
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

    result = evaluate_sender_scalar_polarity_null(
        common,
        structural_common.direct,
        structural_common.signed_direct,
        functional.matrix,
        overlap_kernel,
        max_exact_assignments=args.max_exact_assignments,
        n_null=args.null_count,
        seed=20260916,
        correlation_threshold=args.correlation_threshold,
    )

    nulls = result.null_residuals
    sign_by_region = {
        region: int(sign)
        for region, sign in zip(common, result.observed_signs.tolist())
    }
    magnitude_by_region = {
        region: float(magnitude)
        for region, magnitude in zip(common, result.sender_magnitudes.tolist())
    }
    payload = {
        "status": "real_region_level_sender_scalar_polarity_assignment_test",
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
        "polarity_test": {
            "observed_sender_scalar_residual": result.observed_residual,
            "magnitude_only_sender_scalar_residual": result.magnitude_only_residual,
            "empirical_p_value": result.empirical_p_value,
            "exact_enumeration": result.exact_enumeration,
            "assignment_count": result.assignment_count,
            "negative_sender_count": result.negative_sender_count,
            "zero_sender_count": result.zero_sender_count,
            "null_mean_residual": float(np.mean(nulls)),
            "null_median_residual": float(np.median(nulls)),
            "null_min_residual": float(np.min(nulls)),
            "null_max_residual": float(np.max(nulls)),
            "null_q05_residual": float(np.quantile(nulls, 0.05)),
            "null_q95_residual": float(np.quantile(nulls, 0.95)),
            "observed_sign_by_region": sign_by_region,
            "sender_magnitude_by_region": magnitude_by_region,
        },
        "firewalls": {
            "unsigned_pairwise_connectivity_changed_by_null": False,
            "sender_scalar_magnitudes_changed_by_null": False,
            "number_of_positive_negative_zero_signs_changed_by_null": False,
            "pair_specific_sign_ratio_used_by_test": False,
            "polarity_assignment_result_implies_receptor_resolved_mechanism": False,
            "single_session_result_implies_population_generalization": False,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
