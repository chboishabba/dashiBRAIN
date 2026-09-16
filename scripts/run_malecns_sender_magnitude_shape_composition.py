#!/usr/bin/env python3
"""Run the direct m_i P_ij joint-compression test on real MaleCNS data."""

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
from dashi.analysis.sender_magnitude_shape_composition import (
    evaluate_sender_magnitude_shape_composition,
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

    result = evaluate_sender_magnitude_shape_composition(
        common,
        structural_common.direct,
        structural_common.signed_direct,
        functional.matrix,
        overlap_kernel,
        correlation_threshold=args.correlation_threshold,
    )

    c = result.composition
    payload = {
        "status": "real_region_level_sender_magnitude_shape_joint_compression",
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
        "residuals": {
            "full_signed_reverse": result.full_signed_reverse_residual,
            "signed_sender_density_a_times_D": result.signed_sender_density_residual,
            "magnitude_sender_density_m_times_D": result.magnitude_sender_density_residual,
            "signed_sender_shape_a_times_P": result.signed_sender_shape_residual,
            "magnitude_sender_shape_m_times_P": result.magnitude_sender_shape_residual,
            "relative_shape_P": result.relative_shape_residual,
        },
        "sender_tendency_by_region": {
            region: float(value) for region, value in zip(common, c.sender_tendency)
        },
        "sender_magnitude_by_region": {
            region: float(value) for region, value in zip(common, c.sender_magnitude)
        },
        "firewalls": {
            "separate_drop_scale_and_drop_polarity_imply_joint_adequacy": False,
            "m_times_P_is_minimal_before_direct_comparison": False,
            "joint_predictive_adequacy_implies_causal_mechanism": False,
            "single_session_result_implies_population_generalization": False,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
