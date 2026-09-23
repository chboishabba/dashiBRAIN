#!/usr/bin/env python3
"""Discriminate the compressed signed MaleCNS fibre on the real joined-controlled target."""

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
from dashi.analysis.signed_fibre_discriminator import evaluate_signed_reverse_discriminator
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
    p.add_argument("--null-count", type=int, default=100)
    p.add_argument("--correlation-threshold", type=float, default=0.98)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if args.null_count < 1:
        raise ValueError("--null-count must be >= 1")

    manifest = MaleCNSManifest(base_dir=args.base_dir)
    graph = load_malecns_graph(
        manifest.target_path("connectome_weights_significant"),
        annotations_path=manifest.target_path("body_annotations"),
        signed_by_transmitter=False,
    )
    if not manifest.is_present("body_neurotransmitters"):
        raise RuntimeError("MaleCNS neurotransmitter carrier is required")
    signed_neuron = signed_adjacency_for_graph(
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
        signed_adjacency=signed_neuron,
    )
    if structural.signed_direct is None:
        raise RuntimeError("signed region carrier was not produced")

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
        structural.signed_direct[np.ix_(si, si)],
    )

    traces = np.asarray(producer.traces.traces[:, fi], dtype=float)
    stimulus = published_lbm_stimulus_regressor(traces.shape[0])
    stim_fit = residualize_region_traces_against_stimulus(traces, stimulus)
    functional = functional_correlation(stim_fit.residual_traces, common)
    overlap_kernel = restrict_overlap_kernel(functional_membership, common)

    result = evaluate_signed_reverse_discriminator(
        common,
        structural_common.direct,
        structural_common.signed_direct,
        functional.matrix,
        overlap_kernel,
        n_null=args.null_count,
        seed=20260916,
        correlation_threshold=args.correlation_threshold,
    )

    nulls = result.sender_pattern_null_residuals
    payload = {
        "status": "real_region_level_signed_fibre_discriminator",
        "regions": list(common),
        "region_count": len(common),
        "structural_source": {
            "mode": "malecns_synapse_primary_neuropil_fractional_membership",
            "source_identifier": args.synapse_partners,
            "source_rows": neuropil.source_rows,
            "observed_neuron_count": neuropil.observed_neuron_count,
            "sign_semantics": (
                "presynaptic transmitter sign multiplies neuron-level outgoing rows; "
                "signed_reverse is the transpose of the aggregated sender-signed region matrix"
            ),
            "receptor_resolved_physiology_claimed": False,
        },
        "functional_consumer": {
            "target": "published-stimulus-residualized functional correlation with train-fold atlas-overlap control",
            "atlas_identifier": args.region_functional_atlas,
            "source_identifier": args.region_functional_source,
            "membership_source": args.functional_membership,
            "stimulus_variance_fraction_removed_mean": float(np.mean(stim_fit.variance_fraction_removed)),
        },
        "discriminator": {
            "true_signed_reverse_residual": result.true_residual,
            "unsigned_direct_reverse_residual": result.unsigned_reverse_residual,
            "magnitude_only_abs_signed_reverse_residual": result.magnitude_only_residual,
            "sender_pattern_permutation_null_p": result.sender_pattern_empirical_p_value,
            "sender_pattern_null_mean_residual": float(np.mean(nulls)),
            "sender_pattern_null_median_residual": float(np.median(nulls)),
            "sender_pattern_null_min_residual": float(np.min(nulls)),
            "sender_pattern_null_max_residual": float(np.max(nulls)),
            "null_count": int(args.null_count),
        },
        "null_semantics": {
            "unsigned_pairwise_connectivity_D_fixed": True,
            "sender_signed_ratio_definition": "R_ij = S_ij / D_ij on nonzero D_ij",
            "draw": "permute whole sender rows of R, then reconstruct S_null = D * R_perm",
            "pair_specific_unsigned_magnitude_scrambled": False,
            "functional_target_scrambled": False,
            "atlas_overlap_geometry_scrambled": False,
        },
        "firewalls": {
            "signed_reverse_equals_receptor_resolved_excitation_inhibition": False,
            "compression_selection_alone_proves_sign_information_is_causal": False,
            "better_than_unsigned_alone_establishes_biological_mechanism": False,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
