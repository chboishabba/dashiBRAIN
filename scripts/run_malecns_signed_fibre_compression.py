#!/usr/bin/env python3
"""Compress the winning MaleCNS signed_reverse fibre on real Gauthey data."""

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
from dashi.analysis.signed_fibre_compression import evaluate_signed_fibre_compression
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
    p.add_argument("--max-rank", type=int, default=8)
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
        raise RuntimeError("body_neurotransmitters is required for signed-fibre compression")
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
    if structural.signed_direct is None:
        raise RuntimeError("signed region matrix unavailable")

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

    result = evaluate_signed_fibre_compression(
        common,
        structural_common.direct,
        structural_common.signed_direct,
        functional.matrix,
        overlap_kernel,
        max_rank=args.max_rank,
        correlation_threshold=args.correlation_threshold,
    )

    best_rank = min(result.rank_residuals, key=result.rank_residuals.get)
    payload = {
        "status": "real_region_level_signed_fibre_compression",
        "regions": list(common),
        "region_count": len(common),
        "structural_source": {
            "mode": "malecns_synapse_primary_neuropil_fractional_membership",
            "source_identifier": args.synapse_partners,
            "source_rows": neuropil.source_rows,
            "observed_neuron_count": neuropil.observed_neuron_count,
            "signed_layer": "presynaptic transmitter sign multiplies neuron outgoing rows before region aggregation",
        },
        "consumer": "published-stimulus-residualized functional correlation with foldwise atlas-overlap nuisance removal",
        "true_signed_reverse_residual": result.true_signed_reverse_residual,
        "sender_scalar_residual": result.sender_scalar_residual,
        "sender_scalar_tendencies": {
            region: float(value) for region, value in zip(common, result.sender_scalar_tendencies)
        },
        "ratio_low_rank_residuals": {
            str(rank): float(score) for rank, score in result.rank_residuals.items()
        },
        "best_tested_rank": int(best_rank),
        "best_tested_rank_residual": float(result.rank_residuals[best_rank]),
        "firewalls": {
            "sender_scalar_is_receptor_resolved_excitation_inhibition": False,
            "low_rank_ratio_is_biological_mechanism": False,
            "compression_selected_using_functional_nulls": False,
            "smaller_representation_automatically_mechanistic": False,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
