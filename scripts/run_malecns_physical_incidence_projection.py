#!/usr/bin/env python3
"""Regenerate the MaleCNS NDim chart from nonzero physical region incidence.

This runner uses the same real MaleCNS/Gauthey inputs as the joined benchmark,
but replaces the dense possible-pair presentation with the nonzero unsigned
regional coupling support.  The historical NDim chart is then regenerated from
that sparse carrier and evaluated with the same stimulus+overlap-controlled
leave-one-region-out consumer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dashi.analysis.gauthey_lbm_experiment import published_lbm_stimulus_regressor
from dashi.analysis.malecns_physical_incidence import (
    ndim_chart_from_physical_region_fabric,
    physical_incidence_chart_round_trip_exact,
    physical_region_fabric_from_structural,
)
from dashi.analysis.ndim_structure_function import build_ndim_structural_fibres
from dashi.analysis.overlap_controlled_structure_function import (
    evaluate_overlap_controlled_leave_one_region_out,
    load_overlap_membership_csv,
    restrict_overlap_kernel,
)
from dashi.analysis.stimulus_controlled_functional import residualize_region_traces_against_stimulus
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
    p.add_argument("--output", required=True)
    args = p.parse_args()

    manifest = MaleCNSManifest(base_dir=args.base_dir)
    graph = load_malecns_graph(
        manifest.target_path("connectome_weights_significant"),
        annotations_path=manifest.target_path("body_annotations"),
        signed_by_transmitter=False,
    )
    if not manifest.is_present("body_neurotransmitters"):
        raise RuntimeError("body_neurotransmitters is required for the signed chart")
    signed = signed_adjacency_for_graph(graph, manifest.target_path("body_neurotransmitters"))

    producer = load_region_functional_producer(
        args.region_functional,
        atlas_identifier=args.region_functional_atlas,
        source_identifier=args.region_functional_source,
    )
    membership = load_overlap_membership_csv(args.functional_membership)
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

    common = tuple(r for r in structural.regions if r in set(producer.traces.unit_ids))
    if len(common) < 4:
        raise RuntimeError(f"need >=4 common regions; got {common}")
    si = [structural.regions.index(r) for r in common]
    fi = [producer.traces.unit_ids.index(r) for r in common]
    structural_common = RegionStructuralFeatures(
        common,
        structural.direct[np.ix_(si, si)],
        structural.two_hop[np.ix_(si, si)],
        structural.signed_direct[np.ix_(si, si)] if structural.signed_direct is not None else None,
    )

    original = build_ndim_structural_fibres(structural_common)
    physical = physical_region_fabric_from_structural(structural_common)
    regenerated = ndim_chart_from_physical_region_fabric(physical)

    traces = np.asarray(producer.traces.traces[:, fi], dtype=float)
    stimulus = published_lbm_stimulus_regressor(traces.shape[0])
    stim_fit = residualize_region_traces_against_stimulus(traces, stimulus)
    functional = functional_correlation(stim_fit.residual_traces, common)
    overlap_kernel = restrict_overlap_kernel(membership, common)

    original_loro = evaluate_overlap_controlled_leave_one_region_out(
        original, functional.matrix, overlap_kernel
    )
    regenerated_loro = evaluate_overlap_controlled_leave_one_region_out(
        regenerated, functional.matrix, overlap_kernel
    )

    payload = {
        "status": "real_region_level_sparse_physical_incidence_projection",
        "regions": list(common),
        "region_count": len(common),
        "structural_source": {
            "source_rows": neuropil.source_rows,
            "observed_neuron_count": neuropil.observed_neuron_count,
            "mode": "MaleCNS nonzero membership-aggregated regional direct coupling",
        },
        "physical_incidence": {
            "possible_ordered_pairs": physical.possible_pair_count,
            "nonzero_direct_incidences": physical.incidence_count,
            "density": physical.density,
            "zero_pairs_are_not_physical_edges": True,
            "signed_metadata_creates_edge_when_unsigned_zero": False,
        },
        "chart_regeneration": {
            "coordinate_names": list(original.fibres),
            "exact": physical_incidence_chart_round_trip_exact(structural_common),
            "derived_coordinates_copied_from_dense_chart": False,
            "two_hop_rule": "row-normalise reconstructed direct by absolute row mass, then square",
        },
        "consumer": {
            "description": "published-stimulus-residualized functional correlation with foldwise atlas-overlap control",
            "original_chart_weighted_loro": original_loro.weighted_mean_residual,
            "physical_regenerated_chart_weighted_loro": regenerated_loro.weighted_mean_residual,
            "absolute_difference": abs(
                original_loro.weighted_mean_residual - regenerated_loro.weighted_mean_residual
            ),
        },
        "firewalls": {
            "dense_possible_pair_base_equals_physical_incidence": False,
            "base_incidence_implies_fibre_transport": False,
            "derived_chart_coordinate_is_primitive_physical_edge": False,
            "consumer_invariance_implies_mechanistic_sufficiency": False,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
