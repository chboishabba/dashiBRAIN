#!/usr/bin/env python3
"""Materialise and score real MaleCNS consumer-relative structural carriers.

Alongside the exact hyperfabric lift/project and sender-gain projection, this
runner now reports terminal held-out scorecards and a training-structure-only
latent ladder.  A Turner-style directed path comparator uses inverse unsigned
structural weight as edge distance and all-pairs directed shortest paths; this
imports the graph-distance convention only, not a claim of experiment-level
comparability with Turner et al.

Terminal loss/metric improvements never establish fibre equivalence, universal
consumer sufficiency, physical identity, or biological mechanism.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dashi.analysis.consumer_relative_scorecard import (
    evaluate_overlap_controlled_loro_scorecard,
    scorecard_to_dict,
)
from dashi.analysis.gauthey_lbm_experiment import published_lbm_stimulus_regressor
from dashi.analysis.hyperfabric_sender_gain_projection import (
    project_sender_gain_from_hyperfabric,
    sender_gain_projection_matches_direct_composition,
)
from dashi.analysis.local_fibre_hyperfabric import (
    add_pair_composition_incidences,
    chart_round_trip_exact,
    hyperfabric_from_structural_family,
)
from dashi.analysis.ndim_structure_function import (
    StructuralFibreFamily,
    build_ndim_structural_fibres,
)
from dashi.analysis.overlap_controlled_structure_function import (
    evaluate_overlap_controlled_leave_one_region_out,
    load_overlap_membership_csv,
    restrict_overlap_kernel,
)
from dashi.analysis.sender_magnitude_shape_composition import compose_sender_magnitude_shape
from dashi.analysis.stimulus_controlled_functional import (
    residualize_region_traces_against_stimulus,
)
from dashi.analysis.structural_latent_ladder import (
    evaluate_overlap_controlled_structural_latent_ladder,
    structural_latent_ladder_to_dict,
)
from dashi.analysis.structural_path_baselines import turner_style_path_family
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
    signed = signed_adjacency_for_graph(
        graph, manifest.target_path("body_neurotransmitters")
    )

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
    if structural_common.signed_direct is None:
        raise RuntimeError("signed region structure was not produced")

    family = build_ndim_structural_fibres(structural_common)
    fabric = hyperfabric_from_structural_family(family)
    fabric = add_pair_composition_incidences(fabric)
    chart_names = tuple(family.fibres)
    projected = fabric.project_chart(chart_names)

    gain_projection = project_sender_gain_from_hyperfabric(fabric)
    direct_gain = compose_sender_magnitude_shape(
        structural_common.direct,
        structural_common.signed_direct,
    )
    gain_round_trip_max_abs = float(
        np.max(
            np.abs(
                gain_projection.magnitude_sender_shape
                - direct_gain.magnitude_sender_shape
            )
        )
    )
    gain_family = StructuralFibreFamily(
        common,
        {"magnitude_sender_shape_reverse": gain_projection.magnitude_sender_shape.T},
    )
    direct_family = StructuralFibreFamily(
        common,
        {
            "direct_forward": np.asarray(structural_common.direct, dtype=float),
            "direct_reverse": np.asarray(structural_common.direct, dtype=float).T,
        },
    )
    path_family = turner_style_path_family(common, structural_common.direct)

    traces = np.asarray(producer.traces.traces[:, fi], dtype=float)
    stimulus = published_lbm_stimulus_regressor(traces.shape[0])
    stim_fit = residualize_region_traces_against_stimulus(traces, stimulus)
    functional = functional_correlation(stim_fit.residual_traces, common)
    overlap_kernel = restrict_overlap_kernel(membership, common)

    original_loro = evaluate_overlap_controlled_leave_one_region_out(
        family, functional.matrix, overlap_kernel
    )
    projected_loro = evaluate_overlap_controlled_leave_one_region_out(
        projected, functional.matrix, overlap_kernel
    )
    gain_loro = evaluate_overlap_controlled_leave_one_region_out(
        gain_family, functional.matrix, overlap_kernel
    )

    original_scorecard = evaluate_overlap_controlled_loro_scorecard(
        family,
        functional.matrix,
        overlap_kernel,
        candidate_name="legacy eight-coordinate NDim chart",
        baseline_families={
            "direct_connectivity_bidirectional": direct_family,
            "turner_style_direct_plus_shortest_path": path_family,
        },
        description_length=float(len(chart_names)),
        description_length_unit="declared chart coordinates; not latent dimension or bits",
    )
    gain_scorecard = evaluate_overlap_controlled_loro_scorecard(
        gain_family,
        functional.matrix,
        overlap_kernel,
        candidate_name="hyperfabric-derived magnitude sender gain m_i P_ij",
        baseline_families={
            "direct_connectivity_bidirectional": direct_family,
            "turner_style_direct_plus_shortest_path": path_family,
            "legacy_full_ndim_chart": family,
        },
        description_length=1.0,
        description_length_unit="declared composite matrix field; not latent dimension or bits",
    )
    path_scorecard = evaluate_overlap_controlled_loro_scorecard(
        path_family,
        functional.matrix,
        overlap_kernel,
        candidate_name="Turner-style directed direct + inverse-weight shortest-path comparator",
        baseline_families={
            "direct_connectivity_bidirectional": direct_family,
            "legacy_full_ndim_chart": family,
        },
        description_length=4.0,
        description_length_unit="declared comparator coordinates; not latent dimension or bits",
    )
    latent_ladder = evaluate_overlap_controlled_structural_latent_ladder(
        family,
        functional.matrix,
        overlap_kernel,
    )

    local_counts = [len(coords) for coords in fabric.fibres.values()]
    payload = {
        "status": "real_region_level_local_fibre_hyperfabric_projection",
        "regions": list(common),
        "region_count": len(common),
        "structural_source": {
            "mode": "malecns_synapse_primary_neuropil_fractional_membership",
            "source_rows": neuropil.source_rows,
            "observed_neuron_count": neuropil.observed_neuron_count,
        },
        "base": {
            "kind": "ordered region-pair localities",
            "locality_count": len(fabric.fibres),
            "pair_composition_incidence_count": len(fabric.incidences),
            "incidence_semantics": "(i,j)->(j,k) base composition only; no automatic fibre transport/gluing",
        },
        "legacy_chart": {
            "coordinate_names": list(chart_names),
            "coordinate_count": len(chart_names),
            "local_coordinate_count_min": min(local_counts),
            "local_coordinate_count_max": max(local_counts),
            "matrix_round_trip_exact": chart_round_trip_exact(family),
            "underlying_fibre_cardinality_claim": False,
        },
        "consumer": {
            "description": "published-stimulus-residualized functional correlation with foldwise atlas-overlap control",
            "original_chart_weighted_loro": original_loro.weighted_mean_residual,
            "projected_chart_weighted_loro": projected_loro.weighted_mean_residual,
            "absolute_difference": abs(
                original_loro.weighted_mean_residual - projected_loro.weighted_mean_residual
            ),
            "metric_semantics": "terminal consumer metrics only; lower loss does not establish representation equivalence or adequacy for other consumers",
        },
        "sender_gain_projection": {
            "candidate": "magnitude sender gain m_i times row-relative wiring shape P_ij, evaluated as reverse singleton carrier",
            "derived_only_from_hyperfabric_coordinates": ["direct_forward", "signed_forward"],
            "matches_direct_composition_exactly": sender_gain_projection_matches_direct_composition(
                fabric,
                structural_common.direct,
                structural_common.signed_direct,
            ),
            "max_abs_difference_from_direct_composition": gain_round_trip_max_abs,
            "weighted_loro": gain_loro.weighted_mean_residual,
            "certified_consumer_sufficient": False,
            "anatomical_gain_indexing_distinguished_by_current_null": False,
        },
        "benchmark_scorecards": {
            "legacy_chart": scorecard_to_dict(original_scorecard),
            "sender_gain": scorecard_to_dict(gain_scorecard),
            "turner_style_path": scorecard_to_dict(path_scorecard),
            "baseline_protocol": {
                "zero": "zero prediction on each fold-specific controlled target",
                "fold_train_mean": "mean of the fold training controlled target, frozen onto held-out pairs",
                "direct_connectivity_bidirectional": "same LORO/control fit using only direct-forward and direct-reverse structural coordinates",
                "turner_style_direct_plus_shortest_path": "directed inverse-weight edge distance, all-pairs directed shortest-path distance, with forward/reverse coordinates; graph-distance convention adapted from Turner et al. 2021, not experiment-level reproduction",
                "legacy_full_ndim_chart": "same LORO/control fit using the declared eight-coordinate chart",
            },
        },
        "structural_latent_ladder": structural_latent_ladder_to_dict(latent_ladder),
        "firewalls": {
            "time_is_fibre_ontology": False,
            "hop_is_fibre_ontology": False,
            "eight_is_global_fibre_count": False,
            "pair_composition_implies_fibre_transport": False,
            "crossing_implies_fusion": False,
            "symmetry_implies_quotient_authority": False,
            "consumer_projection_implies_physical_identity": False,
            "sender_gain_projection_implies_sufficiency": False,
            "lower_terminal_loss_implies_better_representation_for_all_consumers": False,
            "declared_matrix_coordinate_count_equals_latent_dimension": False,
            "lowest_discovery_mae_pca_dimension_is_universal_minimal_latent": False,
            "pca_geometry_is_biological_mechanism": False,
            "turner_style_path_comparator_is_apples_to_apples_literature_reproduction": False,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
