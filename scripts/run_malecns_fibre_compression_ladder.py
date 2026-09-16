#!/usr/bin/env python3
"""Find the smallest transferable MaleCNS NDim fibre carrier on real data.

This runner uses the same 26-region Gauthey soft functional carrier, optional
published-stimulus residualization, atlas-overlap control, and real streamed
MaleCNS synapse-derived fractional membership as the joined-observer comparator.

For every outer held-out neuropil, fibre pruning is selected only on the other
regions using inner LORO. The untouched outer neuropil then evaluates whether
the compressed carrier transfers. No null family is run here; this experiment
answers the distinct compression question.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dashi.analysis.gauthey_lbm_experiment import published_lbm_stimulus_regressor
from dashi.analysis.ndim_compression_ladder import (
    evaluate_nested_fibre_compression_overlap_controlled,
)
from dashi.analysis.ndim_structure_function import build_ndim_structural_fibres
from dashi.analysis.overlap_controlled_structure_function import (
    load_overlap_membership_csv,
    restrict_overlap_kernel,
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
    p.add_argument(
        "--stimulus-control",
        choices=("none", "published-gauthey"),
        default="published-gauthey",
    )
    p.add_argument("--compression-tolerance", type=float, default=0.01)
    p.add_argument("--correlation-threshold", type=float, default=0.98)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if args.compression_tolerance < 0:
        raise ValueError("--compression-tolerance must be non-negative")

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
    common = tuple(region for region in structural.regions if region in functional_set)
    if len(common) < 4:
        raise RuntimeError(f"need >=4 common regions for nested compression; got {common}")

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

    traces = np.asarray(producer.traces.traces[:, fi], dtype=float)
    stimulus_summary: dict[str, object] = {
        "mode": args.stimulus_control,
        "applied": False,
    }
    if args.stimulus_control == "published-gauthey":
        stimulus = published_lbm_stimulus_regressor(traces.shape[0])
        fit = residualize_region_traces_against_stimulus(traces, stimulus)
        traces = fit.residual_traces
        stimulus_summary = {
            "mode": args.stimulus_control,
            "applied": True,
            "variance_fraction_removed_mean": float(np.mean(fit.variance_fraction_removed)),
            "variance_fraction_removed_median": float(np.median(fit.variance_fraction_removed)),
            "uses_structural_features": False,
            "uses_region_pair_outcomes": False,
        }

    functional = functional_correlation(traces, common)
    overlap_kernel = restrict_overlap_kernel(functional_membership, common)
    family = build_ndim_structural_fibres(structural_common)

    compressed = evaluate_nested_fibre_compression_overlap_controlled(
        family,
        functional.matrix,
        overlap_kernel,
        tolerance=args.compression_tolerance,
        correlation_threshold=args.correlation_threshold,
    )

    payload = {
        "status": "real_region_level_nested_fibre_compression",
        "regions": list(common),
        "region_count": len(common),
        "structural_source": {
            "mode": "malecns_synapse_primary_neuropil_fractional_membership",
            "source_identifier": args.synapse_partners,
            "source_rows": neuropil.source_rows,
            "observed_neuron_count": neuropil.observed_neuron_count,
        },
        "functional_source": {
            "atlas_identifier": args.region_functional_atlas,
            "source_identifier": args.region_functional_source,
            "membership_source": args.functional_membership,
        },
        "stimulus_control": stimulus_summary,
        "consumer": "stimulus-controlled functional correlation with train-fold atlas-overlap nuisance removed",
        "compression_rule": {
            "selection": "smallest fibre subset within absolute inner-LORO tolerance of full family",
            "tolerance": args.compression_tolerance,
            "outer_held_out_region_visible_to_subset_selection": False,
            "atlas_overlap_refit_inside_inner_and_outer_folds": True,
            "full_fibre_count": len(family.fibres),
            "fibre_names": list(family.fibres.keys()),
        },
        "nested_compression": {
            "weighted_mean_outer_residual": compressed.weighted_mean_residual,
            "mean_selected_fibre_count": compressed.mean_selected_size,
            "selection_frequency": compressed.selection_frequency,
            "folds": [
                {
                    "held_out_region": fold.held_out_region,
                    "selected_fibres": list(fold.selected_fibres),
                    "selected_size": fold.selected_size,
                    "full_size": fold.full_size,
                    "inner_full_residual": fold.inner_full_residual,
                    "inner_selected_residual": fold.inner_selected_residual,
                    "outer_residual": fold.outer_residual,
                    "outer_pair_count": fold.outer_pair_count,
                }
                for fold in compressed.folds
            ],
        },
        "firewalls": {
            "smaller_carrier_automatically_better": False,
            "compression_selected_using_outer_region": False,
            "held_out_adequacy_implies_biological_mechanism": False,
            "one_tolerance_defines_universal_minimal_carrier": False,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
