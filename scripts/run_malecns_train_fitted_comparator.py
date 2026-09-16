#!/usr/bin/env python3
"""Run a train-only fitted structure/function comparator on the real MaleCNS seam.

This is a sibling experiment to ``run_malecns_benchmark.py``.  It preserves the
fixed DASHI/path/direct results and adds one least-privilege comparator whose
coefficients are fit only on training region pairs.  The large synapse partner
table is still consumed through the bounded batchwise membership loader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dashi.analysis.structure_function_real import (
    RegionStructuralFeatures,
    aggregate_connectome_by_membership,
    evaluate_region_structure_function,
    functional_correlation,
    pairwise_train_holdout_masks,
)
from dashi.analysis.train_fitted_structure_function import fit_train_only_structure_mixture
from dashi.io.malecns_loader import load_malecns_graph
from dashi.io.malecns_manifest import MaleCNSManifest
from dashi.io.malecns_neuropil_membership import load_synapse_neuropil_membership
from dashi.io.malecns_signs import signed_adjacency_for_graph
from dashi.io.region_functional_adapter import load_region_functional_producer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train-only MaleCNS structural mixture comparator")
    parser.add_argument("--base-dir", default="data/malecns")
    parser.add_argument("--region-functional", required=True)
    parser.add_argument("--region-functional-atlas", required=True)
    parser.add_argument("--region-functional-source", required=True)
    parser.add_argument("--synapse-partners", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = MaleCNSManifest(base_dir=args.base_dir)
    graph = load_malecns_graph(
        manifest.target_path("connectome_weights_significant"),
        annotations_path=manifest.target_path("body_annotations"),
        signed_by_transmitter=False,
    )
    signed = None
    if manifest.is_present("body_neurotransmitters"):
        signed = signed_adjacency_for_graph(graph, manifest.target_path("body_neurotransmitters"))

    producer = load_region_functional_producer(
        args.region_functional,
        atlas_identifier=args.region_functional_atlas,
        source_identifier=args.region_functional_source,
    )
    membership = load_synapse_neuropil_membership(
        args.synapse_partners,
        graph.idx_to_id,
        allowed_regions=producer.traces.unit_ids,
    )
    structural = aggregate_connectome_by_membership(
        graph.carrier,
        membership.membership,
        membership.regions,
        signed_adjacency=signed,
    )

    common = tuple(r for r in structural.regions if r in set(producer.traces.unit_ids))
    if len(common) < 3:
        raise RuntimeError(f"fewer than three common regions: {common!r}")

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
    fitted = fit_train_only_structure_mixture(
        structural_common,
        functional,
        fit_mask=fit,
        held_out_mask=held,
        include_signed=True,
    )

    payload = {
        "status": "success",
        "experiment": "train_only_linear_structure_mixture",
        "regions": list(common),
        "fit_pair_count": fitted.fit_pair_count,
        "held_out_pair_count": fitted.held_out_pair_count,
        "fixed_comparators": {
            "mean_direct_edge_residual": fixed.mean_direct,
            "mean_path_aware_residual": fixed.mean_path,
            "mean_fixed_dashi_residual": fixed.mean_dashi,
        },
        "train_fitted_mixture": {
            "feature_names": list(fitted.feature_names),
            "coefficients": [float(x) for x in fitted.coefficients],
            "mean_held_out_residual": fitted.mean_residual,
            "beats_direct": fitted.mean_residual < fixed.mean_direct,
            "beats_path": fitted.mean_residual < fixed.mean_path,
            "beats_fixed_dashi": fitted.mean_residual < fixed.mean_dashi,
            "fit_policy": "ordinary least squares on training pairs only; no intercept; no held-out tuning",
        },
        "structural_source": {
            "mode": "malecns_synapse_primary_neuropil_fractional_membership",
            "source_rows": membership.source_rows,
            "observed_neuron_count": membership.observed_neuron_count,
            "membership_semantics": "retained-region synapse incidence divided by each neuron's total synaptic incidence across all MaleCNS primary neuropils",
        },
        "functional_source": {
            "atlas_identifier": producer.atlas_identifier,
            "source_identifier": producer.source_identifier,
            "identity_semantics": "shared atlas region identity; not same-neuron identity",
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
