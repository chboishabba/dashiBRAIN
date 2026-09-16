#!/usr/bin/env python3
"""Compare absolute regional scale, relative wiring shape, and signed relative shape.

This runner reuses the real MaleCNS aggregation and Gauthey joined-control target.
It asks whether held-out structure/function transfer is carried mainly by:

* density-like absolute regional coupling D;
* row-relative wiring shape P;
* sender strength alone;
* full coarse signed coupling S;
* row-relative signed shape P * R, with absolute sender scale removed.

No new predictive family is introduced: every candidate is a singleton structural
carrier evaluated by the same overlap-controlled leave-one-region-out consumer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dashi.analysis.gauthey_lbm_experiment import published_lbm_stimulus_regressor
from dashi.analysis.network_scale_shape_discriminator import (
    decompose_network_scale_shape,
    evaluate_network_scale_shape,
)
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
    p.add_argument("--output", required=True)
    args = p.parse_args()

    manifest = MaleCNSManifest(base_dir=args.base_dir)
    graph = load_malecns_graph(
        manifest.target_path("connectome_weights_significant"),
        annotations_path=manifest.target_path("body_annotations"),
        signed_by_transmitter=False,
    )
    if not manifest.is_present("body_neurotransmitters"):
        raise RuntimeError("body_neurotransmitters is required for signed scale/shape comparison")
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

    traces = np.asarray(producer.traces.traces[:, fi], dtype=float)
    stimulus = published_lbm_stimulus_regressor(traces.shape[0])
    stim_fit = residualize_region_traces_against_stimulus(traces, stimulus)
    functional = functional_correlation(stim_fit.residual_traces, common)
    overlap_kernel = restrict_overlap_kernel(membership, common)

    result = evaluate_network_scale_shape(
        common,
        structural_common.direct,
        structural_common.signed_direct,
        functional.matrix,
        overlap_kernel,
    )
    dec = decompose_network_scale_shape(
        structural_common.direct,
        structural_common.signed_direct,
    )

    payload = {
        "status": "real_region_level_network_scale_shape_discriminator",
        "regions": list(common),
        "region_count": len(common),
        "structural_source": {
            "source_rows": neuropil.source_rows,
            "observed_neuron_count": neuropil.observed_neuron_count,
            "absolute_direct_semantics": (
                "membership-mass-normalized regional coupling density, not raw synapse count"
            ),
        },
        "consumer": (
            "published-stimulus-residualized functional correlation with foldwise atlas-overlap control"
        ),
        "decomposition": {
            "unsigned": "D[i,j] = out_strength[i] * row_shape[i,j]",
            "signed": "S[i,j] = out_strength[i] * row_shape[i,j] * sign_ratio[i,j]",
            "out_strength": dec.out_strength.tolist(),
            "reconstruction_error_unsigned": result.reconstruction_error_unsigned,
            "reconstruction_error_signed": result.reconstruction_error_signed,
        },
        "residuals": {
            "absolute_reverse": result.absolute_reverse_residual,
            "relative_shape_reverse": result.relative_shape_reverse_residual,
            "strength_only_reverse": result.strength_only_reverse_residual,
            "signed_reverse": result.signed_reverse_residual,
            "relative_signed_shape_reverse": result.relative_signed_shape_reverse_residual,
        },
        "firewalls": {
            "absolute_direct_is_raw_synapse_count": False,
            "global_scale_is_identifiable_after_train_fold_standardization": False,
            "relative_shape_preserves_sender_strength": False,
            "strength_only_preserves_target_specific_wiring": False,
            "relative_similarity_implies_physical_identity": False,
            "absolute_similarity_implies_functional_equivalence": False,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
