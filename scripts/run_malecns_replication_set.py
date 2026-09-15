#!/usr/bin/env python3
"""Run frozen-carrier MaleCNS structure/function replication across recordings.

The manifest's region list is the frozen discovery vocabulary. Replicate files
may present those regions in another order, but no replicate may silently add,
remove, or substitute regions.

When ``--latent-encoder`` is supplied, the persisted structure-only LORO encoder
from discovery is loaded and reused unchanged for every independent replicate.
Within-replicate nuisance and linear consumer coefficients are still fit on that
recording's training folds; the representation geometry is not refit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dashi.analysis.frozen_structural_latent_encoder import (
    load_frozen_structural_loro_encoder,
)
from dashi.analysis.malecns_replication import (
    FrozenStructuralCarrier,
    ReplicateFunctionalInput,
    evaluate_frozen_latent_replicate,
    evaluate_replication_set,
)
from dashi.analysis.ndim_structure_function import build_ndim_structural_fibres
from dashi.analysis.overlap_controlled_structure_function import (
    load_overlap_membership_csv,
    restrict_overlap_kernel,
)
from dashi.analysis.structural_latent_ladder import structural_latent_ladder_to_dict
from dashi.analysis.structure_function_real import (
    RegionStructuralFeatures,
    aggregate_connectome_by_membership,
)
from dashi.io.functional_imaging_loader import FunctionalTraceTable
from dashi.io.malecns_loader import load_malecns_graph
from dashi.io.malecns_manifest import MaleCNSManifest
from dashi.io.malecns_neuropil_membership import load_synapse_neuropil_membership
from dashi.io.malecns_signs import signed_adjacency_for_graph
from dashi.io.region_functional_adapter import RegionFunctionalProducer, load_region_functional_producer


def _load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("replication manifest must be a JSON object")
    return payload


def _reorder_producer_exact(
    producer: RegionFunctionalProducer,
    regions: tuple[str, ...],
) -> RegionFunctionalProducer:
    producer.validate()
    observed = tuple(producer.traces.unit_ids)
    if set(observed) != set(regions) or len(observed) != len(regions):
        missing = sorted(set(regions) - set(observed))
        extra = sorted(set(observed) - set(regions))
        raise ValueError(
            f"replicate region vocabulary differs from frozen discovery vocabulary; "
            f"missing={missing}, extra={extra}"
        )
    index = {region: i for i, region in enumerate(observed)}
    ii = [index[region] for region in regions]
    return RegionFunctionalProducer(
        atlas_identifier=producer.atlas_identifier,
        traces=FunctionalTraceTable(
            unit_ids=regions,
            traces=np.asarray(producer.traces.traces[:, ii], dtype=float),
            time=np.asarray(producer.traces.time, dtype=float),
            identity_kind=producer.traces.identity_kind,
        ),
        source_identifier=producer.source_identifier,
        evidence_kind=producer.evidence_kind,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="data/malecns")
    p.add_argument("--synapse-partners", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--latent-encoder",
        help="Optional persisted structure-only frozen LORO encoder (.npz) from discovery",
    )
    args = p.parse_args()

    config = _load_json(args.manifest)
    regions = tuple(str(x) for x in config.get("regions", ()))
    if not regions or len(set(regions)) != len(regions):
        raise ValueError("manifest regions must be a non-empty unique ordered list")
    replicate_specs = config.get("replicates")
    if not isinstance(replicate_specs, list) or not replicate_specs:
        raise ValueError("manifest replicates must be a non-empty list")

    manifest = MaleCNSManifest(base_dir=args.base_dir)
    graph = load_malecns_graph(
        manifest.target_path("connectome_weights_significant"),
        annotations_path=manifest.target_path("body_annotations"),
        signed_by_transmitter=False,
    )
    if not manifest.is_present("body_neurotransmitters"):
        raise RuntimeError("body_neurotransmitters is required for frozen signed candidates")
    signed = signed_adjacency_for_graph(
        graph,
        manifest.target_path("body_neurotransmitters"),
    )
    neuropil = load_synapse_neuropil_membership(
        args.synapse_partners,
        graph.idx_to_id,
        allowed_regions=regions,
    )
    structural = aggregate_connectome_by_membership(
        graph.carrier,
        neuropil.membership,
        neuropil.regions,
        signed_adjacency=signed,
    )
    if tuple(structural.regions) != regions:
        index = {region: i for i, region in enumerate(structural.regions)}
        missing = [region for region in regions if region not in index]
        if missing:
            raise RuntimeError(f"MaleCNS structural carrier lacks frozen regions: {missing}")
        ii = [index[region] for region in regions]
        structural = RegionStructuralFeatures(
            regions,
            structural.direct[np.ix_(ii, ii)],
            structural.two_hop[np.ix_(ii, ii)],
            structural.signed_direct[np.ix_(ii, ii)] if structural.signed_direct is not None else None,
        )
    if structural.signed_direct is None:
        raise RuntimeError("signed structural carrier was not produced")
    frozen = FrozenStructuralCarrier(regions, structural.direct, structural.signed_direct)
    full_ndim_family = build_ndim_structural_fibres(structural)

    latent_encoder = None
    if args.latent_encoder:
        latent_encoder = load_frozen_structural_loro_encoder(args.latent_encoder)
        if latent_encoder.regions != regions:
            raise ValueError(
                "persisted latent encoder region carrier differs from replication manifest"
            )

    replicates: list[ReplicateFunctionalInput] = []
    for spec in replicate_specs:
        if not isinstance(spec, dict):
            raise ValueError("each replicate manifest entry must be an object")
        producer = load_region_functional_producer(
            spec["region_functional"],
            atlas_identifier=spec["atlas_identifier"],
            source_identifier=spec["source_identifier"],
        )
        producer = _reorder_producer_exact(producer, regions)
        membership = load_overlap_membership_csv(spec["functional_membership"])
        overlap_kernel = restrict_overlap_kernel(membership, regions)
        replicates.append(
            ReplicateFunctionalInput(
                replicate_id=str(spec["replicate_id"]),
                producer=producer,
                overlap_kernel=overlap_kernel,
                independent_recording=bool(spec.get("independent_recording", False)),
                independence_receipt=str(spec.get("independence_receipt", "")),
            )
        )

    result = evaluate_replication_set(frozen, replicates)
    latent_scores = (
        [
            evaluate_frozen_latent_replicate(
                latent_encoder,
                replicate,
                full_reference_family=full_ndim_family,
            )
            for replicate in replicates
        ]
        if latent_encoder is not None
        else []
    )
    latent_by_id = {score.replicate_id: score for score in latent_scores}

    payload = {
        "status": "malecns_frozen_carrier_independent_recording_replication",
        "frozen_regions": list(regions),
        "region_count": len(regions),
        "structural_source": {
            "source_rows": neuropil.source_rows,
            "observed_neuron_count": neuropil.observed_neuron_count,
            "representation_reselected_per_replicate": False,
        },
        "candidate_family": [
            "absolute_reverse",
            "relative_shape_reverse",
            "signed_reverse",
            "magnitude_shape_reverse",
        ],
        "frozen_latent_encoder": None
        if latent_encoder is None
        else {
            "artifact_path": str(args.latent_encoder),
            "common_max_dimension": latent_encoder.common_max_dimension,
            "correlation_threshold": latent_encoder.correlation_threshold,
            "functional_outcomes_used_to_fit_encoder": (
                latent_encoder.functional_outcomes_used_to_fit_encoder
            ),
            "reselected_per_replicate": False,
        },
        "replicate_count": result.replicate_count,
        "replicates": [
            {
                "replicate_id": score.replicate_id,
                "source_identifier": score.source_identifier,
                "atlas_identifier": score.atlas_identifier,
                "timepoint_count": score.timepoint_count,
                "residual_absolute_reverse": score.residual_absolute_reverse,
                "residual_relative_shape_reverse": score.residual_relative_shape_reverse,
                "residual_signed_reverse": score.residual_signed_reverse,
                "residual_magnitude_shape_reverse": score.residual_magnitude_shape_reverse,
                "magnitude_shape_gain_over_unsigned": score.magnitude_shape_gain_over_unsigned,
                "magnitude_shape_gain_over_relative_shape": score.magnitude_shape_gain_over_relative_shape,
                "frozen_structural_latent": None
                if score.replicate_id not in latent_by_id
                else {
                    "encoder_reselected_for_replicate": latent_by_id[
                        score.replicate_id
                    ].encoder_reselected_for_replicate,
                    "independent_recording_validated": latent_by_id[
                        score.replicate_id
                    ].independent_recording_validated,
                    "ladder": structural_latent_ladder_to_dict(
                        latent_by_id[score.replicate_id].ladder
                    ),
                },
            }
            for score in result.scores
        ],
        "summary": {
            "magnitude_shape_better_than_absolute_count": result.magnitude_shape_better_than_absolute_count,
            "magnitude_shape_better_than_relative_shape_count": result.magnitude_shape_better_than_relative_shape_count,
            "median_magnitude_shape_gain_over_unsigned": result.median_magnitude_shape_gain_over_unsigned,
        },
        "firewalls": {
            "within_replicate_loro_coefficients_are_fitted": True,
            "representation_reselection_allowed": False,
            "latent_encoder_refit_on_replicate": False,
            "pooled_recording_counts_as_independent_replication": False,
            "replication_implies_mechanism": False,
            "single_animal_or_recording_set_implies_population_generalization": False,
            "best_replicated_dimension_implies_universal_sufficient_latent": False,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
