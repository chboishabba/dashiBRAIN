#!/usr/bin/env python3
"""Test MaleCNS coupling after controlling functional-carrier nuisances.

The ``compare`` mode is the canonical real-data decision run.  It streams the
MaleCNS synapse-partner carrier once, constructs the structural fibre family
once, and then evaluates the same structure against four functional targets:

    raw soft correlation
    overlap-controlled raw correlation
    published-stimulus-residualized correlation
    stimulus+overlap-controlled correlation

The expensive refitted label and strength-preserving wiring nulls are evaluated
against the final joint-controlled target.  This keeps the observer/control
comparison on one same-object structural carrier instead of paying another
311M-row structural aggregation for each nuisance choice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dashi.analysis.gauthey_lbm_experiment import published_lbm_stimulus_regressor
from dashi.analysis.ndim_structure_function import (
    build_ndim_structural_fibres,
    evaluate_leave_one_region_out,
)
from dashi.analysis.overlap_controlled_nulls import (
    overlap_controlled_strength_preserving_null_loro,
)
from dashi.analysis.overlap_controlled_structure_function import (
    evaluate_overlap_controlled_leave_one_region_out,
    load_overlap_membership_csv,
    overlap_controlled_label_permutation_null_loro,
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


def _controlled_summary(result) -> dict[str, object]:
    return {
        "weighted_mean_residual": result.weighted_mean_residual,
        "mean_fold_residual": result.mean_fold_residual,
        "folds": [
            {
                "held_out_region": fold.held_out_region,
                "mean_residual": fold.mean_residual,
                "nuisance_intercept": fold.nuisance.intercept,
                "nuisance_overlap_coefficient": fold.nuisance.coefficient,
                "selected_fibres": list(fold.selected_fibres),
                "coefficients_intercept_then_selected": fold.coefficients.tolist(),
            }
            for fold in result.folds
        ],
    }


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
        choices=("none", "published-gauthey", "compare"),
        default="none",
        help=(
            "none: raw functional target; published-gauthey: regress the published "
            "audio regressor first; compare: evaluate raw, overlap-only, stimulus-only, "
            "and stimulus+overlap targets on one shared real-data structural pass"
        ),
    )
    p.add_argument("--correlation-threshold", type=float, default=0.98)
    p.add_argument("--null-count", type=int, default=100)
    p.add_argument("--output", required=True)
    args = p.parse_args()

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

    # This is the only pass over the very large synapse-partner table.
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
    if len(common) < 3:
        raise RuntimeError(f"need >=3 common regions; got {common}")

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

    common_traces = np.asarray(producer.traces.traces[:, fi], dtype=float)
    functional_raw = functional_correlation(common_traces, common)
    overlap_kernel = restrict_overlap_kernel(functional_membership, common)
    family = build_ndim_structural_fibres(structural_common)

    raw_loro = evaluate_leave_one_region_out(
        family,
        functional_raw.matrix,
        correlation_threshold=args.correlation_threshold,
    )
    raw_overlap_controlled = evaluate_overlap_controlled_leave_one_region_out(
        family,
        functional_raw.matrix,
        overlap_kernel,
        correlation_threshold=args.correlation_threshold,
    )

    stimulus_summary: dict[str, object] = {
        "mode": args.stimulus_control,
        "applied": args.stimulus_control in ("published-gauthey", "compare"),
        "uses_structural_features": False,
        "uses_region_pair_outcomes": False,
    }

    analysis_traces = common_traces
    stimulus_only_loro = None
    stimulus_overlap_controlled = None
    if args.stimulus_control in ("published-gauthey", "compare"):
        stimulus = published_lbm_stimulus_regressor(common_traces.shape[0])
        stim_fit = residualize_region_traces_against_stimulus(common_traces, stimulus)
        analysis_traces = stim_fit.residual_traces
        stimulus_summary.update(
            {
                "regressor": (
                    "published Gauthey binary audio blocks convolved with source "
                    "GCaMP6f kernel"
                ),
                "fit": "intercept + stimulus coefficient independently per region over time",
                "coefficient_mean": float(np.mean(stim_fit.coefficients)),
                "coefficient_std": float(np.std(stim_fit.coefficients)),
                "coefficient_min": float(np.min(stim_fit.coefficients)),
                "coefficient_max": float(np.max(stim_fit.coefficients)),
                "variance_fraction_removed_mean": float(
                    np.mean(stim_fit.variance_fraction_removed)
                ),
                "variance_fraction_removed_median": float(
                    np.median(stim_fit.variance_fraction_removed)
                ),
                "variance_fraction_removed_min": float(
                    np.min(stim_fit.variance_fraction_removed)
                ),
                "variance_fraction_removed_max": float(
                    np.max(stim_fit.variance_fraction_removed)
                ),
            }
        )
        functional_stimulus = functional_correlation(analysis_traces, common)
        stimulus_only_loro = evaluate_leave_one_region_out(
            family,
            functional_stimulus.matrix,
            correlation_threshold=args.correlation_threshold,
        )
        stimulus_overlap_controlled = evaluate_overlap_controlled_leave_one_region_out(
            family,
            functional_stimulus.matrix,
            overlap_kernel,
            correlation_threshold=args.correlation_threshold,
        )
        final_functional = functional_stimulus
        final_controlled = stimulus_overlap_controlled
        final_target_name = "published-stimulus+atlas-overlap-controlled correlation"
    else:
        final_functional = functional_raw
        final_controlled = raw_overlap_controlled
        final_target_name = "atlas-overlap-controlled raw correlation"

    # Nulls are always refit against the final declared analysis target.
    _real_label, label_nulls, label_p = overlap_controlled_label_permutation_null_loro(
        family,
        final_functional.matrix,
        overlap_kernel,
        n_null=args.null_count,
        seed=20260914,
        correlation_threshold=args.correlation_threshold,
    )
    strength = overlap_controlled_strength_preserving_null_loro(
        structural_common,
        final_functional.matrix,
        overlap_kernel,
        n_null=args.null_count,
        seed=20260915,
        correlation_threshold=args.correlation_threshold,
    )

    stages: dict[str, object] = {
        "raw_soft": {
            "weighted_mean_residual": raw_loro.weighted_mean_residual,
            "consumer": "raw soft functional correlation",
        },
        "overlap_controlled_raw": {
            **_controlled_summary(raw_overlap_controlled),
            "consumer": "raw functional correlation after train-fold atlas-overlap control",
        },
    }
    if stimulus_only_loro is not None and stimulus_overlap_controlled is not None:
        stages["stimulus_controlled"] = {
            "weighted_mean_residual": stimulus_only_loro.weighted_mean_residual,
            "consumer": "functional correlation after per-region published-stimulus residualization",
        }
        stages["stimulus_and_overlap_controlled"] = {
            **_controlled_summary(stimulus_overlap_controlled),
            "consumer": (
                "published-stimulus-residualized correlation after train-fold "
                "atlas-overlap control"
            ),
        }

    payload = {
        "status": "real_region_level_joined_observer_ndim_comparator",
        "regions": list(common),
        "region_count": len(common),
        "structural_source": {
            "mode": "malecns_synapse_primary_neuropil_fractional_membership",
            "source_identifier": args.synapse_partners,
            "source_rows": neuropil.source_rows,
            "observed_neuron_count": neuropil.membership.shape[1],
            "streamed_once_for_all_control_stages": True,
        },
        "functional_source": {
            "atlas_identifier": args.region_functional_atlas,
            "source_identifier": args.region_functional_source,
            "membership_source": args.functional_membership,
        },
        "stimulus_nuisance": stimulus_summary,
        "overlap_nuisance": {
            "kernel": "cosine similarity of ROI->painted-domain membership rows",
            "fit": (
                "intercept + one overlap coefficient, fit independently on training "
                "pairs inside each fold"
            ),
            "held_out_pairs_used_for_nuisance_fit": False,
        },
        "joined_observer_stages": stages,
        "decision_target": {
            "name": final_target_name,
            **_controlled_summary(final_controlled),
            "region_label_permutation_null_p": label_p,
            "label_null_mean_residual": float(np.mean(label_nulls)),
            "label_null_min_residual": float(np.min(label_nulls)),
            "strength_preserving_wiring_null_p": strength.empirical_p_value,
            "strength_null_mean_residual": strength.null_mean_residual,
            "strength_null_min_residual": strength.null_min_residual,
            "max_row_strength_error": strength.max_row_strength_error,
            "max_column_strength_error": strength.max_column_strength_error,
        },
        "firewalls": {
            "atlas_overlap_kernel_is_structural_connectome_feature": False,
            "stimulus_regressor_is_structural_connectome_feature": False,
            "stimulus_residualization_uses_region_pair_outcomes": False,
            "nuisance_fit_uses_held_out_functional_pairs": False,
            "label_null_permuted_function_without_overlap_geometry": False,
            "strength_null_scrambles_functional_overlap_geometry": False,
            "residualization_is_automatically_information_refinement": False,
            "low_decision_residual_implies_wiring_specific_signal": False,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
