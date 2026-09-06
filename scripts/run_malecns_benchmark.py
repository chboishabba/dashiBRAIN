#!/usr/bin/env python3
"""Run dependence-aware MaleCNS structure/function/effector benchmark.

Mock mode exercises mechanics only. Real mode never substitutes hash-derived
observations: empirical stages run only when their real artifacts exist.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from dashi.analysis.benchmark import PredictorKind, StructureFunctionBenchmark, dashi_beats_baseline, mean_numeric_residual
from dashi.analysis.benchmark_nulls import train_heldout_permutation_nulls, train_heldout_topology_nulls
from dashi.analysis.benchmark_promotion import BenchmarkRunMode, RunEvidenceStatus, consumer_promotable_for_run, result_wording
from dashi.analysis.consumer_evidence import EvidenceConsumer
from dashi.analysis.provenance_dependence import classify_evidence_relation
from dashi.analysis.structure_function_real import (
    RegionStructuralFeatures,
    aggregate_connectome_by_region,
    evaluate_region_structure_function,
    functional_correlation,
    pairwise_train_holdout_masks,
)
from dashi.io.artifact_verification import verify_consumer_artifacts
from dashi.io.functional_imaging_loader import aggregate_functional_traces_by_region, load_functional_traces, load_registration_map
from dashi.io.malecns_manifest import MaleCNSManifest
from dashi.io.malecns_real_data import MALECNS_REAL_AUTHORITIES


def _canonical_sha256(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(blob).hexdigest()


def _paths(manifest: MaleCNSManifest) -> dict[str, Path]:
    return {k: manifest.target_path(k) for k in manifest.specs}


def _metadata_column(graph, aliases: tuple[str, ...]) -> str | None:
    if not graph.metadata:
        return None
    cols = list(graph.metadata)
    lowered = {c.lower(): c for c in cols}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    for c in cols:
        lc = c.lower()
        if any(alias.lower() in lc for alias in aliases):
            return c
    return None


def _mock_structure_function(nodes: list[str]) -> dict[str, Any]:
    benchmark = StructureFunctionBenchmark(
        direct_feature=lambda a, b: 1.0 if (sum(map(ord, str(a))) + sum(map(ord, str(b)))) % 5 == 0 else 0.0,
        path_feature=lambda a, b: float((sum(map(ord, str(a))) * 31 + sum(map(ord, str(b)))) % 10) / 10.0,
        predict=lambda kind, feat: {
            PredictorKind.DIRECT_EDGE: feat * 0.5,
            PredictorKind.PATH_AWARE: feat * 0.7 + 0.1,
            PredictorKind.DASHI: feat * 0.85 + 0.05,
        }[kind],
        observed=lambda fa, fb: float((sum(map(ord, str(fa))) * 17 + sum(map(ord, str(fb)))) % 10) / 10.0,
        residual=lambda p, o: abs(p - o),
        registration_confidence=lambda _: 0.95,
        training_fold=tuple(nodes[: len(nodes) // 2]),
        held_out_fold=tuple(nodes[len(nodes) // 2 :]),
        folds_disjoint=lambda a, b: set(a).isdisjoint(b),
    )
    held = list(benchmark.held_out_fold)
    pairs = [(held[i], held[(i + 1) % len(held)]) for i in range(min(30, max(0, len(held) - 1)))]
    residuals = {k: [] for k in PredictorKind}
    for pair in pairs:
        fpair = (f"f_{pair[0]}", f"f_{pair[1]}")
        result = benchmark.evaluate_pair(pair, fpair)
        for k, value in result.items():
            residuals[k].append(value)
    return {
        "status": "synthetic_diagnostic",
        "mean_direct_edge_residual": mean_numeric_residual(residuals[PredictorKind.DIRECT_EDGE]),
        "mean_path_aware_residual": mean_numeric_residual(residuals[PredictorKind.PATH_AWARE]),
        "mean_dashi_residual": mean_numeric_residual(residuals[PredictorKind.DASHI]),
        "dashi_beats_direct": dashi_beats_baseline(residuals[PredictorKind.DASHI], residuals[PredictorKind.DIRECT_EDGE]),
    }


def _real_structure_function(manifest: MaleCNSManifest) -> dict[str, Any]:
    required = (
        "connectome_weights_significant",
        "body_annotations",
        "functional_trial_calcium",
        "registration_bifrost_map",
    )
    missing = [k for k in required if not manifest.is_present(k)]
    if missing:
        return {"status": "unavailable", "missing_artifacts": missing}

    from dashi.io.malecns_loader import load_malecns_graph
    from dashi.io.malecns_signs import signed_adjacency_for_graph

    graph = load_malecns_graph(
        manifest.target_path("connectome_weights_significant"),
        annotations_path=manifest.target_path("body_annotations"),
        signed_by_transmitter=False,
    )
    signed_adjacency = None
    if manifest.is_present("body_neurotransmitters"):
        signed_adjacency = signed_adjacency_for_graph(
            graph,
            manifest.target_path("body_neurotransmitters"),
        )

    # Actual MaleCNS v1.0 exposes somaNeuromere. It is a defensible coarse
    # biological grouping, not a claim of direct functional-neuron identity.
    region_col = _metadata_column(
        graph,
        ("somaNeuromere", "neuromere", "neuropil", "region", "primary_neuropil", "soma_neuropil"),
    )
    if region_col is None:
        return {
            "status": "unavailable",
            "reason": "connectome annotations expose no recognized region/neuromere column",
            "available_metadata": sorted(graph.metadata or {}),
        }

    functional = load_functional_traces(manifest.target_path("functional_trial_calcium"))
    registration = load_registration_map(manifest.target_path("registration_bifrost_map"))
    functional_region = aggregate_functional_traces_by_region(functional, registration)

    node_regions = [str(x) for x in graph.metadata[region_col]]
    structural = aggregate_connectome_by_region(
        graph.carrier,
        node_regions,
        signed_adjacency=signed_adjacency,
    )

    functional_region_set = set(functional_region.unit_ids)
    common = tuple(r for r in structural.regions if r in functional_region_set)
    if len(common) < 3:
        return {
            "status": "unavailable",
            "reason": "fewer than three common registered regions",
            "structural_region_count": len(structural.regions),
            "functional_region_count": len(functional_region.unit_ids),
            "structural_region_field": region_col,
        }

    structural_index = {r: i for i, r in enumerate(structural.regions)}
    functional_index = {r: i for i, r in enumerate(functional_region.unit_ids)}
    si = [structural_index[r] for r in common]
    fi = [functional_index[r] for r in common]

    direct = structural.direct[np.ix_(si, si)]
    two_hop = structural.two_hop[np.ix_(si, si)]
    signed = structural.signed_direct[np.ix_(si, si)] if structural.signed_direct is not None else None
    structural_common = RegionStructuralFeatures(common, direct, two_hop, signed)

    traces = functional_region.traces[:, fi]
    functional_assoc = functional_correlation(traces, common)
    fit_mask, held_mask = pairwise_train_holdout_masks(len(common), modulus=3, held_out_residue=0)
    result = evaluate_region_structure_function(
        structural_common,
        functional_assoc,
        fit_mask=fit_mask,
        held_out_mask=held_mask,
    )

    signed_feature = signed if signed is not None else np.zeros_like(direct)
    dashi_raw = 0.5 * direct + 0.4 * two_hop + 0.1 * signed_feature
    registration_null = train_heldout_permutation_nulls(
        dashi_raw,
        functional_assoc.matrix,
        fit_mask=fit_mask,
        held_out_mask=held_mask,
        n_null=100,
        seed=42,
    )
    topology_null = train_heldout_topology_nulls(
        direct,
        functional_assoc.matrix,
        fit_mask=fit_mask,
        held_out_mask=held_mask,
        n_null=100,
        seed=4242,
    )

    return {
        "status": "real_region_level",
        "resolution": "soma_neuromere_or_registered_region",
        "structural_region_field": region_col,
        "direct_neuron_identity_fraction": registration.direct_identity_fraction(),
        "regions": list(common),
        "fit_pair_count": result.fit_pair_count,
        "held_out_pair_count": result.held_out_pair_count,
        "mean_direct_edge_residual": result.mean_direct,
        "mean_path_aware_residual": result.mean_path,
        "mean_dashi_residual": result.mean_dashi,
        "dashi_beats_direct": result.mean_dashi < result.mean_direct,
        "dashi_beats_path": result.mean_dashi < result.mean_path,
        "registration_permutation_null_p": registration_null.empirical_p_value,
        "region_degree_preserving_null_p": topology_null.empirical_p_value,
    }


def run_benchmark(base_dir: str = "data/malecns", mock_run: bool = False, output_path: str | None = None) -> dict:
    manifest = MaleCNSManifest(base_dir=base_dir)
    prov_graph = manifest.build_provenance_graph()
    paths = _paths(manifest)

    print("=== MaleCNS Dependence-Aware Benchmark ===")
    print(f"Base Data Directory: {manifest.base_dir}")
    print(f"Mock Run Mode: {mock_run}")

    rel_same_trial = classify_evidence_relation(prov_graph, "functional_trial_calcium", "behaviour_fictrac_kinematics")
    rel_cross_animal = classify_evidence_relation(prov_graph, "connectome_weights_significant", "functional_trial_calcium")

    tracked_consumers = (
        EvidenceConsumer.STRUCTURE_FUNCTION,
        EvidenceConsumer.NEURAL_STATE,
        EvidenceConsumer.EFFECTOR_STATE,
        EvidenceConsumer.BEHAVIOUR,
    )
    consumer_verification = {
        consumer: verify_consumer_artifacts(consumer, paths, MALECNS_REAL_AUTHORITIES)
        for consumer in tracked_consumers
    }

    if mock_run:
        metrics = _mock_structure_function([f"body_{i:04d}" for i in range(100)])
        global_mode = BenchmarkRunMode.MOCK
    else:
        metrics = _real_structure_function(manifest)
        any_real = any(v.all_present for v in consumer_verification.values())
        any_verified = any(v.all_hash_verified for v in consumer_verification.values())
        global_mode = (
            BenchmarkRunMode.REAL_HASH_VERIFIED if any_verified
            else BenchmarkRunMode.REAL_UNVERIFIED if any_real
            else BenchmarkRunMode.SYNTHETIC
        )

    held_out_verified = metrics.get("held_out_pair_count", 0) > 0 or mock_run
    bundles = manifest.build_consumer_bundles(dependence_relation=rel_cross_animal, provenance_adequate=True)

    consumer_results: dict[str, Any] = {}
    for consumer in (
        EvidenceConsumer.STRUCTURE_FUNCTION,
        EvidenceConsumer.EFFECTOR_STATE,
        EvidenceConsumer.BEHAVIOUR,
        EvidenceConsumer.SEMANTIC,
    ):
        bundle = bundles[consumer]
        verification = consumer_verification.get(consumer)
        active_target = consumer is not EvidenceConsumer.SEMANTIC
        input_verified = bool(verification and verification.all_hash_verified)
        input_present = bool(verification and verification.all_present)
        mode = BenchmarkRunMode.MOCK if mock_run else (
            BenchmarkRunMode.REAL_HASH_VERIFIED if input_verified
            else BenchmarkRunMode.REAL_UNVERIFIED if input_present
            else BenchmarkRunMode.SYNTHETIC
        )
        registration_verified = (
            consumer is not EvidenceConsumer.STRUCTURE_FUNCTION
            or bool(
                verification
                and verification.artifacts.get("registration_bifrost_map")
                and verification.artifacts["registration_bifrost_map"].hash_verified
            )
        )
        # The result payload is digested below. It is an integrity receipt, not
        # an independently pre-pinned scientific authority; empirical promotion
        # still requires all consumer inputs and the real evaluated stage.
        output_verified = (
            not mock_run
            and metrics.get("status") == "real_region_level"
            and input_verified
        )
        run_status = RunEvidenceStatus(
            mode=mode,
            input_hashes_verified=input_verified,
            registration_verified=registration_verified,
            held_out_split_verified=held_out_verified,
            output_hash_verified=output_verified,
            same_trial_only=(consumer is EvidenceConsumer.SEMANTIC),
            synthetic_observations_present=mock_run or metrics.get("status") == "synthetic_diagnostic",
        )
        empirical = active_target and consumer_promotable_for_run(bundle, run_status)
        consumer_results[consumer.value] = {
            "active_target": active_target,
            "policy_sufficient": bundle.promotable,
            "artifacts_present": input_present,
            "artifacts_hash_verified": input_verified,
            "run_authority": run_status.promotion_level.value,
            "run_wording": result_wording(run_status),
            "empirically_promotable": empirical,
        }

    summary: dict[str, Any] = {
        "status": "success",
        "mock_run": mock_run,
        "global_mode": global_mode.value,
        "metrics": metrics,
        "evidence_relations": {
            "calcium_vs_kinematics": rel_same_trial.value,
            "connectome_vs_calcium": rel_cross_animal.value,
        },
        "consumer_promotions": consumer_results,
        "semantic_target_active": False,
        "boundaries": {
            "same_region_is_not_same_neuron": True,
            "cross_animal_motor_atlas_is_not_same_animal_identity": True,
            "repository_doi_is_not_direct_file_receipt": True,
            "fit_pairs_are_disjoint_from_held_out_pairs": True,
        },
    }
    summary["result_payload_sha256"] = _canonical_sha256(summary)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved benchmark run summary to {out}")

    print(f"Structure/function stage: {metrics.get('status')}")
    for consumer, result in consumer_results.items():
        print(
            f"  {consumer.upper()}: active={result['active_target']} "
            f"policy={result['policy_sufficient']} verified={result['artifacts_hash_verified']} "
            f"promotion={result['empirically_promotable']}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MaleCNS dependence-aware benchmark")
    parser.add_argument("--base-dir", default="data/malecns")
    parser.add_argument("--mock-run", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    run_benchmark(args.base_dir, args.mock_run, args.output)


if __name__ == "__main__":
    main()
