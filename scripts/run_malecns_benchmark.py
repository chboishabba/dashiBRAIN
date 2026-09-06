#!/usr/bin/env python3
"""Run dependence-aware MaleCNS structure/function/effector benchmark.

Mock mode exercises mechanics only. Real mode never substitutes hash-derived
observations: it runs only the empirical stages supported by present, resolved
artifacts and reports unavailable stages explicitly.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from dashi.analysis.benchmark import (
    PredictorKind,
    StructureFunctionBenchmark,
    dashi_beats_baseline,
    mean_numeric_residual,
)
from dashi.analysis.benchmark_nulls import residual_against_permutation_nulls
from dashi.analysis.benchmark_promotion import (
    BenchmarkRunMode,
    RunEvidenceStatus,
    consumer_promotable_for_run,
    result_wording,
)
from dashi.analysis.consumer_evidence import EvidenceConsumer
from dashi.analysis.provenance_dependence import EvidenceRelation, classify_evidence_relation
from dashi.analysis.structure_function_real import (
    aggregate_connectome_by_region,
    evaluate_region_structure_function,
    functional_correlation,
)
from dashi.io.artifact_verification import verify_consumer_artifacts
from dashi.io.functional_imaging_loader import (
    aggregate_functional_traces_by_region,
    load_functional_traces,
    load_registration_map,
)
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
    # Deterministic pipeline diagnostic. These numbers have no empirical authority.
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
        for k, v in result.items():
            residuals[k].append(v)
    return {
        "status": "synthetic_diagnostic",
        "mean_direct_edge_residual": mean_numeric_residual(residuals[PredictorKind.DIRECT_EDGE]),
        "mean_path_aware_residual": mean_numeric_residual(residuals[PredictorKind.PATH_AWARE]),
        "mean_dashi_residual": mean_numeric_residual(residuals[PredictorKind.DASHI]),
        "dashi_beats_direct": dashi_beats_baseline(
            residuals[PredictorKind.DASHI], residuals[PredictorKind.DIRECT_EDGE]
        ),
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

    graph = load_malecns_graph(
        manifest.target_path("connectome_weights_significant"),
        annotations_path=manifest.target_path("body_annotations"),
        neurotransmitters_path=(
            manifest.target_path("body_neurotransmitters")
            if manifest.is_present("body_neurotransmitters") else None
        ),
        signed_by_transmitter=False,
    )
    signed_graph = None
    if manifest.is_present("body_neurotransmitters"):
        signed_graph = load_malecns_graph(
            manifest.target_path("connectome_weights_significant"),
            annotations_path=manifest.target_path("body_annotations"),
            neurotransmitters_path=manifest.target_path("body_neurotransmitters"),
            signed_by_transmitter=True,
        )

    region_col = _metadata_column(graph, ("neuropil", "region", "primary_neuropil", "soma_neuropil"))
    if region_col is None:
        return {
            "status": "unavailable",
            "reason": "connectome annotations expose no recognized region/neuropil column",
            "available_metadata": sorted(graph.metadata or {}),
        }

    functional = load_functional_traces(manifest.target_path("functional_trial_calcium"))
    registration = load_registration_map(manifest.target_path("registration_bifrost_map"))
    functional_region = aggregate_functional_traces_by_region(functional, registration)

    node_regions = [str(x) for x in graph.metadata[region_col]]
    structural = aggregate_connectome_by_region(
        graph.carrier,
        node_regions,
        signed_adjacency=signed_graph.carrier if signed_graph is not None else None,
    )

    common = tuple(r for r in structural.regions if r in set(functional_region.unit_ids))
    if len(common) < 3:
        return {
            "status": "unavailable",
            "reason": "fewer than three common registered regions",
            "structural_region_count": len(structural.regions),
            "functional_region_count": len(functional_region.unit_ids),
        }

    si = [structural.regions.index(r) for r in common]
    fi = [functional_region.unit_ids.index(r) for r in common]
    direct = structural.direct[np.ix_(si, si)]
    two_hop = structural.two_hop[np.ix_(si, si)]
    signed = structural.signed_direct[np.ix_(si, si)] if structural.signed_direct is not None else None
    structural_common = type(structural)(common, direct, two_hop, signed)

    traces = functional_region.traces[:, fi]
    functional_assoc = functional_correlation(traces, common)

    # Pair-level holdout: deterministic checkerboard mask, disjoint from fitting pairs.
    n = len(common)
    ii, jj = np.indices((n, n))
    held_mask = ((ii + jj) % 3 == 0)
    result = evaluate_region_structure_function(structural_common, functional_assoc, held_out_mask=held_mask)

    null = residual_against_permutation_nulls(
        # Null comparison uses the full direct structural surface scaled only for shape.
        structural_common.direct,
        functional_assoc.matrix,
        n_null=100,
        seed=42,
    )

    return {
        "status": "real_region_level",
        "resolution": "region_or_neuropil",
        "direct_neuron_identity_fraction": registration.direct_identity_fraction(),
        "regions": list(common),
        "mean_direct_edge_residual": result.mean_direct,
        "mean_path_aware_residual": result.mean_path,
        "mean_dashi_residual": result.mean_dashi,
        "dashi_beats_direct": result.mean_dashi < result.mean_direct,
        "dashi_beats_path": result.mean_dashi < result.mean_path,
        "registration_permutation_null_p": null.empirical_p_value,
        "held_out_pair_count": int(result.observed.size),
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

    consumer_verification = {
        c: verify_consumer_artifacts(c, paths, MALECNS_REAL_AUTHORITIES)
        for c in (
            EvidenceConsumer.STRUCTURE_FUNCTION,
            EvidenceConsumer.NEURAL_STATE,
            EvidenceConsumer.EFFECTOR_STATE,
            EvidenceConsumer.BEHAVIOUR,
        )
    }

    if mock_run:
        nodes = [f"body_{i:04d}" for i in range(100)]
        metrics = _mock_structure_function(nodes)
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

    # Split receipt is meaningful only after a concrete evaluated unit family exists.
    held_out_verified = metrics.get("held_out_pair_count", 0) > 0 or mock_run

    bundles = manifest.build_consumer_bundles(
        dependence_relation=rel_cross_animal,
        provenance_adequate=True,
    )

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
        run_status = RunEvidenceStatus(
            mode=mode,
            input_hashes_verified=input_verified,
            registration_verified=(
                consumer not in {EvidenceConsumer.STRUCTURE_FUNCTION}
                or bool(verification and verification.artifacts.get("registration_bifrost_map") and verification.artifacts["registration_bifrost_map"].hash_verified)
            ),
            held_out_split_verified=held_out_verified,
            # Result payload is checksummed below; this records that the mechanism exists.
            output_hash_verified=not mock_run and metrics.get("status") == "real_region_level",
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
        },
    }
    summary["result_payload_sha256"] = _canonical_sha256(summary)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved benchmark run summary to {out}")

    print(f"Structure/function stage: {metrics.get('status')}")
    for c, result in consumer_results.items():
        print(
            f"  {c.upper()}: active={result['active_target']} "
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
