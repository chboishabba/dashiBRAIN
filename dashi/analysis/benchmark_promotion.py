"""Run-mode-aware promotion boundary for MaleCNS benchmarks.

This module is intentionally independent of the concrete manifest loader so local
manifest implementations can consume it without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from dashi.analysis.consumer_evidence import ConsumerEvidenceBundle, EvidenceConsumer


class BenchmarkRunMode(str, Enum):
    DRY_RUN = "dry_run"
    MOCK = "mock"
    SYNTHETIC = "synthetic"
    REAL_UNVERIFIED = "real_unverified"
    REAL_HASH_VERIFIED = "real_hash_verified"


class PromotionLevel(str, Enum):
    PIPELINE_ONLY = "pipeline_only"
    DIAGNOSTIC = "diagnostic"
    EMPIRICAL_CANDIDATE = "empirical_candidate"
    EMPIRICALLY_PROMOTABLE = "empirically_promotable"


@dataclass(frozen=True)
class RunEvidenceStatus:
    mode: BenchmarkRunMode
    input_hashes_verified: bool
    registration_verified: bool
    held_out_split_verified: bool
    output_hash_verified: bool
    same_trial_only: bool = False
    synthetic_observations_present: bool = False

    @property
    def real_data(self) -> bool:
        return self.mode in {
            BenchmarkRunMode.REAL_UNVERIFIED,
            BenchmarkRunMode.REAL_HASH_VERIFIED,
        }

    @property
    def artifact_verified(self) -> bool:
        return all(
            (
                self.input_hashes_verified,
                self.registration_verified,
                self.held_out_split_verified,
                self.output_hash_verified,
            )
        )

    @property
    def promotion_level(self) -> PromotionLevel:
        if self.mode in {BenchmarkRunMode.DRY_RUN, BenchmarkRunMode.MOCK}:
            return PromotionLevel.PIPELINE_ONLY
        if self.mode is BenchmarkRunMode.SYNTHETIC or self.synthetic_observations_present:
            return PromotionLevel.DIAGNOSTIC
        if self.mode is BenchmarkRunMode.REAL_UNVERIFIED or not self.artifact_verified:
            return PromotionLevel.EMPIRICAL_CANDIDATE
        return PromotionLevel.EMPIRICALLY_PROMOTABLE


def consumer_promotable_for_run(
    bundle: ConsumerEvidenceBundle,
    run: RunEvidenceStatus,
) -> bool:
    """Require both consumer sufficiency and run-level empirical authority.

    A synthetically complete evidence bundle may exercise the pipeline but cannot
    promote an empirical consumer claim. Semantic promotion additionally cannot
    be based on a same-trial-only evidence family even if all named channels are
    present.
    """

    if not bundle.promotable:
        return False
    if run.promotion_level is not PromotionLevel.EMPIRICALLY_PROMOTABLE:
        return False
    if bundle.policy.consumer is EvidenceConsumer.SEMANTIC and run.same_trial_only:
        return False
    return True


def result_wording(run: RunEvidenceStatus) -> str:
    level = run.promotion_level
    if level is PromotionLevel.PIPELINE_ONLY:
        return "pipeline execution receipt only; no empirical claim"
    if level is PromotionLevel.DIAGNOSTIC:
        return "synthetic/diagnostic result; no empirical promotion"
    if level is PromotionLevel.EMPIRICAL_CANDIDATE:
        return "real-data candidate result pending artifact verification"
    return "artifact-verified empirical result eligible for consumer-specific promotion"
