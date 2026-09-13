"""Claim-specific promotion gates for symbolic-learning experiments.

This module reuses the existing MaleCNS ``benchmark_promotion`` run authority
rather than inventing another run-mode lattice.  A training update is not an
empirical learning result unless the run is artifact-verified and the symbolic
task succeeds on a verified held-out split.  Stronger claims require stronger
receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from dashi.analysis.benchmark_promotion import PromotionLevel, RunEvidenceStatus


class LearningClaimKind(str, Enum):
    SYMBOLIC_LEARNING = "symbolic_learning"
    CONNECTOME_ADVANTAGE = "connectome_advantage"
    CROSS_TASK_TRANSFER = "cross_task_transfer"
    GENERAL_PROGRAMMING_COMPETENCE = "general_programming_competence"


@dataclass(frozen=True)
class SymbolicLearningEvidence:
    run: RunEvidenceStatus
    learning_update_applied: bool
    held_out_task_passed: bool
    matched_topology_control_present: bool
    cross_task_transfer_passed: bool

    @property
    def empirical_learning_paid(self) -> bool:
        return (
            self.run.promotion_level is PromotionLevel.EMPIRICALLY_PROMOTABLE
            and self.run.held_out_split_verified
            and not self.run.same_trial_only
            and self.learning_update_applied
            and self.held_out_task_passed
        )


def learning_claim_promotable(
    evidence: SymbolicLearningEvidence,
    claim: LearningClaimKind,
) -> bool:
    """Return whether this exact receipt family can promote ``claim``."""
    if not evidence.empirical_learning_paid:
        return False

    if claim is LearningClaimKind.SYMBOLIC_LEARNING:
        return True
    if claim is LearningClaimKind.CONNECTOME_ADVANTAGE:
        return evidence.matched_topology_control_present
    if claim is LearningClaimKind.CROSS_TASK_TRANSFER:
        return evidence.cross_task_transfer_passed
    if claim is LearningClaimKind.GENERAL_PROGRAMMING_COMPETENCE:
        return False
    return False
