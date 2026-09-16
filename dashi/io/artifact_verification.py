"""Consumer-scoped artifact verification for empirical MaleCNS runs.

A file being present is not equivalent to being an authority-pinned artifact.
Verification is intentionally layered: presence -> digest -> expected digest ->
resolved repository file.  Consumer promotion may depend on different artifact
sets, so verification is indexed by consumer rather than one global boolean.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

from dashi.analysis.consumer_evidence import EvidenceConsumer
from dashi.io.malecns_protocol import sha256_file


class ArtifactVerificationLevel(str, Enum):
    MISSING = "missing"
    PRESENT = "present"
    DIGESTED = "digested"
    HASH_VERIFIED = "hash_verified"


@dataclass(frozen=True)
class ArtifactAuthority:
    key: str
    repository_identifier: str
    resolved_filename: str | None = None
    expected_sha256: str | None = None
    direct_download: bool = False

    @property
    def repository_resolved(self) -> bool:
        return bool(self.resolved_filename)


@dataclass(frozen=True)
class ArtifactVerification:
    key: str
    path: str
    authority: ArtifactAuthority
    level: ArtifactVerificationLevel
    actual_sha256: str | None = None
    size_bytes: int = 0

    @property
    def hash_verified(self) -> bool:
        return self.level is ArtifactVerificationLevel.HASH_VERIFIED


@dataclass(frozen=True)
class ConsumerArtifactVerification:
    consumer: EvidenceConsumer
    required_keys: tuple[str, ...]
    artifacts: Mapping[str, ArtifactVerification]

    @property
    def all_present(self) -> bool:
        return all(self.artifacts[k].level is not ArtifactVerificationLevel.MISSING for k in self.required_keys)

    @property
    def all_hash_verified(self) -> bool:
        return all(self.artifacts[k].hash_verified for k in self.required_keys)


DEFAULT_REQUIRED_ARTIFACTS: Mapping[EvidenceConsumer, tuple[str, ...]] = {
    EvidenceConsumer.STRUCTURE_FUNCTION: (
        "connectome_weights_significant",
        "body_annotations",
        "functional_trial_calcium",
        "registration_bifrost_map",
    ),
    EvidenceConsumer.NEURAL_STATE: (
        "functional_trial_calcium",
        "registration_bifrost_map",
    ),
    EvidenceConsumer.EFFECTOR_STATE: (
        "motor_neuron_muscle_map",
        "behaviour_fictrac_kinematics",
    ),
    EvidenceConsumer.BEHAVIOUR: (
        "behaviour_fictrac_kinematics",
    ),
    # Semantic inference is architecture-only for the present MaleCNS programme.
    # Deliberately no active empirical artifact set is declared here.
    EvidenceConsumer.SEMANTIC: (),
}


def verify_artifact(path: str | Path, authority: ArtifactAuthority) -> ArtifactVerification:
    p = Path(path)
    if not p.is_file() or p.stat().st_size <= 0:
        return ArtifactVerification(authority.key, str(p), authority, ArtifactVerificationLevel.MISSING)

    digest = sha256_file(p)
    size = p.stat().st_size
    if authority.expected_sha256 is None:
        return ArtifactVerification(
            authority.key,
            str(p),
            authority,
            ArtifactVerificationLevel.DIGESTED,
            actual_sha256=digest,
            size_bytes=size,
        )

    level = (
        ArtifactVerificationLevel.HASH_VERIFIED
        if digest.lower() == authority.expected_sha256.lower()
        else ArtifactVerificationLevel.DIGESTED
    )
    return ArtifactVerification(authority.key, str(p), authority, level, digest, size)


def verify_consumer_artifacts(
    consumer: EvidenceConsumer,
    paths: Mapping[str, str | Path],
    authorities: Mapping[str, ArtifactAuthority],
    required: Mapping[EvidenceConsumer, tuple[str, ...]] = DEFAULT_REQUIRED_ARTIFACTS,
) -> ConsumerArtifactVerification:
    keys = required[consumer]
    artifacts = {k: verify_artifact(paths[k], authorities[k]) for k in keys}
    return ConsumerArtifactVerification(consumer, keys, artifacts)
