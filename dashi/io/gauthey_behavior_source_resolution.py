"""Classify possible Gauthey behavior artifacts from archive member names.

This module is deliberately filename-only.  It can surface candidates from a
remote ZIP central directory without downloading large members, but a filename
match does not establish artifact identity, trial binding, or synchronized
neural/behavior timebase alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

from dashi.io.remote_zip import RemoteZipMember


_STRONG_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("behavior", ("behavior", "behaviour")),
    ("fictrac", ("fictrac",)),
    ("kinematic", ("kinematic", "kinematics")),
    ("locomotion", ("locomotion", "locomotor")),
)
_WEAK_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ball", ("ball",)),
    ("treadmill", ("treadmill",)),
)


@dataclass(frozen=True)
class BehaviorSourceCandidate:
    archive_member: str
    basename: str
    matched_terms: tuple[str, ...]
    compressed_size: int
    uncompressed_size: int
    crc32: int


@dataclass(frozen=True)
class BehaviorSourceResolution:
    strong_candidates: tuple[BehaviorSourceCandidate, ...]
    weak_candidates: tuple[BehaviorSourceCandidate, ...]
    exact_behavior_artifact_identity_paid: bool = False
    exact_trial_binding_paid: bool = False
    synchronized_timebase_binding_paid: bool = False


def _matched_terms(name: str, vocabulary: tuple[tuple[str, tuple[str, ...]], ...]) -> tuple[str, ...]:
    lowered = name.lower()
    return tuple(
        canonical
        for canonical, aliases in vocabulary
        if any(alias in lowered for alias in aliases)
    )


def _candidate(member: RemoteZipMember, terms: tuple[str, ...]) -> BehaviorSourceCandidate:
    return BehaviorSourceCandidate(
        archive_member=member.name,
        basename=PurePosixPath(member.name).name,
        matched_terms=terms,
        compressed_size=member.compressed_size,
        uncompressed_size=member.uncompressed_size,
        crc32=member.crc32,
    )


def classify_behavior_archive_members(
    members: Iterable[RemoteZipMember],
) -> BehaviorSourceResolution:
    """Surface behavior-like archive members without promoting filename evidence.

    Strong candidates contain at least one explicit behavior/FicTrac/kinematic/
    locomotion term.  Ball/treadmill-only names are retained as weak candidates,
    because those words can also describe apparatus videos or calibration files.
    """

    strong: list[BehaviorSourceCandidate] = []
    weak: list[BehaviorSourceCandidate] = []
    for member in members:
        strong_terms = _matched_terms(member.name, _STRONG_TERMS)
        if strong_terms:
            strong.append(_candidate(member, strong_terms))
            continue
        weak_terms = _matched_terms(member.name, _WEAK_TERMS)
        if weak_terms:
            weak.append(_candidate(member, weak_terms))

    strong.sort(key=lambda candidate: candidate.archive_member)
    weak.sort(key=lambda candidate: candidate.archive_member)
    return BehaviorSourceResolution(
        strong_candidates=tuple(strong),
        weak_candidates=tuple(weak),
    )
