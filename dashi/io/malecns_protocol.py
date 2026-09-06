"""Receipt protocol for MaleCNS + functional imaging + behaviour benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from dashi.analysis.embodied import ScientificSource


@dataclass(frozen=True)
class ArtifactReceipt:
    role: str
    path: str
    sha256: str
    source: ScientificSource
    dataset_version: str

    def __post_init__(self) -> None:
        if len(self.sha256) != 64:
            raise ValueError("sha256 must be a 64-character hex digest")
        int(self.sha256, 16)
        if not self.dataset_version.strip():
            raise ValueError("dataset_version is required")


@dataclass(frozen=True)
class RegistrationReceipt:
    functional_artifact_sha256: str
    structural_artifact_sha256: str
    method_source: ScientificSource
    mapping_artifact_sha256: str
    evidence_kind: str
    residual_definition: str

    def __post_init__(self) -> None:
        for digest in (
            self.functional_artifact_sha256,
            self.structural_artifact_sha256,
            self.mapping_artifact_sha256,
        ):
            if len(digest) != 64:
                raise ValueError("registration digests must be sha256")
            int(digest, 16)
        if not self.residual_definition.strip():
            raise ValueError("registration residual definition is required")


@dataclass(frozen=True)
class BenchmarkSplitReceipt:
    split_artifact_sha256: str
    train_ids_sha256: str
    held_out_ids_sha256: str
    leakage_check_passed: bool

    def __post_init__(self) -> None:
        if not self.leakage_check_passed:
            raise ValueError("benchmark split cannot promote with leakage")


@dataclass(frozen=True)
class BenchmarkRunReceipt:
    structural: ArtifactReceipt
    functional: ArtifactReceipt
    behaviour: ArtifactReceipt | None
    registration: RegistrationReceipt
    split: BenchmarkSplitReceipt
    predictor_name: str
    predictor_commit: str
    output_sha256: str
    metric_name: str
    metric_value: float

    def __post_init__(self) -> None:
        if not self.predictor_commit.strip():
            raise ValueError("predictor commit is required")
        if not self.metric_name.strip():
            raise ValueError("metric name is required")
        if len(self.output_sha256) != 64:
            raise ValueError("output_sha256 must be sha256")
        int(self.output_sha256, 16)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_ids(ids: Iterable[str]) -> str:
    canonical = "\n".join(sorted(str(x) for x in ids)).encode("utf-8")
    return sha256(canonical).hexdigest()


def assert_disjoint_ids(train_ids: Iterable[str], held_out_ids: Iterable[str]) -> None:
    overlap = set(train_ids).intersection(held_out_ids)
    if overlap:
        sample = sorted(overlap)[:5]
        raise ValueError(f"train/held-out leakage detected: {sample}")
