from pathlib import Path

from dashi.analysis.consumer_evidence import EvidenceConsumer
from dashi.io.artifact_verification import (
    ArtifactAuthority,
    ArtifactVerificationLevel,
    verify_artifact,
    verify_consumer_artifacts,
)


def test_present_file_without_expected_hash_is_only_digested(tmp_path: Path):
    p = tmp_path / "x.dat"
    p.write_bytes(b"abc")
    a = ArtifactAuthority("x", "doi:example", resolved_filename="x.dat")
    v = verify_artifact(p, a)
    assert v.level is ArtifactVerificationLevel.DIGESTED
    assert not v.hash_verified


def test_expected_hash_promotes_to_hash_verified(tmp_path: Path):
    p = tmp_path / "x.dat"
    p.write_bytes(b"abc")
    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    a = ArtifactAuthority("x", "doi:example", resolved_filename="x.dat", expected_sha256=digest)
    assert verify_artifact(p, a).hash_verified


def test_consumer_verification_is_consumer_scoped(tmp_path: Path):
    keys = ("connectome_weights_significant", "body_annotations", "functional_trial_calcium", "registration_bifrost_map")
    paths = {}
    authorities = {}
    for k in keys:
        p = tmp_path / k
        p.write_bytes(k.encode())
        paths[k] = p
        authorities[k] = ArtifactAuthority(k, "doi:test", resolved_filename=k)
    result = verify_consumer_artifacts(EvidenceConsumer.STRUCTURE_FUNCTION, paths, authorities)
    assert result.all_present
    assert not result.all_hash_verified
