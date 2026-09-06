from pathlib import Path

from dashi.analysis.embodied import ScientificSource
from dashi.io.malecns_protocol import (
    ArtifactReceipt,
    BenchmarkSplitReceipt,
    RegistrationReceipt,
    assert_disjoint_ids,
    sha256_file,
    sha256_ids,
)


SOURCE = ScientificSource("Author", "Dataset", "doi:10.1/example")


def test_sha256_ids_is_order_invariant():
    assert sha256_ids(["b", "a"]) == sha256_ids(["a", "b"])


def test_disjoint_split_rejects_leakage():
    try:
        assert_disjoint_ids(["1", "2"], ["2", "3"])
    except ValueError as exc:
        assert "leakage" in str(exc)
    else:
        raise AssertionError("overlapping split must be rejected")


def test_artifact_receipt_requires_version_and_digest():
    r = ArtifactReceipt(
        role="structural_connectome",
        path="male_cns.csv",
        sha256="0" * 64,
        source=SOURCE,
        dataset_version="MaleCNS v1",
    )
    assert r.dataset_version == "MaleCNS v1"


def test_registration_receipt_carries_both_input_hashes():
    r = RegistrationReceipt(
        functional_artifact_sha256="1" * 64,
        structural_artifact_sha256="2" * 64,
        method_source=SOURCE,
        mapping_artifact_sha256="3" * 64,
        evidence_kind="atlas+morphology",
        residual_definition="atlas displacement",
    )
    assert r.functional_artifact_sha256 != r.structural_artifact_sha256


def test_split_receipt_blocks_failed_leakage_check():
    try:
        BenchmarkSplitReceipt(
            split_artifact_sha256="1" * 64,
            train_ids_sha256="2" * 64,
            held_out_ids_sha256="3" * 64,
            leakage_check_passed=False,
        )
    except ValueError as exc:
        assert "leakage" in str(exc)
    else:
        raise AssertionError("failed leakage check must block receipt")


def test_sha256_file(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("abc", encoding="utf-8")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
