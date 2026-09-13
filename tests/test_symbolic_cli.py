import json
from pathlib import Path

from dashi.analysis.symbolic_cli import build_arg_parser, run_from_args


def write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    weights = tmp_path / "weights.csv"
    weights.write_text(
        "body_pre,body_post,weight\n"
        "n0,n1,2\n"
        "n1,n2,2\n"
        "n2,n3,2\n"
        "n3,n0,2\n",
        encoding="utf-8",
    )
    decoder = tmp_path / "decoder.json"
    decoder.write_text(json.dumps({"n0": "x=1\\n"}), encoding="utf-8")
    return weights, decoder


def test_cli_builds_three_arm_packet_and_input_receipt(tmp_path: Path):
    weights, decoder = write_fixture(tmp_path)
    output = tmp_path / "packet.json"
    args = build_arg_parser().parse_args(
        [
            str(weights),
            "--decoder-json",
            str(decoder),
            "--active-body-id",
            "n0",
            "--rewire-swaps",
            "1",
            "--seed",
            "7",
            "--steps",
            "2",
            "--output",
            str(output),
        ]
    )

    payload = run_from_args(args)

    assert output.exists()
    assert payload["input_artifact"]["dataset_version"] == "male-cns:v1.0"
    assert len(payload["input_artifact"]["sha256"]) == 64
    assert payload["packet"]["baseline"]["topology_kind"] == "male_cns"
    assert payload["packet"]["topology_control"]["topology_kind"] == "degree_preserving_rewire"
    assert payload["packet"]["identity_control"]["identity_assignment_kind"] == "shuffled"
    assert payload["packet"]["viral_demo_reproduction_claimed"] is False


def test_cli_rejects_non_object_decoder_json(tmp_path: Path):
    weights, decoder = write_fixture(tmp_path)
    decoder.write_text("[]", encoding="utf-8")
    args = build_arg_parser().parse_args(
        [
            str(weights),
            "--decoder-json",
            str(decoder),
            "--active-body-id",
            "n0",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    try:
        run_from_args(args)
    except ValueError as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("non-object decoder JSON must be rejected")


def test_cli_rejects_empty_decoder_token(tmp_path: Path):
    weights, decoder = write_fixture(tmp_path)
    decoder.write_text(json.dumps({"n0": ""}), encoding="utf-8")
    args = build_arg_parser().parse_args(
        [
            str(weights),
            "--decoder-json",
            str(decoder),
            "--active-body-id",
            "n0",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    try:
        run_from_args(args)
    except ValueError as exc:
        assert "non-empty string" in str(exc)
    else:
        raise AssertionError("empty decoder token must be rejected")
