from pathlib import Path

from dashi.io.malecns_real_data import MALECNS_REAL_AUTHORITIES
from dashi.io.motor_atlas_loader import load_fanc_muscle_json_directory


def test_tier1_authorities_are_hash_pinned():
    expected = {
        "connectome_weights_significant": "5c536423a62a688e59e7b441f9c04d6272c9a1f017e35814cf561f8c275d9e9e",
        "body_annotations": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
        "body_neurotransmitters": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
    }
    for key, digest in expected.items():
        authority = MALECNS_REAL_AUTHORITIES[key]
        assert authority.resolved_filename
        assert authority.expected_sha256 == digest
        assert authority.direct_download


def test_unresolved_downstream_artifacts_do_not_claim_hash_authority():
    for key in (
        "functional_trial_calcium",
        "registration_bifrost_map",
        "motor_neuron_muscle_map",
        "behaviour_fictrac_kinematics",
    ):
        authority = MALECNS_REAL_AUTHORITIES[key]
        assert authority.expected_sha256 is None
        assert not authority.direct_download


def test_fanc_json_loader_preserves_cross_animal_boundary(tmp_path: Path):
    d = tmp_path / "jsons"
    d.mkdir()
    (d / "ti_flexor.json").write_text(
        '{"segments": [101, 102], "hiddenSegments": [103]}',
        encoding="utf-8",
    )
    mapping = load_fanc_muscle_json_directory(d)
    assert mapping.neuron_to_module["101"] == "ti_flexor"
    assert mapping.neuron_to_module["103"] == "ti_flexor"
    assert mapping.module_to_effector["ti_flexor"] == "ti_flexor"
    assert not mapping.same_animal_identity
