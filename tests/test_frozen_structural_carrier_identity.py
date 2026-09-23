import numpy as np
import pytest

from dashi.analysis.frozen_structural_latent_encoder import (
    assert_frozen_encoder_matches_structural_family,
    fit_frozen_structural_loro_encoder,
    load_frozen_structural_loro_encoder,
    save_frozen_structural_loro_encoder,
    structural_fibre_family_sha256,
)
from dashi.analysis.ndim_structure_function import StructuralFibreFamily


def _family(scale: float = 1.0) -> StructuralFibreFamily:
    regions = ("A", "B", "C", "D")
    direct = scale * np.array(
        [
            [0.0, 2.0, 1.0, 0.5],
            [1.0, 0.0, 3.0, 1.0],
            [2.0, 0.5, 0.0, 2.0],
            [1.5, 1.0, 0.5, 0.0],
        ],
        dtype=float,
    )
    signed = direct * np.array(
        [
            [0.0, 1.0, -1.0, 1.0],
            [-1.0, 0.0, 1.0, 1.0],
            [1.0, -1.0, 0.0, 1.0],
            [1.0, 1.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    return StructuralFibreFamily(
        regions,
        {
            "direct_forward": direct,
            "direct_reverse": direct.T,
            "signed_forward": signed,
            "signed_reverse": signed.T,
        },
    )


def test_frozen_encoder_records_exact_structural_family_fingerprint():
    family = _family()
    encoder = fit_frozen_structural_loro_encoder(family)

    assert encoder.structural_carrier_sha256 == structural_fibre_family_sha256(family)
    assert_frozen_encoder_matches_structural_family(encoder, family)


def test_same_region_vocabulary_does_not_authorize_changed_structural_carrier():
    encoder = fit_frozen_structural_loro_encoder(_family())
    changed = _family(scale=1.0001)

    assert changed.regions == encoder.regions
    with pytest.raises(ValueError, match="structural carrier fingerprint"):
        assert_frozen_encoder_matches_structural_family(encoder, changed)


def test_structural_fingerprint_survives_pickle_free_artifact_roundtrip(tmp_path):
    family = _family()
    encoder = fit_frozen_structural_loro_encoder(family)
    target = tmp_path / "encoder.npz"

    save_frozen_structural_loro_encoder(encoder, target)
    restored = load_frozen_structural_loro_encoder(target)

    assert restored.structural_carrier_sha256 == encoder.structural_carrier_sha256
    assert_frozen_encoder_matches_structural_family(restored, family)
