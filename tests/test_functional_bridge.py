from dashi.analysis.functional_bridge import (
    CrossModalityResiduals,
    FlyConnectomeFunctionalAdapter,
    LOGOTHETIS_BOLD_SOURCE,
    UBAGHS_CALCIUM_BOLD_SOURCE,
    mammalian_bridge_does_not_create_fly_bold_receipt,
)


def test_cross_modality_sources_have_stable_dois():
    assert LOGOTHETIS_BOLD_SOURCE.stable_identifier == "doi:10.1038/35084005"
    assert UBAGHS_CALCIUM_BOLD_SOURCE.stable_identifier == "doi:10.1038/s41592-026-03154-2"


def test_cross_modality_residuals_reject_modality_identity():
    try:
        CrossModalityResiduals(
            calcium_bold=lambda c, b: c - b,
            voltage_bold=lambda v, b: v - b,
            ephys_bold=lambda e, b: e - b,
            admissible=lambda _: True,
            modalities_identified=True,
        )
    except ValueError as exc:
        assert "identity" in str(exc).lower()
    else:
        raise AssertionError("calibration must not collapse modalities")


def test_mammalian_bridge_does_not_create_drosophila_bold_receipt():
    assert mammalian_bridge_does_not_create_fly_bold_receipt() is True


def test_fly_adapter_defaults_to_optical_without_bold_receipt():
    adapter = FlyConnectomeFunctionalAdapter(
        connectome_to_latent=lambda x: x,
        optical_readout=lambda x: x,
    )
    assert adapter.fly_bold_receipt_present is False


def test_fly_adapter_rejects_unreceipted_bold_promotion():
    try:
        FlyConnectomeFunctionalAdapter(
            connectome_to_latent=lambda x: x,
            optical_readout=lambda x: x,
            fly_bold_receipt_present=True,
        )
    except ValueError as exc:
        assert "drosophila bold" in str(exc).lower()
    else:
        raise AssertionError("unreceipted fly BOLD promotion must be rejected")
