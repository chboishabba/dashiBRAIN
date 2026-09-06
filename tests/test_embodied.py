from dashi.analysis.embodied import (
    DROSOPHILA_CALCIUM_SOURCE,
    EffectorROM,
    MALE_CNS_SOURCE,
    ObservationModality,
    ObservationQuotient,
    ScientificSource,
    fruit_fly_bold_fmri_receipted,
)


def test_scientific_source_requires_stable_identifier():
    try:
        ScientificSource("Author", "Title", "")
    except ValueError as exc:
        assert "stable identifier" in str(exc)
    else:
        raise AssertionError("missing stable identifier must be rejected")


def test_calcium_is_not_promoted_to_bold_fmri():
    assert fruit_fly_bold_fmri_receipted((DROSOPHILA_CALCIUM_SOURCE,)) is False


def test_explicit_bold_source_is_detected_only_when_supplied():
    source = ScientificSource(
        "Example Author",
        "BOLD fMRI in an example system",
        "doi:10.0000/example",
    )
    assert fruit_fly_bold_fmri_receipted((source,)) is True


def test_effector_rom_defaults_to_non_exact():
    rom = EffectorROM(
        project=lambda x: x,
        reconstruct=lambda x: x,
        residual=lambda _: 0,
        residual_admissible=lambda r: r == 0,
        receipt="toy",
    )
    assert rom.exact_inverse_claim is False


def test_effector_rom_rejects_silent_exactness_promotion():
    try:
        EffectorROM(
            project=lambda x: x,
            reconstruct=lambda x: x,
            residual=lambda _: 0,
            residual_admissible=lambda r: r == 0,
            receipt="toy",
            exact_inverse_claim=True,
        )
    except ValueError as exc:
        assert "exactness" in str(exc).lower()
    else:
        raise AssertionError("silent exactness promotion must be rejected")


def test_observation_quotient_preserves_modality():
    q = ObservationQuotient(
        modality=ObservationModality.OPTICAL_CALCIUM,
        observe=lambda x: x,
        source=DROSOPHILA_CALCIUM_SOURCE,
    )
    assert q.modality is ObservationModality.OPTICAL_CALCIUM
    assert q.modality is not ObservationModality.BOLD_FMRI
    assert q.lossy is True


def test_malecns_source_has_doi():
    assert MALE_CNS_SOURCE.stable_identifier.startswith("doi:")
