from dashi.analysis.functional_registration import (
    BIFROST_SOURCE,
    RegistrationEvidence,
    StructuralFunctionalComparison,
    identity_strength,
)


def test_bifrost_source_has_canonical_doi():
    assert BIFROST_SOURCE.stable_identifier == "doi:10.1073/pnas.2322687121"


def test_identity_strength_keeps_direct_identity_strongest():
    weak = identity_strength((RegistrationEvidence.ATLAS_REGION,))
    strong = identity_strength((RegistrationEvidence.DIRECT_IDENTITY,))
    assert weak < strong


def test_structure_function_identity_promotion_is_rejected():
    try:
        StructuralFunctionalComparison(
            structural=lambda a, b: 1,
            functional=lambda a, b: 1.0,
            residual=lambda s, f: abs(s - f),
            structure_equals_function_claim=True,
        )
    except ValueError as exc:
        assert "not identical" in str(exc).lower()
    else:
        raise AssertionError("structure/function identity must remain blocked")
