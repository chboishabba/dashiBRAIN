from __future__ import annotations

from dashi.io.gauthey_lbm_source_preference import (
    SourceAvailability,
    choose_trial_source,
)


def test_prefer_existing_local_then_pdc_then_zenodo():
    available = SourceAvailability(
        local_zip="/scratch/a1_r6.zip",
        pdc_available=True,
        zenodo_available=True,
    )
    assert choose_trial_source(available).kind == "local"

    available = SourceAvailability(local_zip=None, pdc_available=True, zenodo_available=True)
    assert choose_trial_source(available).kind == "pdc"

    available = SourceAvailability(local_zip=None, pdc_available=False, zenodo_available=True)
    assert choose_trial_source(available).kind == "zenodo"


def test_source_gap_remains_explicit_when_no_repository_can_pay_trial():
    available = SourceAvailability(local_zip=None, pdc_available=False, zenodo_available=False)
    choice = choose_trial_source(available)
    assert choice.kind == "missing"
    assert choice.promotes_zero_contribution is False
