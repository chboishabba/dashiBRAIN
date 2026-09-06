"""Closed-world payment frontier for Gauthey et al. 2026 benchmark inputs.

Scientific source:
Wayan Gauthey, Albert Lin, Osama M. Ahmed, Andrew M. Leifer, Mala Murthy,
Stephan Y. Thiberge, "High-speed whole-brain imaging in Drosophila",
DOI 10.1038/s41467-026-72437-1; preprocessed data DOI 10.5281/zenodo.17618684;
analysis code github:murthylab/lightbead-analysis.

This module turns repository/deposit discovery into explicit benchmark payment
states.  A failed closed-world recovery route is not equivalent to an absent
scientific fact: it means an external producer must supply a receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class PaymentRoute(str, Enum):
    DEPOSIT_RECOVERY = "deposit_recovery"
    EXTERNAL_SCIENTIFIC_RECEIPT = "external_scientific_receipt"
    SOURCE_IMPLEMENTATION_REQUIRED = "source_implementation_required"


@dataclass(frozen=True)
class DepositRecoveryFrontier:
    archive_member_count: int
    required_file_count: int
    uniquely_resolved_file_count: int
    zero_match_file_count: int
    ambiguous_file_count: int
    all_required_uniquely_resolved: bool
    exact_trace_recovery_available: bool
    payment_a_route: PaymentRoute
    required_output_rows: int = 940
    note: str = ""


@dataclass(frozen=True)
class AtlasRegistrationFrontier:
    launcher_present: bool
    implementation_present: bool
    ants_dependency_declared: bool
    payment_b_route: PaymentRoute
    note: str


def classify_deposit_source_resolution(payload: Mapping[str, object]) -> DepositRecoveryFrontier:
    """Classify the output of ``resolve_gauthey_2p_sources.py``.

    The resolver stores a mapping from each required basename to zero, one, or
    multiple matching archive members.  Exact row recovery is available only
    when every required source file resolves uniquely.
    """
    resolved_obj = payload.get("resolved")
    if not isinstance(resolved_obj, Mapping):
        raise ValueError("resolution payload lacks mapping 'resolved'")

    required_obj = payload.get("required_source_files")
    if not isinstance(required_obj, Sequence) or isinstance(required_obj, (str, bytes)):
        raise ValueError("resolution payload lacks sequence 'required_source_files'")

    unique = zero = ambiguous = 0
    for basename in required_obj:
        matches = resolved_obj.get(str(basename), [])
        if not isinstance(matches, Sequence) or isinstance(matches, (str, bytes)):
            raise ValueError(f"resolution matches for {basename!r} are not a sequence")
        n = len(matches)
        if n == 1:
            unique += 1
        elif n == 0:
            zero += 1
        else:
            ambiguous += 1

    all_unique = unique == len(required_obj)
    return DepositRecoveryFrontier(
        archive_member_count=int(payload.get("archive_member_count", 0)),
        required_file_count=len(required_obj),
        uniquely_resolved_file_count=unique,
        zero_match_file_count=zero,
        ambiguous_file_count=ambiguous,
        all_required_uniquely_resolved=all_unique,
        exact_trace_recovery_available=all_unique,
        payment_a_route=(
            PaymentRoute.DEPOSIT_RECOVERY
            if all_unique
            else PaymentRoute.EXTERNAL_SCIENTIFIC_RECEIPT
        ),
        required_output_rows=940,
        note=(
            "All generating source matrices/labels resolve uniquely; conditional exact trace-row recovery may run."
            if all_unique
            else "Closed-world deposit recovery cannot close selected-ROI identity; require an explicit 940-row selected_roi -> (trial, plane, cluster) receipt."
        ),
    )


def canonical_public_atlas_registration_frontier() -> AtlasRegistrationFrontier:
    """Current public-code boundary for the Gauthey local-atlas route.

    The public repository contains ``batch_tiff_to_local_atlas_AL.sh`` and its
    README declares ANTsPy for the signal-extraction pipeline, but the launcher
    references ``tiff_to_local_atlas.py`` and that implementation is not present
    in the public repository tree audited for this tranche.
    """
    return AtlasRegistrationFrontier(
        launcher_present=True,
        implementation_present=False,
        ants_dependency_declared=True,
        payment_b_route=PaymentRoute.SOURCE_IMPLEMENTATION_REQUIRED,
        note=(
            "Public launcher exists, but the referenced tiff_to_local_atlas.py implementation is not source-owned in the audited public repository; do not infer an executable trial->atlas transform from the launcher alone."
        ),
    )
