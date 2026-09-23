"""Closed-world payment frontier for Gauthey et al. 2026 benchmark inputs.

Scientific source:
Wayan Gauthey, Albert Lin, Osama M. Ahmed, Andrew M. Leifer, Mala Murthy,
Stephan Y. Thiberge, "High-speed whole-brain imaging in Drosophila",
DOI 10.1038/s41467-026-72437-1; preprocessed data DOI 10.5281/zenodo.17618684;
analysis code github:murthylab/lightbead-analysis.

This module turns repository/deposit discovery into explicit benchmark payment
states. A failed closed-world recovery route is not equivalent to an absent
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


@dataclass(frozen=True)
class PublicRawAcquisitionFrontier:
    """Public-source boundary for non-discovery same-trial anatomy/raw access.

    Generic repository capabilities and historical internal filesystem paths are
    retained as discovery metadata only. Neither is promoted into a public
    dataset-specific access receipt.
    """

    representative_raw_public: bool
    all_trial_preprocessed_public: bool
    princeton_mirror_public: bool
    pdc_large_deposits_may_use_globus: bool
    historical_internal_storage_namespace_observed: bool
    historical_internal_storage_root: str
    public_non_discovery_raw_route_found: bool
    public_non_discovery_anatomy_route_found: bool
    dataset_specific_public_globus_endpoint_found: bool
    internal_path_is_public_access_receipt: bool
    next_payment_route: PaymentRoute
    note: str


def classify_deposit_source_resolution(payload: Mapping[str, object]) -> DepositRecoveryFrontier:
    """Classify the output of ``resolve_gauthey_2p_sources.py``.

    The resolver stores a mapping from each required basename to zero, one, or
    multiple matching archive members. Exact row recovery is available only
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


def canonical_deposited_2p_recovery_frontier() -> DepositRecoveryFrontier:
    """Pinned closed-world audit for the published Gauthey ``Data.zip``.

    The execution receipt inspected the full 35-member central directory. All
    eight source-code-declared conventional-2p generating inputs had zero
    matches. This pins the *deposit route* closed-negative while leaving open an
    external scientific receipt from the data producers or another authoritative
    carrier.
    """
    return DepositRecoveryFrontier(
        archive_member_count=35,
        required_file_count=8,
        uniquely_resolved_file_count=0,
        zero_match_file_count=8,
        ambiguous_file_count=0,
        all_required_uniquely_resolved=False,
        exact_trace_recovery_available=False,
        payment_a_route=PaymentRoute.EXTERNAL_SCIENTIFIC_RECEIPT,
        required_output_rows=940,
        note=(
            "Audited deposited source set contains none of the eight generating conventional-2p inputs; Payment A requires an external 940-row selected_roi -> (trial, plane, cluster) receipt."
        ),
    )


def canonical_public_atlas_registration_frontier() -> AtlasRegistrationFrontier:
    """Current public-code boundary for the Gauthey local-atlas route.

    The public repository contains ``batch_tiff_to_local_atlas_AL.sh`` and its
    README declares ANTsPy for the signal-extraction pipeline. The launcher
    invokes ``tiff_to_local_atlas.py``; that referenced implementation was not
    present in the public repository tree audited for this tranche. This is a
    public-source audit result, not a claim that the implementation never
    existed elsewhere.
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


def canonical_public_raw_acquisition_frontier() -> PublicRawAcquisitionFrontier:
    """Pinned public-source audit for the current replication wall.

    The paper advertises raw data for one representative trial and preprocessed
    data for all trials, with Princeton Data Commons as another public location
    for the deposited material. Princeton Data Commons supports Globus for large
    deposits in general, but the current audit did not locate a dataset-specific
    public Globus endpoint, non-discovery raw TIFF carrier, anatomical/reference
    volume, or saved transform.

    Historical ``lightbead-analysis`` commits preserve Princeton HPC namespaces
    including ``/scratch/gpfs/albertl/rigE_data/`` and trial-like internal paths.
    Those strings establish historical processing provenance only; they are not
    publicly routable object identifiers or access authority.
    """
    return PublicRawAcquisitionFrontier(
        representative_raw_public=True,
        all_trial_preprocessed_public=True,
        princeton_mirror_public=True,
        pdc_large_deposits_may_use_globus=True,
        historical_internal_storage_namespace_observed=True,
        historical_internal_storage_root="/scratch/gpfs/albertl/rigE_data/",
        public_non_discovery_raw_route_found=False,
        public_non_discovery_anatomy_route_found=False,
        dataset_specific_public_globus_endpoint_found=False,
        internal_path_is_public_access_receipt=False,
        next_payment_route=PaymentRoute.EXTERNAL_SCIENTIFIC_RECEIPT,
        note=(
            "Current public audit found no dataset-specific route from the paper, Zenodo/Princeton deposits, indexed Google-hosted objects, or published repository history to non-discovery raw/anatomical bytes or an executed same-trial transform. This is a bounded search result, not proof that such files do not exist privately or unindexed."
        ),
    )
