"""Authority registry for the real MaleCNS benchmark tranche.

Repository-level identifiers are not treated as direct file locators.  Exact
files and expected hashes can be pinned later without changing benchmark logic.
"""

from __future__ import annotations

from dashi.io.artifact_verification import ArtifactAuthority


MALECNS_REAL_AUTHORITIES: dict[str, ArtifactAuthority] = {
    "connectome_weights_significant": ArtifactAuthority(
        key="connectome_weights_significant",
        repository_identifier="doi:10.1016/j.cell.2026.08.015",
        resolved_filename="connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
        direct_download=True,
    ),
    "body_annotations": ArtifactAuthority(
        key="body_annotations",
        repository_identifier="doi:10.1016/j.cell.2026.08.015",
        resolved_filename="body-annotations-male-cns-v1.0-minconf-0.5.feather",
        direct_download=True,
    ),
    "body_neurotransmitters": ArtifactAuthority(
        key="body_neurotransmitters",
        repository_identifier="doi:10.1016/j.cell.2026.08.015",
        resolved_filename="body-neurotransmitters-male-cns-v1.0.feather",
        direct_download=True,
    ),
    "functional_trial_calcium": ArtifactAuthority(
        key="functional_trial_calcium",
        repository_identifier="doi:10.5281/zenodo.17618684",
        # Repository is authoritative, but the exact benchmark file must be
        # resolved from the deposit before hash verification can be claimed.
        resolved_filename=None,
        direct_download=False,
    ),
    "registration_bifrost_map": ArtifactAuthority(
        key="registration_bifrost_map",
        repository_identifier="doi:10.5061/dryad.8pk0p2nx1;doi:10.5281/zenodo.11097259",
        resolved_filename=None,
        direct_download=False,
    ),
    "motor_neuron_muscle_map": ArtifactAuthority(
        key="motor_neuron_muscle_map",
        repository_identifier="doi:10.1038/s41586-024-07389-x;github:EllenLesser/Azevedo_Lesser_Phelps_Mark_2023",
        resolved_filename=None,
        direct_download=False,
    ),
    "behaviour_fictrac_kinematics": ArtifactAuthority(
        key="behaviour_fictrac_kinematics",
        repository_identifier="doi:10.5281/zenodo.17618684",
        resolved_filename=None,
        direct_download=False,
    ),
}
