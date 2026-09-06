"""Authority registry for the real MaleCNS benchmark tranche.

Repository-level identifiers are not treated as direct file locators. Exact
files become empirically authoritative only when their filenames and expected
hashes are pinned. Large upstream repositories can also expose concrete source
resources without pretending those resources are already benchmark-ready.
"""

from __future__ import annotations

from dataclasses import dataclass

from dashi.io.artifact_verification import ArtifactAuthority


@dataclass(frozen=True)
class UpstreamRepositoryArtifact:
    """Concrete repository file discovered during data-resolution work.

    This is intentionally distinct from ``ArtifactAuthority``: an upstream file
    can be source-authoritative while still requiring a derived/adapted benchmark
    artifact (for example a large BIFROST transform rather than a compact region
    registration table).
    """

    repository_identifier: str
    filename: str
    size_bytes: int
    sha256: str
    role: str


# Tier-1 hashes are receipts from the canonical MaleCNS v1.0 object downloads.
MALECNS_REAL_AUTHORITIES: dict[str, ArtifactAuthority] = {
    "connectome_weights_significant": ArtifactAuthority(
        key="connectome_weights_significant",
        repository_identifier="doi:10.1016/j.cell.2026.08.015",
        resolved_filename="connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
        expected_sha256="5c536423a62a688e59e7b441f9c04d6272c9a1f017e35814cf561f8c275d9e9e",
        direct_download=True,
    ),
    "body_annotations": ArtifactAuthority(
        key="body_annotations",
        repository_identifier="doi:10.1016/j.cell.2026.08.015",
        resolved_filename="body-annotations-male-cns-v1.0-minconf-0.5.feather",
        expected_sha256="2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
        direct_download=True,
    ),
    "body_neurotransmitters": ArtifactAuthority(
        key="body_neurotransmitters",
        repository_identifier="doi:10.1016/j.cell.2026.08.015",
        resolved_filename="body-neurotransmitters-male-cns-v1.0.feather",
        expected_sha256="95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
        direct_download=True,
    ),
    "functional_trial_calcium": ArtifactAuthority(
        key="functional_trial_calcium",
        repository_identifier="doi:10.5281/zenodo.17618684",
        # The repository contains Data.zip (~48.75 GB). A compact benchmark
        # trace file must be resolved/extracted before this key can be verified.
        resolved_filename=None,
        direct_download=False,
    ),
    "registration_bifrost_map": ArtifactAuthority(
        key="registration_bifrost_map",
        repository_identifier="doi:10.5061/dryad.8pk0p2nx1;doi:10.5281/zenodo.11097259",
        # BIFROST publishes large NIfTI/HDF5 transform resources. The benchmark
        # consumes a derived compact region mapping, not a fictitious CSV from
        # the paper DOI.
        resolved_filename=None,
        direct_download=False,
    ),
    "motor_neuron_muscle_map": ArtifactAuthority(
        key="motor_neuron_muscle_map",
        repository_identifier="doi:10.1038/s41586-024-07389-x;github:EllenLesser/Azevedo_Lesser_Phelps_Mark_2023",
        # Concrete upstream mappings exist (escape_df.pkl, jsons/*.json and
        # data/synapse_tables/*), but the cross-animal atlas still requires an
        # explicit derived module mapping before same-benchmark use.
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


# Exact BIFROST source resources discovered from Dryad version 295150. These
# hashes are repository receipts, not claims that the files are compact
# registration maps suitable for direct ingestion by this benchmark.
BIFROST_DRYAD_RESOURCES: tuple[UpstreamRepositoryArtifact, ...] = (
    UpstreamRepositoryArtifact(
        "doi:10.5061/dryad.8pk0p2nx1",
        "README.md",
        5_900,
        "6e692ab3c5f1da4e3e8ce26816a2ca9b4e01b90de36ed8ce0cdda7eb05263417",
        "repository_readme",
    ),
    UpstreamRepositoryArtifact(
        "doi:10.5061/dryad.8pk0p2nx1",
        "replication_dataset.tar.gz",
        6_390_000_000,
        "0a49a0a0d674670932286938d832e83c8fc78ee1e1206cc7d003699fa07da057",
        "replication_dataset",
    ),
    UpstreamRepositoryArtifact(
        "doi:10.5061/dryad.8pk0p2nx1",
        "nifti1_compliant_FDA.nii",
        2_430_000_000,
        "82eae16603e8055c0212751f556afd2c02a86e3f2dde1348d428ec4dd8e562d5",
        "functional_data_atlas",
    ),
    UpstreamRepositoryArtifact(
        "doi:10.5061/dryad.8pk0p2nx1",
        "thresholded_FDA_to_JRC2018_female.h5",
        13_800_000_000,
        "1c096f5baf488a59c8deaddf13bd989a21b94e978c14ac0626192cfcaa8868fb",
        "registration_transform",
    ),
)


# Concrete Azevedo/Lesser resources discovered in the public analysis
# repository. No immutable hash is asserted here until a repository commit is
# pinned; this registry only removes the previous fiction that a paper DOI was a
# ready-made motor_neuron_to_muscle_targets.csv file.
AZEVEDO_LESSER_RESOLVED_FILES: tuple[str, ...] = (
    "escape_df.pkl",
    "jsons/ti_flexor.json",
    "jsons/ti_extensor.json",
    "jsons/acc_ti_flexor.json",
    "jsons/sternal_adductor.json",
    "jsons/fe_reductor.json",
    "data/synapse_tables/",
)
