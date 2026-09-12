"""Concrete MaleCNS experiment manifest and receipt generation.

This module instantiates the abstract MaleCNS benchmark protocol, provenance
graphs, and consumer evidence policies with concrete artifact specifications,
dataset versions, and SHA-256 verification boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from dashi.analysis.consumer_evidence import (
    BEHAVIOUR_POLICY,
    EFFECTOR_POLICY,
    SEMANTIC_POLICY,
    STRUCTURE_FUNCTION_POLICY,
    ConsumerEvidenceBundle,
    EvidenceChannel,
    EvidenceConsumer,
)
from dashi.analysis.embodied import (
    FICTRAC_METHOD_SOURCE,
    MALE_CNS_SOURCE,
    ScientificSource,
)
from dashi.analysis.functional_registration import BIFROST_SOURCE
from dashi.analysis.motor_effector import AZEVEDO_MOTOR_ATLAS_SOURCE
from dashi.analysis.provenance_dependence import (
    EvidenceRelation,
    ProvenanceGraph,
    ProvenanceRoot,
    RootKind,
)
from dashi.io.malecns_protocol import (
    ArtifactReceipt,
    BenchmarkSplitReceipt,
    RegistrationReceipt,
    assert_disjoint_ids,
    sha256_file,
    sha256_ids,
)

GAUTHEY_WHOLE_BRAIN_SOURCE = ScientificSource(
    "Wayan Gauthey; Albert Lin; Osama M. Ahmed; Andrew M. Leifer; Mala Murthy; Stephan Y. Thiberge",
    "High-speed whole-brain imaging in Drosophila",
    "doi:10.1038/s41467-026-72437-1",
)


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    role: str
    tier: str
    relative_path: str
    download_url: str
    expected_size_bytes: int
    source: ScientificSource
    dataset_version: str
    provenance_roots: tuple[ProvenanceRoot, ...]
    description: str


# Canonical dataset roots.
ROOT_MALECNS_DATASET = ProvenanceRoot("dataset:malecns:v1.0", RootKind.DATASET, MALE_CNS_SOURCE.stable_identifier)
ROOT_MALECNS_ANIMAL = ProvenanceRoot("animal:flyem-male-cns-01", RootKind.ANIMAL, "specimen:Z0720-07")
ROOT_MALECNS_EM_ACQUISITION = ProvenanceRoot("acq:fibsem-male-cns", RootKind.ACQUISITION, "method:fibsem-hotknife")

ROOT_GAUTHEY_DATASET = ProvenanceRoot("dataset:gauthey:2026", RootKind.DATASET, "doi:10.5281/zenodo.17618684")
ROOT_GAUTHEY_AGGREGATED_FUNCTIONAL = ProvenanceRoot(
    "artifact:gauthey:dffs_audio_2p_corr_top05_all",
    RootKind.PREPROCESSING,
    "archive:Data.zip/Data/Dffs/Audio correlated/dffs_audio_2p_corr_top05_all.pkl",
)
ROOT_GAUTHEY_ANIMAL = ProvenanceRoot("animal:gauthey-fly-01", RootKind.ANIMAL, "specimen:gauthey-live-01")
ROOT_GAUTHEY_TRIAL = ProvenanceRoot("trial:gauthey-fly-01-t01", RootKind.TRIAL, "trial:t01")
ROOT_GAUTHEY_OPTICAL_ACQ = ProvenanceRoot("acq:lightbead-wholebrain", RootKind.ACQUISITION, "method:lightbead-two-photon")
ROOT_GAUTHEY_PREPROCESSING = ProvenanceRoot("prep:lightbead-dff", RootKind.PREPROCESSING, "pipeline:lightbead-analysis-v1")

ROOT_BIFROST_REGISTRATION = ProvenanceRoot("reg:bifrost-jrc2018", RootKind.REGISTRATION, BIFROST_SOURCE.stable_identifier)
ROOT_MOTOR_TARGETING = ProvenanceRoot("dataset:azevedo-lesser-vnc", RootKind.DATASET, AZEVEDO_MOTOR_ATLAS_SOURCE.stable_identifier)
ROOT_FICTRAC_METHOD = ProvenanceRoot("method:fictrac-spherical-motion", RootKind.OBSERVER_PROTOCOL, FICTRAC_METHOD_SOURCE.stable_identifier)


CANONICAL_ARTIFACT_SPECS: dict[str, ArtifactSpec] = {
    "connectome_weights_significant": ArtifactSpec(
        key="connectome_weights_significant",
        role="structural_adjacency",
        tier="connectome",
        relative_path="connectome/connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
        download_url="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
        expected_size_bytes=502_169_298,
        source=MALE_CNS_SOURCE,
        dataset_version="male-cns:v1.0",
        provenance_roots=(ROOT_MALECNS_DATASET, ROOT_MALECNS_ANIMAL, ROOT_MALECNS_EM_ACQUISITION),
        description="MaleCNS v1.0 significant connection graph (weights and segment pairs)",
    ),
    "body_annotations": ArtifactSpec(
        key="body_annotations",
        role="structural_annotations",
        tier="connectome",
        relative_path="connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather",
        download_url="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather",
        expected_size_bytes=14_483_314,
        source=MALE_CNS_SOURCE,
        dataset_version="male-cns:v1.0",
        provenance_roots=(ROOT_MALECNS_DATASET, ROOT_MALECNS_ANIMAL),
        description="MaleCNS curated neuron annotations",
    ),
    "body_neurotransmitters": ArtifactSpec(
        key="body_neurotransmitters",
        role="transmitter_signs",
        tier="connectome",
        relative_path="connectome/body-neurotransmitters-male-cns-v1.0.feather",
        download_url="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather",
        expected_size_bytes=43_282_834,
        source=MALE_CNS_SOURCE,
        dataset_version="male-cns:v1.0",
        provenance_roots=(ROOT_MALECNS_DATASET, ROOT_MALECNS_ANIMAL),
        description="Aggregate predicted neurotransmitters per body for sign determination",
    ),
    "functional_trial_calcium": ArtifactSpec(
        key="functional_trial_calcium",
        role="functional_observation",
        tier="functional",
        relative_path="functional/gauthey_compact/dffs_audio_2p_corr_top05_all.pkl",
        download_url="https://zenodo.org/api/records/17618684/files/Data.zip/content",
        expected_size_bytes=5_020_000,
        source=GAUTHEY_WHOLE_BRAIN_SOURCE,
        dataset_version="zenodo:17618684:Data.zip:dffs_audio_2p_corr_top05_all",
        # The runner resolved this as an aggregate preprocessed matrix. Do not
        # attach a particular animal/trial root until the deposit proves it.
        provenance_roots=(ROOT_GAUTHEY_DATASET, ROOT_GAUTHEY_AGGREGATED_FUNCTIONAL),
        description="Gauthey preprocessed audio-correlated 2-photon matrix; 940 x 668; archive-unit identities remain unregistered",
    ),
    "registration_bifrost_map": ArtifactSpec(
        key="registration_bifrost_map",
        role="registration_mapping",
        tier="registration",
        relative_path="registration/bifrost_mcns_to_jrc2018.csv",
        download_url="https://doi.org/10.1073/pnas.2322687121",
        expected_size_bytes=35_000_000,
        source=BIFROST_SOURCE,
        dataset_version="bifrost-v1",
        provenance_roots=(ROOT_BIFROST_REGISTRATION,),
        description="Derived compact functional-to-atlas region mapping; not identical to the upstream multi-GB BIFROST transform",
    ),
    "motor_neuron_muscle_map": ArtifactSpec(
        key="motor_neuron_muscle_map",
        role="effector_mapping",
        tier="effector",
        relative_path="effector/motor_neuron_to_muscle_targets.csv",
        download_url="https://doi.org/10.1038/s41586-024-07389-x",
        expected_size_bytes=8_500_000,
        source=AZEVEDO_MOTOR_ATLAS_SOURCE,
        dataset_version="vnc-motor-map-2024",
        provenance_roots=(ROOT_MOTOR_TARGETING,),
        description="Derived cross-animal VNC motor-module to muscle-target mapping; not same-animal MaleCNS identity",
    ),
    "behaviour_fictrac_kinematics": ArtifactSpec(
        key="behaviour_fictrac_kinematics",
        role="behaviour_observation",
        tier="behaviour",
        relative_path="behaviour/gauthey_trial01_fictrac_kinematics.csv",
        download_url="https://doi.org/10.1038/s41467-026-72437-1",
        expected_size_bytes=22_000_000,
        source=GAUTHEY_WHOLE_BRAIN_SOURCE,
        dataset_version="gauthey-fictrac-2026",
        provenance_roots=(ROOT_GAUTHEY_DATASET, ROOT_GAUTHEY_ANIMAL, ROOT_GAUTHEY_TRIAL, ROOT_FICTRAC_METHOD),
        description="Synchronized spherical treadmill velocities extracted via FicTrac; exact deposit member still unresolved",
    ),
}


class MaleCNSManifest:
    def __init__(self, base_dir: str | Path = "data/malecns", specs: Mapping[str, ArtifactSpec] | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.specs: dict[str, ArtifactSpec] = dict(specs or CANONICAL_ARTIFACT_SPECS)

    def get_tier_specs(self, tier: str) -> list[ArtifactSpec]:
        return [spec for spec in self.specs.values() if spec.tier == tier]

    def target_path(self, key: str) -> Path:
        return self.base_dir / self.specs[key].relative_path

    def is_present(self, key: str) -> bool:
        path = self.target_path(key)
        return path.is_file() and path.stat().st_size > 0

    def total_expected_bytes(self, tiers: Iterable[str] | None = None) -> int:
        selected_tiers = set(tiers) if tiers is not None else None
        return sum(spec.expected_size_bytes for spec in self.specs.values() if selected_tiers is None or spec.tier in selected_tiers)

    def build_provenance_graph(self) -> ProvenanceGraph[str]:
        return ProvenanceGraph({spec.key: frozenset(spec.provenance_roots) for spec in self.specs.values()})

    def create_artifact_receipt(self, key: str, sha256_digest: str | None = None) -> ArtifactReceipt:
        spec = self.specs[key]
        path = self.target_path(key)
        if sha256_digest is None:
            if not path.exists():
                raise FileNotFoundError(f"Artifact file not found: {path}")
            sha256_digest = sha256_file(path)
        return ArtifactReceipt(role=spec.role, path=str(path), sha256=sha256_digest, source=spec.source, dataset_version=spec.dataset_version)

    def create_registration_receipt(
        self,
        functional_key: str = "functional_trial_calcium",
        structural_key: str = "connectome_weights_significant",
        mapping_key: str = "registration_bifrost_map",
        evidence_kind: str = "neuropil_morphology_and_coordinates",
        residual_definition: str = "l2_coordinate_registration_error",
        functional_sha256: str | None = None,
        structural_sha256: str | None = None,
        mapping_sha256: str | None = None,
    ) -> RegistrationReceipt:
        fn_receipt = self.create_artifact_receipt(functional_key, functional_sha256)
        st_receipt = self.create_artifact_receipt(structural_key, structural_sha256)
        mp_receipt = self.create_artifact_receipt(mapping_key, mapping_sha256)
        return RegistrationReceipt(
            functional_artifact_sha256=fn_receipt.sha256,
            structural_artifact_sha256=st_receipt.sha256,
            method_source=BIFROST_SOURCE,
            mapping_artifact_sha256=mp_receipt.sha256,
            evidence_kind=evidence_kind,
            residual_definition=residual_definition,
        )

    def create_benchmark_split_receipt(
        self,
        train_neuron_ids: Sequence[str],
        held_out_neuron_ids: Sequence[str],
        split_artifact_name: str = "isomorphic_vs_dimorphic_split",
    ) -> BenchmarkSplitReceipt:
        assert_disjoint_ids(train_neuron_ids, held_out_neuron_ids)
        train_hash = sha256_ids(train_neuron_ids)
        held_out_hash = sha256_ids(held_out_neuron_ids)
        from hashlib import sha256
        split_hash = sha256(f"{split_artifact_name}\n{train_hash}\n{held_out_hash}".encode("utf-8")).hexdigest()
        return BenchmarkSplitReceipt(split_hash, train_hash, held_out_hash, True)

    def build_consumer_bundles(
        self,
        dependence_relation: EvidenceRelation = EvidenceRelation.CROSS_DATASET_REPLICATION,
        provenance_adequate: bool = True,
    ) -> dict[EvidenceConsumer, ConsumerEvidenceBundle]:
        return {
            EvidenceConsumer.STRUCTURE_FUNCTION: ConsumerEvidenceBundle(
                policy=STRUCTURE_FUNCTION_POLICY,
                channels=frozenset({EvidenceChannel.CONNECTOME, EvidenceChannel.CALCIUM, EvidenceChannel.REGISTRATION}),
                dependence_relation=dependence_relation,
                provenance_adequate=provenance_adequate,
            ),
            EvidenceConsumer.EFFECTOR_STATE: ConsumerEvidenceBundle(
                policy=EFFECTOR_POLICY,
                channels=frozenset({EvidenceChannel.MUSCLE_ACTIVATION, EvidenceChannel.KINEMATICS}),
                dependence_relation=dependence_relation,
                provenance_adequate=provenance_adequate,
            ),
            EvidenceConsumer.BEHAVIOUR: ConsumerEvidenceBundle(
                policy=BEHAVIOUR_POLICY,
                channels=frozenset({EvidenceChannel.KINEMATICS}),
                dependence_relation=dependence_relation,
                provenance_adequate=provenance_adequate,
            ),
            EvidenceConsumer.SEMANTIC: ConsumerEvidenceBundle(
                policy=SEMANTIC_POLICY,
                channels=frozenset({EvidenceChannel.KINEMATICS, EvidenceChannel.INTERACTION, EvidenceChannel.ENVIRONMENT}),
                dependence_relation=dependence_relation,
                provenance_adequate=provenance_adequate,
            ),
        }
