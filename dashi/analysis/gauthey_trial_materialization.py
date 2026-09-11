"""Trial-wise materialization coordinates for Gauthey LBM replication.

The deposited selected functional carrier is pooled across six LBM recordings.
The selected-row reconstruction already recovers the exact source trial, plane,
and cluster for each retained row, while the deposit exposes one segmentation
artifact per trial.  Therefore trial-native functional fields can be built for
all six recordings without asserting a common-atlas registration.

Common-atlas replication is a later gate.  A trial is registration-ready only
when a same-trial anatomical image/transform route is explicitly supplied; a
mean brain from another recording must never be silently reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from dashi.analysis.gauthey_lbm_experiment import (
    LBM_LABEL_MEMBERS,
    LBM_SOURCE_PICKLE_NAMES,
    LBM_TRIALS,
)
from dashi.analysis.gauthey_lbm_native_field import RecoveredSelectedROI


@dataclass(frozen=True)
class GautheyTrialMaterializationSpec:
    trial_id: str
    source_pickle_name: str
    segmentation_name: str
    discovery_recording: bool
    same_trial_mean_brain_name: str | None = None

    @property
    def has_pinned_same_trial_anatomy(self) -> bool:
        return self.same_trial_mean_brain_name is not None


# Only the a2_r5 mean brain is currently pinned by the compact working-set route.
# This is an acquisition state, not a statement that other trial anatomies do not
# exist upstream.
_A2_R5_MEAN_BRAIN = "04032024_GCamp6f_a2_r5_w3_mean_G.nii"


def canonical_trial_specs() -> tuple[GautheyTrialMaterializationSpec, ...]:
    if not (
        len(LBM_TRIALS)
        == len(LBM_SOURCE_PICKLE_NAMES)
        == len(LBM_LABEL_MEMBERS)
        == 6
    ):
        raise RuntimeError("Gauthey six-trial source coordinates are misaligned")

    specs: list[GautheyTrialMaterializationSpec] = []
    for trial, source_name, label_member in zip(
        LBM_TRIALS, LBM_SOURCE_PICKLE_NAMES, LBM_LABEL_MEMBERS
    ):
        specs.append(
            GautheyTrialMaterializationSpec(
                trial_id=trial,
                source_pickle_name=source_name,
                segmentation_name=Path(label_member).name,
                discovery_recording=(trial == "04032024_6f_a2_r5"),
                same_trial_mean_brain_name=(
                    _A2_R5_MEAN_BRAIN if trial == "04032024_6f_a2_r5" else None
                ),
            )
        )
    return tuple(specs)


def identities_by_trial(
    identities: Sequence[RecoveredSelectedROI],
) -> dict[str, tuple[RecoveredSelectedROI, ...]]:
    known = {spec.trial_id for spec in canonical_trial_specs()}
    grouped: dict[str, list[RecoveredSelectedROI]] = {trial: [] for trial in known}
    for identity in identities:
        if identity.trial_id not in known:
            raise ValueError(f"identity references unknown Gauthey trial {identity.trial_id!r}")
        grouped[identity.trial_id].append(identity)
    return {
        trial: tuple(sorted(rows, key=lambda item: item.selected_row))
        for trial, rows in grouped.items()
    }


@dataclass(frozen=True)
class GautheyTrialReadiness:
    trial_id: str
    selected_roi_count: int
    segmentation_present: bool
    same_trial_mean_brain_present: bool
    native_field_ready: bool
    common_atlas_registration_ready: bool
    discovery_recording: bool


def trial_readiness(
    identities: Sequence[RecoveredSelectedROI],
    *,
    segmentation_paths: Mapping[str, str | Path],
    mean_brain_paths: Mapping[str, str | Path] | None = None,
) -> tuple[GautheyTrialReadiness, ...]:
    grouped = identities_by_trial(identities)
    mean_brains = {} if mean_brain_paths is None else dict(mean_brain_paths)
    out: list[GautheyTrialReadiness] = []
    for spec in canonical_trial_specs():
        selected_count = len(grouped[spec.trial_id])
        seg_path = segmentation_paths.get(spec.trial_id)
        seg_present = seg_path is not None and Path(seg_path).exists()
        mean_path = mean_brains.get(spec.trial_id)
        mean_present = mean_path is not None and Path(mean_path).exists()
        native_ready = selected_count > 0 and seg_present
        out.append(
            GautheyTrialReadiness(
                trial_id=spec.trial_id,
                selected_roi_count=selected_count,
                segmentation_present=seg_present,
                same_trial_mean_brain_present=mean_present,
                native_field_ready=native_ready,
                common_atlas_registration_ready=native_ready and mean_present,
                discovery_recording=spec.discovery_recording,
            )
        )
    return tuple(out)


def default_segmentation_paths(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    return {
        spec.trial_id: base / spec.segmentation_name
        for spec in canonical_trial_specs()
    }


def default_mean_brain_paths(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    return {
        spec.trial_id: base / spec.same_trial_mean_brain_name
        for spec in canonical_trial_specs()
        if spec.same_trial_mean_brain_name is not None
    }
