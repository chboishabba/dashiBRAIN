from pathlib import Path

from dashi.analysis.gauthey_lbm_native_field import RecoveredSelectedROI
from dashi.analysis.gauthey_trial_materialization import (
    canonical_trial_specs,
    default_segmentation_paths,
    identities_by_trial,
    trial_readiness,
)


def _identity(selected_row: int, trial_id: str) -> RecoveredSelectedROI:
    # Geometry consistency is tested in gauthey_lbm_native_field; this module only
    # groups already recovered identities by their declared trial.
    return RecoveredSelectedROI(
        selected_row=selected_row,
        pooled_source_row=selected_row,
        trial_id=trial_id,
        plane_index=0,
        cluster_index=0,
        correlation=0.5,
    )


def test_canonical_specs_cover_exactly_six_distinct_recordings():
    specs = canonical_trial_specs()
    assert len(specs) == 6
    assert len({spec.trial_id for spec in specs}) == 6
    assert len({spec.segmentation_name for spec in specs}) == 6
    assert sum(spec.discovery_recording for spec in specs) == 1
    assert next(spec for spec in specs if spec.discovery_recording).trial_id == "04032024_6f_a2_r5"


def test_only_pinned_same_trial_anatomy_is_marked_available_by_default():
    specs = canonical_trial_specs()
    pinned = [spec for spec in specs if spec.has_pinned_same_trial_anatomy]
    assert [spec.trial_id for spec in pinned] == ["04032024_6f_a2_r5"]
    assert pinned[0].same_trial_mean_brain_name == "04032024_GCamp6f_a2_r5_w3_mean_G.nii"


def test_identity_grouping_preserves_trials_and_zero_count_trials():
    specs = canonical_trial_specs()
    rows = [_identity(3, specs[0].trial_id), _identity(4, specs[0].trial_id), _identity(9, specs[2].trial_id)]
    grouped = identities_by_trial(rows)
    assert len(grouped) == 6
    assert [x.selected_row for x in grouped[specs[0].trial_id]] == [3, 4]
    assert [x.selected_row for x in grouped[specs[2].trial_id]] == [9]
    assert grouped[specs[1].trial_id] == ()


def test_native_and_common_atlas_readiness_are_separate_gates(tmp_path: Path):
    specs = canonical_trial_specs()
    first, discovery = specs[0], next(spec for spec in specs if spec.discovery_recording)
    rows = [_identity(1, first.trial_id), _identity(2, discovery.trial_id)]

    segmentation_paths = default_segmentation_paths(tmp_path)
    Path(segmentation_paths[first.trial_id]).touch()
    Path(segmentation_paths[discovery.trial_id]).touch()

    mean = tmp_path / discovery.same_trial_mean_brain_name
    mean.touch()
    states = trial_readiness(
        rows,
        segmentation_paths=segmentation_paths,
        mean_brain_paths={discovery.trial_id: mean},
    )
    by_trial = {state.trial_id: state for state in states}

    assert by_trial[first.trial_id].native_field_ready is True
    assert by_trial[first.trial_id].common_atlas_registration_ready is False
    assert by_trial[discovery.trial_id].native_field_ready is True
    assert by_trial[discovery.trial_id].common_atlas_registration_ready is True


def test_foreign_trial_anatomy_is_not_inferred_from_discovery_mean_brain(tmp_path: Path):
    specs = canonical_trial_specs()
    foreign = next(spec for spec in specs if not spec.discovery_recording)
    rows = [_identity(1, foreign.trial_id)]
    segmentation_paths = default_segmentation_paths(tmp_path)
    Path(segmentation_paths[foreign.trial_id]).touch()

    discovery = next(spec for spec in specs if spec.discovery_recording)
    discovery_mean = tmp_path / discovery.same_trial_mean_brain_name
    discovery_mean.touch()

    states = trial_readiness(
        rows,
        segmentation_paths=segmentation_paths,
        mean_brain_paths={discovery.trial_id: discovery_mean},
    )
    foreign_state = next(state for state in states if state.trial_id == foreign.trial_id)
    assert foreign_state.native_field_ready is True
    assert foreign_state.same_trial_mean_brain_present is False
    assert foreign_state.common_atlas_registration_ready is False
