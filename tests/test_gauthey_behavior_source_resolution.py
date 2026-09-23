from dashi.io.gauthey_behavior_source_resolution import (
    classify_behavior_archive_members,
)
from dashi.io.remote_zip import RemoteZipMember


def _member(name: str) -> RemoteZipMember:
    return RemoteZipMember(
        name=name,
        compressed_size=10,
        uncompressed_size=20,
        compression_method=8,
        local_header_offset=0,
        crc32=123,
    )


def test_strong_behavior_terms_surface_candidates_without_claiming_trial_binding():
    result = classify_behavior_archive_members(
        [
            _member("Data/Behavior/FicTrac/04192024_a1_r9_fictrac.dat"),
            _member("Data/Dffs/Aligned/GCaMP6f_04192024_a1_r9.zip"),
        ]
    )
    assert [c.archive_member for c in result.strong_candidates] == [
        "Data/Behavior/FicTrac/04192024_a1_r9_fictrac.dat"
    ]
    assert result.strong_candidates[0].matched_terms == ("behavior", "fictrac")
    assert result.exact_trial_binding_paid is False
    assert result.synchronized_timebase_binding_paid is False


def test_kinematic_or_locomotion_names_are_strong_candidates():
    result = classify_behavior_archive_members(
        [
            _member("Data/Kinematics/ball_kinematics.csv"),
            _member("Data/locomotion_trial.pkl"),
        ]
    )
    assert len(result.strong_candidates) == 2
    assert result.weak_candidates == ()


def test_ball_or_treadmill_alone_remains_weak_not_exact_behavior_identity():
    result = classify_behavior_archive_members(
        [
            _member("Data/Video/tracking_ball_camera.zip"),
            _member("Data/treadmill_movie.avi"),
        ]
    )
    assert result.strong_candidates == ()
    assert len(result.weak_candidates) == 2
    assert result.exact_behavior_artifact_identity_paid is False


def test_unrelated_members_do_not_become_behavior_candidates():
    result = classify_behavior_archive_members(
        [
            _member("Data/Dffs/Audio correlated/dffs_audio_LB_corr_top05_all.pkl"),
            _member("Data/Segmentation/04032024_6f_a2_r5_seg.npy"),
        ]
    )
    assert result.strong_candidates == ()
    assert result.weak_candidates == ()
    assert result.exact_behavior_artifact_identity_paid is False
