import numpy as np

from dashi.io.gauthey_registration_staging import (
    BIFROST_FDA_TO_JRC2018_BOUNDARY,
    GAUTHEY_TRIAL_TO_FDA_BOUNDARY,
    _labels_to_xyz_volume,
    label_centroids,
)


def test_labels_planes_align_to_unique_spatial_axis():
    # spatial x,y,z = 2,3,4; deposited labels are z x (x*y)
    labels = np.arange(24).reshape(4, 6)
    volume = _labels_to_xyz_volume(labels, (2, 3, 4))
    assert volume.shape == (2, 3, 4)
    # Deposited plane 0 should occupy z=0 after reshaping.
    assert np.array_equal(volume[:, :, 0], labels[0].reshape(2, 3))


def test_label_centroids_use_affine_without_identity_promotion():
    labels = np.zeros((3, 3, 2), dtype=int)
    labels[0, 0, 0] = 1
    labels[2, 2, 0] = 1
    labels[1, 1, 1] = 2
    affine = np.array([
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 3.0, 0.0, 20.0],
        [0.0, 0.0, 4.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    centroids = {c.label_id: c for c in label_centroids(labels, affine)}
    assert centroids[1].voxel_count == 2
    assert centroids[1].voxel_x == 1.0
    assert centroids[1].voxel_y == 1.0
    assert centroids[1].world_x == 12.0
    assert centroids[1].world_y == 23.0
    assert centroids[1].coordinate_space == "gauthey_trial_mean_brain"
    assert "unregistered" in centroids[1].evidence_kind


def test_registration_chain_keeps_missing_trial_to_fda_payment_explicit():
    assert not GAUTHEY_TRIAL_TO_FDA_BOUNDARY.verified
    assert GAUTHEY_TRIAL_TO_FDA_BOUNDARY.transform_identifier is None
    assert BIFROST_FDA_TO_JRC2018_BOUNDARY.verified
    assert "thresholded_FDA_to_JRC2018_female.h5" in BIFROST_FDA_TO_JRC2018_BOUNDARY.transform_identifier
