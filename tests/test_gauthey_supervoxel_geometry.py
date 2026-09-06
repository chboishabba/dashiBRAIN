import numpy as np

from dashi.io.gauthey_supervoxel_geometry import plane_local_centroids


def test_same_cluster_integer_on_different_planes_remains_distinct():
    # Two planes, each with local cluster labels 0 and 1.
    labels = np.array([
        [0, 0, 1, 1],
        [0, 1, 0, 1],
    ])
    spatial_shape = (2, 2, 2)
    centroids = plane_local_centroids(labels, spatial_shape, np.eye(4))
    keys = {(c.plane_index, c.cluster_index) for c in centroids}
    assert keys == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert len(centroids) == 4
