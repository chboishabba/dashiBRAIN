from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import scipy.sparse as sp

from dashi.analysis.structure_function_real import aggregate_connectome_by_membership
from dashi.io.malecns_neuropil_membership import (
    canonicalize_malecns_neuropil,
    load_synapse_neuropil_membership,
)


def test_canonicalize_malecns_neuropil_collapses_laterality_only():
    assert canonicalize_malecns_neuropil("AL(L)") == "AL"
    assert canonicalize_malecns_neuropil("AMMC(R)") == "AMMC"
    assert canonicalize_malecns_neuropil("SAD") == "SAD"
    assert canonicalize_malecns_neuropil("CentralBrain-unspecified") == "CentralBrain-unspecified"


def test_synapse_membership_is_fractional_and_vocab_restricted(tmp_path):
    path = tmp_path / "partners.feather"
    feather.write_feather(
        pa.table(
            {
                "body_pre": pa.array([1, 1, 2, 3], type=pa.int64()),
                "body_post": pa.array([2, 3, 3, 1], type=pa.int64()),
                "primary_post": ["AL(L)", "AMMC(R)", "AL(R)", "GNG"],
            }
        ),
        path,
    )

    out = load_synapse_neuropil_membership(
        path,
        ["1", "2", "3"],
        allowed_regions={"AL", "AMMC"},
    )
    assert out.regions == ("AL", "AMMC")
    assert out.source_rows == 4
    assert out.observed_neuron_count == 3

    dense = out.membership.toarray()
    # Body 1 participates once in AL and once in AMMC after GNG is excluded.
    assert np.allclose(dense[:, 0], [0.5, 0.5])
    # Body 2 participates in two retained AL contacts only.
    assert np.allclose(dense[:, 1], [1.0, 0.0])
    # Body 3 participates once in AMMC and once in AL.
    assert np.allclose(dense[:, 2], [0.5, 0.5])
    assert np.allclose(np.asarray(out.membership.sum(axis=0)).ravel(), 1.0)


def test_fractional_membership_drives_region_structural_carrier():
    adjacency = sp.csr_matrix(
        np.array(
            [
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [1.0, 0.0, 0.0],
            ]
        )
    )
    membership = sp.csr_matrix(
        np.array(
            [
                [0.5, 1.0, 0.5],
                [0.5, 0.0, 0.5],
            ]
        )
    )
    structural = aggregate_connectome_by_membership(
        adjacency,
        membership,
        ("AL", "AMMC"),
    )
    assert structural.regions == ("AL", "AMMC")
    assert structural.direct.shape == (2, 2)
    assert structural.two_hop.shape == (2, 2)
    assert np.isfinite(structural.direct).all()
    assert np.isfinite(structural.two_hop).all()
