from __future__ import annotations

import numpy as np

from dashi.analysis.fly_streaming_optimization_certificates import (
    PartnerAccumulator,
    PartnerContribution,
    jrc_domain_local_count_law,
    jrc_stream_residency_bound,
    malecns_membership_from_counts,
    malecns_partner_local_certificate,
    malecns_stream_residency_bound,
)
from dashi.analysis.local_optimization_compiler import (
    ExecutionResourceWitness,
    compile_fold_observational_equivalence,
    source_count_does_not_scale_live_stream_item,
)


def test_jrc_local_bincount_matches_literal_count():
    labels = np.array(
        [
            [[0, 1, 1], [2, 0, 3]],
            [[3, 3, 1], [0, 2, 2]],
        ],
        dtype=np.int64,
    )
    domain = np.array(
        [
            [[0, 1, 0], [1, 1, 1]],
            [[1, 0, 1], [0, 1, 0]],
        ],
        dtype=bool,
    )
    assert jrc_domain_local_count_law(labels, domain, max_label=3)


def test_stream_residency_bound_does_not_multiply_by_source_count():
    bound = jrc_stream_residency_bound(
        selected_label_bytes=120,
        one_domain_bytes=100,
        score_state_bytes=20,
        scratch_bytes=10,
    )
    assert bound.peak_bound_bytes == 250
    assert source_count_does_not_scale_live_stream_item(46, bound)
    # 46 source domains still means 46 units of work; it does not create a
    # 46*one_domain_bytes residency term in this streaming bound.
    assert bound.peak_bound_bytes != 120 + 20 + 46 * 100 + 10


def test_partner_row_local_law_compiles_to_global_observation():
    initial_a = PartnerAccumulator(
        counts=np.zeros((2, 3), dtype=np.int64),
        total_incidence=np.zeros(3, dtype=np.int64),
    )
    initial_b = initial_a.copy()
    items = [
        PartnerContribution(0, 1, 0),
        PartnerContribution(1, 2, 1),
        PartnerContribution(0, 2, None),  # outside retained vocabulary
    ]
    result = compile_fold_observational_equivalence(
        initial_a,
        initial_b,
        items,
        malecns_partner_local_certificate(),
    )
    assert result.equal
    assert result.item_count == 3

    counts_flat, total = result.optimized_observation
    counts = np.asarray(counts_flat, dtype=np.int64).reshape(2, 3)
    total_arr = np.asarray(total, dtype=np.int64)
    membership = malecns_membership_from_counts(counts, total_arr)

    # Body 0 has one retained regional incidence and one outside-vocabulary
    # incidence, so retained mass is 1/2 rather than renormalized to 1.
    assert np.allclose(membership[:, 0], [0.5, 0.0])
    assert np.allclose(membership[:, 1], [0.5, 0.5])
    assert np.allclose(membership[:, 2], [0.0, 0.5])


def test_malecns_residency_formula_is_batch_count_independent():
    bound = malecns_stream_residency_bound(
        region_count=13,
        neuron_count=164_740,
        one_batch_bytes=8_000_000,
        scratch_bytes=4_000_000,
    )
    expected_persistent = 13 * 164_740 * 8 + 164_740 * 8
    assert bound.persistent_state_bytes == expected_persistent
    assert source_count_does_not_scale_live_stream_item(10_000, bound)


def test_empirical_resource_witness_is_separate_from_semantic_certificate():
    bound = jrc_stream_residency_bound(
        selected_label_bytes=100,
        one_domain_bytes=100,
        score_state_bytes=50,
        fixed_runtime_overhead_bytes=50,
    )
    completed = ExecutionResourceWitness(completed=True, peak_rss_bytes=250)
    exhausted = ExecutionResourceWitness(completed=False, resource_exhausted=True)
    assert completed.fits(bound) is True
    assert exhausted.fits(bound) is None
