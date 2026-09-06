from dashi.io.gauthey_payment_frontier import (
    PaymentRoute,
    canonical_public_atlas_registration_frontier,
    classify_deposit_source_resolution,
)


def test_zero_match_source_files_require_external_payment():
    payload = {
        "archive_member_count": 35,
        "required_source_files": ["a.pkl", "a_labels.h5", "b.pkl", "b_labels.h5"],
        "resolved": {
            "a.pkl": [],
            "a_labels.h5": [],
            "b.pkl": [],
            "b_labels.h5": [],
        },
    }
    frontier = classify_deposit_source_resolution(payload)
    assert frontier.archive_member_count == 35
    assert frontier.required_file_count == 4
    assert frontier.uniquely_resolved_file_count == 0
    assert frontier.zero_match_file_count == 4
    assert frontier.ambiguous_file_count == 0
    assert not frontier.all_required_uniquely_resolved
    assert not frontier.exact_trace_recovery_available
    assert frontier.payment_a_route is PaymentRoute.EXTERNAL_SCIENTIFIC_RECEIPT
    assert frontier.required_output_rows == 940


def test_all_unique_source_files_leave_conditional_recovery_route_open():
    payload = {
        "archive_member_count": 12,
        "required_source_files": ["a.pkl", "a_labels.h5"],
        "resolved": {
            "a.pkl": [{"archive_member": "Data/a.pkl"}],
            "a_labels.h5": [{"archive_member": "Data/a_labels.h5"}],
        },
    }
    frontier = classify_deposit_source_resolution(payload)
    assert frontier.all_required_uniquely_resolved
    assert frontier.exact_trace_recovery_available
    assert frontier.payment_a_route is PaymentRoute.DEPOSIT_RECOVERY


def test_ambiguous_source_file_does_not_count_as_resolved():
    payload = {
        "required_source_files": ["a.pkl"],
        "resolved": {"a.pkl": [{"archive_member": "x/a.pkl"}, {"archive_member": "y/a.pkl"}]},
    }
    frontier = classify_deposit_source_resolution(payload)
    assert frontier.ambiguous_file_count == 1
    assert frontier.uniquely_resolved_file_count == 0
    assert frontier.payment_a_route is PaymentRoute.EXTERNAL_SCIENTIFIC_RECEIPT


def test_public_atlas_launcher_is_not_an_executable_transform_receipt():
    frontier = canonical_public_atlas_registration_frontier()
    assert frontier.launcher_present
    assert frontier.ants_dependency_declared
    assert not frontier.implementation_present
    assert frontier.payment_b_route is PaymentRoute.SOURCE_IMPLEMENTATION_REQUIRED
