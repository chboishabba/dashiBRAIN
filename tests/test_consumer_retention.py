"""Finite toy receipts only: no NSD, Brain-IT checkpoints or fly BOLD data."""
import pytest

from dashi.analysis.consumer_retention import (
    audit_commutation,
    audit_consumer_retention,
    render_vertex_map_orbit_svg,
    write_vertex_map_orbit_svg,
    classify_topic_token,
    audit_spectral_stratification,
    FleetGoal,
    schedule_full_twenty,
    audit_max_cut,
)


def coarse_regional(state):
    sensory, association, planning = state
    return sensory + association, planning


def laplacian_variation(state):
    sensory, association, planning = state
    return abs(sensory - association) + abs(association - planning)


def test_neural_collision_matches_dashi_agda_witness():
    states = [(1, 2, 3), (2, 1, 3)]
    report = audit_consumer_retention(states, coarse_regional, laplacian_variation)
    assert report.fibre_count == 1
    assert report.ambiguous_fibre_count == 1
    assert not report.factorises_on_sample
    assert report.collision is not None
    assert (report.collision.left_target, report.collision.right_target) == (2, 3)


def test_observed_consumer_factors_on_same_quotient():
    states = [(1, 2, 3), (2, 1, 3)]
    report = audit_consumer_retention(states, coarse_regional, coarse_regional)
    assert report.factorises_on_sample
    assert report.collision is None


def test_finite_consistency_is_not_global_proof():
    # One sample hides the collision revealed by adding the second sample.
    one = audit_consumer_retention([(1, 2, 3)], coarse_regional, laplacian_variation)
    two = audit_consumer_retention([(1, 2, 3), (2, 1, 3)],
                                    coarse_regional, laplacian_variation)
    assert one.factorises_on_sample
    assert not two.factorises_on_sample


def test_commutation_good_and_bad_coarse_dynamics():
    states = [(1, 2, 3), (2, 1, 3)]
    fine = lambda state: (state[0] + 1, state[1], state[2])
    good = audit_commutation(states, fine, coarse_regional,
                             lambda b: (b[0] + 1, b[1]))
    bad = audit_commutation(states, fine, coarse_regional, lambda b: b)
    assert good.commutes_on_sample and good.mismatches == 0
    assert not bad.commutes_on_sample and bad.mismatches == 2


def test_empty_ensemble_rejected():
    with pytest.raises(ValueError):
        audit_consumer_retention([], coarse_regional, laplacian_variation)
    with pytest.raises(ValueError):
        audit_commutation([], lambda x: x, coarse_regional, lambda x: x)

def test_viewed_stimulus_does_not_decode_independent_concealed_target():
    # Cross-lane contact with dashi_agda DMT PR #1243. These four worlds
    # are invented, not EEG/fMRI participants or Brain-IT image examples.
    # An observation revealing the viewed stimulus is perfect for *that*
    # consumer but is not sufficient to reconstruct a concealed target.
    worlds = [(viewed, secret) for viewed in (0, 1) for secret in (0, 1)]
    viewed_observation = lambda world: world[0]
    hidden_target = lambda world: world[1]
    viewed = audit_consumer_retention(worlds, viewed_observation, viewed_observation)
    hidden = audit_consumer_retention(worlds, viewed_observation, hidden_target)
    assert viewed.factorises_on_sample
    assert hidden.state_count == 4
    assert hidden.fibre_count == 2
    assert hidden.ambiguous_fibre_count == 2
    assert not hidden.factorises_on_sample
    assert hidden.collision is not None
    assert hidden.collision.left_target != hidden.collision.right_target
    # An invalid experiment leaking the secret into the observed channel
    # would appear to achieve complete information recovery.
    leaked = audit_consumer_retention(worlds, lambda world: world, hidden_target)
    assert leaked.factorises_on_sample
    # A finite retained/nonretained consumer receipt does not supply
    # independent physical agent identity or an experimental data result.



def test_vertex_map_orbit_svg_colours_target_by_map_distance():
    source = [(0.0, 0.0), (1.0, 0.0), (0.0, 2.0)]
    target = [(0.0, 0.0), (2.0, 0.0), (0.0, 1.0)]
    svg = render_vertex_map_orbit_svg(
        source,
        target,
        {0: 0, 1: 1, 2: 2},
        show_source=False,
    )
    assert '<svg' in svg
    assert 'data-layer="target"' in svg
    assert 'data-map-distance="0"' in svg
    assert svg.count('data-map-distance="1"') == 2
    assert 'data-layer="source"' not in svg
    assert 'data-layer="correspondence"' not in svg


def test_vertex_map_orbit_svg_exposes_collision_and_subsamples_lines():
    source = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
    target = [(0.0, 0.0), (4.0, 0.0)]
    mapping = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
    svg = render_vertex_map_orbit_svg(
        source,
        target,
        mapping,
        show_source=True,
        max_correspondence_lines=2,
    )
    assert 'data-layer="source"' in svg
    assert svg.count('data-layer="correspondence"') == 2
    assert 'data-map-preimage-count="3"' in svg
    assert 'data-map-preimage-count="2"' in svg
    # Colliding target vertices use the maximum source-to-target map distance.
    assert 'data-map-distance="2"' in svg
    assert 'data-map-distance="1"' in svg


def test_vertex_map_orbit_svg_write_is_deterministic(tmp_path):
    source = [(0.0, 0.0), (1.0, 1.0)]
    target = [(0.0, 0.0), (2.0, 2.0)]
    mapping = {0: 0, 1: 1}
    expected = render_vertex_map_orbit_svg(source, target, mapping)
    output = tmp_path / "vertex-map-orbit.svg"
    written = write_vertex_map_orbit_svg(
        output,
        source,
        target,
        mapping,
    )
    assert written == output
    assert output.read_text(encoding="utf-8") == expected


def test_vertex_map_orbit_svg_rejects_bad_correspondence_indices():
    with pytest.raises(ValueError):
        render_vertex_map_orbit_svg([(0.0, 0.0)], [(0.0, 0.0)], {1: 0})
    with pytest.raises(ValueError):
        render_vertex_map_orbit_svg([(0.0, 0.0)], [(0.0, 0.0)], {0: 1})


def test_topic_filter_rejects_hex_memory_ui_and_dom_noise():
    assert classify_topic_token("0xffff1000") == "hex_literal_noise"
    assert classify_topic_token("0xffff1000()") == "memory_address_noise"
    assert classify_topic_token("collapse(property: auto)") == "raw_ui_token_noise"
    assert classify_topic_token("<div class='topic'>") == "raw_dom_token_noise"
    assert classify_topic_token("Yang-Mills reflection positivity") == "semantic_token"


def test_spectral_stratification_supports_only_k50_or_k100_and_filters_noise():
    receipt = audit_spectral_stratification(
        eigenvalues=[12.0, 7.0, 4.0, 1.0],
        lambda_plus=3.0,
        k=50,
        primary_labels=[
            "Yang-Mills",
            "0xffff1000",
            "Navier-Stokes",
            "collapse(property: auto)",
        ],
        subtopic_labels=["OS positivity", "<span>", "commutator"],
        fibre_count=9,
        unblocked_goal_count=20,
    )
    assert receipt.k == 50
    assert receipt.selected_eigen_indices == (0, 1, 2)
    assert receipt.primary_topics == ("Yang-Mills", "Navier-Stokes")
    assert receipt.subtopics == ("OS positivity", "commutator")
    assert receipt.levels == (
        "eigenspace",
        "primary_topic",
        "subtopic",
        "fibre",
        "goal",
    )
    with pytest.raises(ValueError):
        audit_spectral_stratification(
            eigenvalues=[4.0],
            lambda_plus=3.0,
            k=64,
            primary_labels=["topic"],
            subtopic_labels=["subtopic"],
            fibre_count=1,
            unblocked_goal_count=1,
        )


def test_full_twenty_schedule_uses_all_slots_in_pareto_order():
    goals = [
        FleetGoal(
            goal_id=f"g{i:02d}",
            pareto_rank=i,
            unblocked=True,
            pareto_eligible=True,
            typed_support=frozenset({f"surface-{i:02d}"}),
        )
        for i in range(25)
    ]
    schedule = schedule_full_twenty(goals)
    assert len(schedule) == 20
    assert tuple(goal.goal_id for goal in schedule) == tuple(
        f"g{i:02d}" for i in range(20)
    )
    assert len({surface for goal in schedule for surface in goal.typed_support}) == 20


def test_full_twenty_schedule_fails_when_independent_capacity_is_insufficient():
    goals = [
        FleetGoal(
            goal_id=f"g{i:02d}",
            pareto_rank=i,
            unblocked=True,
            pareto_eligible=True,
            typed_support=frozenset({"shared"} if i < 2 else {f"surface-{i:02d}"}),
        )
        for i in range(20)
    ]
    with pytest.raises(ValueError, match="20 pairwise noTypedMeet"):
        schedule_full_twenty(goals)


def test_full_twenty_schedule_ignores_blocked_and_ineligible_goals():
    goals = [
        FleetGoal("blocked", -2, False, True, frozenset({"blocked"})),
        FleetGoal("ineligible", -1, True, False, frozenset({"ineligible"})),
    ] + [
        FleetGoal(
            goal_id=f"g{i:02d}",
            pareto_rank=i,
            unblocked=True,
            pareto_eligible=True,
            typed_support=frozenset({f"surface-{i:02d}"}),
        )
        for i in range(20)
    ]
    schedule = schedule_full_twenty(goals)
    assert "blocked" not in {goal.goal_id for goal in schedule}
    assert "ineligible" not in {goal.goal_id for goal in schedule}


def test_finite_max_cut_matches_laplacian_quadratic_identity():
    audit = audit_max_cut(
        vertex_count=4,
        edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
        left_vertices={0, 2},
    )
    assert audit.cut_edge_count == 4
    assert audit.boundary_vertices == frozenset({0, 1, 2, 3})
    assert audit.laplacian_quadratic == 4 * audit.cut_edge_count
    assert audit.laplacian_quadratic == 16


def test_max_cut_optional_sdp_receipt_is_only_an_upper_bound_check():
    audit = audit_max_cut(
        vertex_count=3,
        edges=[(0, 1), (1, 2), (0, 2)],
        left_vertices={0},
        sdp_upper_bound=3.0,
    )
    assert audit.cut_edge_count == 2
    assert audit.sdp_upper_bound == 3.0
    assert audit.cut_within_sdp_bound
    with pytest.raises(ValueError, match="SDP upper bound"):
        audit_max_cut(
            vertex_count=3,
            edges=[(0, 1), (1, 2), (0, 2)],
            left_vertices={0},
            sdp_upper_bound=1.0,
        )
