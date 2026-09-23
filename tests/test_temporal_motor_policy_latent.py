import numpy as np

from dashi.analysis.temporal_motor_policy_latent import (
    build_temporal_motor_samples,
    evaluate_temporal_motor_latent_ladder,
)


def test_temporal_samples_use_past_neural_history_and_future_behaviour():
    neural = np.arange(24, dtype=float).reshape(8, 3)
    behaviour = np.arange(16, dtype=float).reshape(8, 2)
    samples = build_temporal_motor_samples(
        neural,
        behaviour,
        history_bins=2,
        future_lag=2,
    )
    assert samples.neural_history.shape == (5, 6)
    assert samples.behaviour_history.shape == (5, 4)
    assert np.array_equal(samples.source_time_index, np.array([1, 2, 3, 4, 5]))
    assert np.array_equal(samples.target_time_index, np.array([3, 4, 5, 6, 7]))
    assert np.array_equal(samples.future_behaviour[0], behaviour[3])
    assert np.array_equal(samples.current_behaviour[0], behaviour[1])
    assert np.array_equal(samples.behaviour_history[0], behaviour[0:2].reshape(-1))


def test_structure_of_heldout_behaviour_cannot_change_training_fitted_encoder():
    rng = np.random.default_rng(7)
    neural = rng.normal(size=(120, 5))
    behaviour = rng.normal(size=(120, 2))
    original = evaluate_temporal_motor_latent_ladder(
        neural,
        behaviour,
        dimensions=(1, 2),
        history_bins=3,
        future_lag=2,
        holdout_fraction=0.25,
    )

    changed = behaviour.copy()
    changed[95:] += 10_000.0
    perturbed = evaluate_temporal_motor_latent_ladder(
        neural,
        changed,
        dimensions=(1, 2),
        history_bins=3,
        future_lag=2,
        holdout_fraction=0.25,
    )

    assert np.array_equal(original.components, perturbed.components)
    assert np.array_equal(original.feature_means, perturbed.feature_means)
    assert np.array_equal(original.feature_scales, perturbed.feature_scales)


def test_predictive_latent_can_capture_a_neural_lead_signal_beyond_behaviour_history():
    rng = np.random.default_rng(9)
    n = 240
    latent = rng.normal(size=n)
    neural = np.column_stack(
        (
            latent + rng.normal(scale=0.03, size=n),
            latent + rng.normal(scale=0.03, size=n),
        )
    )
    behaviour = np.zeros((n, 1), dtype=float)
    behaviour[2:, 0] = 3.0 * latent[:-2]

    result = evaluate_temporal_motor_latent_ladder(
        neural,
        behaviour,
        dimensions=(1, 2),
        history_bins=1,
        future_lag=2,
        holdout_fraction=0.25,
    )
    d1 = result.scores[0]
    assert d1.future_mae < d1.persistence_mae
    assert d1.future_mae < d1.behaviour_history_mae
    assert d1.neural_improves_over_behaviour_history is True
    assert d1.motor_policy_identity_certified is False
    assert d1.causal_plan_certified is False


def test_literal_memory_or_motor_plan_identity_is_never_promoted_by_prediction():
    rng = np.random.default_rng(11)
    neural = rng.normal(size=(100, 4))
    behaviour = rng.normal(size=(100, 2))
    result = evaluate_temporal_motor_latent_ladder(
        neural,
        behaviour,
        dimensions=(1,),
        history_bins=2,
        future_lag=1,
        holdout_fraction=0.2,
    )
    assert result.motor_policy_identity_certified is False
    assert result.memory_content_certified is False
    assert result.physical_latent_dimension_identified is False
