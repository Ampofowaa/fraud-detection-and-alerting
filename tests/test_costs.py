import numpy as np
import pytest

from fraud.costs import best_threshold, outcome, total_cost


@pytest.fixture
def scored():
    """Synthetic scored transactions where fraud tends to score higher."""
    rng = np.random.default_rng(0)
    n = 20_000
    y = rng.binomial(1, 0.04, n)
    scores = np.clip(rng.normal(0.2 + 0.5 * y, 0.15), 0, 1)
    amounts = rng.exponential(120, n)
    return y, scores, amounts


def test_total_cost_arithmetic():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    amounts = np.array([100.0, 50.0, 30.0, 20.0])
    # threshold 0.5: alerts on rows 0 and 2; misses the $50 fraud; 2 reviews at $5
    assert total_cost(0.5, y, scores, amounts, review_cost = 5) == pytest.approx(50 + 2 * 5)


def test_outcome_matches_hand_calculation():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    amounts = np.array([100.0, 50.0, 30.0, 20.0])
    o = outcome(0.5, y, scores, amounts, review_cost = 5)
    assert o['alerts per 1,000 transactions'] == pytest.approx(500)
    assert o['precision (alerts that are fraud)'] == pytest.approx(0.5)
    assert o['fraud value caught ($)'] == pytest.approx(100)
    assert o['fraud value caught (%)'] == pytest.approx(100 * 100 / 150)
    assert o['total cost with model ($)'] == pytest.approx(60)
    assert o['total cost without model ($)'] == pytest.approx(150)


def test_higher_review_cost_never_means_more_alerts(scored):
    y, scores, amounts = scored
    alert_volumes = []
    for cost in [1, 5, 10, 25, 50]:
        t, _, _ = best_threshold(y, scores, amounts, cost)
        alert_volumes.append((scores >= t).mean())
    assert alert_volumes == sorted(alert_volumes, reverse = True)


def test_chosen_threshold_is_lowest_volume_within_tolerance(scored):
    y, scores, amounts = scored
    t, rates, costs = best_threshold(y, scores, amounts, review_cost = 5, tolerance = 0.01)
    i = int(np.argmin(np.abs(np.quantile(scores, 1 - rates) - t)))
    assert costs[i] <= costs.min() * 1.01
    assert (costs[:i] > costs.min() * 1.01).all()


def test_model_beats_no_model_at_the_chosen_threshold(scored):
    y, scores, amounts = scored
    t, _, _ = best_threshold(y, scores, amounts, review_cost = 5)
    o = outcome(t, y, scores, amounts, review_cost = 5)
    assert o['total cost with model ($)'] < o['total cost without model ($)']
