"""Cost-based alert thresholds: missed fraud value vs analyst review cost."""

import numpy as np

DEFAULT_ALERT_RATES = np.arange(0.001, 0.30, 0.001)


def total_cost(threshold: float, y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, review_cost: float) -> float:
    """Missed fraud value plus the cost of reviewing every alert (score >= threshold)."""
    alert = scores >= threshold
    return float(amounts[(y == 1) & ~alert].sum() + review_cost * alert.sum())


def best_threshold(y, scores, amounts, review_cost: float, alert_rates = DEFAULT_ALERT_RATES, tolerance: float = 0.01):
    """Choose an alert threshold on labelled data.

    Searches the given alert rates (share of transactions alerted) and returns the threshold with the
    *lowest alert volume* whose total cost is within `tolerance` of the minimum. The cost curve is usually
    flat near its minimum, so this saves analyst time at almost no extra cost.

    Returns (threshold, alert_rates, costs).
    """
    y, scores, amounts = np.asarray(y), np.asarray(scores), np.asarray(amounts)
    thresholds = np.quantile(scores, 1 - np.asarray(alert_rates))
    costs = np.array([total_cost(t, y, scores, amounts, review_cost) for t in thresholds])
    i = int(np.argmax(costs <= costs.min() * (1 + tolerance)))
    return float(thresholds[i]), np.asarray(alert_rates), costs


def outcome(threshold: float, y, scores, amounts, review_cost: float) -> dict:
    """Business summary of alerting at `threshold`."""
    y, scores, amounts = np.asarray(y), np.asarray(scores), np.asarray(amounts)
    alert = scores >= threshold
    fraud = y == 1
    caught_value = float(amounts[alert & fraud].sum())
    total_fraud_value = float(amounts[fraud].sum())
    return {
        'alerts per 1,000 transactions': 1000 * alert.mean(),
        'precision (alerts that are fraud)': (alert & fraud).sum() / max(alert.sum(), 1),
        'fraud cases caught (%)': 100 * (alert & fraud).sum() / max(fraud.sum(), 1),
        'fraud value caught ($)': caught_value,
        'fraud value caught (%)': 100 * caught_value / total_fraud_value if total_fraud_value else 0.0,
        'total cost with model ($)': (total_fraud_value - caught_value) + review_cost * alert.sum(),
        'total cost without model ($)': total_fraud_value,
    }
