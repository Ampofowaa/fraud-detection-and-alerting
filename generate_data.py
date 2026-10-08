import numpy as np
import pandas as pd

np.random.seed(42)   # reproducibility
N = 500_000

# -----------------------------
# 1. Transaction Amount (skewed)
# -----------------------------
# Most values < 500, long tail up to ~14,000
transaction_amount = np.random.exponential(scale=120, size=N)
transaction_amount = np.clip(transaction_amount, 0, None)

# Add rare extreme values
extreme_idx = np.random.choice(N, size=300, replace=False)
transaction_amount[extreme_idx] *= 40   # push some values into 5k-14k range

# Introduce missing values (match your counts)
missing_amt_idx = np.random.choice(N, size=N - 460_041, replace=False)
transaction_amount[missing_amt_idx] = np.nan

# --------------------------------
# 2. Location Distance (skewed)
# --------------------------------
# Most values < 50km, rare up to 1000km
location_distance = np.random.exponential(scale=10, size=N)
location_distance = np.clip(location_distance, 0, None)

# Add rare extreme values
extreme_loc_idx = np.random.choice(N, size=300, replace=False)
location_distance[extreme_loc_idx] *= 30   # push into 300-1200km range

# Missing values (match your counts)
missing_loc_idx = np.random.choice(N, size=N - 459_930, replace=False)
location_distance[missing_loc_idx] = np.nan

# --------------------------------
# 3. Account age (days)
# --------------------------------
account_age_days = np.random.randint(30, 4000, size=N)

# --------------------------------
# 4. Number of previous transactions
# --------------------------------
num_prev_transactions = np.random.poisson(lam=20, size=N)

# --------------------------------
# 5. Hour of day (0-23)
# --------------------------------
hour_of_day = np.random.randint(0, 24, size=N)

# --------------------------------
# 6. Weekend flag (0/1)
# --------------------------------
is_weekend = np.random.binomial(1, p=0.28, size=N)

# --------------------------------
# 7. Device type
# --------------------------------
device_type = np.random.choice(["mobile", "tablet", "desktop"], size=N, p=[0.80, 0.15, 0.05])

# --------------------------------
# 8. Fraud label (rare, and driven by the features)
# --------------------------------
# Fraud is generated from a logistic risk model so that it actually correlates
# with the observable features. Without this, is_fraud is just noise and no
# model (supervised or unsupervised) can do better than the base rate.
#
# Risk drivers (all realistic for card fraud):
#   - large transaction amounts
#   - large location distance (impossible-travel / card-not-present)
#   - transactions in the small hours (0-5)
#   - young accounts
#   - few prior transactions (little history to compare against)
#   - desktop slightly riskier than mobile/tablet
#   - interaction: a young account making a large payment

# Fill missing values with the median for the risk computation only
amt_filled = np.where(np.isnan(transaction_amount),
                      np.nanmedian(transaction_amount), transaction_amount)
loc_filled = np.where(np.isnan(location_distance),
                      np.nanmedian(location_distance), location_distance)

def _z(x):
    return (x - x.mean()) / x.std()

# log-transform the heavy-tailed features before standardising
amt_z = _z(np.log1p(amt_filled))
loc_z = _z(np.log1p(loc_filled))
age_z = _z(account_age_days.astype(float))
prev_z = _z(num_prev_transactions.astype(float))
night = ((hour_of_day >= 0) & (hour_of_day <= 5)).astype(float)

device_effect = np.select(
    [device_type == "desktop", device_type == "tablet", device_type == "mobile"],
    [0.35, 0.00, -0.15],
)

young_and_large = ((age_z < -0.5) & (amt_z > 1.0)).astype(float)

# Linear predictor (no intercept yet). Coefficients are moderate on purpose:
# strong enough to be learnable, weak enough that the label is not trivially
# separable (achievable ROC-AUC lands around 0.90, not ~1.0).
lin = (
    1.10 * amt_z
    + 0.90 * loc_z
    - 0.55 * age_z
    - 0.45 * prev_z
    + 0.80 * night
    + 0.20 * is_weekend
    + device_effect
    + 0.90 * young_and_large
)

# Solve for the intercept so the overall fraud rate is ~1.5%
TARGET_RATE = 0.015
lo, hi = -20.0, 5.0
for _ in range(60):
    mid = (lo + hi) / 2
    rate = (1.0 / (1.0 + np.exp(-(mid + lin)))).mean()
    if rate > TARGET_RATE:
        hi = mid
    else:
        lo = mid
intercept = (lo + hi) / 2

p_fraud = 1.0 / (1.0 + np.exp(-(intercept + lin)))
is_fraud = np.random.binomial(1, p_fraud)

print(f"fraud rate: {is_fraud.mean():.4f}  ({is_fraud.sum()} of {N})")

# --------------------------------
# Build DataFrame
# --------------------------------
df = pd.DataFrame({
    "transaction_amount": transaction_amount,
    "account_age_days": account_age_days,
    "num_prev_transactions": num_prev_transactions,
    "location_distance_km": location_distance,
    "hour_of_day": hour_of_day,
    "is_weekend": is_weekend,
    "device_type": device_type,
    "is_fraud": is_fraud
})

df.to_csv('anomaly.csv', index=False)
