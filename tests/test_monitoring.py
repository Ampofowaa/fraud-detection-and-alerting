import numpy as np
import pandas as pd

from fraud.monitoring import SIGNIFICANT, STABLE, psi


def test_identical_distributions_are_stable():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(size = 50_000))
    current = pd.Series(rng.normal(size = 50_000))
    assert psi(reference, current) < STABLE


def test_shifted_distribution_is_significant():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(size = 50_000))
    current = pd.Series(rng.normal(loc = 1.0, size = 50_000))
    assert psi(reference, current) > SIGNIFICANT


def test_psi_is_zero_for_the_same_sample():
    s = pd.Series(np.arange(1_000, dtype = float))
    assert psi(s, s) == 0.0


def test_missing_values_count_as_their_own_bin():
    reference = pd.Series(np.r_[np.arange(1_000, dtype = float), np.full(10, np.nan)])
    mostly_missing = pd.Series(np.r_[np.arange(500, dtype = float), np.full(500, np.nan)])
    assert psi(reference, mostly_missing) > SIGNIFICANT


def test_categorical_variables():
    reference = pd.Series(['mobile'] * 800 + ['desktop'] * 200, dtype = 'category')
    same_mix = pd.Series(['mobile'] * 80 + ['desktop'] * 20, dtype = 'category')
    flipped = pd.Series(['mobile'] * 200 + ['desktop'] * 800, dtype = 'category')
    assert psi(reference, same_mix) < STABLE
    assert psi(reference, flipped) > SIGNIFICANT


def test_constant_reference_does_not_crash():
    reference = pd.Series(np.ones(100))
    assert psi(reference, pd.Series(np.ones(100))) == 0.0
