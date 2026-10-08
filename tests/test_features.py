import numpy as np
import pandas as pd
import pytest

from fraud.features import card_history_features

DAY = 86_400


def make_transactions(rows):
    """rows: list of (seconds, amount, card1, addr1); card2 and card6 are fixed."""
    df = pd.DataFrame(rows, columns = ['TransactionDT', 'TransactionAmt', 'card1', 'addr1'])
    df['card2'] = 100.0
    df['card6'] = 'debit'
    return df


def test_transaction_never_sees_itself():
    df = make_transactions([(0, 50.0, 1, 10)])
    f = card_history_features(df)
    assert f.loc[0, 'card_txn_count_30d'] == 0
    assert np.isnan(f.loc[0, 'card_amt_mean_30d'])
    assert np.isnan(f.loc[0, 'secs_since_card_prev'])


def test_uses_only_earlier_transactions_of_the_same_card():
    df = make_transactions([
        (0, 10.0, 1, 10),
        (100, 99.0, 2, 10),        # different card: must not count
        (200, 30.0, 1, 10),
        (300, 1000.0, 1, 10),
    ])
    f = card_history_features(df)
    assert f.loc[2, 'card_txn_count_30d'] == 1
    assert f.loc[2, 'card_amt_mean_30d'] == pytest.approx(10.0)
    assert f.loc[3, 'card_txn_count_30d'] == 2
    assert f.loc[3, 'card_amt_mean_30d'] == pytest.approx(20.0)
    assert f.loc[3, 'amt_vs_card_mean_30d'] == pytest.approx(50.0)
    assert f.loc[3, 'secs_since_card_prev'] == 100


def test_future_transactions_do_not_change_past_features():
    """The core leakage guarantee: appending later rows leaves every earlier row's features unchanged."""
    rng = np.random.default_rng(0)
    n = 400
    rows = sorted((int(t), float(a), int(c), 10) for t, a, c in
                  zip(rng.integers(0, 90 * DAY, n), rng.exponential(100, n), rng.integers(1, 15, n)))
    df = make_transactions(rows)
    cols = ['card_txn_count_30d', 'card_amt_mean_30d', 'amt_vs_card_mean_30d', 'secs_since_card_prev']
    full = card_history_features(df)[cols]
    past_only = card_history_features(df.iloc[:200])[cols]
    pd.testing.assert_frame_equal(full.iloc[:200], past_only)


def test_window_excludes_older_transactions():
    df = make_transactions([
        (0, 500.0, 1, 10),          # 31 days before the last row: outside the window
        (20 * DAY, 10.0, 1, 10),
        (31 * DAY, 20.0, 1, 10),
    ])
    f = card_history_features(df)
    assert f.loc[2, 'card_txn_count_30d'] == 1
    assert f.loc[2, 'card_amt_mean_30d'] == pytest.approx(10.0)


def test_rows_without_address_get_no_card_features():
    df = make_transactions([(0, 10.0, 1, np.nan), (100, 20.0, 1, np.nan)])
    f = card_history_features(df)
    assert f['card_key'].isna().all()
    assert f['card_txn_count_30d'].isna().all()


def test_matches_brute_force_on_random_data():
    rng = np.random.default_rng(1)
    n = 500
    rows = sorted((int(t), float(a), int(c), int(addr)) for t, a, c, addr in
                  zip(rng.integers(0, 120 * DAY, n), rng.exponential(100, n), rng.integers(1, 30, n), rng.integers(1, 4, n)))
    df = make_transactions(rows)
    f = card_history_features(df)
    for i in rng.choice(n, 60, replace = False):
        row = df.iloc[i]
        earlier = df[(df['card1'] == row['card1']) & (df['addr1'] == row['addr1'])
                     & (df['TransactionDT'] < row['TransactionDT']) & (df['TransactionDT'] >= row['TransactionDT'] - 30 * DAY)]
        assert f.iloc[i]['card_txn_count_30d'] == len(earlier)
        if len(earlier):
            assert f.iloc[i]['card_amt_mean_30d'] == pytest.approx(earlier['TransactionAmt'].mean(), rel = 1e-4)


def test_rejects_unsorted_input():
    df = make_transactions([(100, 10.0, 1, 10), (0, 20.0, 1, 10)])
    with pytest.raises(ValueError):
        card_history_features(df)
