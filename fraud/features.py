"""Leakage-safe per-card history features."""

import numpy as np
import pandas as pd

CARD_COLUMNS = ['card1', 'card2', 'addr1', 'card6']


def card_key(df: pd.DataFrame) -> pd.Series:
    """Approximate card identifier from card1, card2, addr1 and card6.

    Missing (NaN) when card1 or addr1 is missing, so those rows are not lumped together as one "card".
    """
    has_key = df['card1'].notna() & df['addr1'].notna()
    key = (df['card1'].astype('Int64').astype(str) + '|' + df['card2'].astype('Int64').astype(str) + '|'
           + df['addr1'].astype('Int64').astype(str) + '|' + df['card6'].astype(str))
    return pd.Series(pd.factorize(key)[0], index = df.index, dtype = 'float64').where(has_key)


def card_history_features(df: pd.DataFrame, window: str = '30D') -> pd.DataFrame:
    """Per-card features from strictly earlier transactions within `window`.

    Expects rows in time order, with columns TransactionDT (seconds), TransactionAmt and CARD_COLUMNS.
    A transaction never contributes to its own features, and later transactions never contribute to
    earlier ones, so the features are safe to use for time-based evaluation.
    """
    if not df['TransactionDT'].is_monotonic_increasing:
        raise ValueError('rows must be sorted by TransactionDT')

    key = card_key(df)
    has_key = key.notna()
    out = pd.DataFrame(index = df.index)
    out['card_key'] = key

    keyed = pd.DataFrame({'card_key': key[has_key], 'TransactionAmt': df.loc[has_key, 'TransactionAmt']})
    keyed.index = pd.to_datetime(df.loc[has_key, 'TransactionDT'], unit = 's')
    rolling = keyed.groupby('card_key')['TransactionAmt'].rolling(window, closed = 'left')   # 'left' excludes the current row
    count = rolling.count().fillna(0)   # an empty window means 0 earlier transactions
    mean = rolling.mean()

    # groupby-rolling returns rows grouped by card (stable within each card); map them back to the original order
    order = df.index[has_key][np.argsort(key[has_key].to_numpy(), kind = 'stable')]
    out['card_txn_count_30d'] = pd.Series(count.to_numpy(), index = order).reindex(df.index).astype('float32')
    out['card_amt_mean_30d'] = pd.Series(mean.to_numpy(), index = order).reindex(df.index).astype('float32')
    out['amt_vs_card_mean_30d'] = (df['TransactionAmt'] / out['card_amt_mean_30d']).astype('float32')
    out['secs_since_card_prev'] = (df['TransactionDT'] - df['TransactionDT'].groupby(key).shift()).astype('float32')
    return out
