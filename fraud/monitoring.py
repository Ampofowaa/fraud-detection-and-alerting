"""Drift monitoring with the Population Stability Index (PSI)."""

import numpy as np
import pandas as pd

# Conventional PSI bands
STABLE, SIGNIFICANT = 0.1, 0.25


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index of `current` against `reference`.

    Numeric variables use `bins` quantile bins of the reference distribution; categorical variables use
    their categories. Missing values form their own bin in both cases. Shares are floored at 1e-4 so
    empty bins don't produce infinite values.
    """
    categorical = isinstance(reference.dtype, pd.CategoricalDtype)
    edges = None
    if not categorical:
        inner = np.unique(np.nanquantile(reference.to_numpy(dtype = float), np.linspace(0, 1, bins + 1)))[1:-1]
        edges = np.concatenate([[-np.inf], inner, [np.inf]])

    def shares(s: pd.Series) -> pd.Series:
        if categorical:
            counts = s.astype(object).fillna('missing').value_counts()
        else:
            assert edges is not None
            counts = pd.cut(s, edges.tolist(), include_lowest = True).value_counts(sort = False)
            counts.index = counts.index.astype(str)
            counts['missing'] = s.isna().sum()
        return counts / len(s)

    e, a = shares(reference).align(shares(current), fill_value = 0)
    e, a = e.clip(lower = 1e-4), a.clip(lower = 1e-4)
    return float(((a - e) * np.log(a / e)).sum())
