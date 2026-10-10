"""Feature drift between a model's training data and the inventory it scores. Pure functions, no I/O.

At train time `reference_profile` summarises every model input on the training rows, exactly as the
model sees them (after `model.model_input`):
* numeric features: decile bin edges, the share of rows in each bin, and the share of missing values
* categorical features: the share of each category

The profile is stored with the model, so a scoring job compares against the data of the model it
actually used. At predict time `psi_report` bins the scored rows the same way and computes the
population stability index per feature:

    PSI = sum over bins of (actual - expected) * ln(actual / expected)

Rule of thumb: below 0.1 stable, 0.1 to 0.2 a moderate shift, above 0.2 a shift worth a look.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OTHER = "__other__"  # a category the model never saw in training
EPS = 1e-4           # floor for empty bins, so ln() stays finite


def _numeric_fractions(values: pd.Series, edges: list[float]) -> list[float]:
    """Shares per bin (-inf, e1], (e1, e2], ..., (ek, inf), then the missing share last."""
    v = values.astype("float64")
    n = max(len(v), 1)
    missing = v.isna()
    idx = np.searchsorted(np.asarray(edges, dtype="float64"), v[~missing].to_numpy(), side="left")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return [*(counts / n).tolist(), float(missing.sum() / n)]


def _categorical_fractions(values: pd.Series, categories: list[str]) -> dict[str, float]:
    shares = values.astype(str).value_counts(normalize=True)
    out = {c: float(shares.get(c, 0.0)) for c in categories}
    out[OTHER] = float(shares[~shares.index.isin(categories)].sum())
    return out


def reference_profile(x: pd.DataFrame, numeric: list[str], categorical: list[str], bins: int = 10) -> dict:
    features = {}
    for col in numeric:
        present = x[col].astype("float64").dropna()
        quantiles = np.linspace(0, 1, bins + 1)[1:-1]
        edges = np.unique(np.quantile(present, quantiles)).tolist() if len(present) else []
        features[col] = {"type": "numeric", "edges": edges, "fractions": _numeric_fractions(x[col], edges)}
    for col in categorical:
        shares = x[col].astype(str).value_counts(normalize=True)
        features[col] = {"type": "categorical", "fractions": {str(k): float(v) for k, v in shares.items()}}
    return {"rows": int(len(x)), "bins": bins, "features": features}


def psi(expected, actual) -> float:
    e = np.clip(np.asarray(expected, dtype="float64"), EPS, None)
    a = np.clip(np.asarray(actual, dtype="float64"), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_report(profile: dict, x: pd.DataFrame, threshold: float) -> dict:
    """PSI per feature of `x` against `profile`, and the features above `threshold`."""
    values = {}
    for col, ref in profile["features"].items():
        if col not in x.columns:
            continue
        if ref["type"] == "numeric":
            expected = ref["fractions"]
            actual = _numeric_fractions(x[col], ref["edges"])
        else:
            categories = list(ref["fractions"])
            actual_by_cat = _categorical_fractions(x[col], categories)
            expected = [*(ref["fractions"][c] for c in categories), 0.0]
            actual = [*(actual_by_cat[c] for c in categories), actual_by_cat[OTHER]]
        values[col] = round(psi(expected, actual), 4)
    drifted = sorted((c for c, v in values.items() if v > threshold), key=lambda c: -values[c])
    return {
        "status": "drift" if drifted else "ok",
        "threshold": threshold,
        "reference_rows": profile["rows"],
        "rows": int(len(x)),
        "psi": values,
        "drifted": drifted,
    }
