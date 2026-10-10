"""Model definition, evaluation split and metrics. Pure functions, no MLflow and no I/O.

Evaluation design (see DECISIONS.md, requirement 4):
* Pipes are grouped into square spatial blocks. A deterministic hash of (utility, block) puts
  about holdout_pct% of blocks in a holdout set that NO model version ever trains on. The
  champion and a challenger can therefore be compared on the same unseen pipes, even though
  they were trained at different times.
* Primary metric: average precision (ranking quality, at 16% prevalence).
  Business metric: precision in the top 10% riskiest pipes (an inspection budget).
  Calibration guard: Brier score, because customers read the scores as probabilities.
* Baseline: rank pipes by age. A model that cannot beat that rule should not ship.

Scoring design (cross-fitting): the registered model is a CrossFitModel. Besides the full model it holds
one model per spatial CV fold over the training blocks, and which fold held out each block. A pipe in a
training block is scored by the fold model that never saw that block; any other pipe by the full model,
which only trained on training blocks. No pipe is ever scored by a model that trained on it.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pipesense.features import LABEL


def build_model(params: dict) -> Pipeline:
    numeric = list(params["numeric_features"])
    categorical = list(params["categorical_features"])
    preprocess = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )
    clf = GradientBoostingClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params.get("learning_rate", 0.1)),
        random_state=int(params["random_state"]),
    )
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def model_input(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    cols = list(params["numeric_features"]) + list(params["categorical_features"])
    x = df[cols].copy()
    for c in params["numeric_features"]:
        x[c] = x[c].astype("float64")
    for c in params["categorical_features"]:
        x[c] = x[c].astype(str)
    return x


class CrossFitModel:
    """Full model + one model per spatial CV fold, and the fold that held out each training block.

    predict / predict_proba use the full model: that is what the gate evaluates on the holdout.
    score_pipes routes each pipe to a model that never trained on it.
    """

    def __init__(self, full, fold_models: list, block_fold: dict[str, int], block_size_m: float):
        self.full = full
        self.fold_models = list(fold_models)
        self.block_fold = dict(block_fold)
        self.block_size_m = float(block_size_m)

    @property
    def classes_(self):
        return self.full.classes_

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.full.predict(x)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.full.predict_proba(x)

    def score_pipes(self, x: pd.DataFrame, blocks: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Leak scores plus their source: "oof" (fold model that held out the pipe's block) or "full"."""
        fold = blocks.map(self.block_fold).to_numpy()  # NaN: holdout or a block unseen in training
        scores = np.zeros(len(x))
        source = np.full(len(x), "full", dtype=object)
        rows = pd.isna(fold)
        if rows.any():
            scores[rows] = self.full.predict_proba(x[rows])[:, 1]
        for k, fold_model in enumerate(self.fold_models):
            rows = fold == k
            if rows.any():
                scores[rows] = fold_model.predict_proba(x[rows])[:, 1]
                source[rows] = "oof"
        return scores, source


def fit_cross_fitted(df: pd.DataFrame, params: dict, build=build_model) -> tuple[CrossFitModel, np.ndarray]:
    """Fit one model per spatial GroupKFold fold over df's blocks, plus the full model on all of df.

    Returns the model and the out-of-fold scores of df's rows: the spatial CV predictions, which are
    exactly what the fold models will serve for these pipes.
    """
    block_size_m = float(params["block_size_m"])
    groups = spatial_blocks(df, block_size_m)
    folds = min(int(params.get("cv_folds", 5)), groups.nunique())
    x, y = model_input(df, params), labels(df)
    oof = np.zeros(len(df))
    fold_models, block_fold = [], {}
    for k, (tr, te) in enumerate(GroupKFold(folds).split(x, y, groups)):
        fold_model = build(params).fit(x.iloc[tr], y[tr])
        oof[te] = fold_model.predict_proba(x.iloc[te])[:, 1]
        fold_models.append(fold_model)
        block_fold.update({block: k for block in groups.iloc[te].unique()})
    return CrossFitModel(build(params).fit(x, y), fold_models, block_fold, block_size_m), oof


def spatial_blocks(df: pd.DataFrame, block_size_m: float) -> pd.Series:
    bx = np.floor(df["x_m"] / block_size_m).astype(int).astype(str)
    by = np.floor(df["y_m"] / block_size_m).astype(int).astype(str)
    return bx + "_" + by


def holdout_mask(df: pd.DataFrame, utility: str, block_size_m: float, holdout_pct: int) -> pd.Series:
    """True for pipes in held-out blocks. Stable across runs, model versions and row order."""
    blocks = spatial_blocks(df, block_size_m)

    def bucket(block: str) -> int:
        return int(hashlib.sha256(f"{utility}:{block_size_m}:{block}".encode()).hexdigest(), 16) % 100

    return blocks.map(lambda b: bucket(b) < holdout_pct).astype(bool)


def holdout_spec(params: dict) -> str:
    """Identifies the holdout definition; versions are only comparable if it matches."""
    return f"spatial-block:{params['block_size_m']}m:{params['holdout_pct']}pct:v1"


def precision_at_frac(y: np.ndarray, scores: np.ndarray, frac: float) -> float:
    k = max(1, int(round(len(y) * frac)))
    top = np.argsort(-scores, kind="stable")[:k]
    return float(np.mean(y[top]))


def baseline_scores(df: pd.DataFrame) -> np.ndarray:
    """No-model rule a utility engineer could apply: older pipes first (unknown age = median)."""
    age = df["age_years"].astype(float)
    return age.fillna(age.median()).to_numpy()


def evaluate(y: np.ndarray, scores: np.ndarray, top_k_frac: float) -> dict:
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        f"precision_at_{int(top_k_frac * 100)}pct": precision_at_frac(y, scores, top_k_frac),
    }
    if scores.min() >= 0 and scores.max() <= 1:
        out["brier"] = float(brier_score_loss(y, scores))
    return out


def paired_bootstrap_ap_diff(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, n_samples: int, seed: int = 0
) -> dict:
    """Bootstrap AP(a) - AP(b) over the same resampled rows. Reported for audit, not gated on."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    diffs = []
    for _ in range(n_samples):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        diffs.append(average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx]))
    d = np.array(diffs)
    return {
        "mean": float(d.mean()),
        "ci90_low": float(np.quantile(d, 0.05)),
        "ci90_high": float(np.quantile(d, 0.95)),
        "p_better": float((d > 0).mean()),
    }


def labels(df: pd.DataFrame) -> np.ndarray:
    return df[LABEL].astype(int).to_numpy()
