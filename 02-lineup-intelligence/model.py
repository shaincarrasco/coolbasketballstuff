"""
Lineup net-rating model.

* Target  : lineup net rating (OffRtg - DefRtg)
* Weights : minutes played (a 400-minute lineup is far more informative than a 15-minute one)
* Models  : Ridge baseline (standardised features) vs XGBoost vs HistGradientBoosting
* CV      : GroupKFold by team (held-out TEAMS, not rows) so the model is scored on
            rosters it has never seen -- the setting the optimizer runs in.
* Metrics : minutes-weighted RMSE, minutes-weighted R^2, Spearman rank correlation
            (rank quality is what matters for picking a best five).
* Lab     : learning curve (fraction of training rows vs held-out Spearman) for the
            three models on a 4-fold team split.

The production model (used by the optimizer and the report builder) is XGBoost;
it is the one that ships `pred_contribs` attributions. A second XGBoost model
predicts OFFENSIVE rating so the report can split a lineup's predicted net into
offence / defence (defence is derived as off - net so the three always agree).

Fit score
---------
Predicted net rating is mapped to a 0-100 "Fit score" with a logistic transform:

    fit = 100 * sigmoid((net_pred - FIT_CENTER) / FIT_SCALE)

FIT_CENTER and FIT_SCALE are calibrated at training time to the minutes-weighted
mean and standard deviation of historical lineup net rating, so 50 = "an average
lineup in this league", 73 = one standard deviation better, 88 = two, 95 = three.
It is strictly monotone, bounded, and keeps elite lineups from pinning at 100.
The calibrated values are stored in models/meta.json.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from features import build_feature_matrix, feature_names

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")

# Defaults; overwritten by calibrate_fit() during train() and reloaded by load().
FIT_CENTER = 0.0
FIT_SCALE = 10.0

XGB_PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=2,               # shallow trees: pairwise interactions, little capacity to memorise 500 rows
    min_child_weight=3.0,      # in units of normalised sample weight (mean weight = 1 lineup)
    subsample=0.7,
    colsample_bytree=0.5,
    reg_lambda=2.0,
    reg_alpha=0.5,
    gamma=0.0,
    objective="reg:squarederror",
    random_state=7,
    n_jobs=1,                  # single thread: on a 2-core box OpenMP oversubscription made fits 20x slower
)
HGB_PARAMS = dict(
    max_iter=300,
    learning_rate=0.04,
    max_depth=3,
    max_leaf_nodes=6,
    min_samples_leaf=10,
    l2_regularization=1.0,
    random_state=7,
)
RIDGE_ALPHA = 30.0
MODEL_NAMES = ("ridge", "xgboost", "hgb")
MODEL_LABELS = {"ridge": "Ridge", "xgboost": "XGBoost", "hgb": "HistGradientBoosting", "mean_baseline": "Predict the mean"}
LEARNING_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def fit_score(net_pred: np.ndarray | float, center: float | None = None, scale: float | None = None) -> np.ndarray | float:
    c = FIT_CENTER if center is None else center
    s = FIT_SCALE if scale is None else scale
    z = (np.asarray(net_pred, dtype=float) - c) / s
    return 100.0 / (1.0 + np.exp(-z))


def calibrate_fit(y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Set the fit-score transform from the training target distribution."""
    global FIT_CENTER, FIT_SCALE
    FIT_CENTER = float(np.average(y, weights=w))
    FIT_SCALE = float(np.sqrt(np.average((y - FIT_CENTER) ** 2, weights=w)))
    return FIT_CENTER, FIT_SCALE


def _weighted_rmse(y, p, w):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def _weighted_r2(y, p, w):
    ybar = np.average(y, weights=w)
    ss_res = np.average((y - p) ** 2, weights=w)
    ss_tot = np.average((y - ybar) ** 2, weights=w)
    return float(1 - ss_res / ss_tot)


def _spearman(y, p):
    return float(spearmanr(y, p).correlation)


def make_ridge() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))])


def make_xgb(**over) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(**{**XGB_PARAMS, **over})


def make_hgb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**HGB_PARAMS)


def _fit(name: str, X, y, w):
    with threadpool_limits(limits=1):          # HGB's OpenMP pool fights XGBoost's on small boxes
        if name == "ridge":
            return make_ridge().fit(X, y, ridge__sample_weight=w)
        if name == "xgboost":
            return make_xgb().fit(X, y, sample_weight=w)
        if name == "hgb":
            return make_hgb().fit(X, y, sample_weight=w)
    raise KeyError(name)


@dataclass
class Evaluation:
    n_rows: int
    n_teams: int
    n_features: int
    folds: int
    metrics: dict = field(default_factory=dict)        # {"ridge": {...}, "xgboost": {...}, "hgb": {...}}
    per_fold: list = field(default_factory=list)       # [{"held_out_teams": [...], "ridge_spearman":..., ...}]
    learning_curve: dict = field(default_factory=dict) # {"fractions": [...], "n_train": [...], "ridge": [...], ...}
    weighted_minutes_total: float = 0.0


def cross_validate(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, groups: np.ndarray, n_splits: int | None = None) -> Evaluation:
    n_groups = len(np.unique(groups))
    n_splits = n_splits or n_groups
    gkf = GroupKFold(n_splits=n_splits)
    oof = {name: np.zeros(len(y)) for name in MODEL_NAMES}
    per_fold = []
    for tr, te in gkf.split(X, y, groups):
        row = {"held_out_teams": sorted(set(groups[te])), "n": int(len(te))}
        for name in MODEL_NAMES:
            mdl = _fit(name, X.iloc[tr], y[tr], w[tr])
            oof[name][te] = mdl.predict(X.iloc[te])
            key = "xgb" if name == "xgboost" else name
            row[f"{key}_spearman"] = _spearman(y[te], oof[name][te])
            row[f"{key}_rmse"] = _weighted_rmse(y[te], oof[name][te], w[te])
        per_fold.append(row)
    metrics = {}
    for name, p in oof.items():
        key = "xgb" if name == "xgboost" else name
        metrics[name] = {
            "rmse_weighted": _weighted_rmse(y, p, w),
            "r2_weighted": _weighted_r2(y, p, w),
            "spearman_pooled": _spearman(y, p),
            "spearman_fold_mean": float(np.mean([f[f"{key}_spearman"] for f in per_fold])),
            "folds_won": int(sum(1 for f in per_fold if f[f"{key}_spearman"] >= max(f[f"{k}_spearman"] for k in ("ridge", "xgb", "hgb")))),
        }
    # Naive baseline for context: predict the training weighted mean
    metrics["mean_baseline"] = {
        "rmse_weighted": _weighted_rmse(y, np.full_like(y, np.average(y, weights=w)), w),
        "r2_weighted": 0.0, "spearman_pooled": 0.0, "spearman_fold_mean": 0.0, "folds_won": 0,
    }
    return Evaluation(n_rows=len(y), n_teams=n_groups, n_features=X.shape[1], folds=n_splits,
                      metrics=metrics, per_fold=per_fold, weighted_minutes_total=float(w.sum()))


def learning_curve(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, groups: np.ndarray,
                   fractions=LEARNING_FRACTIONS, n_splits: int = 4, seed: int = 11) -> dict:
    """Held-out-team Spearman as a function of the fraction of training rows used.

    4 folds (two held-out teams each); for each fraction the training rows are a
    seeded random subsample of the other six teams' lineups. Mean +- sd over folds.
    """
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups))
    rng = np.random.default_rng(seed)
    out = {"fractions": list(fractions), "n_train": [], "folds": n_splits}
    for name in MODEL_NAMES:
        out[name] = []
        out[name + "_sd"] = []
    for frac in fractions:
        scores = {name: [] for name in MODEL_NAMES}
        n_used = []
        for tr, te in splits:
            k = max(40, int(round(frac * len(tr))))
            sub = np.sort(rng.choice(tr, size=min(k, len(tr)), replace=False))
            n_used.append(len(sub))
            for name in MODEL_NAMES:
                mdl = _fit(name, X.iloc[sub], y[sub], w[sub])
                scores[name].append(_spearman(y[te], mdl.predict(X.iloc[te])))
        out["n_train"].append(int(np.mean(n_used)))
        for name in MODEL_NAMES:
            out[name].append(float(np.mean(scores[name])))
            out[name + "_sd"].append(float(np.std(scores[name])))
    return out


def train(players: pd.DataFrame, lineups: pd.DataFrame, model_dir: str = MODEL_DIR, verbose: bool = True,
          with_learning_curve: bool = True) -> dict:
    X = build_feature_matrix(lineups, players)
    y = lineups["net_rating"].to_numpy(dtype=float)
    y_off = lineups["off_rating"].to_numpy(dtype=float)
    w = lineups["minutes"].to_numpy(dtype=float)
    w = w / w.mean()                                   # normalise so min_child_weight is in "lineups"
    groups = lineups["team"].to_numpy()

    ev = cross_validate(X, y, w, groups)
    if with_learning_curve:
        ev.learning_curve = learning_curve(X, y, w, groups)
    calibrate_fit(y, w)

    # Final models on all data
    ridge = _fit("ridge", X, y, w)
    booster = _fit("xgboost", X, y, w)
    hgb = _fit("hgb", X, y, w)
    with threadpool_limits(limits=1):
        booster_off = make_xgb(n_estimators=800).fit(X, y_off, sample_weight=w)

    # Feature importance (gain), normalised
    gain = booster.get_booster().get_score(importance_type="gain")
    total = sum(gain.values()) or 1.0
    importance = sorted(((k, v / total) for k, v in gain.items()), key=lambda kv: -kv[1])

    os.makedirs(model_dir, exist_ok=True)
    booster.save_model(os.path.join(model_dir, "model.json"))
    booster_off.save_model(os.path.join(model_dir, "model_off.json"))
    with open(os.path.join(model_dir, "ridge.pkl"), "wb") as fh:
        pickle.dump(ridge, fh)
    with open(os.path.join(model_dir, "hgb.pkl"), "wb") as fh:
        pickle.dump(hgb, fh)
    meta = {
        "feature_names": feature_names(),
        "fit_center": FIT_CENTER,
        "fit_scale": FIT_SCALE,
        "xgb_params": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in XGB_PARAMS.items()},
        "hgb_params": HGB_PARAMS,
        "ridge_alpha": RIDGE_ALPHA,
        "off_rating_model": "xgboost, same params, 800 rounds; def = off - net",
        "evaluation": {
            "n_rows": ev.n_rows, "n_teams": ev.n_teams, "n_features": ev.n_features, "folds": ev.folds,
            "metrics": ev.metrics, "per_fold": ev.per_fold, "learning_curve": ev.learning_curve,
        },
        "feature_importance": importance,
        "target_summary": {
            "net_rating_mean_weighted": float(np.average(y, weights=w)),
            "net_rating_std": float(np.std(y)),
            "off_rating_mean_weighted": float(np.average(y_off, weights=w)),
            "minutes_total": float(w.sum()),
        },
    }
    with open(os.path.join(model_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    if verbose:
        print(f"Trained on {ev.n_rows} lineups / {ev.n_teams} teams / {ev.n_features} features "
              f"(GroupKFold by team, {ev.folds} folds)")
        print(f"{'model':<22}{'RMSE(w)':>10}{'R2(w)':>10}{'Spearman':>12}{'Spearman/fold':>16}{'folds won':>11}")
        for name in ("mean_baseline",) + MODEL_NAMES:
            m = ev.metrics[name]
            print(f"{MODEL_LABELS[name]:<22}{m['rmse_weighted']:>10.2f}{m['r2_weighted']:>10.3f}"
                  f"{m['spearman_pooled']:>12.3f}{m['spearman_fold_mean']:>16.3f}{m['folds_won']:>11}")
        if ev.learning_curve:
            lc = ev.learning_curve
            print("Learning curve (held-out Spearman, mean of 4 team-folds):")
            print("  " + " ".join(f"{'n=' + str(n):>10}" for n in lc["n_train"]))
            for name in MODEL_NAMES:
                print(f"  {MODEL_LABELS[name]:<20}" + " ".join(f"{v:>10.3f}" for v in lc[name]))
        print("Top features (gain share):")
        for k, v in importance[:8]:
            print(f"  {k:<28}{v:6.3f}")
        print(f"Fit score: 100*sigmoid((net - {FIT_CENTER:.2f}) / {FIT_SCALE:.2f})")
        print(f"Saved -> {model_dir}/model.json, model_off.json, ridge.pkl, hgb.pkl, meta.json")
    return meta


def load(model_dir: str = MODEL_DIR) -> tuple[xgb.XGBRegressor, Pipeline, dict]:
    """Load the production booster, the Ridge baseline and meta (sets the fit transform)."""
    global FIT_CENTER, FIT_SCALE
    with open(os.path.join(model_dir, "meta.json")) as fh:
        meta = json.load(fh)
    FIT_CENTER, FIT_SCALE = float(meta["fit_center"]), float(meta["fit_scale"])
    booster = xgb.XGBRegressor()
    booster.load_model(os.path.join(model_dir, "model.json"))
    with open(os.path.join(model_dir, "ridge.pkl"), "rb") as fh:
        ridge = pickle.load(fh)
    return booster, ridge, meta


def load_off(model_dir: str = MODEL_DIR) -> xgb.XGBRegressor | None:
    path = os.path.join(model_dir, "model_off.json")
    if not os.path.exists(path):
        return None
    b = xgb.XGBRegressor()
    b.load_model(path)
    return b


if __name__ == "__main__":
    players = pd.read_csv(os.path.join(ROOT, "data", "players.csv"))
    lineups = pd.read_csv(os.path.join(ROOT, "data", "lineups.csv"))
    train(players, lineups)
