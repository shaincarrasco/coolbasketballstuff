"""
Training / evaluation for the NBA Game Predictor.

Models
  * home_baseline : "home team always wins" with p = train home win rate
  * elo_logit     : LogisticRegression on Elo difference only
  * xgb           : XGBoost classifier on all pre-game features (+ optional
                    calibrator chosen on a validation slice by Brier score)
  * hgb           : sklearn HistGradientBoostingClassifier on the same features
                    (third model for the "model lab" comparison; LightGBM is
                    not a dependency, HGB is its stdlib-ish cousin)

Protocol
  * time-based split: train on seasons 1-2, test on season 3
  * calibration choice: train on seasons 1-2 minus the final ~2 months of
    season 2 (validation slice), fit isotonic / sigmoid calibrators on
    out-of-fold predictions, pick whichever minimises Brier on the slice
  * walk-forward by month inside season 3 (refit each month on all prior data)

Artifacts: models/model.json, models/calibrator.pkl, models/metrics.json,
           models/feature_columns.json, output/metrics.json
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import FEATURE_COLUMNS, assert_no_leakage, build_features

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUTPUT = ROOT / "output"

XGB_PARAMS = dict(
    # heavily regularised: ~2.5k training rows, noisy target. Chosen on the
    # validation slice (last 60 days of season 2), not on the test season.
    n_estimators=150,
    learning_rate=0.03,
    max_depth=2,
    min_child_weight=30,
    subsample=0.7,
    colsample_bytree=0.6,
    reg_lambda=10.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=7,
    n_jobs=1,  # 2-core box: more threads is slower (oversubscription)
)


HGB_PARAMS = dict(
    max_iter=200,
    learning_rate=0.03,
    max_depth=3,
    min_samples_leaf=40,
    l2_regularization=5.0,
    early_stopping=False,
    random_state=7,
)


def fit_hgb(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(**HGB_PARAMS)
    clf.fit(X, y)
    return clf


# ---- calibrators -------------------------------------------------------
class SigmoidCalibrator:
    """Platt scaling on the logit of the raw probability."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e6)

    @staticmethod
    def _z(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y):
        self.lr.fit(self._z(p), y)
        return self

    def predict(self, p):
        return self.lr.predict_proba(self._z(p))[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)

    def fit(self, p, y):
        self.iso.fit(np.asarray(p, dtype=float), y)
        return self

    def predict(self, p):
        return self.iso.predict(np.asarray(p, dtype=float))


class IdentityCalibrator:
    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.asarray(p, dtype=float)


CALIBRATORS = {"none": IdentityCalibrator, "isotonic": IsotonicCalibrator, "sigmoid": SigmoidCalibrator}


# ---- helpers -----------------------------------------------------------
def metrics(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }


def reliability_table(y, p, bins: int = 10) -> list[dict]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": int(m.sum()),
            "pred_mean": float(p[m].mean()) if m.any() else None,
            "actual_rate": float(y[m].mean()) if m.any() else None,
        })
    return rows


def expected_calibration_error(table: list[dict]) -> float:
    n = sum(r["n"] for r in table)
    return float(sum(r["n"] * abs(r["pred_mean"] - r["actual_rate"]) for r in table if r["n"]) / max(n, 1))


def fit_xgb(X: pd.DataFrame, y: pd.Series, seed: int = 7) -> xgb.XGBClassifier:
    params = dict(XGB_PARAMS, random_state=seed)
    clf = xgb.XGBClassifier(**params)
    clf.fit(X, y)
    return clf


def oof_predictions(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 11) -> np.ndarray:
    """Out-of-fold raw probabilities on the training set (used to fit calibrators
    without reusing in-sample, over-confident predictions)."""
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        m = fit_xgb(X.iloc[tr], y.iloc[tr], seed=seed)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def choose_calibrator(X_tr, y_tr, X_val, y_val) -> tuple[str, dict]:
    """Fit on X_tr, calibrate on OOF(X_tr), score each option on X_val by Brier."""
    base = fit_xgb(X_tr, y_tr)
    oof = oof_predictions(X_tr, y_tr)
    raw_val = base.predict_proba(X_val)[:, 1]
    scores = {}
    for name, cls in CALIBRATORS.items():
        cal = cls().fit(oof, y_tr)
        scores[name] = metrics(y_val, cal.predict(raw_val))
    best = min(scores, key=lambda k: scores[k]["brier"])
    return best, scores


# ---- main training routine --------------------------------------------
def train(games_path: Path | None = None, verbose: bool = True) -> dict:
    games_path = Path(games_path) if games_path else ROOT / "data" / "games.csv"
    games = pd.read_csv(games_path)
    feats = build_features(games)
    assert_no_leakage(games, feats)
    played = feats.dropna(subset=["home_win"]).copy()
    played["home_win"] = played["home_win"].astype(int)

    seasons = sorted(played["season"].unique())
    if len(seasons) < 3:
        raise SystemExit("need at least 3 seasons for the train(1-2)/test(3) protocol")
    train_seasons, test_season = seasons[:-1], seasons[-1]
    tr = played[played["season"].isin(train_seasons)].copy()
    te = played[played["season"] == test_season].copy()

    X_tr, y_tr = tr[FEATURE_COLUMNS], tr["home_win"]
    X_te, y_te = te[FEATURE_COLUMNS], te["home_win"]
    log = print if verbose else (lambda *a, **k: None)
    log(f"train: seasons {train_seasons} ({len(tr)} games) | test: season {test_season} ({len(te)} games)")

    # ---- baseline 1: home team always wins ----------------------------
    p_home = float(y_tr.mean())
    res_home = metrics(y_te, np.full(len(y_te), p_home))

    # ---- baseline 2: logistic regression on Elo diff only -------------
    elo_lr = LogisticRegression(C=1.0).fit(tr[["elo_diff"]], y_tr)
    p_elo = elo_lr.predict_proba(te[["elo_diff"]])[:, 1]
    res_elo = metrics(y_te, p_elo)

    # ---- calibration choice on a validation slice ---------------------
    last_train_season = tr[tr["season"] == train_seasons[-1]]
    val_cut = last_train_season["date"].max() - pd.Timedelta(days=60)
    val_mask = (tr["season"] == train_seasons[-1]) & (tr["date"] > val_cut)
    cal_name, cal_scores = choose_calibrator(X_tr[~val_mask], y_tr[~val_mask], X_tr[val_mask], y_tr[val_mask])
    log(f"calibration on validation slice ({int(val_mask.sum())} games): "
        + ", ".join(f"{k} brier={v['brier']:.4f}" for k, v in cal_scores.items()) + f" -> {cal_name}")

    # ---- final XGBoost on all training data ---------------------------
    model = fit_xgb(X_tr, y_tr)
    calibrator = CALIBRATORS[cal_name]().fit(oof_predictions(X_tr, y_tr), y_tr)
    raw_te = model.predict_proba(X_te)[:, 1]
    p_xgb = calibrator.predict(raw_te)
    res_xgb = metrics(y_te, p_xgb)
    res_xgb_raw = metrics(y_te, raw_te)

    # ---- third model: sklearn HistGradientBoosting, same split --------
    hgb = fit_hgb(X_tr, y_tr)
    p_hgb = hgb.predict_proba(X_te)[:, 1]
    res_hgb = metrics(y_te, p_hgb)

    # ---- walk-forward by month inside the test season -----------------
    te["month"] = te["date"].dt.to_period("M").astype(str)
    walk = []
    for month, grp in te.groupby("month", sort=True):
        hist = played[played["date"] < grp["date"].min()]
        m = fit_xgb(hist[FEATURE_COLUMNS], hist["home_win"])
        cal = CALIBRATORS[cal_name]().fit(oof_predictions(hist[FEATURE_COLUMNS], hist["home_win"], n_splits=4), hist["home_win"]) \
            if cal_name != "none" else IdentityCalibrator()
        p = cal.predict(m.predict_proba(grp[FEATURE_COLUMNS])[:, 1])
        elo_p = LogisticRegression(C=1.0).fit(hist[["elo_diff"]], hist["home_win"]).predict_proba(grp[["elo_diff"]])[:, 1]
        hgb_p = fit_hgb(hist[FEATURE_COLUMNS], hist["home_win"]).predict_proba(grp[FEATURE_COLUMNS])[:, 1]
        walk.append({
            "month": month,
            "n_train": int(len(hist)),
            **{f"xgb_{k}": v for k, v in metrics(grp["home_win"], p).items() if k != "n"},
            **{f"elo_{k}": v for k, v in metrics(grp["home_win"], elo_p).items() if k != "n"},
            **{f"hgb_{k}": v for k, v in metrics(grp["home_win"], hgb_p).items() if k != "n"},
            "home_accuracy": float(grp["home_win"].mean()),
            "n": int(len(grp)),
        })
        log(f"  walk-forward {month}: n={len(grp):3d} xgb acc={walk[-1]['xgb_accuracy']:.3f} "
            f"ll={walk[-1]['xgb_log_loss']:.3f} | elo acc={walk[-1]['elo_accuracy']:.3f}")

    # ---- feature importance (gain) -------------------------------------
    gain = model.get_booster().get_score(importance_type="gain")
    total = sum(gain.values()) or 1.0
    importance = sorted(({"feature": k, "gain_share": v / total} for k, v in gain.items()),
                        key=lambda r: -r["gain_share"])
    # permutation importance (log-loss drop) for HGB on the test season; 5 repeats
    perm = permutation_importance(hgb, X_te, y_te, scoring="neg_log_loss", n_repeats=5, random_state=3, n_jobs=1)
    hgb_importance = sorted(({"feature": FEATURE_COLUMNS[i], "logloss_increase": float(perm.importances_mean[i])}
                             for i in range(len(FEATURE_COLUMNS))), key=lambda r: -r["logloss_increase"])
    # Elo-logit "importance" is trivially one coefficient; report it for completeness
    elo_coef = float(elo_lr.coef_[0][0])

    out = {
        "data": {
            "games_total": int(len(played)),
            "train_seasons": train_seasons,
            "test_season": test_season,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "home_win_rate_train": p_home,
            "home_win_rate_test": float(y_te.mean()),
        },
        "models": {
            "home_baseline": res_home,
            "elo_logit": res_elo,
            "xgb_raw": res_xgb_raw,
            "xgb": res_xgb,
            "hgb": res_hgb,
        },
        "calibration": {
            "chosen": cal_name,
            "validation_scores": cal_scores,
            "reliability_xgb": reliability_table(y_te, p_xgb),
            "reliability_elo": reliability_table(y_te, p_elo),
            "ece_xgb": expected_calibration_error(reliability_table(y_te, p_xgb)),
            "ece_elo": expected_calibration_error(reliability_table(y_te, p_elo)),
        },
        "walk_forward": walk,
        "feature_importance": importance[:15],
        "hgb_permutation_importance": hgb_importance[:15],
        "elo_logit_coef_per_100": elo_coef * 100,
        "hgb_params": HGB_PARAMS,
        "xgb_params": {k: v for k, v in XGB_PARAMS.items() if k not in ("n_jobs",)},
        "features": FEATURE_COLUMNS,
    }

    MODELS.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)
    model.get_booster().save_model(MODELS / "model.json")
    with open(MODELS / "calibrator.pkl", "wb") as f:
        pickle.dump({"name": cal_name, "calibrator": calibrator}, f)
    with open(MODELS / "elo_logit.pkl", "wb") as f:
        pickle.dump(elo_lr, f)
    with open(MODELS / "hgb.pkl", "wb") as f:
        pickle.dump(hgb, f)
    (MODELS / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    for p in (MODELS / "metrics.json", OUTPUT / "metrics.json"):
        p.write_text(json.dumps(out, indent=2))

    log("\nTEST SEASON RESULTS  (season %s)" % test_season)
    log(f"{'model':<14}{'acc':>8}{'logloss':>10}{'brier':>8}{'auc':>8}")
    for name, r in out["models"].items():
        log(f"{name:<14}{r['accuracy']:>8.3f}{r['log_loss']:>10.4f}{r['brier']:>8.4f}{r['roc_auc']:>8.3f}")
    return out


def load_model():
    booster = xgb.Booster()
    booster.load_model(MODELS / "model.json")
    with open(MODELS / "calibrator.pkl", "rb") as f:
        cal = pickle.load(f)
    cols = json.loads((MODELS / "feature_columns.json").read_text())
    return booster, cal["calibrator"], cal["name"], cols


if __name__ == "__main__":
    # run through the importable module so pickled calibrators resolve to
    # `model.SigmoidCalibrator` instead of `__main__.SigmoidCalibrator`
    import model as _m
    _m.train()
