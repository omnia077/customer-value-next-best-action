"""Propensity: P(recharge of >= 500 SDG next calendar month), as-of features, time-based split.

Train cutoffs whose label month is a campaign month (April, June) are excluded by construction
(campaign offers inflate recharge counts of treated subscribers). Test = untouched later cutoffs.

Note: segment ID is NOT used as a feature. Segments are fit as of 30 Sep; using them at earlier cutoffs
would leak future behaviour, and per-cutoff refits give non-comparable ids.

Run:  python src/propensity.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).parent))
from features import ASOF_FEATURES, asof_features, load_tables, recharge500_label  # noqa: E402
from segment import load_cfg  # noqa: E402

CAMPAIGN_LABEL_MONTHS = (4, 6)


def next_month_window(cutoff):
    start = pd.Timestamp(cutoff) + pd.Timedelta(days=1)
    return start, start + pd.offsets.MonthEnd(0)


def build_xy(tables, cutoff):
    cust, rech, usage = tables
    X = asof_features(cust, rech, usage, cutoff)
    s, e = next_month_window(cutoff)
    y = pd.Series(X.index.isin(list(recharge500_label(rech, s, e))).astype(int), index=X.index, name="y")
    return X, y, s.month


def logistic_model():
    return make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log1p), StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))


def lgbm_model(seed):
    return LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                          random_state=seed, verbose=-1)


def evaluate(y, p, base):
    top = int(len(y) * 0.1)
    order = np.argsort(-np.asarray(p))
    return {"roc_auc": round(roc_auc_score(y, p), 3), "pr_auc": round(average_precision_score(y, p), 3),
            "top_decile_lift": round(float(np.asarray(y)[order[:top]].mean() / y.mean()), 2),
            "brier": round(brier_score_loss(y, p), 4), "base_rate": round(float(y.mean()), 3),
            "n": int(len(y))}


def run(cfg_path):
    cfg = load_cfg(cfg_path)
    pc = cfg["propensity"]
    tables = load_tables(cfg["_data"])
    data = {}
    for name in ("train", "val", "test"):
        parts = []
        for c in pc[f"{name}_cutoffs"]:
            X, y, lm = build_xy(tables, c)
            if name == "train":
                assert lm not in CAMPAIGN_LABEL_MONTHS, f"cutoff {c}: label month {lm} is a campaign month"
            parts.append((c, X, y))
        data[name] = parts
        print(name, [(c, len(X), round(float(y.mean()), 3)) for c, X, y in parts])

    Xtr = pd.concat([X for _, X, _ in data["train"]])[ASOF_FEATURES]
    ytr = pd.concat([y for _, _, y in data["train"]])
    Xva = pd.concat([X for _, X, _ in data["val"]])[ASOF_FEATURES]
    yva = pd.concat([y for _, _, y in data["val"]])
    Xte = pd.concat([X.assign(_c=c) for c, X, _ in data["test"]])
    yte = pd.concat([y for _, _, y in data["test"]])
    cut_te = Xte.pop("_c")
    Xte = Xte[ASOF_FEATURES]

    lr = logistic_model().fit(Xtr, ytr)
    gb = lgbm_model(cfg["seed"]).fit(Xtr, ytr)
    p_base = np.full(len(yte), ytr.mean())
    p_lr = lr.predict_proba(Xte)[:, 1]
    p_gb_raw = gb.predict_proba(Xte)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(gb.predict_proba(Xva)[:, 1], yva)
    p_gb = iso.predict(p_gb_raw)

    res = {"Base rate": evaluate(yte, p_base + np.random.default_rng(0).normal(0, 1e-9, len(yte)), None),
           "Logistic": evaluate(yte, p_lr, None),
           "LightGBM": evaluate(yte, p_gb_raw, None),
           "LightGBM + isotonic (val)": evaluate(yte, p_gb, None)}
    per_cut = {}
    for c in sorted(cut_te.unique()):
        m = (cut_te == c).values
        per_cut[str(pd.Timestamp(c).date())] = evaluate(yte[m], p_gb_raw[m], None)
    print(pd.DataFrame(res).T.to_string())
    print("per test cutoff (LightGBM):", per_cut)

    d = pd.qcut(pd.Series(p_gb, index=yte.index).rank(method="first"), 10, labels=False)
    cal = pd.DataFrame({"pred": p_gb, "obs": yte.values, "decile": d.values}).groupby("decile").mean().round(3)
    cal.to_csv(cfg["_out"] / "propensity_calibration_test.csv")
    imp = pd.Series(gb.booster_.feature_importance("gain"), index=ASOF_FEATURES).sort_values(ascending=False)
    imp = (100 * imp / imp.sum()).round(1)
    print("gain importance %:", imp.head(8).to_dict())

    # scores as of 30 Sep (calibration map was fit on the same-cutoff labels: low-capacity monotone map, disclosed)
    sc = Xva.copy()
    out = pd.DataFrame({"customer_id": sc.index, "p_recharge500_next_month": iso.predict(gb.predict_proba(sc)[:, 1]),
                        "as_of": pc["val_cutoffs"][0]})
    out.to_parquet(cfg["_out"] / "propensity_scores.parquet", index=False)
    json.dump({"test": res, "per_test_cutoff": per_cut, "importance_gain_pct": imp.head(10).to_dict(),
               "train_rows": int(len(ytr)), "features": ASOF_FEATURES},
              open(cfg["_out"] / "propensity_metrics.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    run(Path(__file__).resolve().parent.parent / "config" / "customer_value.yaml")
