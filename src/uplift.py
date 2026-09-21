"""Uplift on the simulated randomized recharge-bonus campaigns.

Design: train on the April campaign, test on the June campaign. Outcome = recharge >= 500 SDG within
30 days of the send date. Features are as of the day BEFORE sending. ground_truth/campaign_truth is read only
inside the evaluation block, never for features, training or model selection.

Learner selection uses 5-fold cross-validated Qini on APRIL ONLY; June is never used to choose a learner.
All learners are then reported on June so the choice can be audited.

Run:  python src/uplift.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
from features import ASOF_FEATURES, asof_features, load_tables, recharge500_label  # noqa: E402
from segment import load_cfg  # noqa: E402

P = dict(n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=60, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)
P_REG = {**P, "n_estimators": 150, "num_leaves": 7, "min_child_samples": 150, "reg_lambda": 20.0}


# ------------------------------------------------------------------ learners ---
class TLearner:
    """Separate outcome models for treated and control; uplift = difference."""
    def __init__(self, params=P):
        self.params = params

    def fit(self, X, t, y, seed):
        self.m1 = LGBMClassifier(random_state=seed, **self.params).fit(X[t == 1], y[t == 1])
        self.m0 = LGBMClassifier(random_state=seed, **self.params).fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]


class SLearner:
    """One model with the treatment flag as a feature; uplift = f(x, 1) - f(x, 0)."""
    def fit(self, X, t, y, seed):
        self.m = LGBMClassifier(random_state=seed, **P).fit(X.assign(_t=np.asarray(t)), y)
        return self

    def predict(self, X):
        return self.m.predict_proba(X.assign(_t=1))[:, 1] - self.m.predict_proba(X.assign(_t=0))[:, 1]


class XLearner:
    """Kunzel et al.: impute individual effects with the opposite arm's model, regress them, average (g = 0.5)."""
    def fit(self, X, t, y, seed):
        t = np.asarray(t)
        X1, X0, y1, y0 = X[t == 1], X[t == 0], np.asarray(y)[t == 1], np.asarray(y)[t == 0]
        m1 = LGBMClassifier(random_state=seed, **P).fit(X1, y1)
        m0 = LGBMClassifier(random_state=seed, **P).fit(X0, y0)
        d1 = y1 - m0.predict_proba(X1)[:, 1]
        d0 = m1.predict_proba(X0)[:, 1] - y0
        self.r1 = LGBMRegressor(random_state=seed, **P).fit(X1, d1)
        self.r0 = LGBMRegressor(random_state=seed, **P).fit(X0, d0)
        return self

    def predict(self, X):
        return 0.5 * self.r1.predict(X) + 0.5 * self.r0.predict(X)


class ClassTransform:
    """Jaskowski & Jaroszewicz: with ~50% randomization, uplift = 2 * P(z = 1 | x) - 1 where z = 1 if y == t."""
    def fit(self, X, t, y, seed):
        z = (np.asarray(y) == np.asarray(t)).astype(int)
        self.m = LGBMClassifier(random_state=seed, **P).fit(X, z)
        return self

    def predict(self, X):
        return 2 * self.m.predict_proba(X)[:, 1] - 1


LEARNERS = {"T-learner": lambda: TLearner(P), "T-learner (regularized)": lambda: TLearner(P_REG),
            "S-learner": SLearner, "X-learner": XLearner, "Class transformation": ClassTransform}


# ------------------------------------------------------------------ helpers ----
def campaign_frame(tables, hist, cid, horizon_days):
    """Features (as of send date - 1), treatment flag and outcome for one campaign."""
    cust, rech, usage = tables
    h = hist[hist.campaign_id == cid].set_index("customer_id")
    sent = pd.Timestamp(h.sent_date.iloc[0])
    X = asof_features(cust, rech, usage, sent - pd.Timedelta(days=1)).reindex(h.index)   # NaN if no recent activity
    y = pd.Series(h.index.isin(list(recharge500_label(rech, sent, sent + pd.Timedelta(days=horizon_days - 1)))).astype(int),
                  index=h.index)
    return X[ASOF_FEATURES], h.treated.astype(int), y, sent


def qini(score, t, y):
    """Cumulative incremental conversions when contacting the top-x share by score (randomized data)."""
    o = np.argsort(-np.asarray(score), kind="stable")
    t, y = np.asarray(t)[o], np.asarray(y)[o]
    cT, cC = np.cumsum(t), np.cumsum(1 - t)
    inc = np.cumsum(y * t) - np.cumsum(y * (1 - t)) * np.where(cC > 0, cT / np.maximum(cC, 1), 0.0)
    return np.arange(1, len(y) + 1) / len(y), inc


def qini_coef(score, t, y):
    """Area between the model's Qini curve and the random-targeting line, per customer."""
    x, inc = qini(score, t, y)
    return float((np.trapezoid(inc, x) - 0.5 * inc[-1]) / len(y))


def observed_uplift_top(score, t, y, frac):
    n = int(len(y) * frac)
    o = np.argsort(-np.asarray(score), kind="stable")[:n]
    t_, y_ = np.asarray(t)[o], np.asarray(y)[o]
    return float(y_[t_ == 1].mean() - y_[t_ == 0].mean())


def balance_table(X, t):
    """Standardized mean difference treated vs control for each feature (|SMD| < 0.1 = balanced)."""
    a, b = X[t == 1], X[t == 0]
    sd = np.sqrt((a.var() + b.var()) / 2).replace(0, np.nan)
    return ((a.mean() - b.mean()) / sd).round(3)


def cv_select(X, t, y, seed, folds=5):
    """5-fold cross-validated Qini per learner on ONE campaign (April). Returns table and the winner's name."""
    rows = []
    kf = KFold(folds, shuffle=True, random_state=seed)
    for name, make in LEARNERS.items():
        q = []
        for tr, va in kf.split(X):
            m = make().fit(X.iloc[tr], t.iloc[tr], y.iloc[tr], seed)
            q.append(qini_coef(m.predict(X.iloc[va]), t.iloc[va].values, y.iloc[va].values))
        rows.append({"learner": name, "cv_qini_mean": round(float(np.mean(q)), 4), "cv_qini_sd": round(float(np.std(q)), 4)})
        print("  CV", rows[-1])
    cv = pd.DataFrame(rows)
    return cv, cv.loc[cv.cv_qini_mean.idxmax(), "learner"]


def run(cfg_path):
    cfg = load_cfg(cfg_path)
    uc = cfg["uplift"]
    tables = load_tables(cfg["_data"])
    hist = pd.read_parquet(cfg["_data"] / "campaigns" / "campaign_history.parquet")
    Xa, ta, ya, _ = campaign_frame(tables, hist, uc["train_campaign"], uc["horizon_days"])
    Xj, tj, yj, sent_j = campaign_frame(tables, hist, uc["test_campaign"], uc["horizon_days"])

    # ---- randomization + average effect ----
    chk = {}
    for nm, X, t, y in [("april", Xa, ta, ya), ("june", Xj, tj, yj)]:
        ate = y[t == 1].mean() - y[t == 0].mean()
        se = np.sqrt(y[t == 1].var() / (t == 1).sum() + y[t == 0].var() / (t == 0).sum())
        smd = balance_table(X, t)
        chk[nm] = {"n": int(len(y)), "treated_share": round(float(t.mean()), 3),
                   "control_rate": round(float(y[t == 0].mean()), 3), "treated_rate": round(float(y[t == 1].mean()), 3),
                   "ate": round(float(ate), 4), "ate_se": round(float(se), 4), "max_abs_smd": round(float(smd.abs().max()), 3)}
    print("randomization / ATE:", json.dumps(chk))

    # ---- learner selection on April only (5-fold CV Qini) ----
    print("learner selection (April CV):")
    cv, selected = cv_select(Xa, ta, ya, cfg["seed"])
    print("selected:", selected)
    cv.to_csv(cfg["_out"] / "uplift_learner_cv.csv", index=False)

    # ---- fit everything on April, predict June ----
    fitted = {n: make().fit(Xa, ta, ya, cfg["seed"]) for n, make in LEARNERS.items()}
    preds = {n: m.predict(Xj) for n, m in fitted.items()}
    s_up = preds[selected]
    prop = LGBMClassifier(random_state=cfg["seed"], **P).fit(Xa, ya)          # ignores treatment: "who will buy"
    s_pr = prop.predict_proba(Xj)[:, 1]

    # ---- evaluation against ground truth (validation only) ----
    truth = (pd.read_parquet(cfg["_data"] / "ground_truth" / "campaign_truth.parquet")
             .query("campaign_id == @uc['test_campaign']").set_index("customer_id").reindex(Xj.index))
    tu = truth.true_uplift.values
    rng = np.random.default_rng(cfg["seed"])
    frac = uc["contact_frac"]
    n_sel = int(len(tu) * frac)
    ranks = {"Uplift model": s_up, "Propensity model": s_pr, "Oracle (true uplift)": tu, "Random": rng.random(len(tu))}
    for n in LEARNERS:
        if n != selected:
            ranks[f"Alt learner: {n}"] = preds[n]
    rows, curves = [], {}
    yv, tv = yj.values, tj.values
    for nm, s in ranks.items():
        sel = np.argsort(-s, kind="stable")[:n_sel]
        rows.append({"policy": nm, "learner": selected if nm == "Uplift model" else nm.replace("Alt learner: ", "") if nm.startswith("Alt") else "",
                     "spearman_vs_true_uplift": round(float(spearmanr(s, tu)[0]), 3),
                     "expected_incremental_recharges_top20": round(float(tu[sel].sum()), 0),
                     "observed_uplift_top20_pts": round(100 * observed_uplift_top(s, tv, yv, frac), 1),
                     "qini_coef": round(qini_coef(s, tv, yv), 4)})
        x, inc = qini(s, tv, yv)
        curves[nm] = np.interp(np.linspace(0, 1, 101), np.r_[0, x], np.r_[0, inc])
    res = pd.DataFrame(rows)
    res.loc[res.policy == "Random", "expected_incremental_recharges_top20"] = round(float(tu.sum() * frac), 0)   # exact expectation

    # ---- bootstrap CIs (resample subscribers; scores fixed) ----
    B = uc["n_bootstrap"]
    boot = {nm: {"qini": [], "obs_top20": []} for nm in ranks if nm != "Random"}
    for _ in range(B):
        i = rng.integers(0, len(tu), len(tu))
        for nm in boot:
            boot[nm]["qini"].append(qini_coef(ranks[nm][i], tv[i], yv[i]))
            boot[nm]["obs_top20"].append(observed_uplift_top(ranks[nm][i], tv[i], yv[i], frac))
    ci = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    res["qini_coef_ci95"] = res.policy.map(lambda n: ci(boot[n]["qini"]) if n in boot else None)
    res["observed_uplift_top20_ci95_pts"] = res.policy.map(
        lambda n: [round(100 * v, 1) for v in ci(boot[n]["obs_top20"])] if n in boot else None)
    # is the selected model's Qini advantage over propensity / random significant? paired bootstrap difference
    dq = np.array(boot["Uplift model"]["qini"]) - np.array(boot["Propensity model"]["qini"])
    print(res.to_string(index=False))
    print(f"all customers: observed uplift {100*chk['june']['ate']:.1f} pts, true mean {100*tu.mean():.1f} pts | "
          f"Qini uplift - propensity: {dq.mean():.4f} (95% CI {np.percentile(dq, 2.5):.4f}, {np.percentile(dq, 97.5):.4f})")

    # ---- calibration of the selected model: quintiles of predicted uplift ----
    q = pd.qcut(pd.Series(s_up).rank(method="first"), 5, labels=False).values
    cal = pd.DataFrame({"q": q, "pred": s_up, "true": tu, "y": yv, "t": tv})
    calt = cal.groupby("q").apply(lambda d: pd.Series({
        "pred_uplift_pts": 100 * d.pred.mean(), "true_uplift_pts": 100 * d.true.mean(),
        "observed_uplift_pts": 100 * (d.y[d.t == 1].mean() - d.y[d.t == 0].mean())}), include_groups=False).round(1)
    print("\nquintiles (5 = highest predicted uplift):\n", calt.to_string())
    calt.to_csv(cfg["_out"] / "uplift_calibration_quintiles.csv")
    res.to_csv(cfg["_out"] / "uplift_evaluation.csv", index=False)
    pd.DataFrame({"share_contacted": np.linspace(0, 1, 101), **curves}).to_csv(cfg["_out"] / "uplift_qini_curves.csv", index=False)
    pd.DataFrame({"customer_id": Xj.index, "uplift_hat": s_up, "prop_hat": s_pr, "true_uplift": tu,
                  "treated": tv, "y": yv}).to_parquet(cfg["_out"] / "uplift_june_test_scored.parquet", index=False)

    # ---- final scoring model: selected learner on both campaigns, applied to features as of 30 Sep ----
    Xall, tall, yall = pd.concat([Xa, Xj]), pd.concat([ta, tj]), pd.concat([ya, yj])
    final = LEARNERS[selected]().fit(Xall, tall, yall, cfg["seed"])
    cust, rech, usage = tables
    Xs = asof_features(cust, rech, usage, cfg["segmentation"]["as_of"])[ASOF_FEATURES]
    p0 = LGBMClassifier(random_state=cfg["seed"], **P).fit(Xall[tall == 0], yall[tall == 0]).predict_proba(Xs)[:, 1]
    pd.DataFrame({"customer_id": Xs.index, "uplift": final.predict(Xs), "p_recharge500_no_offer": p0,
                  "as_of": cfg["segmentation"]["as_of"]}).to_parquet(cfg["_out"] / "uplift_scores.parquet", index=False)
    json.dump({"randomization": chk, "selected_learner": selected, "learner_cv": cv.to_dict("records"),
               "evaluation": json.loads(res.to_json(orient="records")),
               "qini_diff_uplift_minus_propensity": {"mean": round(float(dq.mean()), 4),
                                                     "ci95": [round(float(np.percentile(dq, 2.5)), 4), round(float(np.percentile(dq, 97.5)), 4)]},
               "quintiles": json.loads(calt.reset_index().to_json(orient="records")),
               "true_mean_uplift_june": round(float(tu.mean()), 4)},
              open(cfg["_out"] / "uplift_metrics.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    run(Path(__file__).resolve().parent.parent / "config" / "customer_value.yaml")
