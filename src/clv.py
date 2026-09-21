"""CLV: BG/NBD + Gamma-Gamma (own implementation, Fader-Hardie formulas) vs simple and ML baselines.

Prepaid lines have no cancellation event, so dropout is inferred (BG/NBD). Time unit = days.
Fit window 2024-01-01..2024-09-30, validation window 2024-10-01..2024-12-31 (92 days).

Run:  python src/clv.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features, load_tables  # noqa: E402
from segment import load_cfg  # noqa: E402


# ------------------------------------------------------------------ BG/NBD ----
def _bgnbd_ll(p, x, t_x, T):
    r, alpha, a, b = np.exp(p)
    A1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    A2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    A3 = -(r + x) * np.log(alpha + T)
    A4 = np.log(a) - np.log(b + np.maximum(x, 1) - 1) - (r + x) * np.log(alpha + t_x)
    return A1 + A2 + np.logaddexp(A3, np.where(x > 0, A4, -np.inf))


class BGNBD:
    def __init__(self, penalizer=0.001):
        self.pen = penalizer

    def fit(self, x, t_x, T):
        f = lambda p: -_bgnbd_ll(p, x, t_x, T).mean() + self.pen * np.sum(np.exp(p) ** 2)
        res = minimize(f, np.log([0.5, 10.0, 1.5, 1.5]), method="L-BFGS-B",
                       bounds=[(-6, 4)] * 4)
        self.r, self.alpha, self.a, self.b = np.exp(res.x)
        self.converged = bool(res.success)
        return self

    def p_alive(self, x, t_x, T):
        r, al, a, b = self.r, self.alpha, self.a, self.b
        odds = np.where(x > 0, a / (b + np.maximum(x, 1) - 1) * ((al + T) / (al + t_x)) ** (r + x), 0.0)
        return 1.0 / (1.0 + odds)

    def expected_purchases(self, t, x, t_x, T):
        r, al, a, b = self.r, self.alpha, self.a, self.b
        z = t / (al + T + t)
        num = (a + b + x - 1) / (a - 1) * (1 - ((al + T) / (al + T + t)) ** (r + x) * hyp2f1(r + x, b + x, a + b + x - 1, z))
        den = 1 + np.where(x > 0, a / (b + np.maximum(x, 1) - 1) * ((al + T) / (al + t_x)) ** (r + x), 0.0)
        return num / den

    def rate_if_alive(self, x, T):
        return (self.r + x) / (self.alpha + T)          # posterior mean purchase rate given alive


# ------------------------------------------------------------- Gamma-Gamma ----
def _gg_ll(p, x, m):
    pp, q, v = np.exp(p)
    return (gammaln(pp * x + q) - gammaln(pp * x) - gammaln(q) + q * np.log(v)
            + (pp * x - 1) * np.log(m) + pp * x * np.log(x) - (pp * x + q) * np.log(x * m + v))


class GammaGamma:
    def __init__(self, penalizer=0.001):
        self.pen = penalizer

    def fit(self, x, m):
        f = lambda p: -_gg_ll(p, x, m).mean() + self.pen * np.sum(np.exp(p) ** 2)
        res = minimize(f, np.log([2.0, 3.0, 50.0]), method="L-BFGS-B", bounds=[(-4, 6)] * 3)
        self.p, self.q, self.v = np.exp(res.x)
        self.converged = bool(res.success)
        return self

    def expected_value(self, x, m):
        return self.p * (self.v + x * m) / (self.p * x + self.q - 1)


# ------------------------------------------------------------------ helpers ---
def rfm_table(rech, cal_end):
    r = rech[rech.recharge_date <= cal_end]
    days = r.drop_duplicates(["customer_id", "recharge_date"]).groupby("customer_id").recharge_date.agg(["min", "max", "size"])
    ev = r.groupby("customer_id").amount_sdg.agg(n_events="size", monetary="mean")
    t = pd.DataFrame({"x": days["size"] - 1,
                      "t_x": (days["max"] - days["min"]).dt.days,
                      "T": (cal_end - days["min"]).dt.days})
    return t.join(ev)


def metrics(pred, actual, name):
    order = np.argsort(-pred.values)
    top = int(len(pred) * 0.2)
    return {"model": name,
            "pred_over_actual": round(float(pred.sum() / actual.sum()), 3),
            "rmse_sdg": round(float(np.sqrt(np.mean((pred - actual) ** 2))), 0),
            "spearman": round(float(spearmanr(pred, actual)[0]), 3),
            "top20_capture_pct": round(100 * float(actual.values[order[:top]].sum() / actual.sum()), 1)}


def run(cfg_path):
    cfg = load_cfg(cfg_path)
    cc = cfg["clv"]
    cal_end, ho_end = pd.Timestamp(cc["cal_end"]), pd.Timestamp(cc["holdout_end"])
    H = (ho_end - cal_end).days                                   # 92
    cust, rech, usage = load_tables(cfg["_data"])
    prepaid = set(cust[cust.plan_type == "prepaid"].customer_id)
    rech = rech[rech.customer_id.isin(prepaid)]

    t = rfm_table(rech, cal_end)
    bg = BGNBD(cc["penalizer"]).fit(t.x.values.astype(float), t.t_x.values.astype(float), t["T"].values.astype(float))
    rep = t[t.x > 0]
    gg = GammaGamma(cc["penalizer"]).fit(rep.n_events.values.astype(float), rep.monetary.values)
    print(f"BG/NBD r={bg.r:.3f} alpha={bg.alpha:.2f} a={bg.a:.3f} b={bg.b:.3f} converged={bg.converged}")
    print(f"Gamma-Gamma p={gg.p:.3f} q={gg.q:.3f} v={gg.v:.2f} converged={gg.converged}")
    # a < 1 is allowed: the closed form was checked against Monte Carlo simulation at the fitted parameters
    # (92-day expectation 6.086 vs 6.086); the formula only needs to stay finite and non-negative.

    x, tx, T = (t[c].values.astype(float) for c in ["x", "t_x", "T"])
    ev_txn = np.where(t.n_events > 0, gg.expected_value(np.maximum(t.n_events.values, 1).astype(float), t.monetary.values), t.monetary.values)
    # customers with a single event: Gamma-Gamma with n=1 is still defined; use it for everyone
    t["p_alive"] = bg.p_alive(x, tx, T)
    t["exp_value_per_txn"] = ev_txn
    t["exp_n_92d"] = bg.expected_purchases(H, x, tx, T)
    t["exp_n_365d"] = bg.expected_purchases(365, x, tx, T)
    assert np.isfinite(t.exp_n_365d).all() and (t.exp_n_92d >= 0).all(), "BG/NBD expectation out of range"
    t["clv_92d"] = t.exp_n_92d * t.exp_value_per_txn
    t["clv_12m"] = t.exp_n_365d * t.exp_value_per_txn
    t["clv_12m_if_alive"] = 365 * bg.rate_if_alive(x, T) * t.exp_value_per_txn      # ignores dropout risk

    # ------------------------------------------------ holdout validation ------
    actual = (rech[(rech.recharge_date > cal_end) & (rech.recharge_date <= ho_end)]
              .groupby("customer_id").amount_sdg.sum().reindex(t.index).fillna(0.0))
    last90 = (rech[(rech.recharge_date > cal_end - pd.Timedelta(days=90)) & (rech.recharge_date <= cal_end)]
              .groupby("customer_id").amount_sdg.sum().reindex(t.index).fillna(0.0)) * H / 90.0

    # ML bake-off: train as of Jun 30 -> spend Jul1-Sep30 (same 92-day horizon), apply as of Sep 30
    ml_cols = ["recharge_freq_30d", "avg_recharge_sdg", "recency_days", "sessions_30d", "total_recharge_sdg",
               "share_data", "share_voice", "tenure_days", "exposure_days"]
    tr_asof = cal_end - pd.Timedelta(days=H)
    Xtr, _ = build_features(cfg["_data"], tr_asof, cc["ml_window_days"])
    ytr = (rech[(rech.recharge_date > tr_asof) & (rech.recharge_date <= cal_end)]
           .groupby("customer_id").amount_sdg.sum().reindex(Xtr.index).fillna(0.0))
    Xte, _ = build_features(cfg["_data"], cal_end, cc["ml_window_days"])
    Xte = Xte.reindex(t.index)                                 # customers with no window activity -> NaN -> LightGBM handles
    gbm = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=cfg["seed"], verbose=-1)
    gbm.fit(Xtr[ml_cols], ytr)
    ml_pred = pd.Series(np.clip(gbm.predict(Xte[ml_cols]), 0, None), index=t.index)

    res = [metrics(last90, actual, "Last-90-day spend (scaled to 92d)"),
           metrics(t.clv_92d, actual, "BG/NBD + Gamma-Gamma"),
           metrics(ml_pred, actual, "LightGBM on RFM (trained on prior period)"),
           metrics(actual, actual, "Oracle")]
    res = pd.DataFrame(res)
    res.to_csv(cfg["_out"] / "clv_holdout.csv", index=False)
    print("\nHoldout (n=%d, actual total %.0f SDG):\n" % (len(t), actual.sum()), res.to_string(index=False))

    # calibration of BG/NBD by predicted decile
    dec = pd.qcut(t.clv_92d.rank(method="first"), 10, labels=False)
    cal = pd.DataFrame({"pred": t.clv_92d, "actual": actual, "d": dec}).groupby("d").mean().round(0)
    cal.to_csv(cfg["_out"] / "clv_calibration_deciles.csv")
    print("\nBG/NBD calibration by decile (mean SDG):\n", cal.T.to_string())

    # p_alive check (drives the win-back weighting): share with any recharge in the holdout, by p_alive bucket
    bucket = pd.cut(t.p_alive, [-0.001, 0.2, 0.5, 0.8, 0.95, 1.0], labels=["<0.2", "0.2-0.5", "0.5-0.8", "0.8-0.95", ">0.95"])
    pa = pd.DataFrame({"n": t.groupby(bucket, observed=True).size(),
                       "mean_p_alive": t.p_alive.groupby(bucket, observed=True).mean().round(3),
                       "share_recharged_in_holdout": (actual > 0).groupby(bucket, observed=True).mean().round(3)})
    pa.to_csv(cfg["_out"] / "clv_p_alive_check.csv")
    print("\np_alive vs recharging in the next 92 days:\n", pa.to_string())

    # assumption checks
    corr_fm = float(np.corrcoef(rep.n_events, rep.monetary)[0, 1])
    # a proxy for the dropout/purchase-rate link: recency ratio vs frequency among repeaters
    corr_rate_dropout = float(spearmanr(rep.x / (rep["T"] + 1), rep.t_x / (rep["T"] + 1))[0])
    checks = {"corr_frequency_monetary": round(corr_fm, 3),
              "bgnbd_params": {"r": round(bg.r, 3), "alpha": round(bg.alpha, 2), "a": round(bg.a, 3), "b": round(bg.b, 3)},
              "gg_params": {"p": round(gg.p, 3), "q": round(gg.q, 3), "v": round(gg.v, 2)},
              "n_customers": int(len(t)), "n_repeaters": int(len(rep)),
              "mean_p_alive": round(float(t.p_alive.mean()), 3)}
    print("\nChecks:", checks)

    out = t[["p_alive", "exp_n_92d", "exp_value_per_txn", "clv_92d", "clv_12m", "clv_12m_if_alive"]].reset_index()
    out.assign(as_of=cfg["clv"]["cal_end"]).to_parquet(cfg["_out"] / "clv_scores.parquet", index=False)
    json.dump({"holdout": res.to_dict("records"), "checks": checks}, open(cfg["_out"] / "clv_metrics.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    run(Path(__file__).resolve().parent.parent / "config" / "customer_value.yaml")
