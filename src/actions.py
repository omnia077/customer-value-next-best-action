"""Decision layer: one next-best-action per subscriber under a single contact budget.

Actions and expected net value (SDG), all cost/rate parameters are ASSUMPTIONS from config:
  grow      = uplift x avg_added_recharge - offer_cost                       (uplift model; validated vs truth)
  protect   = churn_prob x save_rate x clv_12m_if_alive - retention_cost     (save_rate is an assumption)
  win_back  = winback_rate x clv_12m_if_alive - winback_cost                 (only recency >= threshold; assumption)
  maintain  = 0 (automated messages / no paid contact)
CLV is never multiplied by churn probability to build a "risk-adjusted CLV": BG/NBD CLV already prices dropout, so the
protect action uses the if-alive value and the churn probability supplies the risk side once.

Run:  python src/actions.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features  # noqa: E402
from segment import load_cfg  # noqa: E402


def load_inputs(cfg, churn=None):
    """`churn`: optional DataFrame (customer_id, churn_prob_30d) that replaces outputs/churn_scores.parquet."""
    out = cfg["_out"]
    seg = pd.read_parquet(out / "segment_assignments.parquet").set_index("customer_id")
    df, _ = build_features(cfg["_data"], cfg["segmentation"]["as_of"], cfg["segmentation"]["window_days"],
                           cfg["segmentation"]["active_within_days"])
    df = df.join(seg[["segment_id", "persona"]], how="inner")
    clv = pd.read_parquet(out / "clv_scores.parquet").set_index("customer_id")
    upl = pd.read_parquet(out / "uplift_scores.parquet").set_index("customer_id")
    chn = (churn if churn is not None else pd.read_parquet(out / "churn_scores.parquet")).set_index("customer_id")
    prp = pd.read_parquet(out / "propensity_scores.parquet").set_index("customer_id")
    df = (df.join(clv[["p_alive", "clv_92d", "clv_12m", "clv_12m_if_alive"]])
            .join(upl[["uplift"]]).join(chn[["churn_prob_30d"]]).join(prp[["p_recharge500_next_month"]]))
    df[["clv_92d", "clv_12m", "clv_12m_if_alive"]] = df[["clv_92d", "clv_12m", "clv_12m_if_alive"]].fillna(0)
    return df


def allocate(df, a):
    """Best positive-EV action per subscriber, then keep the top max_contact_share by EV."""
    d = df.copy()
    d["ev_grow"] = (d.uplift * a["avg_added_recharge_sdg"] - a["offer_cost_sdg"]).where(d.uplift.notna())
    d["ev_protect"] = (d.churn_prob_30d * a["save_rate"] * d.clv_12m_if_alive - a["retention_cost_sdg"]).where(d.churn_prob_30d.notna())
    # an offer only adds value if the line has really lapsed: weight by (1 - p_alive) from BG/NBD
    d["ev_win_back"] = (a["winback_rate"] * (1 - d.p_alive.fillna(0)) * d.clv_12m_if_alive
                        - a["winback_cost_sdg"]).where(d.recency_days >= a["winback_min_recency_days"])
    evs = d[["ev_grow", "ev_protect", "ev_win_back"]].fillna(-np.inf)
    best = evs.max(axis=1)
    d["best_action"] = evs.idxmax(axis=1).str.replace("ev_", "", regex=False)
    d["best_ev"] = best
    eligible = best > 0
    n_budget = int(len(d) * a["max_contact_share"])
    order = best.where(eligible).sort_values(ascending=False).dropna().index[:n_budget]
    d["action"] = "maintain"
    d.loc[order, "action"] = d.loc[order, "best_action"]
    d["ev"] = np.where(d.action == "maintain", 0.0, d.best_ev)
    return d


def quadrants(d, clv_top, risk_top):
    hi_clv = d.clv_12m >= d.clv_12m.quantile(1 - clv_top)
    hi_risk = d.churn_prob_30d >= d.churn_prob_30d.quantile(1 - risk_top)
    return np.select([hi_clv & ~hi_risk, hi_clv & hi_risk, ~hi_clv & hi_risk],
                     ["Grow", "Protect", "Don't overspend"], "Maintain")


def evaluate_grow_policy(cfg):
    """June holdout: net value of contacting by different rules, against the simulator's true uplift."""
    a = cfg["actions"]
    j = pd.read_parquet(cfg["_out"] / "uplift_june_test_scored.parquet")
    val = lambda idx: (j.true_uplift.values[idx] * a["avg_added_recharge_sdg"] - a["offer_cost_sdg"])
    rule = np.where((j.uplift_hat * a["avg_added_recharge_sdg"] - a["offer_cost_sdg"]) > 0)[0]
    n = int(len(j) * a["max_contact_share"])                      # budget-matched comparison
    pols = {"Contact everyone": np.arange(len(j)),
            f"Uplift model, EV>0 rule": rule,
            f"Uplift model, top {n:,} (budget)": np.argsort(-j.uplift_hat.values)[:n],
            f"Propensity, top {n:,} (budget)": np.argsort(-j.prop_hat.values)[:n],
            f"Oracle, top {n:,} (budget)": np.argsort(-j.true_uplift.values)[:n]}
    rows = [{"policy": nm, "contacted": len(idx),
             "expected_incremental_recharges": round(float(j.true_uplift.values[idx].sum()), 0),
             "net_value_sdg": round(float(val(idx).sum()), 0)} for nm, idx in pols.items()]
    # random at the same budget: exact expectation (mean over subscribers x n)
    rows.insert(4, {"policy": f"Random, {n:,} (budget, expectation)", "contacted": n,
                    "expected_incremental_recharges": round(float(j.true_uplift.mean() * n), 0),
                    "net_value_sdg": round(float(val(np.arange(len(j))).mean() * n), 0)})
    return pd.DataFrame(rows)


def grow_sweeps(cfg):
    """When does uplift targeting beat 'contact everyone'? Net value on the June holdout (true uplift) as the
    offer cost and the contact budget vary. Returns (cost sweep, budget sweep, break-even offer cost)."""
    a = cfg["actions"]
    V = a["avg_added_recharge_sdg"]
    j = pd.read_parquet(cfg["_out"] / "uplift_june_test_scored.parquet")
    tu, N = j.true_uplift.values, len(j)
    net = lambda idx, c: float(tu[idx].sum() * V - c * len(idx))
    by_up, by_pr, by_or = (np.argsort(-j[c].values) for c in ("uplift_hat", "prop_hat", "true_uplift"))
    n = int(N * a["max_contact_share"])
    rows = []
    for c in [5, 10, 25, 50, 75, 100, 150, 200, 300]:
        rule = np.where(j.uplift_hat.values * V - c > 0)[0]
        rows.append({"offer_cost_sdg": c, "contact_everyone": round(net(np.arange(N), c)),
                     "uplift_ev_rule": round(net(rule, c)), "uplift_rule_contacted": len(rule),
                     f"uplift_top{int(100*a['max_contact_share'])}pct": round(net(by_up[:n], c)),
                     f"propensity_top{int(100*a['max_contact_share'])}pct": round(net(by_pr[:n], c)),
                     f"random_{int(100*a['max_contact_share'])}pct": round(float(tu.mean() * V * n - c * n)),
                     "oracle_ev_rule": round(net(np.where(tu * V - c > 0)[0], c))})
    cost = pd.DataFrame(rows)
    # break-even: first offer cost (step 1 SDG) where the uplift EV rule beats contacting everyone
    be = next((c for c in range(1, 400) if net(np.where(j.uplift_hat.values * V - c > 0)[0], c) > net(np.arange(N), c)), None)
    c0 = a["offer_cost_sdg"]
    bud = pd.DataFrame([{"contact_share": s, "uplift": round(net(by_up[:int(N * s)], c0)),
                         "propensity": round(net(by_pr[:int(N * s)], c0)),
                         "random": round(float(tu.mean() * V * int(N * s) - c0 * int(N * s))),
                         "oracle": round(net(by_or[:int(N * s)], c0))} for s in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)])
    return cost, bud, be


def breakeven_rates(d, a):
    """How wrong can the assumed (unvalidated) rates be? Per allocated subscriber, the rate at which the action's
    expected value is exactly zero. `robust_share` = share of allocated subscribers still positive if the true rate is half the assumed one."""
    rows = []
    for act, rate_key, denom in [("protect", "save_rate", d.churn_prob_30d * d.clv_12m_if_alive),
                                 ("win_back", "winback_rate", (1 - d.p_alive.fillna(0)) * d.clv_12m_if_alive)]:
        cost = a["retention_cost_sdg"] if act == "protect" else a["winback_cost_sdg"]
        m = d.action == act
        be = (cost / denom[m]).replace([np.inf], np.nan)
        rows.append({"action": act, "allocated": int(m.sum()), "assumed_rate": a[rate_key],
                     "median_breakeven_rate": round(float(be.median()), 3), "p90_breakeven_rate": round(float(be.quantile(0.9)), 3),
                     "robust_share_if_rate_halves": round(float((be < a[rate_key] / 2).mean()), 3),
                     "ev_sdg": round(float(d.ev[m].sum()))})
    return pd.DataFrame(rows)


def run(cfg_path):
    cfg = load_cfg(cfg_path)
    a = cfg["actions"]
    df = load_inputs(cfg)
    d = allocate(df, a)
    d["quadrant"] = quadrants(d, a["clv_top_share"], a["risk_top_share"])
    print(f"population {len(d):,} | contacted {int((d.action != 'maintain').sum()):,} | total EV {d.ev.sum():,.0f} SDG")
    print(d.action.value_counts().to_string())
    print("\nquadrant x action:\n", pd.crosstab(d.quadrant, d.action).to_string())

    # persona cards
    g = d.groupby("persona")
    cards = pd.DataFrame({
        "subscribers": g.size(),
        "pct_subs": (100 * g.size() / len(d)).round(1),
        "pct_revenue_180d": (100 * g.total_recharge_sdg.sum() / d.total_recharge_sdg.sum()).round(1),
        "avg_clv_12m": g.clv_12m.mean().round(0),
        "avg_churn_prob_30d": (100 * g.churn_prob_30d.mean()).round(1),
        "avg_propensity_500": (100 * g.p_recharge500_next_month.mean()).round(1),
        "avg_uplift_pts": (100 * g.uplift.mean()).round(1),
        "pct_contacted": (100 * g.apply(lambda x: (x.action != "maintain").mean(), include_groups=False)).round(1),
        "expected_net_value_sdg": g.ev.sum().round(0)})
    for act in ("grow", "protect", "win_back"):
        cards[f"pct_{act}"] = (100 * g.apply(lambda x, act=act: (x.action == act).mean(), include_groups=False)).round(1)
    cards["recommended_play"] = cards[["pct_grow", "pct_protect", "pct_win_back"]].idxmax(axis=1).str.replace("pct_", "", regex=False)
    cards.to_csv(cfg["_out"] / "persona_cards.csv")
    print("\n", cards.to_string())

    # sensitivity 1: matrix cut-offs -> quadrant sizes and allocated EV
    rows = []
    for c in (0.20, 0.30, 0.40):
        for r in (0.05, 0.10, 0.15):
            q = quadrants(d, c, r)
            for name in ("Grow", "Protect", "Maintain", "Don't overspend"):
                m = q == name
                rows.append({"clv_top_share": c, "risk_top_share": r, "quadrant": name, "subscribers": int(m.sum()),
                             "allocated_ev_sdg": round(float(d.ev[m].sum()), 0)})
    pd.DataFrame(rows).to_csv(cfg["_out"] / "sensitivity_matrix_cutoffs.csv", index=False)
    # sensitivity 2: assumptions -> total EV and action mix
    rows = []
    for oc in (15, 25, 40):
        for sr in (0.10, 0.15, 0.25):
            x = allocate(df, {**a, "offer_cost_sdg": oc, "save_rate": sr})
            vc = x.action.value_counts()
            rows.append({"offer_cost_sdg": oc, "save_rate": sr, "total_ev_sdg": round(float(x.ev.sum()), 0),
                         **{k: int(vc.get(k, 0)) for k in ("grow", "protect", "win_back")}})
    sens = pd.DataFrame(rows)
    sens.to_csv(cfg["_out"] / "sensitivity_assumptions.csv", index=False)
    print("\nassumption sensitivity:\n", sens.to_string(index=False))

    # Grow policy evaluated against truth (June holdout)
    gp = evaluate_grow_policy(cfg)
    gp.to_csv(cfg["_out"] / "grow_policy_evaluation.csv", index=False)
    print("\nGrow policy on June holdout (true uplift):\n", gp.to_string(index=False))

    # when does targeting beat contact-everyone? (cost and budget sweeps on the June holdout)
    cost_sw, bud_sw, be_cost = grow_sweeps(cfg)
    cost_sw.to_csv(cfg["_out"] / "grow_sweep_offer_cost.csv", index=False)
    bud_sw.to_csv(cfg["_out"] / "grow_sweep_budget.csv", index=False)
    print(f"\nGrow policy vs offer cost (June, true uplift):\n{cost_sw.to_string(index=False)}"
          f"\nbreak-even offer cost, uplift rule beats contact-everyone from: {be_cost} SDG")
    print("\nbudget sweep at assumed cost:\n", bud_sw.to_string(index=False))
    be = breakeven_rates(d, a)
    be.to_csv(cfg["_out"] / "assumption_breakeven.csv", index=False)
    print("\nbreak-even rates for unvalidated actions:\n", be.to_string(index=False))
    ev_by = d.groupby("action").ev.sum().round(0).to_dict()

    # fairness audit: contact rate by age band and region (age/gender are NOT model inputs)
    cu = pd.read_parquet(cfg["_data"] / "sftp_drop" / "customers.parquet", columns=["customer_id", "age", "region"]).set_index("customer_id")
    d = d.join(cu)
    d["age_band"] = pd.cut(d.age, [17, 29, 44, 59, 120], labels=["18-29", "30-44", "45-59", "60+"])
    aud = pd.concat([d.groupby("age_band", observed=True).apply(lambda x: (x.action != "maintain").mean(), include_groups=False).rename("contact_rate").to_frame().assign(dim="age_band"),
                     d.groupby("region").apply(lambda x: (x.action != "maintain").mean(), include_groups=False).rename("contact_rate").to_frame().assign(dim="region")])
    aud["contact_rate"] = (100 * aud.contact_rate).round(1)
    aud.to_csv(cfg["_out"] / "fairness_audit.csv")
    print("\nfairness audit (contact rate %):\n", aud.to_string())

    keep = ["persona", "segment_id", "quadrant", "action", "ev", "best_action", "clv_92d", "clv_12m", "clv_12m_if_alive",
            "p_alive", "churn_prob_30d", "p_recharge500_next_month", "uplift", "recency_days"]
    d[keep].reset_index().assign(as_of=cfg["segmentation"]["as_of"]).to_parquet(cfg["_out"] / "customer_actions.parquet", index=False)
    json.dump({"population": int(len(d)), "contacted": int((d.action != "maintain").sum()),
               "total_ev_sdg": round(float(d.ev.sum()), 0), "action_counts": d.action.value_counts().to_dict(),
               "ev_by_action_sdg": ev_by,
               "share_ev_validated_vs_truth": round(float(ev_by.get("grow", 0) / max(d.ev.sum(), 1)), 3),
               "breakeven_offer_cost_uplift_rule_vs_everyone": be_cost,
               "grow_policy": gp.to_dict("records")},open(cfg["_out"] / "actions_metrics.json", "w"), indent=2)
    return d


if __name__ == "__main__":
    run(Path(__file__).resolve().parent.parent / "config" / "customer_value.yaml")
