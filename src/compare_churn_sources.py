"""Compare the stand-in churn scores with Project 1's real ones, on the same subscribers and the same outcome.

Needs Project 1 scored on THIS dataset (see docs/project1_integration.md) and the salt in CHURN_HASH_SALT. Pass the
score file on the command line (--scores/--key/--prob) or set churn.external_scores_path in the config.
This script never overwrites the pipeline's own outputs (outputs/churn_scores.parquet etc.).

Outcome (validation only, read from ground truth): did the subscriber churn in the calendar month after the
as-of date (October)? Project 1 defines its label the same way.

Run:  python src/compare_churn_sources.py
Out:  outputs/project1_comparison.json, project1_calibration.csv, project1_actions.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import actions  # noqa: E402
import churn_proxy  # noqa: E402
from features import load_tables  # noqa: E402
from segment import load_cfg  # noqa: E402


def october_label(cfg, ids):
    as_of = pd.Timestamp(cfg["segmentation"]["as_of"])
    end = as_of + pd.offsets.MonthEnd(1)
    lab = pd.read_parquet(cfg["_data"] / "ground_truth" / "churn_labels.parquet").set_index("customer_id")
    cd = pd.to_datetime(lab.churn_date, errors="coerce").reindex(ids)
    return ((cd > as_of) & (cd <= end)).astype(int).values


def metrics(y, p):
    top = int(len(y) * 0.1)
    return {"roc_auc": round(float(roc_auc_score(y, p)), 3), "pr_auc": round(float(average_precision_score(y, p)), 3),
            "top_decile_lift": round(float(y[np.argsort(-p, kind="stable")[:top]].mean() / y.mean()), 2),
            "mean_predicted": round(float(np.mean(p)), 4), "observed_rate": round(float(y.mean()), 4),
            "calibration_ratio": round(float(np.mean(p) / y.mean()), 2)}


def run(cfg_path, scores=None, key=None, prob=None):
    cfg = load_cfg(cfg_path)
    for k, v in (("external_scores_path", scores), ("key_column", key), ("prob_column", prob)):
        if v:
            cfg["churn"][k] = v
    tables = load_tables(cfg["_data"])
    p1 = churn_proxy.external_scores(cfg, tables)
    cached = cfg["_out"] / "churn_scores.parquet"
    st = pd.read_parquet(cached) if cached.exists() and (pd.read_parquet(cached, columns=["source"]).source == "stand-in").all() \
        else churn_proxy.standin_scores(cfg, tables)[0]
    m = p1.merge(st[["customer_id", "churn_prob_30d"]], on="customer_id", suffixes=("_p1", "_standin"))
    y = october_label(cfg, m.customer_id)
    a, b = m.churn_prob_30d_p1.values, m.churn_prob_30d_standin.values
    out = {"n_common": int(len(m)), "n_churners": int(y.sum()), "spearman_p1_vs_standin": round(float(spearmanr(a, b)[0]), 3),
           "project1": metrics(y, a), "standin": metrics(y, b)}

    # paired bootstrap on the AUC gap (few churners: the gap is noisy)
    rng = np.random.default_rng(cfg["seed"])
    gaps = []
    for _ in range(300):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() > 0:
            gaps.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    out["auc_gap_p1_minus_standin"] = {"mean": round(float(np.mean(gaps)), 4),
                                       "ci95": [round(float(np.percentile(gaps, 2.5)), 4), round(float(np.percentile(gaps, 97.5)), 4)]}

    # calibration by decile of each score
    rows = []
    for nm, p in (("project1", a), ("standin", b)):
        dec = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
        g = pd.DataFrame({"p": p, "y": y, "d": dec}).groupby("d").agg(mean_predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
        rows.append(g.round(4).reset_index().assign(source=nm))
    pd.concat(rows).to_csv(cfg["_out"] / "project1_calibration.csv", index=False)

    # decision layer under each source; then check who the retention action actually picks
    a_cfg = cfg["actions"]
    plan = []
    lab = pd.Series(october_label(cfg, m.customer_id), index=m.customer_id.values)
    for nm, scores in (("stand-in", st), ("project1", p1)):
        d = actions.allocate(actions.load_inputs(cfg, churn=scores), a_cfg)
        vc = d.action.value_counts()
        prot = d.index[d.action == "protect"]
        yl = lab.reindex(prot).dropna()
        plan.append({"source": nm, "contacted": int((d.action != "maintain").sum()), "grow": int(vc.get("grow", 0)),
                     "protect": int(vc.get("protect", 0)), "win_back": int(vc.get("win_back", 0)),
                     "total_ev_sdg": round(float(d.ev.sum())), "protect_ev_sdg": round(float(d.ev[d.action == "protect"].sum())),
                     "protect_group_october_churn_rate_pct": round(100 * float(yl.mean()), 1) if len(yl) else None,
                     "protect_group_scored_n": int(len(yl))})
    plan = pd.DataFrame(plan)
    plan.to_csv(cfg["_out"] / "project1_actions.csv", index=False)
    out["decision_layer"] = plan.to_dict("records")
    json.dump(out, open(cfg["_out"] / "project1_comparison.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "decision_layer"}, indent=2))
    print(plan.to_string(index=False))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compare stand-in churn scores with Project 1's (CHURN_HASH_SALT must be set for sub_hash keys)")
    ap.add_argument("--scores", help="Project 1 churn_scores.parquet (default: churn.external_scores_path from config)")
    ap.add_argument("--key", default=None, help="key column, e.g. sub_hash")
    ap.add_argument("--prob", default=None, help="probability column, e.g. probability")
    a = ap.parse_args()
    run(Path(__file__).resolve().parent.parent / "config" / "customer_value.yaml", a.scores, a.key, a.prob)
