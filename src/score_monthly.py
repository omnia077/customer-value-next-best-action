"""Monthly customer-value pipeline. The same step functions are used by the CLI here and by the Airflow DAG
(src/customer_value_dag.py), so the DAG and a local run cannot drift apart.

Steps: validate inputs -> segment -> CLV -> propensity -> uplift -> churn scores -> actions -> persona KPIs ->
       drift check -> MLflow log

Run:  python src/score_monthly.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import actions, churn_proxy, clv, persona_kpis, propensity, segment, uplift  # noqa: E402
from features import ASOF_FEATURES, asof_features, load_tables  # noqa: E402

CFG = Path(__file__).resolve().parent.parent / "config" / "customer_value.yaml"


def psi(expected, actual, bins=10):
    """Population stability index; <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift."""
    e, a = pd.Series(expected).dropna(), pd.Series(actual).dropna()
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def step_validate_inputs():
    cfg = segment.load_cfg(CFG)
    sftp = cfg["_data"] / "sftp_drop"
    issues = []
    for t, keys in {"customers": ["customer_id"], "recharge_records": ["customer_id", "recharge_date", "amount_sdg"],
                    "usage_events": ["customer_id", "event_date", "usage_type"]}.items():
        p = sftp / f"{t}.parquet"
        if not p.exists():
            issues.append(f"missing {t}")
            continue
        df = pd.read_parquet(p, columns=keys)
        if df[keys].isna().any().any():
            issues.append(f"{t}: nulls in key columns")
    h = cfg["_data"] / "campaigns" / "campaign_history.parquet"
    if not h.exists():
        issues.append("missing campaign_history")
    else:
        hh = pd.read_parquet(h)
        if hh.duplicated(["campaign_id", "customer_id"]).any():
            issues.append("campaign_history: duplicate customer within a campaign")
        if not hh.treated.isin([0, 1]).all():
            issues.append("campaign_history: treated not binary")
    if issues:
        raise ValueError("input validation failed: " + "; ".join(issues))
    print("inputs validated")


def step_drift_check():
    """PSI of input features (30 Sep vs the previous month-end) and, if a baseline exists, of the score distributions."""
    cfg = segment.load_cfg(CFG)
    out = cfg["_out"]
    tables = load_tables(cfg["_data"])
    prev = asof_features(*tables, "2024-08-31")[ASOF_FEATURES]
    cur = asof_features(*tables, cfg["segmentation"]["as_of"])[ASOF_FEATURES]
    rows = [{"metric": f"feature:{c}", "psi": round(psi(prev[c], cur[c]), 4)} for c in ASOF_FEATURES]
    act = pd.read_parquet(out / "customer_actions.parquet")
    base_p = out / "baseline_scores.parquet"
    if base_p.exists():
        base = pd.read_parquet(base_p)
        for c in ["clv_12m", "churn_prob_30d", "uplift", "p_recharge500_next_month"]:
            rows.append({"metric": f"score:{c}", "psi": round(psi(base[c], act[c]), 4)})
        rows.append({"metric": "segment_shares",
                     "psi": round(psi(base.segment_id.astype(float), act.segment_id.astype(float), bins=act.segment_id.nunique()), 4)})
    else:
        act.to_parquet(base_p, index=False)          # first run defines the reference distribution
        print("baseline score distribution saved:", base_p.name)
    res = pd.DataFrame(rows)
    res["status"] = np.where(res.psi > 0.25, "ALERT", np.where(res.psi > 0.1, "WATCH", "OK"))
    res.to_csv(out / "drift_check.csv", index=False)
    print(res.to_string(index=False))
    return res


def step_mlflow_log():
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed: skipping")
        return
    cfg = segment.load_cfg(CFG)
    out = cfg["_out"]
    mlflow.set_tracking_uri(f"sqlite:///{(cfg['_root'] / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name("customer_value_monthly") is None:
        mlflow.create_experiment("customer_value_monthly", artifact_location=(cfg["_root"] / "mlartifacts").as_uri())
    mlflow.set_experiment("customer_value_monthly")
    with mlflow.start_run(run_name=f"as_of_{cfg['segmentation']['as_of']}"):
        mlflow.log_params({"as_of": cfg["segmentation"]["as_of"], "k_final": cfg["segmentation"]["k_final"],
                           **{f"assump_{k}": v for k, v in cfg["actions"].items()}})
        seg = json.load(open(out / "segmentation_metrics.json"))
        mlflow.log_metrics({"seg_silhouette": seg["silhouette"], "seg_recovery_ari": seg["recovery_ari"]})
        for f, prefix in [("propensity_metrics.json", "prop"), ("churn_proxy_metrics.json", "churn")]:
            m = json.load(open(out / f))
            m = m.get("test", {}).get("LightGBM", m)
            mlflow.log_metrics({f"{prefix}_{k}": v for k, v in m.items() if isinstance(v, (int, float))})
        up = json.load(open(out / "uplift_metrics.json"))
        mlflow.log_param("uplift_selected_learner", up["selected_learner"])
        for r in up["evaluation"]:
            if r["policy"].startswith("Alt"):
                continue
            key = r["policy"].split(" ")[0].lower()
            mlflow.log_metrics({f"uplift_{key}_spearman": r["spearman_vs_true_uplift"], f"uplift_{key}_qini": r["qini_coef"]})
        for f in ["clv_holdout.csv", "drift_check.csv", "persona_cards.csv", "k_selection_active90.csv"]:
            mlflow.log_artifact(str(out / f))
    print("logged to MLflow (sqlite):", cfg["_root"] / "mlflow.db")


STEPS = [("validate_inputs", step_validate_inputs),
         ("segment", lambda: segment.fit_final(segment.load_cfg(CFG))),
         ("clv", lambda: clv.run(CFG)),
         ("propensity", lambda: propensity.run(CFG)),
         ("uplift", lambda: uplift.run(CFG)),
         ("churn_scores", lambda: churn_proxy.run(CFG)),     # DAG waits on Project 1's churn DAG before this step
         ("actions", lambda: actions.run(CFG)),
         ("persona_kpis", lambda: persona_kpis.run(CFG)),
         ("drift_check", step_drift_check),
         ("mlflow_log", step_mlflow_log)]


if __name__ == "__main__":
    only = sys.argv[1:]                    # e.g. python src/score_monthly.py drift_check mlflow_log
    for name, fn in STEPS:
        if only and name not in only:
            continue
        print(f"\n=== {name} ===")
        fn()
