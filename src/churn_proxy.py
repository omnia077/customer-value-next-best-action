"""STAND-IN for Project 1's churn scores.

Project 1's ML model is not in this repo yet. This module trains a simple churn model (as-of features -> churn in
the next 30 days) so the decision layer has a risk input. It is deliberately simpler than the Project 1 model.
If `churn.external_scores_path` in the config points to a parquet with columns customer_id, churn_prob_30d
(Project 1's real scores as of the same date), those are used instead and this model is skipped.

The label comes from ground_truth/churn_labels.parquet, which is the label source Project 1 also uses.
Features are as-of only.

Run:  python src/churn_proxy.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import ASOF_FEATURES, asof_features, load_tables  # noqa: E402
from segment import load_cfg  # noqa: E402


def churn_xy(tables, churn_date, cutoff, horizon=30):
    cust, rech, usage = tables
    X = asof_features(cust, rech, usage, cutoff)
    end = pd.Timestamp(cutoff) + pd.Timedelta(days=horizon)
    d = churn_date.reindex(X.index)
    y = ((d > pd.Timestamp(cutoff)) & (d <= end)).astype(int)
    return X[ASOF_FEATURES], y


def run(cfg_path):
    cfg = load_cfg(cfg_path)
    cc = cfg["churn"]
    tables = load_tables(cfg["_data"])
    ext = cc.get("external_scores_path")
    if ext:
        if not Path(ext).exists():
            raise FileNotFoundError(f"churn.external_scores_path is set but missing: {ext}")
        from ids import msisdn_to_customer_id, standardize_external_scores
        raw = pd.read_csv(ext) if str(ext).endswith(".csv") else pd.read_parquet(ext)
        key = cc.get("key_column", "customer_id")
        id_map = msisdn_to_customer_id(cfg["_data"] / "sftp_drop" / "customers.parquet") if key == "sub_hash" else None
        pop = asof_features(*tables, cfg["segmentation"]["as_of"]).index
        std = standardize_external_scores(raw, key, cc.get("prob_column", "churn_prob_30d"), id_map, pop, cc.get("min_coverage", 0.9),
                                          expected_cutoff=cfg["segmentation"]["as_of"])
        std.assign(as_of=cfg["segmentation"]["as_of"], source="project1").to_parquet(cfg["_out"] / "churn_scores.parquet", index=False)
        print(f"using Project 1 churn scores from {ext}: {len(std):,} subscribers")
        return None
    lab = pd.read_parquet(cfg["_data"] / "ground_truth" / "churn_labels.parquet").set_index("customer_id")
    cd = pd.to_datetime(lab.churn_date, errors="coerce")
    P = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=cfg["seed"])

    def stack(cuts):
        xy = [churn_xy(tables, cd, c) for c in cuts]
        return pd.concat([a for a, _ in xy]), pd.concat([b for _, b in xy])

    Xtr, ytr = stack(cc["train_cutoffs"])
    Xte, yte = stack(cc["test_cutoffs"])
    m = LGBMClassifier(**P).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    top = int(len(yte) * 0.1)
    met = {"roc_auc": round(roc_auc_score(yte, p), 3), "pr_auc": round(average_precision_score(yte, p), 3),
           "top_decile_lift": round(float(yte.values[np.argsort(-p)[:top]].mean() / yte.mean()), 2),
           "base_rate": round(float(yte.mean()), 4), "n_test": int(len(yte)),
           "note": "stand-in for Project 1; simpler features"}
    print(met)
    # final: refit on all cutoffs up to and including the test cutoff, score as of 30 Sep
    Xall, yall = pd.concat([Xtr, Xte]), pd.concat([ytr, yte])
    fin = LGBMClassifier(**P).fit(Xall, yall)
    Xs = asof_features(*tables, cfg["segmentation"]["as_of"])[ASOF_FEATURES]
    pd.DataFrame({"customer_id": Xs.index, "churn_prob_30d": fin.predict_proba(Xs)[:, 1],
                  "as_of": cfg["segmentation"]["as_of"], "source": "stand-in"}).to_parquet(
        cfg["_out"] / "churn_scores.parquet", index=False)
    json.dump(met, open(cfg["_out"] / "churn_proxy_metrics.json", "w"), indent=2)
    return met


if __name__ == "__main__":
    run(Path(__file__).resolve().parent.parent / "config" / "customer_value.yaml")
