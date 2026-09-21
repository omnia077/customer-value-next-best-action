"""Robustness across simulator seeds: are the headline results a lucky draw of seed 42?

For each seed: regenerate the simulated world (archetypes + campaigns), then run segmentation, CLV and uplift with
the same code and settings as the main pipeline, plus the Grow-policy sweep. Only the metrics are kept; the
generated data is deleted after each seed. Seeds 1-10 are aggregated; the main run (seed 42) is shown as a reference
row and is NOT included in the aggregate.

Not re-run per seed (cost, and not part of the headline claims): propensity, churn stand-in, actions allocation.
Model seeds (LightGBM, k-means, folds) are set to the simulator seed.

Run:  python src/multiseed.py --seeds 1-10 --workdir <scratch dir>
Out:  outputs/multiseed_per_seed.csv, outputs/multiseed_summary.csv, outputs/multiseed_claims.json
"""
import argparse
import contextlib
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import actions, clv, segment, uplift  # noqa: E402


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def make_cfg(seed, workdir):
    base = yaml.safe_load(open(ROOT / "config" / "customer_value.yaml"))
    d = Path(workdir) / f"seed_{seed}"
    (d / "config").mkdir(parents=True, exist_ok=True)
    base.update(data_root=str(d / "data"), seed=seed)
    base["segmentation"]["seed"] = seed
    base["uplift"]["n_bootstrap"] = 30            # per-seed CIs are not reported; the across-seed spread is
    cfg_path = d / "config" / "customer_value.yaml"
    yaml.safe_dump(base, open(cfg_path, "w"))
    return d, cfg_path


def collect(seed, out_dir, cfg):
    """Flatten the metrics that back the headline claims."""
    o = Path(out_dir)
    seg = json.load(open(o / "segmentation_metrics.json"))
    c = pd.read_csv(o / "clv_holdout.csv").set_index("model")
    up = json.load(open(o / "uplift_metrics.json"))
    ev = pd.DataFrame(up["evaluation"]).set_index("policy")
    tl = pd.DataFrame(up["evaluation"]).query("learner == 'T-learner'").iloc[0]
    cost, bud, be = actions.grow_sweeps(cfg)
    r25 = cost[cost.offer_cost_sdg == cfg["actions"]["offer_cost_sdg"]].iloc[0]
    b30 = bud[np.isclose(bud.contact_share, cfg["actions"]["max_contact_share"])].iloc[0]
    row = {"seed": seed,
           "seg_silhouette": seg["silhouette"], "seg_recovery_ari": seg["recovery_ari"], "seg_n": seg["n"],
           "clv_rmse_last90": c.loc["Last-90-day spend (scaled to 92d)", "rmse_sdg"],
           "clv_rmse_bgnbd": c.loc["BG/NBD + Gamma-Gamma", "rmse_sdg"],
           "clv_rmse_lgbm": c.loc["LightGBM on RFM (trained on prior period)", "rmse_sdg"],
           "clv_top20_last90": c.loc["Last-90-day spend (scaled to 92d)", "top20_capture_pct"],
           "clv_top20_bgnbd": c.loc["BG/NBD + Gamma-Gamma", "top20_capture_pct"],
           "clv_top20_lgbm": c.loc["LightGBM on RFM (trained on prior period)", "top20_capture_pct"],
           "clv_top20_oracle": c.loc["Oracle", "top20_capture_pct"],
           "uplift_selected_learner": up["selected_learner"],
           "ate_observed_june_pts": round(100 * up["randomization"]["june"]["ate"], 2),
           "ate_true_june_pts": round(100 * up["true_mean_uplift_june"], 2),
           "uplift_spearman": ev.loc["Uplift model", "spearman_vs_true_uplift"],
           "tlearner_spearman": tl.spearman_vs_true_uplift,
           "propensity_spearman": ev.loc["Propensity model", "spearman_vs_true_uplift"],
           "uplift_qini": ev.loc["Uplift model", "qini_coef"], "propensity_qini": ev.loc["Propensity model", "qini_coef"],
           "qini_diff_mean": up["qini_diff_uplift_minus_propensity"]["mean"],
           "qini_diff_ci_lo": up["qini_diff_uplift_minus_propensity"]["ci95"][0],
           "uplift_top20_obs_pts": ev.loc["Uplift model", "observed_uplift_top20_pts"],
           "uplift_top20_ci_lo": ev.loc["Uplift model", "observed_uplift_top20_ci95_pts"][0],
           "propensity_top20_obs_pts": ev.loc["Propensity model", "observed_uplift_top20_pts"],
           "grow30_uplift_sdg": b30.uplift, "grow30_propensity_sdg": b30.propensity, "grow30_random_sdg": b30.random,
           "grow30_oracle_sdg": b30.oracle, "everyone_at25_sdg": r25.contact_everyone, "uplift_rule_at25_sdg": r25.uplift_ev_rule,
           "breakeven_offer_cost_sdg": be}
    return row


def run_seed(seed, workdir):
    d, cfg_path = make_cfg(seed, workdir)
    mfile = d / "metrics.json"
    if mfile.exists():
        return json.load(open(mfile))
    t0 = time.time()
    subprocess.run([sys.executable, str(ROOT / "data_generation" / "phase1b_generate_data.py"), "--archetypes", "--campaign",
                    "--seed", str(seed), "--out", str(d / "data")], check=True, capture_output=True)
    cfg = segment.load_cfg(cfg_path)
    with contextlib.redirect_stdout(io.StringIO()):            # the stage modules are chatty
        segment.fit_final(cfg)
        clv.run(cfg_path)
        uplift.run(cfg_path)
    row = collect(seed, cfg["_out"], cfg)
    row["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(row, open(mfile, "w"), indent=2, default=float)
    shutil.rmtree(d / "data", ignore_errors=True)               # keep results only
    return row


def reference_row():
    """Seed 42 from the main pipeline outputs (same code path, main settings)."""
    cfg = segment.load_cfg(ROOT / "config" / "customer_value.yaml")
    return collect(42, cfg["_out"], cfg)


def summarize(rows, ref):
    df = pd.DataFrame(rows).sort_values("seed")
    num = df.drop(columns=["seed", "uplift_selected_learner", "minutes"], errors="ignore")
    summ = pd.DataFrame({"mean": num.mean(), "sd": num.std(ddof=1), "min": num.min(), "max": num.max(),
                         "seed_42_reference": pd.Series({k: ref[k] for k in num.columns})}).round(4)
    n = len(df)
    claims = {
        "n_seeds": n,
        "propensity_rank_corr_negative": int((df.propensity_spearman < 0).sum()),
        "uplift_rank_corr_beats_propensity": int((df.uplift_spearman > df.propensity_spearman).sum()),
        "uplift_rank_corr_positive": int((df.uplift_spearman > 0).sum()),
        "qini_gap_vs_propensity_ci_above_zero": int((df.qini_diff_ci_lo > 0).sum()),
        "uplift_top20_ci_above_overall_effect": int((df.uplift_top20_ci_lo > df.ate_observed_june_pts).sum()),
        "bgnbd_rmse_beats_last90": int((df.clv_rmse_bgnbd < df.clv_rmse_last90).sum()),
        "bgnbd_rmse_beats_lgbm": int((df.clv_rmse_bgnbd < df.clv_rmse_lgbm).sum()),
        "grow30_uplift_beats_random": int((df.grow30_uplift_sdg > df.grow30_random_sdg).sum()),
        "grow30_propensity_below_random": int((df.grow30_propensity_sdg < df.grow30_random_sdg).sum()),
        "grow30_propensity_negative": int((df.grow30_propensity_sdg < 0).sum()),
        "uplift_rule_beats_everyone_at_25": int((df.uplift_rule_at25_sdg > df.everyone_at25_sdg).sum()),
        "cv_selected_beats_tlearner_rank_corr": int((df.uplift_spearman > df.tlearner_spearman).sum()),
        "selected_learner_counts": df.uplift_selected_learner.value_counts().to_dict(),
        "seg_recovery_ari_range": [float(df.seg_recovery_ari.min()), float(df.seg_recovery_ari.max())],
        "breakeven_offer_cost_range_sdg": [float(df.breakeven_offer_cost_sdg.min()), float(df.breakeven_offer_cost_sdg.max())]}
    return df, summ, claims


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()
    Path(a.workdir).mkdir(parents=True, exist_ok=True)
    rows = []
    for s in parse_seeds(a.seeds):
        r = run_seed(s, a.workdir)
        rows.append(r)
        print(f"seed {s} done ({r.get('minutes', '?')} min): uplift rho {r['uplift_spearman']}, propensity rho {r['propensity_spearman']}, "
              f"recovery ARI {r['seg_recovery_ari']}", flush=True)
    ref = reference_row()
    df, summ, claims = summarize(rows, ref)
    out = ROOT / "outputs"
    pd.concat([df, pd.DataFrame([{**ref, "seed": "42 (reference)"}])]).to_csv(out / "multiseed_per_seed.csv", index=False)
    summ.to_csv(out / "multiseed_summary.csv")
    json.dump(claims, open(out / "multiseed_claims.json", "w"), indent=2)
    print(json.dumps(claims, indent=2))
