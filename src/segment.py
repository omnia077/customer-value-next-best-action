"""Segmentation: k selection, stability, recovery check against planted archetypes, profiles, personas.

Run from the customer_value/ folder:  python src/segment.py [--select]

--select  runs the k grid for BOTH populations (all active in window / active in last N days)
          and writes outputs/k_selection_<tag>.csv. Without it, only the final k is fitted.

ground_truth/archetypes.parquet is read ONLY to report recovery (ARI, cross-tab). It is never used
to pick k or as a feature.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).parent))
from features import SEG_FEATURES, build_features, segmentation_matrix  # noqa: E402


def load_cfg(cfg_path):
    root = Path(cfg_path).resolve().parent.parent          # customer_value/
    cfg = yaml.safe_load(open(cfg_path))
    cfg["_root"] = root
    cfg["_out"] = root / cfg["output_dir"]
    cfg["_out"].mkdir(exist_ok=True)
    cfg["_data"] = (root / cfg["data_root"]).resolve()
    return cfg


def kmeans_fit(Z, k, seed):
    return KMeans(k, n_init=10, random_state=seed).fit(Z)


def bootstrap_stability(Z, k, base_labels, n_boot, seed):
    """Refit on bootstrap resamples, assign ALL points to the refit centroids, compare with the full-data labels."""
    rng = np.random.default_rng(seed)
    aris = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(Z), len(Z))
        km = KMeans(k, n_init=3, random_state=int(rng.integers(1e9))).fit(Z[idx])
        aris.append(adjusted_rand_score(base_labels, km.predict(Z)))
    return float(np.mean(aris)), float(np.std(aris))


def population(cfg, active_days):
    sc = cfg["segmentation"]
    df, info = build_features(cfg["_data"], sc["as_of"], sc["window_days"], active_days)
    truth = (pd.read_parquet(cfg["_data"] / "ground_truth" / "archetypes.parquet")
             .set_index("customer_id").reindex(df.index))
    return df, info, truth


def select_k(cfg, active_days, tag):
    sc = cfg["segmentation"]
    df, info, truth = population(cfg, active_days)
    print(f"[{tag}] population:", info)
    Z = segmentation_matrix(df).to_numpy()
    rows = []
    for k in sc["k_range"]:
        km = kmeans_fit(Z, k, sc["seed"])
        sil = silhouette_score(Z, km.labels_, sample_size=sc["silhouette_sample"], random_state=sc["seed"])
        st_m, st_s = bootstrap_stability(Z, k, km.labels_, sc["n_bootstrap"], sc["seed"])
        rows.append({"k": k, "silhouette": round(sil, 3), "boot_ari": round(st_m, 3), "boot_ari_sd": round(st_s, 3),
                     "smallest_seg_pct": round(100 * np.bincount(km.labels_).min() / len(Z), 1),
                     "recovery_ari": round(adjusted_rand_score(truth.archetype_id, km.labels_), 3)})
        print(f"[{tag}]", rows[-1])
    res = pd.DataFrame(rows)
    res.to_csv(cfg["_out"] / f"k_selection_{tag}.csv", index=False)
    return res


def name_personas(prof):
    """Data-driven names from the profile (not the planted archetype names).
    Rules use the segment's rank on value and the dominant usage mix; ties are resolved by rank."""
    names = {}
    by_rev = prof.pct_revenue.rank(ascending=False)
    for sid, r in prof.iterrows():
        lapsing = r.recency_days >= prof.recency_days.quantile(0.75) and r.recharge_freq_30d <= prof.recharge_freq_30d.median()
        if lapsing:
            base = "Lapsing"
        elif by_rev[sid] == 1:
            base = "Core high-value"
        elif r.sessions_30d <= prof.sessions_30d.quantile(0.25):
            base = "Light"
        else:
            base = "Steady"
        mix = "data-led" if r.share_data >= 0.55 else "voice-led" if r.share_voice >= 0.5 else "mixed-use"
        names[sid] = f"{base} {mix}"
    seen = {}
    for sid in list(names):                         # keep names unique
        seen[names[sid]] = seen.get(names[sid], 0) + 1
        if seen[names[sid]] > 1:
            names[sid] += f" #{seen[names[sid]]}"
    return names


def fit_final(cfg):
    sc = cfg["segmentation"]
    k = sc["k_final"]
    df, info, truth = population(cfg, sc["active_within_days"])
    Zdf = segmentation_matrix(df)
    km = kmeans_fit(Zdf.to_numpy(), k, sc["seed"])
    df["segment_id"] = km.labels_
    prof = (df.groupby("segment_id")
              .agg(n=("segment_id", "size"), revenue_sdg=("total_recharge_sdg", "sum"),
                   **{f: (f, "mean") for f in SEG_FEATURES}).round(2))
    prof["pct_subs"] = (100 * prof.n / prof.n.sum()).round(1)
    prof["pct_revenue"] = (100 * prof.revenue_sdg / prof.revenue_sdg.sum()).round(1)
    prof["persona"] = pd.Series(name_personas(prof))
    df["persona"] = df.segment_id.map(prof.persona)
    xt = pd.crosstab(df.segment_id, truth.archetype, normalize="index").mul(100).round(0)
    rec = adjusted_rand_score(truth.archetype_id, km.labels_)
    sil = silhouette_score(Zdf.to_numpy(), km.labels_, sample_size=sc["silhouette_sample"], random_state=sc["seed"])
    prof.to_csv(cfg["_out"] / "segment_profile.csv")
    xt.to_csv(cfg["_out"] / "segment_vs_archetype.csv")
    df[["segment_id", "persona"]].reset_index().assign(as_of=sc["as_of"], k=k).to_parquet(
        cfg["_out"] / "segment_assignments.parquet", index=False)
    # centroids in original units for later scoring / drift checks
    json.dump({"k": k, "silhouette": round(sil, 3), "recovery_ari": round(rec, 3), "n": len(df), "info": info,
               "active_within_days": sc["active_within_days"]},
              open(cfg["_out"] / "segmentation_metrics.json", "w"), indent=2)
    print(f"final k={k} n={len(df):,} silhouette={sil:.3f} recovery_ARI={rec:.3f}")
    print(prof.to_string()); print("\nrow % vs planted archetype\n", xt.to_string())
    return df, prof


if __name__ == "__main__":
    cfg = load_cfg(Path(__file__).resolve().parent.parent / "config" / "customer_value.yaml")
    if "--select" in sys.argv:
        select_k(cfg, None, "all_active")
        select_k(cfg, cfg["segmentation"]["active_within_days"], "active90")
    else:
        fit_final(cfg)
