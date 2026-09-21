"""As-of RFM + usage features for prepaid subscribers.

Only events with date <= as_of and inside the trailing window are used. Nothing here reads
ground_truth/ and nothing uses the churn label or churn_date.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def load_tables(data_root):
    sftp = Path(data_root) / "sftp_drop"
    cust = pd.read_parquet(sftp / "customers.parquet",
                           columns=["customer_id", "plan_type", "activation_date", "monthly_arpu_sdg"])
    rech = pd.read_parquet(sftp / "recharge_records.parquet",
                           columns=["customer_id", "recharge_date", "amount_sdg"])
    usage = pd.read_parquet(sftp / "usage_events.parquet",
                            columns=["customer_id", "event_date", "usage_type"])
    for df, c in [(cust, "activation_date"), (rech, "recharge_date"), (usage, "event_date")]:
        df[c] = pd.to_datetime(df[c])
    return cust, rech, usage


def build_features(data_root, as_of, window_days, active_within_days=None):
    """One row per prepaid subscriber active in the window. Returns (raw features, info dict).

    active_within_days: if set, also require a recharge or usage event in the last N days
    (removes lines that already went quiet, so segments describe behaviour not lifecycle)."""
    as_of = pd.Timestamp(as_of)
    start = as_of - pd.Timedelta(days=window_days - 1)
    cust, rech, usage = load_tables(data_root)

    cust = cust[(cust.plan_type == "prepaid") & (cust.activation_date <= as_of)]
    rech = rech[(rech.recharge_date >= start) & (rech.recharge_date <= as_of)]
    usage = usage[(usage.event_date >= start) & (usage.event_date <= as_of)]

    # exposure: days the subscriber could have been observed in the window (new activations are shorter)
    first_day = cust.activation_date.clip(lower=start)
    exposure = ((as_of - first_day).dt.days + 1).clip(lower=1)
    df = pd.DataFrame({"customer_id": cust.customer_id.values,
                       "exposure_days": exposure.values,
                       "tenure_days": (as_of - cust.activation_date).dt.days.values}).set_index("customer_id")

    g = rech.groupby("customer_id")
    df["n_recharges"] = g.size()
    df["total_recharge_sdg"] = g.amount_sdg.sum()
    df["avg_recharge_sdg"] = g.amount_sdg.mean()
    df["last_recharge"] = g.recharge_date.max()
    df["n_sessions"] = usage.groupby("customer_id").size()
    df["last_activity"] = pd.concat([g.recharge_date.max(),
                                     usage.groupby("customer_id").event_date.max()], axis=1).max(axis=1)
    mix = pd.crosstab(usage.customer_id, usage.usage_type)
    for t in ["data", "voice", "sms"]:
        df[f"n_{t}"] = mix[t] if t in mix else 0

    df[["n_recharges", "total_recharge_sdg", "n_sessions", "n_data", "n_voice", "n_sms"]] = \
        df[["n_recharges", "total_recharge_sdg", "n_sessions", "n_data", "n_voice", "n_sms"]].fillna(0)

    # population rule: at least one recharge or usage event in the window (inactive lines are out of scope)
    n_before = len(df)
    df = df[(df.n_recharges + df.n_sessions) > 0].copy()
    if active_within_days:
        df = df[(as_of - df.last_activity).dt.days < active_within_days].copy()

    per30 = 30.0 / df.exposure_days
    df["recharge_freq_30d"] = df.n_recharges * per30
    df["sessions_30d"] = df.n_sessions * per30
    df["recency_days"] = (as_of - df.last_recharge).dt.days.fillna(df.exposure_days)   # no recharge -> full exposure
    for t in ["data", "voice", "sms"]:
        df[f"share_{t}"] = df[f"n_{t}"] / df.n_sessions.where(df.n_sessions > 0)

    info = {"prepaid_activated": n_before, "kept": len(df),
            "no_recharge_in_window": int((df.n_recharges == 0).sum()),
            "no_usage_in_window": int((df.n_sessions == 0).sum())}
    # missing values: average size for no-recharge lines, mix for no-usage lines -> population median / mean
    df["avg_recharge_sdg"] = df.avg_recharge_sdg.fillna(df.avg_recharge_sdg.median())
    for t in ["data", "voice", "sms"]:
        df[f"share_{t}"] = df[f"share_{t}"].fillna(df[f"share_{t}"].mean())
    return df.drop(columns=["last_recharge", "last_activity"]), info


ASOF_FEATURES = ["rech_n_30", "rech_n_prev30", "rech_sum_30", "rech_sum_60", "rech_avg_60", "rech_max_60",
                 "n500_60", "days_since_recharge", "sess_30", "sess_prev30", "sess_ratio", "share_data_60",
                 "share_voice_60", "tenure_days", "monthly_arpu_sdg"]


def asof_features(cust, rech, usage, cutoff):
    """Point-in-time features for prepaid subscribers at `cutoff` (inclusive). Uses ONLY events dated <= cutoff.
    Windows: last 30 days, previous 30 days (31-60). Population: activated by cutoff with any
    recharge/usage event in the last 60 days. No ground truth, no labels, no future data."""
    cutoff = pd.Timestamp(cutoff)
    s30, s60 = cutoff - pd.Timedelta(days=29), cutoff - pd.Timedelta(days=59)
    c = cust[(cust.plan_type == "prepaid") & (cust.activation_date <= cutoff)].set_index("customer_id")
    r = rech[rech.recharge_date <= cutoff]
    u = usage[(usage.event_date >= s60) & (usage.event_date <= cutoff)]
    r60 = r[r.recharge_date >= s60]
    r30 = r60[r60.recharge_date >= s30]
    df = pd.DataFrame(index=c.index)
    df["rech_n_30"] = r30.groupby("customer_id").size()
    df["rech_n_60"] = r60.groupby("customer_id").size()
    df["rech_sum_30"] = r30.groupby("customer_id").amount_sdg.sum()
    df["rech_sum_60"] = r60.groupby("customer_id").amount_sdg.sum()
    df["rech_avg_60"] = r60.groupby("customer_id").amount_sdg.mean()
    df["rech_max_60"] = r60.groupby("customer_id").amount_sdg.max()
    df["n500_60"] = r60[r60.amount_sdg >= 500].groupby("customer_id").size()
    last = r.groupby("customer_id").recharge_date.max()
    df["tenure_days"] = (cutoff - c.activation_date).dt.days
    df["days_since_recharge"] = (cutoff - last).dt.days.reindex(df.index).fillna(df.tenure_days).clip(upper=400)
    u30 = u[u.event_date >= s30]
    df["sess_30"] = u30.groupby("customer_id").size()
    df["sess_60"] = u.groupby("customer_id").size()
    mix = pd.crosstab(u.customer_id, u.usage_type)
    for t in ["data", "voice"]:
        df[f"n_{t}"] = mix[t] if t in mix else 0
    zero = ["rech_n_30", "rech_n_60", "rech_sum_30", "rech_sum_60", "n500_60", "sess_30", "sess_60", "n_data", "n_voice"]
    df[zero] = df[zero].fillna(0)
    df = df[(df.rech_n_60 + df.sess_60) > 0].copy()
    df["rech_n_prev30"] = df.rech_n_60 - df.rech_n_30
    df["sess_prev30"] = df.sess_60 - df.sess_30
    df["sess_ratio"] = (df.sess_30 + 1) / (df.sess_prev30 + 1)
    df["share_data_60"] = (df.n_data / df.sess_60.where(df.sess_60 > 0))
    df["share_voice_60"] = (df.n_voice / df.sess_60.where(df.sess_60 > 0))
    df["monthly_arpu_sdg"] = c.monthly_arpu_sdg.reindex(df.index)
    return df[ASOF_FEATURES + ["rech_n_60", "sess_60"]]


def recharge500_label(rech, start, end):
    """1 if the customer has >=1 recharge of >=500 SDG dated in [start, end] (both inclusive)."""
    m = rech[(rech.recharge_date >= pd.Timestamp(start)) & (rech.recharge_date <= pd.Timestamp(end))
             & (rech.amount_sdg >= 500)]
    return set(m.customer_id.unique())


SEG_FEATURES = ["recency_days", "recharge_freq_30d", "avg_recharge_sdg", "sessions_30d",
                "share_data", "share_voice", "share_sms"]
LOG_FEATURES = ["recency_days", "recharge_freq_30d", "avg_recharge_sdg", "sessions_30d"]


def segmentation_matrix(df):
    """log1p on skewed variables, then standardize. The three usage shares sum to 1 (rank 2), so
    keeping all three gives the usage mix somewhat more weight than one column per dimension."""
    X = df[SEG_FEATURES].copy()
    X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])
    Z = (X - X.mean()) / X.std(ddof=0)
    return Z
