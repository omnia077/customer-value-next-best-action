"""Identifier handling and the adapter for Project 1's churn scores.

Project 1 (Telecom-Churn-Platform, ml_churn/score_monthly.py) writes `churn_scores.parquet` for ONE cutoff with columns
sub_hash, cutoff_date, region, probability, decile, ev, reasons. `sub_hash` = sha256(salt + msisdn)[:16] with the salt
in the environment variable CHURN_HASH_SALT (never in a repo). This module reproduces that scheme so the two projects
join on the same key. If Project 1's scheme changes, change `hash_msisdn` here in ONE place.

MSISDN policy: raw MSISDNs are never written to any output. They are read only inside `msisdn_to_customer_id`,
hashed, and discarded.
"""
import hashlib
import os

import pandas as pd

SALT_ENV = "CHURN_HASH_SALT"


def hash_msisdn(msisdn, salt=None):
    salt = salt if salt is not None else os.environ.get(SALT_ENV)
    if not salt:
        raise EnvironmentError(f"{SALT_ENV} is not set: refusing to hash without the shared salt")
    return hashlib.sha256((salt + str(msisdn)).encode()).hexdigest()[:16]


def msisdn_to_customer_id(customers_parquet, salt=None):
    """Map sub_hash -> customer_id. The raw MSISDN column is dropped before returning."""
    c = pd.read_parquet(customers_parquet, columns=["customer_id", "msisdn"])
    c["sub_hash"] = [hash_msisdn(m, salt) for m in c.msisdn]
    return c.drop(columns="msisdn").set_index("sub_hash").customer_id


def standardize_external_scores(df, key_column, prob_column, id_map=None, population_ids=None, min_coverage=0.9,
                                expected_cutoff=None):
    """Validate Project 1's scores and return a frame with columns customer_id, churn_prob_30d.

    Fails loudly (ValueError) on: missing columns, duplicate keys, probabilities outside [0, 1] or missing,
    unmapped hashes (wrong salt), scores from a different cutoff date than `expected_cutoff`, or coverage of the
    scored population below `min_coverage`."""
    for col in (key_column, prob_column):
        if col not in df.columns:
            raise ValueError(f"external churn scores: column '{col}' not found (have {list(df.columns)})")
    if expected_cutoff is not None and "cutoff_date" in df.columns:
        got = pd.to_datetime(df["cutoff_date"]).dt.normalize().unique()
        if len(got) != 1 or got[0] != pd.Timestamp(expected_cutoff):
            raise ValueError(f"external churn scores are for cutoff {[str(g)[:10] for g in got]}, expected {expected_cutoff}: "
                             "re-run Project 1 scoring with --as-of " + str(expected_cutoff))
    out = df[[key_column, prob_column]].rename(columns={prob_column: "churn_prob_30d"}).copy()
    if out[key_column].duplicated().any():
        raise ValueError("external churn scores: duplicate keys")
    if out.churn_prob_30d.isna().any() or not out.churn_prob_30d.between(0, 1).all():
        raise ValueError("external churn scores: probabilities must be non-null and within [0, 1]")
    if key_column == "sub_hash":
        if id_map is None:
            raise ValueError("external churn scores keyed by sub_hash need an id_map")
        out["customer_id"] = out.sub_hash.map(id_map)
        if out.customer_id.isna().any():
            raise ValueError(f"external churn scores: {int(out.customer_id.isna().sum())} hashes did not map "
                             "(wrong salt or hashing scheme?)")
        out = out.drop(columns="sub_hash")
    elif key_column != "customer_id":
        raise ValueError("key_column must be 'customer_id' or 'sub_hash'")
    if population_ids is not None:
        cov = float(pd.Index(population_ids).isin(out.customer_id).mean())
        if cov < min_coverage:
            raise ValueError(f"external churn scores cover only {cov:.1%} of the scored population (< {min_coverage:.0%})")
    return out[["customer_id", "churn_prob_30d"]]
