"""Identifier handling and the adapter for Project 1's churn scores.

MSISDN policy: raw MSISDNs are never written to any output. They are read only inside `msisdn_to_customer_id`,
hashed with a salt from the environment (MSISDN_SALT, never stored in the repo), and discarded. The hashing scheme
below must match Project 1's; if Project 1 uses a different scheme, change `hash_msisdn` in ONE place.

Scheme: sha256(salt + msisdn), first 32 hex characters.
"""
import hashlib
import os

import pandas as pd


def hash_msisdn(msisdn, salt=None):
    salt = salt if salt is not None else os.environ.get("MSISDN_SALT")
    if not salt:
        raise EnvironmentError("MSISDN_SALT is not set: refusing to hash without the shared salt")
    return hashlib.sha256(f"{salt}{msisdn}".encode("utf-8")).hexdigest()[:32]


def msisdn_to_customer_id(customers_parquet, salt=None):
    """Map hashed MSISDN -> customer_id. The raw MSISDN column is dropped before returning."""
    c = pd.read_parquet(customers_parquet, columns=["customer_id", "msisdn"])
    c["msisdn_hash"] = [hash_msisdn(m, salt) for m in c.msisdn]
    return c.drop(columns="msisdn").set_index("msisdn_hash").customer_id


def standardize_external_scores(df, key_column, prob_column, id_map=None, population_ids=None, min_coverage=0.9):
    """Validate Project 1's scores and return a frame with columns customer_id, churn_prob_30d.

    Fails loudly (ValueError) on: missing columns, duplicate keys, probabilities outside [0, 1] or missing,
    unmapped hashed IDs, or coverage of the scored population below `min_coverage`."""
    for col in (key_column, prob_column):
        if col not in df.columns:
            raise ValueError(f"external churn scores: column '{col}' not found (have {list(df.columns)})")
    out = df[[key_column, prob_column]].rename(columns={prob_column: "churn_prob_30d"}).copy()
    if out[key_column].duplicated().any():
        raise ValueError("external churn scores: duplicate keys")
    if out.churn_prob_30d.isna().any() or not out.churn_prob_30d.between(0, 1).all():
        raise ValueError("external churn scores: probabilities must be non-null and within [0, 1]")
    if key_column == "msisdn_hash":
        if id_map is None:
            raise ValueError("external churn scores keyed by msisdn_hash need an id_map")
        out["customer_id"] = out.msisdn_hash.map(id_map)
        if out.customer_id.isna().any():
            raise ValueError(f"external churn scores: {int(out.customer_id.isna().sum())} hashed IDs did not map "
                             "(wrong salt or hashing scheme?)")
        out = out.drop(columns="msisdn_hash")
    elif key_column != "customer_id":
        raise ValueError("key_column must be 'customer_id' or 'msisdn_hash'")
    if population_ids is not None:
        cov = float(pd.Index(population_ids).isin(out.customer_id).mean())
        if cov < min_coverage:
            raise ValueError(f"external churn scores cover only {cov:.1%} of the scored population (< {min_coverage:.0%})")
    return out[["customer_id", "churn_prob_30d"]]
