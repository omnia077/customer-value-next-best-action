import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from clv import BGNBD  # noqa: E402
from features import ASOF_FEATURES, asof_features, load_tables  # noqa: E402
from score_monthly import psi  # noqa: E402
from uplift import qini_coef  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config" / "customer_value.yaml"))
DATA = (ROOT / CFG["data_root"]).resolve()
OUT = ROOT / CFG["output_dir"]


@pytest.fixture(scope="module")
def tables():
    return load_tables(DATA)


@pytest.mark.parametrize("cutoff", ["2024-03-31", "2024-06-30", "2024-09-30"])
def test_features_use_no_future_data(tables, cutoff):
    """Point-in-time safety: deleting every event after the cutoff must not change any feature."""
    cust, rech, usage = tables
    c = pd.Timestamp(cutoff)
    full = asof_features(cust, rech, usage, c)
    trunc = asof_features(cust, rech[rech.recharge_date <= c], usage[usage.event_date <= c], c)
    pd.testing.assert_frame_equal(full, trunc)
    assert set(ASOF_FEATURES) <= set(full.columns)


def test_propensity_train_labels_avoid_campaign_months():
    for c in CFG["propensity"]["train_cutoffs"]:
        label_month = (pd.Timestamp(c) + pd.Timedelta(days=1)).month
        assert label_month not in (4, 6), f"train cutoff {c} has a campaign label month"


def test_campaign_history_only_in_campaign_months():
    h = pd.read_parquet(DATA / "campaigns" / "campaign_history.parquet")
    assert sorted(pd.to_datetime(h.sent_date).dt.month.unique()) == [4, 6]
    assert not h.duplicated(["campaign_id", "customer_id"]).any()
    assert 0.45 < h.groupby("campaign_id").treated.mean().min() and h.groupby("campaign_id").treated.mean().max() < 0.55


def test_ground_truth_only_read_where_allowed():
    """Ground truth may be read to validate (segment recovery, uplift evaluation) or as the churn label source."""
    allowed = {"segment.py", "uplift.py", "churn_proxy.py", "compare_churn_sources.py"}
    for p in (ROOT / "src").glob("*.py"):
        reads = re.search(r"ground_truth", p.read_text(encoding="utf-8"))
        # docstrings in features/others may mention it; actual reads use the folder name in a path expression
        code_read = re.search(r'/\s*"ground_truth"', p.read_text(encoding="utf-8"))
        if p.name not in allowed:
            assert not code_read, f"{p.name} reads ground_truth"


def test_features_do_not_contain_identifiers_or_pii():
    cols = set(ASOF_FEATURES)
    assert not ({"age", "gender", "region", "msisdn", "customer_id"} & cols)


def test_actions_one_per_subscriber_and_budget():
    a = pd.read_parquet(OUT / "customer_actions.parquet")
    assert a.customer_id.is_unique                                    # contact-frequency cap: one action each
    assert set(a.action) <= {"grow", "protect", "win_back", "maintain"}
    contacted = a[a.action != "maintain"]
    assert (contacted.ev > 0).all()
    assert len(contacted) <= CFG["actions"]["max_contact_share"] * len(a) + 1
    assert (a.loc[a.action == "maintain", "ev"] == 0).all()


def test_scores_are_probabilities_and_ids_align():
    clv = pd.read_parquet(OUT / "clv_scores.parquet")
    assert clv.p_alive.between(0, 1).all() and (clv.clv_92d >= 0).all()
    for f, col in [("propensity_scores.parquet", "p_recharge500_next_month"), ("churn_scores.parquet", "churn_prob_30d")]:
        s = pd.read_parquet(OUT / f)
        assert s[col].between(0, 1).all() and s.customer_id.is_unique
    seg = pd.read_parquet(OUT / "segment_assignments.parquet")
    assert seg.customer_id.is_unique and seg.segment_id.nunique() == CFG["segmentation"]["k_final"]


def test_bgnbd_expectation_properties():
    m = BGNBD()
    m.r, m.alpha, m.a, m.b = 0.8, 11.0, 0.07, 2.5
    x, tx, T = (np.array([v]) for v in (3.0, 60.0, 90.0))
    e30, e92, e365 = (m.expected_purchases(t, x, tx, T)[0] for t in (30, 92, 365))
    assert 0 < e30 < e92 < e365
    assert 0 <= m.p_alive(x, tx, T)[0] <= 1
    # fresh customer, tiny horizon: expected purchases ~ (r / alpha) * t
    e = m.expected_purchases(1.0, np.array([0.0]), np.array([0.0]), np.array([0.0]))[0]
    assert e == pytest.approx(m.r / m.alpha, rel=0.05)


def test_qini_coefficient_orders_scores_correctly():
    rng = np.random.default_rng(0)
    n = 40000
    t = rng.integers(0, 2, n)
    tau = rng.uniform(0, 0.3, n)
    y = (rng.random(n) < 0.1 + t * tau).astype(int)
    assert qini_coef(tau, t, y) > 0.01                    # informative score
    assert qini_coef(-tau, t, y) < -0.01                  # reversed score is worse than random
    assert abs(qini_coef(rng.random(n), t, y)) < 0.01     # random score ~ 0


def test_msisdn_hash_needs_salt_and_is_deterministic(monkeypatch):
    from ids import hash_msisdn
    monkeypatch.delenv("CHURN_HASH_SALT", raising=False)
    with pytest.raises(EnvironmentError):
        hash_msisdn("249912345678")
    assert hash_msisdn("249912345678", "s1") == hash_msisdn("249912345678", "s1")
    assert hash_msisdn("249912345678", "s1") != hash_msisdn("249912345678", "s2")
    assert len(hash_msisdn("249912345678", "s1")) == 16 and "2499" not in hash_msisdn("249912345678", "s1")
    # documented Project 1 scheme (ml_churn/common.py hash_id): sha256(salt + msisdn) truncated to 16 hex chars
    import hashlib
    assert hash_msisdn("249912345678", "s1") == hashlib.sha256(b"s1249912345678").hexdigest()[:16]


def test_msisdn_map_round_trip_and_no_raw_msisdn():
    from ids import hash_msisdn, msisdn_to_customer_id
    m = msisdn_to_customer_id(DATA / "sftp_drop" / "customers.parquet", salt="test-salt")
    assert m.index.is_unique and len(m) > 10000
    assert (m.index.str.len() == 16).all() and not m.index.str.startswith("2499").any()   # hashes, not raw numbers
    raw = pd.read_parquet(DATA / "sftp_drop" / "customers.parquet", columns=["customer_id", "msisdn"]).iloc[0]
    assert m[hash_msisdn(raw.msisdn, "test-salt")] == raw.customer_id


def test_external_churn_adapter_accepts_valid_and_rejects_bad_input():
    from ids import standardize_external_scores
    pop = [f"C{i}" for i in range(100)]
    good = pd.DataFrame({"customer_id": pop, "churn_prob_30d": np.linspace(0, 1, 100)})
    out = standardize_external_scores(good, "customer_id", "churn_prob_30d", population_ids=pop)
    assert list(out.columns) == ["customer_id", "churn_prob_30d"] and len(out) == 100
    with pytest.raises(ValueError, match="within"):
        standardize_external_scores(good.assign(churn_prob_30d=1.5), "customer_id", "churn_prob_30d")
    with pytest.raises(ValueError, match="duplicate"):
        standardize_external_scores(pd.concat([good, good]), "customer_id", "churn_prob_30d")
    with pytest.raises(ValueError, match="cover only"):
        standardize_external_scores(good.head(50), "customer_id", "churn_prob_30d", population_ids=pop)
    with pytest.raises(ValueError, match="not found"):
        standardize_external_scores(good, "customer_id", "prob")
    hashed = good.rename(columns={"customer_id": "sub_hash"}).assign(sub_hash=[f"h{i}" for i in range(100)])
    with pytest.raises(ValueError, match="did not map"):
        standardize_external_scores(hashed, "sub_hash", "churn_prob_30d", id_map=pd.Series({"h0": "C0"}))
    # Project 1 writes scores for one cutoff: a file from another cutoff must be rejected, the right one accepted
    p1 = good.assign(cutoff_date=pd.Timestamp("2024-12-31"))
    with pytest.raises(ValueError, match="expected 2024-09-30"):
        standardize_external_scores(p1, "customer_id", "churn_prob_30d", expected_cutoff="2024-09-30")
    ok = standardize_external_scores(p1.assign(cutoff_date=pd.Timestamp("2024-09-30")), "customer_id", "churn_prob_30d",
                                     expected_cutoff="2024-09-30")
    assert len(ok) == 100


def test_uplift_learner_chosen_from_april_cv_only():
    import json
    m = json.load(open(OUT / "uplift_metrics.json"))
    cv = pd.DataFrame(m["learner_cv"])
    assert m["selected_learner"] == cv.loc[cv.cv_qini_mean.idxmax(), "learner"]
    ev = pd.DataFrame(m["evaluation"])
    assert ev.loc[ev.policy == "Uplift model", "learner"].iloc[0] == m["selected_learner"]
    # every learner beats the propensity ranking on true-uplift correlation
    assert (ev[ev.policy.str.startswith(("Uplift", "Alt"))].spearman_vs_true_uplift
            > ev.loc[ev.policy == "Propensity model", "spearman_vs_true_uplift"].iloc[0]).all()


def test_multiseed_results_support_the_headline_claims():
    """Regression guard on the committed 10-seed study: the qualitative claims must hold in every seed."""
    import json
    c = json.load(open(OUT / "multiseed_claims.json"))
    n = c["n_seeds"]
    assert n >= 10
    for k in ("propensity_rank_corr_negative", "uplift_rank_corr_beats_propensity", "qini_gap_vs_propensity_ci_above_zero",
              "uplift_top20_ci_above_overall_effect", "grow30_uplift_beats_random", "grow30_propensity_below_random",
              "grow30_propensity_negative", "bgnbd_rmse_beats_last90"):
        assert c[k] == n, f"{k} holds in only {c[k]}/{n} seeds"
    assert c["bgnbd_rmse_beats_lgbm"] < n          # documented: BG/NBD does NOT reliably beat LightGBM


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 20000)
    assert psi(a, rng.normal(0, 1, 20000)) < 0.05
    assert psi(a, rng.normal(1.0, 1, 20000)) > 0.25
