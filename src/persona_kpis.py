"""Persona KPIs in DuckDB, mirroring the billing DWH's definitions (src/phase_6_mart/financial_mart.py):
  ARPU = SUM(revenue) / COUNT(DISTINCT customer_id)   -> revenue here = recharge amount (prepaid top-ups)
  MOU  = SUM(voice minutes)                             -> usage_value of voice events (assumed minutes per session)
Also writes the Gold-layer additions: dim_segment and fact_customer_value.

Difference from the DWH: the DWH counts customers with a billing event in the month (ARPU per paying user). We also
report revenue per persona member (`arpu_per_member`) because persona sizes differ.

Run:  python src/persona_kpis.py
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from segment import load_cfg  # noqa: E402


def run(cfg_path):
    cfg = load_cfg(cfg_path)
    out, sftp = cfg["_out"], cfg["_data"] / "sftp_drop"
    mart = out / "mart"
    mart.mkdir(exist_ok=True)
    seg = pd.read_parquet(out / "segment_assignments.parquet")
    con = duckdb.connect()
    con.register("seg", seg[["customer_id", "segment_id", "persona"]])
    con.execute(f"CREATE VIEW recharge AS SELECT customer_id, CAST(recharge_date AS DATE) d, amount_sdg FROM read_parquet('{(sftp / 'recharge_records.parquet').as_posix()}')")
    con.execute(f"CREATE VIEW usage AS SELECT customer_id, CAST(event_date AS DATE) d, usage_type, usage_value FROM read_parquet('{(sftp / 'usage_events.parquet').as_posix()}')")

    arpu = con.execute("""
        SELECT s.persona, strftime(r.d, '%Y-%m') AS month,
               ROUND(SUM(r.amount_sdg), 2)                                   AS total_revenue,
               COUNT(DISTINCT r.customer_id)                                 AS active_subscribers,
               ROUND(SUM(r.amount_sdg) / NULLIF(COUNT(DISTINCT r.customer_id), 0), 2) AS arpu,
               ROUND(SUM(r.amount_sdg) / (SELECT COUNT(*) FROM seg s2 WHERE s2.persona = s.persona), 2) AS arpu_per_member
        FROM recharge r JOIN seg s USING (customer_id)
        GROUP BY 1, 2 ORDER BY 2, 1""").df()
    mou = con.execute("""
        SELECT s.persona, strftime(u.d, '%Y-%m') AS month,
               ROUND(SUM(u.usage_value), 1) AS total_mou,
               ROUND(SUM(u.usage_value) / COUNT(DISTINCT u.customer_id), 2) AS mou_per_active_subscriber
        FROM usage u JOIN seg s USING (customer_id)
        WHERE u.usage_type = 'voice'
        GROUP BY 1, 2 ORDER BY 2, 1""").df()
    kpi = arpu.merge(mou, on=["persona", "month"], how="outer")
    kpi.to_parquet(mart / "persona_kpis_monthly.parquet", index=False)
    kpi.to_csv(mart / "persona_kpis_monthly.csv", index=False)

    # Gold additions (star-schema style)
    prof = pd.read_csv(out / "segment_profile.csv")
    dim = prof[["segment_id", "persona", "n", "pct_subs", "pct_revenue"]].assign(
        as_of=cfg["segmentation"]["as_of"], k=cfg["segmentation"]["k_final"],
        definition="k-means on log-scaled RFM+usage mix, 180d window, active in last %d days" % cfg["segmentation"]["active_within_days"])
    dim.to_parquet(mart / "dim_segment.parquet", index=False)
    fact = pd.read_parquet(out / "customer_actions.parquet")
    fact.to_parquet(mart / "fact_customer_value.parquet", index=False)
    print(f"persona KPI rows: {len(kpi)} | dim_segment {len(dim)} | fact_customer_value {len(fact):,}")
    print(kpi[kpi.month == "2024-09"].to_string(index=False))
    return kpi


if __name__ == "__main__":
    run(Path(__file__).resolve().parent.parent / "config" / "customer_value.yaml")
