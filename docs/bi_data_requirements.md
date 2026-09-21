# BI data requirements

Source: `outputs/mart/` (Parquet; DuckDB/Power BI/Tableau readable). Refreshed monthly by DAG `customer_value_monthly` (2nd of month, 05:00), after the churn DAG.

| Table | Grain | Key | Main columns |
|---|---|---|---|
| `fact_customer_value` | subscriber x as-of month | `customer_id`, `as_of` | persona, segment_id, quadrant, action, ev, clv_92d, clv_12m, clv_12m_if_alive, p_alive, churn_prob_30d, p_recharge500_next_month, uplift, recency_days |
| `dim_segment` | persona per as-of | `segment_id`, `as_of` | persona, subscribers, pct_subs, pct_revenue, k, definition |
| `persona_kpis_monthly` | persona x calendar month | `persona`, `month` | total_revenue, active_subscribers, arpu, arpu_per_member, total_mou, mou_per_active_subscriber |

**KPI definitions** (mirror `financial_mart.py` in the billing DWH): ARPU = revenue / distinct customers with revenue that month; MOU = sum of voice minutes. Revenue here is prepaid recharge amount, MOU assumes `usage_value` for voice is minutes.

**Required measures:** subscribers, % of subscribers, % of revenue, average 12-month value, average churn probability, contacted subscribers, expected net value.
**Filters:** persona, action, region (via join to customers), as-of month. **Refresh:** monthly; full reload per as-of.
**Security:** row keys are surrogate IDs only. Restrict `fact_customer_value` by role (commercial, retention). Do not export subscriber-level rows outside the retention team.
**Labels required on every page:** "simulated data"; "expected values use assumed costs and rates".
