# outputs/

Everything here is produced by `python src/score_monthly.py` (plus `src/multiseed.py`, `src/compare_churn_sources.py`, `src/dashboard.py`). All data is simulated.

**Committed** (small, aggregate, no subscriber IDs): the results below. **Not committed** (regenerated, listed in `.gitignore`): the subscriber-level Parquet tables `segment_assignments`, `clv_scores`, `propensity_scores`, `uplift_scores`, `churn_scores`, `customer_actions`, `uplift_june_test_scored`, `baseline_scores` and `mart/*.parquet`.

| Area | Files |
|---|---|
| Dashboard | `dashboard.html` (open in a browser; deep links `#t0` to `#t4`) |
| Segmentation | `segment_profile.csv`, `segment_vs_archetype.csv`, `segmentation_metrics.json`, `k_selection_all_active.csv`, `k_selection_active90.csv`, `persona_cards.csv` |
| CLV | `clv_holdout.csv` (bake-off), `clv_calibration_deciles.csv`, `clv_p_alive_check.csv`, `clv_metrics.json` |
| Propensity | `propensity_metrics.json`, `propensity_calibration_test.csv` |
| Uplift | `uplift_evaluation.csv`, `uplift_learner_cv.csv`, `uplift_calibration_quintiles.csv`, `uplift_qini_curves.csv`, `uplift_metrics.json` |
| Decisions | `actions_metrics.json`, `grow_policy_evaluation.csv`, `grow_sweep_offer_cost.csv`, `grow_sweep_budget.csv`, `assumption_breakeven.csv`, `sensitivity_assumptions.csv`, `sensitivity_matrix_cutoffs.csv`, `fairness_audit.csv` |
| Churn input | `churn_proxy_metrics.json` (stand-in), `project1_comparison.json`, `project1_actions.csv`, `project1_calibration.csv` (Project 1 vs stand-in) |
| Monitoring | `drift_check.csv` |
| Robustness | `multiseed_summary.csv`, `multiseed_per_seed.csv`, `multiseed_claims.json` (10 simulated worlds) |
| BI mart | `mart/persona_kpis_monthly.csv` (ARPU, MOU by persona and month) |

Regenerating changes numbers only if data, seeds or config change; a fresh clone reproduces these files exactly on the machine they were produced on (CI compares and reports differences without failing, since last digits can differ across platforms).
