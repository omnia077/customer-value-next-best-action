# Customer Value Map and Next-Best-Action

For each prepaid subscriber: **who they are** (persona), **what they are worth** (CLV), **who is persuaded by an offer** (uplift) and **what to do** (one action under a budget).

> A propensity model tells you who will buy. An uplift model tells you who buys *because of the offer*. On the simulated randomized campaign, targeting by propensity did worse than random.

**All data is simulated** (`data_generation/phase1b_generate_data.py --archetypes --campaign`, 20,000 customers, seed 42). Results demonstrate the method, not real-world performance.

## Headline results (from `outputs/`)

| Area | Result |
|---|---|
| Population | 12,921 prepaid subscribers active in the last 90 days (as of 30 Sep 2024) |
| Segments (k=4) | Silhouette 0.18, bootstrap stability ARI 0.89, **recovery of planted archetypes ARI 0.33** (0 = random). Segments are soft profiles |
| CLV (holdout Oct-Dec) | BG/NBD + Gamma-Gamma RMSE 2,106 SDG vs 2,412 (last-90-day) and 2,113 (LightGBM): modest gain, ties ML |
| Propensity (Nov, Dec) | LightGBM ROC-AUC 0.870, PR-AUC 0.672, top-decile lift 3.2x, well calibrated |
| Uplift (June, randomized) | Learner chosen on April cross-validation only (S-learner). Rank correlation with true uplift **+0.67** vs **-0.62** for propensity. Top-20% observed uplift **+14.1 pts (95% CI 11.9-16.5)** vs +8.2 overall. Qini coefficient 0.0055 vs -0.0054; difference 0.0111 (95% CI 0.0071-0.0143) |
| Grow policy on June holdout | 30% budget: uplift **207k SDG**, random 89k, propensity **-24k**, oracle 294k. Without a budget cap, the uplift EV rule nets 317k vs 296k for contacting everyone (25 SDG offer cost) |
| Actions | 3,876 subscribers (30%) get one action: 2,963 recharge bonus, 692 win-back, 221 retention. Expected net value 365k SDG, of which **166k (46%) is validated against simulator truth** and 199k rests on assumed rates |

## Caveats (read before quoting any number)

- **Uplift direction is by design.** The simulated offer converts only people who would not otherwise have recharged 500+ SDG. The method and its evaluation against known truth are the demonstration, not the effect size.
- **Learner choice is not decisive.** April cross-validated Qini: S-learner 0.0054, regularised T-learner 0.0050, X-learner 0.0033, T-learner 0.0026, class transformation 0.0015, each with sd about 0.003. The top two are within noise. All five beat propensity targeting on June.
- **54% of expected value rests on assumed rates** (win-back 174k, retention 25k). Break-even analysis (`outputs/assumption_breakeven.csv`): retention stays positive for 93% of allocated subscribers and win-back for 100% even if the true rate is half the assumed one, but the value scales down with the rate. The `p_alive` check (`outputs/clv_p_alive_check.csv`) shows BG/NBD ranks lapse risk correctly (84% of "alive" lines recharge in the next 92 days vs 27% of "probably lapsed" lines), but that 27% means many win-back contacts would be wasted: treat win-back value as an upper estimate until a pilot measures the real rate.
- **Uplift beats contact-everyone only above an offer cost of about 19 SDG** (assumed: 25). Below that, contacting everyone is as good; at 50 SDG contacting everyone loses 41k while the uplift rule earns 109k (`outputs/grow_sweep_offer_cost.csv`).
- **Churn input is a stand-in** (`src/churn_proxy.py`, AUC 0.83). Project 1's ML scores are not in this repo; a validated adapter is ready (`src/ids.py`, see below).
- BG/NBD assumes purchase rate and dropout are independent; the simulator ties both to one hidden state, so that assumption is violated by design. Campaign-induced recharges (April, June) are in the CLV fit window.
- **Billing DWH is reused for KPI definitions only** (ARPU, MOU mirrored in DuckDB). It holds a different population and no recharge events, so the two cannot be joined until both are generated from the same customers.

## Run it

```bash
pip install -r requirements.txt
python data_generation/phase1b_generate_data.py --archetypes --campaign    # writes data_v2/ (gitignored, ~150 MB)
python src/score_monthly.py                                                # whole pipeline; name steps to run a subset
python src/dashboard.py                                                    # outputs/dashboard.html
python src/build_deck.py                                                   # docs/customer_value_deck.pptx
python -m pytest tests -q                                                  # 16 tests
```

Individual stages: `python src/segment.py [--select]`, `clv.py`, `propensity.py`, `uplift.py`, `churn_proxy.py`, `actions.py`, `persona_kpis.py`.

## Connect Project 1's churn scores

1. Export Project 1's scores as `.parquet` or `.csv` with one row per subscriber and a probability column.
2. In `config/customer_value.yaml` set `churn.external_scores_path`, and `key_column` (`customer_id` or `msisdn_hash`) and `prob_column`.
3. If keyed by `msisdn_hash`, export `MSISDN_SALT` (the same salt and scheme as Project 1; scheme is in one function, `ids.hash_msisdn`). Raw MSISDNs are read only inside `ids.msisdn_to_customer_id`, hashed, and discarded.
4. `python src/score_monthly.py churn_scores actions`. The adapter fails loudly on out-of-range probabilities, duplicate keys, unmapped hashes (wrong salt) or under 90% coverage.

## CRISP-DM map

| Phase | Where |
|---|---|
| Business understanding | This README; `docs/director_brief.md` (EN) / `docs/director_brief_ar.md` (AR) |
| Data understanding | `docs/uplift_design.md` (randomization checks); `src/features.py` |
| Data preparation | `src/features.py`: point-in-time features (a test proves no future data) |
| Modelling | `src/segment.py`, `clv.py`, `propensity.py`, `uplift.py`, `churn_proxy.py`, `actions.py` |
| Evaluation | `outputs/*.csv|json`; checks against simulator truth in `segment.py`, `uplift.py`, `actions.py` |
| Deployment | `src/score_monthly.py`, `src/customer_value_dag.py`, `src/persona_kpis.py`, `outputs/dashboard.html`, MLflow (`mlflow.db`, gitignored) |

## Pipeline and outputs

`validate inputs -> segment -> CLV -> propensity -> uplift -> churn scores -> actions -> persona KPIs -> drift check -> MLflow`

Tables in `outputs/` (Parquet, gitignored, regenerated by the pipeline): `segment_assignments`, `clv_scores`, `propensity_scores`, `uplift_scores`, `churn_scores`, `customer_actions`; `outputs/mart/`: `persona_kpis_monthly`, `dim_segment`, `fact_customer_value`. Results as CSV/JSON and the dashboard are committed.

## Decisions and deviations from the working spec

| Item | What was done | Why |
|---|---|---|
| k | 4 (k=3 was more stable: ARI 0.95 vs 0.89) | k=4 separates a very low-value group (4.8% of revenue) from a lapsing data-led group (10.7%), which need different actions. k was **not** chosen on the recovery score |
| GMM | Tried in an early run (BIC unreliable for k>=6); not in the final module | Adds nothing over k-means here |
| Population | Active in last 90 days | Sensitivity run: without the rule 11% of lines had already churned and skewed two segments; conclusions otherwise similar (`outputs/k_selection_*.csv`) |
| CLV library | Own BG/NBD + Gamma-Gamma (`src/clv.py`) | `lifetimes` is unmaintained and fragile with pandas 3. Formula checked against Monte Carlo (92-day expectation 6.086 vs 6.086); fitted a=0.07 (<1) is valid |
| Uplift learners | 5 compared, chosen by April cross-validation | The spec's first learner (T-learner) had rank correlation +0.47; the CV-selected S-learner +0.67. June was never used to choose |
| Propensity features | Segment ID **not** used | Segments are fit as of 30 Sep; using them at earlier cutoffs would leak the future |
| Churn | Stand-in model + adapter for Project 1 | Project 1 scores not available |
| Win-back value | Weighted by (1 - p_alive) | First version priced every lapsed line as truly gone and over-allocated budget |
| Airflow | DAG written, **not executed** | Airflow not installed here; the same steps are run and tested through `score_monthly.py` |
| Model registry | Not used; metrics and artifacts logged to MLflow (sqlite) | Scores are produced by the run that trains the models |
| Billing DWH | KPI definitions mirrored; `dim_segment` / `fact_customer_value` ready for its Gold layer | Different population, see caveats |
| ID hashing | `customer_id` everywhere; hashing helper for the Project 1 join | No MSISDN reaches any output |

## Governance

Segments and CLV are commercially sensitive: restrict by role. `age`, `gender`, region and MSISDN are not model inputs; `outputs/fairness_audit.csv` shows contact rates by age band (27.2-31.5%) and region (27.7-33.0%). One action per subscriber per period (`test_actions_one_per_subscriber_and_budget`). Document retention and permitted use of campaign data before real use.

## Repository layout

```
config/customer_value.yaml      windows, cut-offs, seeds, assumptions (all cost/rate assumptions live here)
data_generation/                simulator (archetypes + randomized campaigns)
src/                            features, segment, clv, propensity, uplift, churn_proxy, actions, ids,
                                persona_kpis, score_monthly, customer_value_dag, dashboard, build_deck
tests/                          16 tests (point-in-time safety, ground-truth access, allocation invariants, adapters)
docs/                           model card, uplift design, BI requirements, director briefs (EN/AR), deck
outputs/                        results (CSV/JSON/dashboard committed; Parquet tables regenerated)
```
