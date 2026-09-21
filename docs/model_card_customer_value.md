# Model card: Customer Value Map and Next-Best-Action

**Status:** research demo on simulated data. Not validated for production.
**Owner:** data science. **As of:** 2024-09-30. **Code:** `src/`. **Config:** `config/customer_value.yaml`.

## Intended use
Rank prepaid subscribers for retention and recharge-bonus offers, and describe the base in a few personas. Not for credit, pricing, eligibility or any decision about an individual's access to service.

## Components

| Model | Method | Target / output | Training data | Validation |
|---|---|---|---|---|
| Segmentation | k-means, k=4, log1p + standardised RFM and usage mix (180d) | persona | 12,921 active prepaid subscribers | Silhouette 0.18; bootstrap ARI 0.89 (sd 0.17); recovery vs planted archetypes ARI 0.33 |
| CLV | BG/NBD + Gamma-Gamma (own implementation) | expected recharge value, 92d and 12m | Recharges 1 Jan-30 Sep | Holdout 1 Oct-31 Dec |
| Propensity | LightGBM (logistic baseline) | recharge >= 500 SDG next month | Cutoffs 29 Feb, 31 Jul, 31 Aug | Val Sep; test Oct, Nov cutoffs |
| Uplift | S-learner (LightGBM), chosen among 5 learners by April 5-fold CV Qini | change in P(recharge >= 500 in 30d) from the offer | April campaign (final model: April + June) | June campaign; ground-truth uplift |
| Churn (stand-in) | LightGBM | churn in next 30 days | Cutoffs Feb-Jul | Cutoff 31 Aug: AUC 0.825 |
| Actions | Expected-value allocation under a 30% contact budget | one action per subscriber | outputs of the above | Grow value vs truth; others assumption-based |

## Key metrics

CLV holdout (n = 13,800; actual spend 18.5M SDG):

| Model | Predicted / actual | RMSE (SDG) | Top-20% capture |
|---|---|---|---|
| Last-90-day spend | 1.09 | 2,412 | 63.6% |
| BG/NBD + Gamma-Gamma | 1.09 | 2,106 | 64.8% |
| LightGBM on RFM | 0.99 | 2,113 | 65.3% |
| Oracle | 1.00 | 0 | 76.4% |

BG/NBD over-predicts the top decile (7,315 vs 6,240 SDG actual). Gamma-Gamma fit has q = 0.90 (<1), i.e. a heavy-tailed spend distribution across customers.

Propensity (test, n = 24,141, base rate 24.1%): logistic ROC-AUC 0.859 / PR-AUC 0.657; LightGBM 0.870 / 0.672, top-decile lift 3.17, Brier 0.121. Isotonic calibration did not improve Brier (0.1216), so raw LightGBM scores are used.

Uplift (June, n = 13,482, randomized; average effect observed +8.2 pts, true +7.5):

| Ranking | Spearman vs true uplift | Expected incremental recharges, top 20% | Observed uplift, top 20% (pts, 95% CI) | Qini coef. (95% CI) |
|---|---|---|---|---|
| Uplift model (S-learner) | +0.67 | 359 | 14.1 (11.9, 16.5) | 0.0055 (0.0033, 0.0078) |
| Propensity model | -0.62 | 59 | 2.1 (-1.2, 5.7) | -0.0054 (-0.0076, -0.0035) |
| Random | 0 | 202 | ~8.2 | ~0 |
| Oracle | 1 | 489 | 17.6 (15.3, 20.0) | 0.0077 (0.0054, 0.0103) |

Other learners (T, regularised T, X, class transformation) are in `docs/uplift_design.md`; all beat propensity. Grow policy net value on June (true uplift, 25 SDG offer): uplift top 30% = 207k SDG, random 89k, propensity -24k, oracle 294k; uplift EV>0 rule 317k vs contact-everyone 296k. Uplift targeting beats contact-everyone from an offer cost of 19 SDG.

Expected value of the current action plan (365k SDG): 166k recharge bonus (validated against truth), 25k retention and 174k win-back (assumption-based, 54% of total).

## Limitations
- Simulated data; mechanisms are known to the author. The uplift effect exists only for people who would not otherwise buy, so propensity-negative correlation is built in.
- Uplift is calibrated reasonably on the simulated June data (quintile means predicted vs true: -0.1/1.9 lowest, 14.1/13.3 highest), but calibration must be re-checked on every real campaign. No negative responders are simulated. The choice between the S-learner and the regularised T-learner is within cross-validation noise.
- Segments are soft (silhouette 0.18). They are descriptive, not natural clusters.
- BG/NBD independence assumption is violated by construction; campaign recharges enter the CLV fit. `p_alive` ranks correctly (84% of lines with p_alive > 0.95 recharge in the next 92 days vs 27% of lines below 0.2), but a quarter of "probably lapsed" lines still recharge, so win-back value is an upper estimate.
- Save rate, win-back rate and offer costs are assumptions. Sensitivity: `outputs/sensitivity_assumptions.csv` (total EV ranges 311k-422k SDG across offer cost 15-40 and save rate 0.10-0.25). Break-even rates: `outputs/assumption_breakeven.csv` (retention 0.049 median vs 0.15 assumed; win-back 0.012 vs 0.06).
- The churn input is a stand-in with simpler features than Project 1's model. A validated adapter (`src/ids.py`) swaps in Project 1's scores through config.
- Prepaid only; postpaid has no recharge events.

## Fairness and privacy
`age`, `gender`, `region` and MSISDN are not model inputs. Contact-rate audit by age band and region: `outputs/fairness_audit.csv` (age 27.2-31.5%, region 27.7-33.0%). Region can still act through behaviour features (network stress in the simulator affects usage), so keep monitoring. Only surrogate IDs appear in outputs.

## Monitoring
`drift_check` step: PSI on 15 input features (month over month) and on score and segment distributions against a saved baseline. Thresholds: <0.1 OK, 0.1-0.25 WATCH, >0.25 ALERT. Current: all OK except `tenure_days` (0.10, expected). Retrain uplift after every campaign that has a randomized holdout.
