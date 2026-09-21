# Uplift design

## Data (simulated)
Two randomized "recharge bonus" campaigns: April (12,051 eligible) and June (13,482 eligible). Eligible = prepaid and active at the send date. About 50% were randomly treated. Source: `data_v2/campaigns/campaign_history` (campaign, customer, send date, treated). `ground_truth/campaign_truth` holds the true individual uplift and is read **only** for validation.

Mechanism (from the generator): treatment converts a subscriber who would *not* otherwise have recharged >= 500 SDG that month, with probability tau (depends on archetype and price sensitivity). True uplift = tau x (1 - baseline probability). Sure things and lost causes get no incremental effect. No negative responders. Both campaigns share one effect structure.

## Outcome and features
Outcome: at least one recharge >= 500 SDG in [send date, send date + 29 days]. Features: as of the day before sending (`asof_features`: 30/60-day recharge and usage counts and amounts, recency, usage mix, tenure, monthly ARPU). A test (`test_features_use_no_future_data`) deletes all later events and checks features do not change. June features include April conversions, which is legitimate: June assignment is independent of April's.

## Checks
| | April | June |
|---|---|---|
| Treated share | 49.9% | 49.8% |
| Control / treated rate | 25.7% / 33.2% | 24.6% / 32.8% |
| Observed effect (se) | +7.5 (0.8) pts | +8.2 (0.8) pts |
| True mean uplift | +7.5 pts | +7.5 pts |
| Max abs. standardised mean difference | 0.034 | 0.049 |

Balanced (|SMD| < 0.1) and the observed effect matches truth within one standard error.

## Learners and selection
Five learners on the same LightGBM base: T-learner, regularised T-learner, S-learner, X-learner, class transformation. **Selection uses 5-fold cross-validated Qini on the April campaign only**; June is never used to choose. The comparison model ("propensity") predicts the outcome ignoring treatment.

| Learner | April CV Qini (mean, sd) | June Spearman vs true | June top-20% observed uplift (pts) | June Qini (95% CI) |
|---|---|---|---|---|
| **S-learner (selected)** | 0.0054 (0.0032) | +0.67 | 14.1 | 0.0055 (0.0033, 0.0078) |
| T-learner (regularized) | 0.0050 (0.0036) | +0.58 | 13.4 | 0.0050 (0.0028, 0.0076) |
| X-learner | 0.0033 (0.0037) | +0.52 | 12.5 | 0.0047 (0.0027, 0.0072) |
| T-learner | 0.0026 (0.0034) | +0.47 | 10.3 | 0.0042 (0.0021, 0.0064) |
| Class transformation | 0.0015 (0.0030) | +0.37 | 10.3 | 0.0027 (0.0008, 0.0050) |
| Propensity model | | -0.62 | 2.1 | -0.0054 (-0.0076, -0.0035) |
| Oracle | | +1.00 | 17.6 | 0.0077 (0.0054, 0.0103) |

The S-learner's April CV lead over the regularised T-learner is within one standard deviation: treat the two as equivalent. Every learner beats propensity targeting. **These numbers are for one simulated world (seed 42), which is a favourable draw.** Across seeds 1-10 the CV-selected learner's rank correlation with the true effect is +0.56 on average (0.50 to 0.65), a fixed T-learner +0.39, propensity -0.62; the selected learner was the S-learner in 6 seeds, the regularised T-learner in 3, the X-learner in 1 (see `docs/model_card_customer_value.md`). Uplift minus propensity Qini (paired bootstrap): 0.0111 (95% CI 0.0071, 0.0143).

Calibration of the selected model on June (quintile means, points): predicted -0.1 / 2.8 / 6.0 / 8.7 / 14.1, true 1.9 / 5.1 / 7.9 / 9.3 / 13.3, observed 1.7 / 5.7 / 8.3 / 10.4 / 14.1.

The final scoring model is refit on both campaigns and applied to 30 Sep features. Feature distributions have moved since March/May (tenure), so re-check calibration on each new campaign.

## What this does and does not show
- Shows: targeting on propensity picks "sure things"; the method recovers most of the ordering; Qini and randomized data detect this without ground truth.
- Does not show: real effect sizes, negative responders, or effect changes over time. In a real campaign, keep a randomized holdout (for example 10%) every time so the model can be re-estimated.
