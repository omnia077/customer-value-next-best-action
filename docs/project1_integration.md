# Connecting Project 1's churn scores

Project 2 needs a churn probability for each subscriber. It ships with a simple stand-in (`src/churn_proxy.py`). Project 1 (Telecom-Churn-Platform, `ml_churn/`) has the real model. This page says how to connect it and what was verified.

## Why the data must be shared
Project 1's checked-in data and Project 2's data have the **same 20,000 customers** (IDs, MSISDNs, regions, plans, activation dates are identical) but **different behaviour**: Project 1 was run on the plain simulator (212,164 recharges), Project 2 on the simulator with planted archetypes and campaigns (205,823 recharges). Scores from one world do not describe the other, so Project 1 must be re-run on Project 2's dataset (`data_v2/`).

## Runbook (do this on a copy or a branch of Project 1: it overwrites Project 1's data and results)
1. Replace Project 1's `src/data/sftp_drop` and `src/data/ground_truth` with Project 2's `data_v2/sftp_drop` and `data_v2/ground_truth`.
2. In Project 1: `cd src/phase_2_etl && python run_phase2.py`.
3. From Project 1's root, with the real salt exported (`CHURN_HASH_SALT`): `python -m ml_churn.build_panel`, `leakage_checks`, `train`, `evaluate`, then `python -m ml_churn.score_monthly --as-of 2024-09-30`. The cutoff must be 30 Sep: Project 2 rejects scores from any other cutoff.
4. In Project 2's `config/customer_value.yaml` set:
   ```yaml
   churn:
     external_scores_path: <Project 1>/src/data/ml/churn_scores.parquet
     key_column: sub_hash
     prob_column: probability
   ```
5. Export the same `CHURN_HASH_SALT`, then `python src/score_monthly.py churn_scores actions persona_kpis drift_check mlflow_log` and `python src/dashboard.py`.

The adapter (`src/ids.py`) reproduces Project 1's scheme, `sha256(salt + msisdn)[:16]` (a test compares it with Project 1's own `hash_id` output), and fails loudly on: wrong salt (unmapped hashes), a different cutoff date, probabilities outside [0, 1], duplicate keys, or under 90% coverage of the scored population. The raw MSISDN is read only to build the hash map and is never written.

## What was verified (scratch copy of Project 1, throwaway salt; your Project 1 folder was not touched)
| Check | Result |
|---|---|
| Project 1 phases 2, panel, leakage checks, train, evaluate, score run on Project 2's data | all passed |
| Project 1 test performance on the shared data | ROC-AUC 0.81 (logistic and LightGBM), top-decile lift 4.85, base rate 1.96%: in line with Project 1's own reported numbers, so planted archetypes do not disturb it |
| Adapter on Project 1's real output (17,340 subscribers, cutoff 2024-09-30) | mapped every hash, passed all validations |
| Agreement with the stand-in (12,332 prepaid subscribers in both) | Spearman 0.79 |
| Actual October churn on those subscribers (237 churners) | Project 1: AUC 0.772, top-decile lift 3.5. Stand-in: AUC 0.746, lift 2.9. The real model is modestly better |
| Calibration on this population | Project 1 mean predicted churn 2.8% vs 1.9% observed (about 1.5x too high); stand-in 2.0% |

## Effect on the decision layer
With Project 1's scores the plan changes from 221 to 439 retention actions (grow 2,963 to 2,796, win-back 692 to 641) and expected value rises from 365k to 420k SDG. The rise comes from higher churn probabilities, and Project 1's probabilities run about 1.5x above observed churn on this population, so **the retention part of that increase is probably overstated**. Check calibration on the actual scoring population (or re-fit Project 1's isotonic calibration on prepaid-only rows) before trusting retention value. With 237 churners this comparison is noisy: treat it as a flag, not a finding.

Committed results in this repository still use the stand-in, because the real scores need your salt and a re-run of Project 1 on the shared data.
