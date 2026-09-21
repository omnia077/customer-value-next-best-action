# Director brief: who to send the recharge bonus to

*All figures are from simulated data. They show the method, not real-world performance.*

## Recommendation
1. **Target the recharge bonus by uplift, not by likelihood to recharge.** Customers most likely to recharge 500+ SDG anyway are the worst targets. In the June test, contacting the top 30% by propensity **lost** 24k SDG, random earned 89k SDG and uplift targeting earned 207k SDG.
2. **Give each subscriber at most one action, within a budget.** For 12,921 active prepaid subscribers, 3,876 (30%) get an action: 2,963 recharge bonus, 692 win-back, 221 retention. Expected net value is about 365k SDG for the cycle: **166k SDG from the recharge bonus (checked against the simulator's true effects)** and 199k SDG from win-back and retention (these use assumed success rates).
3. **Run the next campaign with a randomized 10% holdout** so the model is re-estimated on real results and the win-back and retention rates are measured, not assumed.

## What we found
- **Four personas.** Core high-value data-led: 26% of subscribers, 54% of revenue, highest propensity but below-average uplift (6.3 points, second lowest: they mostly recharge anyway). Steady voice-led: 34% / 30%, lowest uplift (5.2 points). Lapsing data-led: 23% / 11%, highest win-back value. Light mixed-use: 17% / 5%, cheapest to serve.
- **Propensity is not persuasion.** Rank correlation with the true effect of the offer: uplift model +0.67, propensity model -0.62. In the top 20% by uplift, the offer added 14.1 points to the recharge rate (interval 11.9 to 16.5) against 8.2 points for everyone.
- **Targeting pays once the offer costs more than about 19 SDG per contact.** At the assumed 25 SDG it beats contacting everyone (317k vs 296k SDG). At 50 SDG, contacting everyone loses 41k SDG while targeting still earns 109k SDG.

- **The result is not a lucky draw.** The simulation was repeated in 10 independent worlds. In all 10, propensity targeting lost money (average -23k SDG at a 30% budget) and uplift targeting beat random (average +192k vs +88k SDG). The rank correlation with the true effect averaged +0.56 for the uplift model (range +0.50 to +0.65); the +0.67 quoted above is the most favourable of the runs.
- **The edge over contacting everyone is small at 25 SDG** (about 5% on average) and grows quickly with offer cost, so confirm the real cost per contact before relying on it.

## Chart to show
Uplift vs propensity (dashboard page "Uplift vs propensity"): cumulative incremental recharges as more subscribers are contacted. The propensity line sits below random.

## Assumptions and risks
- Offer cost 25 SDG per contact; retention save rate 15%; win-back rate 6%. Changing them moves total value between 311k and 422k SDG. The retention and win-back actions stay worthwhile even if the true rates are half the assumed ones, but their value halves too. About a quarter of lines flagged as lapsed still recharge on their own, so win-back value is an upper estimate.
- Segments are soft profiles, not natural groups.
- Churn risk comes from a simplified stand-in model until the churn platform's scores are connected (a checked adapter is ready).

## Pilot needed to confirm
A 4-week pilot on one region: 50% of the uplift-selected group gets the offer, 50% is held out. Define success before the pilot (for example, observed uplift in the selected group above the all-customer average, with the confidence interval above zero) and record win-back and retention outcomes to replace the assumed rates.
