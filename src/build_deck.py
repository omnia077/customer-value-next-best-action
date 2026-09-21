"""5-slide deck from outputs/. Run: python src/build_deck.py  ->  docs/customer_value_deck.pptx"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, str(Path(__file__).parent))
from segment import load_cfg  # noqa: E402

INK, INK2, BG = RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E), RGBColor(0xFC, 0xFC, 0xFB)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def add_text(slide, x, y, w, h, text, size=18, bold=False, color=INK, bold_first=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text if isinstance(text, list) else [text]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size, p.font.color.rgb = Pt(size), color
        p.font.bold = bold or (bold_first and i == 0)
        p.space_after = Pt(6)
    return tb


def add_table(slide, df, x, y, w, col_w=None, size=12):
    rows, cols = len(df) + 1, len(df.columns)
    t = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(0.36 * rows)).table
    for j, c in enumerate(df.columns):
        t.cell(0, j).text = str(c)
    for i, r in enumerate(df.itertuples(index=False), 1):
        for j, v in enumerate(r):
            t.cell(i, j).text = str(v)
    for i in range(rows):
        for j in range(cols):
            for p in t.cell(i, j).text_frame.paragraphs:
                p.font.size = Pt(size)
    if col_w:
        for j, cw in enumerate(col_w):
            t.columns[j].width = Inches(cw)


def slide(prs, title):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    add_text(s, 0.6, 0.35, 12.1, 0.9, title, size=28, bold=True)
    add_text(s, 0.6, 7.0, 12.1, 0.35, "Simulated data: results demonstrate the method, not real-world performance.", size=11, color=INK2)
    return s


def main(cfg_path):
    cfg = load_cfg(cfg_path)
    o = cfg["_out"]
    docs = cfg["_root"] / "docs"
    cards = pd.read_csv(o / "persona_cards.csv")
    qc = pd.read_csv(o / "uplift_qini_curves.csv")
    grow = pd.read_csv(o / "grow_policy_evaluation.csv")
    up = pd.DataFrame(json.load(open(o / "uplift_metrics.json"))["evaluation"]).set_index("policy")
    acts = json.load(open(o / "actions_metrics.json"))

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # 1. answer
    s = slide(prs, "Send the offer to people it persuades, not to people who would buy anyway")
    ms_path = o / "multiseed_summary.csv"
    ms = pd.read_csv(ms_path, index_col=0) if ms_path.exists() else None
    mc = json.load(open(o / "multiseed_claims.json")) if ms is not None else None
    if ms is not None:      # headline = mean over independent simulated worlds (more conservative than the main run)
        sp_up, sp_pr = ms.loc["uplift_spearman", "mean"], ms.loc["propensity_spearman", "mean"]
        v_up, v_pr = round(ms.loc["grow30_uplift_sdg", "mean"] / 1000), round(ms.loc["grow30_propensity_sdg", "mean"] / 1000)
        basis = f"mean of {mc['n_seeds']} simulated worlds"
    else:
        sp_up, sp_pr = up.loc["Uplift model", "spearman_vs_true_uplift"], up.loc["Propensity model", "spearman_vs_true_uplift"]
        g30 = grow.set_index("policy")["net_value_sdg"]
        v_up = int(g30[[i for i in g30.index if i.startswith("Uplift model, top")][0]] / 1000)
        v_pr = int(g30[[i for i in g30.index if i.startswith("Propensity")][0]] / 1000)
        basis = "single simulated world"
    for i, (big, small) in enumerate([(f"{sp_up:+.2f} vs {sp_pr:+.2f}", f"rank correlation with the true effect of the offer: uplift model vs propensity model ({basis})"),
                                      (f"{v_pr:+d}k vs {v_up:+d}k SDG", f"net value of the offer at a 30% budget: propensity targeting vs uplift targeting ({basis})"),
                                      (f"{acts['contacted']:,} of {acts['population']:,}", f"subscribers get one action; about {round(acts['total_ev_sdg']/1000)}k SDG expected net value (assumed costs)")]):
        add_text(s, 0.6 + i * 4.15, 1.7, 3.9, 0.9, big, size=30, bold=True, color=RGBColor(0x2A, 0x78, 0xD6))
        add_text(s, 0.6 + i * 4.15, 2.6, 3.9, 1.6, small, size=16, color=INK2)
    add_text(s, 0.6, 4.6, 12, 2, ["Who they are: 4 personas.  What they are worth: BG/NBD lifetime value.",
                                  "Who is persuaded: uplift model checked against a simulated randomized campaign.",
                                  "What to do: one action per subscriber under a contact budget."], size=18)

    # 2. personas
    core = cards.loc[cards.pct_revenue_180d.idxmax()]
    lapse = cards.loc[cards.avg_churn_prob_30d.idxmax()]
    low_up = cards.loc[cards.avg_uplift_pts.idxmin()]
    avg_prop = float((cards.avg_propensity_500 * cards.subscribers).sum() / cards.subscribers.sum())
    s = slide(prs, "Four personas: the largest revenue group is mostly people who would recharge anyway")
    t = cards[["persona", "pct_subs", "pct_revenue_180d", "avg_clv_12m", "avg_churn_prob_30d", "avg_propensity_500", "avg_uplift_pts", "pct_contacted"]].copy()
    t.columns = ["Persona", "% subs", "% revenue", "12m value (SDG)", "Churn % (30d)", "P(recharge 500+) %", "Uplift (pts)", "% contacted"]
    t["12m value (SDG)"] = t["12m value (SDG)"].map("{:,.0f}".format)
    add_table(s, t, 0.6, 1.6, 12.1, [3.2, 1.0, 1.2, 1.7, 1.4, 1.7, 1.1, 1.3][:len(t.columns)], size=13)
    add_text(s, 0.6, 3.9, 12, 3, [f"{core.persona}: {core.pct_revenue_180d}% of revenue; P(recharge 500+) {core.avg_propensity_500}% vs {avg_prop:.1f}% average, uplift {core.avg_uplift_pts} pts: mostly sure things.",
                                  f"{lapse.persona}: {lapse.pct_revenue_180d}% of revenue, highest churn risk ({lapse.avg_churn_prob_30d}% in 30 days), {lapse.pct_win_back}% get a win-back offer.",
                                  f"Lowest uplift: {low_up.persona} ({low_up.avg_uplift_pts} pts), which is why only {low_up.pct_contacted}% of it is contacted.",
                                  "Segments are soft profiles (silhouette 0.18; recovery of planted types ARI 0.33), so use them to describe, not to prove."], size=16, color=INK2)

    # 3. uplift chart
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.plot(qc.share_contacted * 100, qc["Uplift model"], color=BLUE, lw=2, label="Uplift model")
    ax.plot(qc.share_contacted * 100, qc["Propensity model"], color=ORANGE, lw=2, label="Propensity model")
    ax.plot(qc.share_contacted * 100, qc["Oracle (true uplift)"], color=AQUA, lw=2, ls="--", label="Oracle")
    ax.plot([0, 100], [0, qc["Oracle (true uplift)"].iloc[-1]], color="#898781", lw=1.5, ls=":", label="Random")
    ax.set_xlabel("share of eligible subscribers contacted (%)", color="#52514e"); ax.set_ylabel("cumulative incremental recharges", color="#52514e")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", lw=.8); ax.tick_params(colors="#898781"); ax.legend(frameon=False)
    fig.tight_layout(); img = o / "qini_slide.png"; fig.savefig(img); plt.close(fig)
    s = slide(prs, "Propensity targeting did worse than random; uplift targeting did better")
    s.shapes.add_picture(str(img), Inches(0.6), Inches(1.4), width=Inches(7.6))
    u = up.loc["Uplift model"]
    add_text(s, 8.5, 1.6, 4.4, 5, [f"Qini coefficient (95% CI):",
                                   f"Uplift {u.qini_coef} {u.qini_coef_ci95}",
                                   f"Propensity {up.loc['Propensity model','qini_coef']} {up.loc['Propensity model','qini_coef_ci95']}",
                                   "June campaign, randomized, 13,482 subscribers. The effect direction is built into the simulation; the method and its check against known truth are the point.",
                                   f"Top-20% observed uplift {u.observed_uplift_top20_pts:+.1f} pts (CI {u.observed_uplift_top20_ci95_pts[0]}-{u.observed_uplift_top20_ci95_pts[1]}) vs +8.2 overall. Learner ({u.learner}) chosen on April data only."], size=14, color=INK2)

    # 4. action plan
    s = slide(prs, "One action per subscriber under a 30% budget")
    g = grow.rename(columns={"policy": "Grow policy (June holdout)", "expected_incremental_recharges": "Incremental recharges", "net_value_sdg": "Net value (SDG)"})
    g["Net value (SDG)"] = g["Net value (SDG)"].map("{:,.0f}".format)
    g["Incremental recharges"] = g["Incremental recharges"].map("{:,.0f}".format)
    g["contacted"] = g["contacted"].map("{:,}".format)
    g = g.rename(columns={"contacted": "Contacted"})
    add_table(s, g, 0.6, 1.5, 12.1, [5.5, 2.0, 2.3, 2.3], size=13)
    add_text(s, 0.6, 4.3, 12, 2.6, [f"Actions: {acts['action_counts'].get('grow', 0):,} recharge bonus, {acts['action_counts'].get('win_back', 0):,} win-back, {acts['action_counts'].get('protect', 0):,} retention.",
                                    f"Uplift targeting beats contacting everyone from an offer cost of {acts['breakeven_offer_cost_uplift_rule_vs_everyone']} SDG (assumed: 25). At 50 SDG, contacting everyone loses money while the uplift rule still earns.",
                                    "Recharge-bonus value is checked against simulator truth. Retention and win-back values use assumed rates; they stay positive even if the true rates are half the assumed ones (values scale down)."], size=15, color=INK2)

    # 5. limits + pilot
    s = slide(prs, "Limits, and the pilot that would confirm it")
    bg_vs_gb = f"BG/NBD beats last-90-day spend in {mc['bgnbd_rmse_beats_last90']}/{mc['n_seeds']} seeds but LightGBM in only {mc['bgnbd_rmse_beats_lgbm']}/{mc['n_seeds']}." if mc else "BG/NBD roughly ties a plain ML model on holdout."
    add_text(s, 0.6, 1.5, 6, 3.4, ["Limits", "- Simulated data; uplift direction is by design.", "- Segments are soft (ARI 0.33-0.36 vs planted types).",
                                   f"- {bg_vs_gb}", "- Churn input is a stand-in; Project 1's real model ranks risk better (AUC +0.03) but its probabilities run high.",
                                   "- Retention and win-back rates are assumptions (win-back is an upper estimate)."], size=15, bold_first=True)
    if mc:
        add_text(s, 0.6, 5.0, 12.1, 1.8, [f"Robustness: {mc['n_seeds']} independent simulated worlds",
                                          f"Propensity rank correlation negative, uplift beats propensity, Qini gap above zero and uplift top-30% beats random in {mc['n_seeds']}/{mc['n_seeds']} seeds.",
                                          f"Uplift rank correlation {ms.loc['uplift_spearman','mean']:.2f} on average (range {ms.loc['uplift_spearman','min']:.2f}-{ms.loc['uplift_spearman','max']:.2f}); the main run (0.67) is a favourable draw.",
                                          "Uplift beats contacting everyone by about 5% at a 25 SDG offer and decisively from 50 SDG."], size=15, bold_first=True)
    add_text(s, 7.0, 1.5, 5.8, 3.4, ["Pilot", "- 4 weeks, one region.", "- 50% of the uplift-selected group gets the offer, 50% held out.",
                                     "- Define success first: uplift above the average with the CI above zero.",
                                     "- Keep a randomized holdout in every campaign and retrain after each."], size=15, bold_first=True)
    prs.save(docs / "customer_value_deck.pptx")
    print("wrote", docs / "customer_value_deck.pptx")


if __name__ == "__main__":
    main(Path(__file__).resolve().parent.parent / "config" / "customer_value.yaml")
