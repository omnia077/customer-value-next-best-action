"""Static dashboard (single HTML file, inline SVG, no external requests).
Pages: Segments, Value map, Action list, Uplift vs propensity, Model health.
Run:  python src/dashboard.py   ->  outputs/dashboard.html
"""
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from segment import load_cfg  # noqa: E402

SERIES = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"]


def esc(x):
    return html.escape(str(x))


def table(df, cls=""):
    th = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>'


def hbars(cats, series, colors, fmt="{:.1f}%", w=680, label_w=190):
    """Horizontal grouped bars. series: {name: [values]}."""
    n_s, n_c = len(series), len(cats)
    bh, gap, grp = 14, 2, 18
    h = 20 + n_c * (n_s * (bh + gap) + grp)
    vmax = max(max(v) for v in series.values()) * 1.12 or 1
    pw = w - label_w - 60
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">']
    y = 10
    for i, c in enumerate(cats):
        out.append(f'<text x="{label_w - 8}" y="{y + n_s * (bh + gap) / 2 + 3}" text-anchor="end" class="lbl">{esc(c)}</text>')
        for j, (nm, vals) in enumerate(series.items()):
            bw = max(vals[i] / vmax * pw, 0)
            out.append(f'<path d="M{label_w},{y} h{max(bw - 4, 0)} a4,4 0 0 1 4,4 v{bh - 8} a4,4 0 0 1 -4,4 h-{max(bw - 4, 0)} z" fill="{colors[j]}">'
                       f'<title>{esc(c)} | {esc(nm)}: {fmt.format(vals[i])}</title></path>')
            out.append(f'<text x="{label_w + bw + 6}" y="{y + bh - 3}" class="val">{fmt.format(vals[i])}</text>')
            y += bh + gap
        y += grp
    out.append(f'<line x1="{label_w}" x2="{label_w}" y1="6" y2="{h - 6}" class="axis"/></svg>')
    return "".join(out)


def legend(names, colors):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{c}"></i>{esc(n)}</span>' for n, c in zip(names, colors)) + "</div>"


def scatter_panels(df, groups, colors, w=330, h=250):
    """One small-multiple panel per persona. Log-log: churn probability (x) vs 12-month CLV (y)."""
    x = np.log10(df.churn_prob_30d.clip(lower=1e-3)); y = np.log10(df.clv_12m.clip(lower=10))
    xl, xh, yl, yh = -3, math.ceil(x.max()), 1, math.ceil(y.max())
    ml, mb, mt, mr = 62, 34, 10, 22
    sx = lambda v: ml + (v - xl) / (xh - xl) * (w - ml - mr)
    sy = lambda v: h - mb - (v - yl) / (yh - yl) * (h - mb - mt)
    panels = []
    rng = np.random.default_rng(0)
    for g, col in zip(groups, colors):
        s = ['<svg viewBox="0 0 %d %d" role="img" class="chart panel">' % (w, h)]
        for t in range(yl, yh + 1):
            s.append(f'<line x1="{ml}" x2="{w - mr}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="grid"/>'
                     f'<text x="{ml - 6}" y="{sy(t) + 3:.1f}" text-anchor="end" class="tick">{10 ** t:,.0f}</text>')
        for t in range(xl, xh + 1):
            s.append(f'<text x="{sx(t):.1f}" y="{h - mb + 14}" text-anchor="middle" class="tick">{100 * 10 ** t:g}%</text>')
        others = df[df.persona != g]
        io = rng.choice(len(others), min(400, len(others)), replace=False)
        for a, b in zip(x.loc[others.index[io]], y.loc[others.index[io]]):
            s.append(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="1.6" class="bg"/>')
        mine = df[df.persona == g]
        im = rng.choice(len(mine), min(450, len(mine)), replace=False)
        for idx in mine.index[im]:
            s.append(f'<circle cx="{sx(x[idx]):.1f}" cy="{sy(y[idx]):.1f}" r="2.6" fill="{col}" fill-opacity=".7" class="pt">'
                     f'<title>{esc(g)}: churn risk {100*df.churn_prob_30d[idx]:.1f}%, 12m value {df.clv_12m[idx]:,.0f} SDG</title></circle>')
        s.append(f'<text x="{ml}" y="{mt + 8}" class="ptitle">{esc(g)}</text>')
        s.append(f'<text x="{(ml + w - mr) / 2}" y="{h - 4}" text-anchor="middle" class="tick">30-day churn probability (log)</text></svg>')
        panels.append("".join(s))
    return '<div class="panels">' + "".join(panels) + "</div>"


def lines(xs, series, colors, styles, w=680, h=300, ylabel="", end_labels=True):
    ml, mb, mt, mr = 56, 34, 12, 120
    ymax = max(max(v) for v in series.values()) * 1.05
    ymin = min(0, min(min(v) for v in series.values()))
    sx = lambda v: ml + v * (w - ml - mr)
    sy = lambda v: h - mb - (v - ymin) / (ymax - ymin) * (h - mb - mt)
    s = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">']
    for t in np.linspace(ymin, ymax, 5):
        s.append(f'<line x1="{ml}" x2="{w - mr}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="grid"/>'
                 f'<text x="{ml - 6}" y="{sy(t) + 3:.1f}" text-anchor="end" class="tick">{t:,.0f}</text>')
    for t in (0, .25, .5, .75, 1):
        s.append(f'<text x="{sx(t):.1f}" y="{h - mb + 14}" text-anchor="middle" class="tick">{int(t*100)}%</text>')
    s.append(f'<text x="{(ml + w - mr) / 2}" y="{h - 4}" text-anchor="middle" class="tick">share of eligible subscribers contacted</text>')
    s.append(f'<text x="12" y="{(h - mb) / 2}" class="tick" transform="rotate(-90 12 {(h - mb) / 2})" text-anchor="middle">{esc(ylabel)}</text>')
    for (nm, v), col, st in zip(series.items(), colors, styles):
        pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(xs, v))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" {st}/>')
        for a, b in list(zip(xs, v))[::10]:
            s.append(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="8" fill="transparent"><title>{esc(nm)} @ {a:.0%}: {b:,.0f} incremental recharges</title></circle>')
        if end_labels:
            s.append(f'<text x="{sx(xs[-1]) + 6:.1f}" y="{sy(v[-1]) + 3:.1f}" class="lbl">{esc(nm)}</text>')
    s.append("</svg>")
    return "".join(s)


CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--dot:#c3c2b7}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--dot:#4a4a46}}
:root[data-theme="dark"]{--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--dot:#4a4a46}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:20px 16px 8px;max-width:1080px;margin:0 auto}h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0 0 6px}
.sub{color:var(--ink2);margin:0}.warn{color:var(--ink2);font-size:12.5px;border-left:3px solid var(--s4);padding:4px 10px;margin:10px 0}
nav{max-width:1080px;margin:8px auto 0;padding:0 16px;display:flex;gap:6px;flex-wrap:wrap}
nav button{background:transparent;border:1px solid var(--border);color:var(--ink2);padding:6px 12px;border-radius:8px;cursor:pointer;font:inherit}
nav button[aria-selected=true]{background:var(--surface);color:var(--ink);border-color:var(--axis);font-weight:600}
main{max-width:1080px;margin:0 auto;padding:12px 16px 40px}section{display:none}section.on{display:block}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;margin:0 0 14px}
.chart{width:100%;height:auto;display:block}.lbl{fill:var(--ink);font-size:11.5px}.val{fill:var(--ink2);font-size:11px}.tick{fill:var(--muted);font-size:10.5px}
.ptitle{fill:var(--ink);font-size:12px;font-weight:600}.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis)}
.bg{fill:var(--dot);opacity:.7}.pt:hover{stroke:var(--ink);stroke-width:1.5}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:4px 0 8px}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{padding:5px 10px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th{color:var(--ink2);font-weight:600}td:first-child,th:first-child{text-align:left}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:0 0 14px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 14px}.kpi b{display:block;font-size:24px}.kpi span{color:var(--ink2);font-size:12px}
.ok{color:var(--ink)}.watch::before{content:"\\25B2 ";}.alert::before{content:"\\25CF ";}
details{margin-top:8px}summary{cursor:pointer;color:var(--ink2);font-size:12.5px}
@media (max-width:600px){h1{font-size:18px}}
"""


def build(cfg_path):
    cfg = load_cfg(cfg_path)
    o = cfg["_out"]
    cards = pd.read_csv(o / "persona_cards.csv")
    prof = pd.read_csv(o / "segment_profile.csv")
    acts = pd.read_parquet(o / "customer_actions.parquet")
    clv = pd.read_csv(o / "clv_holdout.csv")
    up = json.load(open(o / "uplift_metrics.json"))
    prop = json.load(open(o / "propensity_metrics.json"))
    churn = json.load(open(o / "churn_proxy_metrics.json"))
    drift = pd.read_csv(o / "drift_check.csv")
    grow = pd.read_csv(o / "grow_policy_evaluation.csv")
    qc = pd.read_csv(o / "uplift_qini_curves.csv")
    quint = pd.read_csv(o / "uplift_calibration_quintiles.csv")
    segm = json.load(open(o / "segmentation_metrics.json"))
    order = list(prof.sort_values("segment_id").persona)
    col = dict(zip(order, SERIES))
    cards = cards.set_index("persona").loc[order].reset_index()

    # ---- page 1
    p1 = (f'<div class="card"><h2>Who is in each persona: share of subscribers and of revenue</h2>'
          + legend(["% of subscribers", "% of 180-day recharge revenue"], ["var(--s1)", "var(--s2)"])
          + hbars(order, {"subs": list(cards.pct_subs), "rev": list(cards.pct_revenue_180d)}, ["var(--s1)", "var(--s2)"])
          + '</div><div class="card"><h2>Persona profile (as of 30 Sep, 180-day window)</h2>'
          + table(prof[["persona", "n", "pct_subs", "pct_revenue", "recency_days", "recharge_freq_30d", "avg_recharge_sdg", "sessions_30d",
                        "share_data", "share_voice", "share_sms"]].rename(columns={"n": "subscribers"}))
          + f'<p class="sub">Segments are soft profiles: silhouette {segm["silhouette"]}, recovery of the planted archetypes ARI {segm["recovery_ari"]} (0 = random).</p></div>')
    # ---- page 2
    p2 = ('<div class="card"><h2>Value map: 12-month value against churn risk</h2>'
          '<p class="sub">Each panel highlights one persona over a grey sample of the others. Both axes are logarithmic.</p>'
          + scatter_panels(acts.dropna(subset=["churn_prob_30d"]).set_index(acts.dropna(subset=["churn_prob_30d"]).index), order, [col[p] for p in order])
          + '</div><div class="card"><h2>Persona cards</h2>'
          + table(cards[["persona", "subscribers", "avg_clv_12m", "avg_churn_prob_30d", "avg_propensity_500", "avg_uplift_pts", "recommended_play", "pct_contacted", "expected_net_value_sdg"]]
                  .rename(columns={"avg_clv_12m": "avg 12m value (SDG)", "avg_churn_prob_30d": "avg churn %", "avg_propensity_500": "avg P(recharge>=500) %",
                                   "avg_uplift_pts": "avg uplift (pts)", "pct_contacted": "% contacted", "expected_net_value_sdg": "expected net value (SDG)"})) + "</div>")
    # ---- page 3
    mix = pd.crosstab(acts.persona, acts.action).reindex(order).fillna(0)
    acts_names = [a for a in ["grow", "protect", "win_back"] if a in mix]
    top = acts[acts.action != "maintain"].sort_values("ev", ascending=False).head(20)
    top = top[["customer_id", "persona", "action", "ev", "clv_12m", "churn_prob_30d", "uplift"]].round(3)
    top["churn_prob_30d"] = (100 * top.churn_prob_30d).round(1)
    top["uplift"] = (100 * top.uplift).round(1)
    top = top.rename(columns={"ev": "expected net value (SDG)", "clv_12m": "12m value (SDG)", "churn_prob_30d": "churn %", "uplift": "uplift (pts)"})
    n_c = int((acts.action != "maintain").sum())
    p3 = (f'<div class="kpis"><div class="kpi"><b>{n_c:,}</b><span>subscribers with an action ({100*n_c/len(acts):.0f}% of {len(acts):,})</span></div>'
          f'<div class="kpi"><b>{acts.ev.sum():,.0f}</b><span>expected net value, SDG (assumption-based)</span></div>'
          f'<div class="kpi"><b>1</b><span>action per subscriber (contact cap)</span></div></div>'
          '<div class="card"><h2>Actions by persona</h2>'
          + legend([a.replace("_", " ") for a in acts_names], ["var(--s1)", "var(--s2)", "var(--s3)"][:len(acts_names)])
          + hbars(order, {a: list(mix[a]) for a in acts_names}, ["var(--s1)", "var(--s2)", "var(--s3)"][:len(acts_names)], fmt="{:,.0f}")
          + '</div><div class="card"><h2>Top 20 by expected net value</h2>' + table(top)
          + '<p class="sub">IDs are synthetic surrogate keys; no MSISDN is stored in these tables.</p></div>')
    # ---- page 4
    ev = pd.DataFrame(up["evaluation"])
    names = ["Uplift model", "Propensity model", "Oracle (true uplift)"]
    sweep = pd.read_csv(o / "grow_sweep_offer_cost.csv")
    brk = pd.read_csv(o / "assumption_breakeven.csv")
    cv = pd.DataFrame(up["learner_cv"])
    alt = ev[ev.policy.str.startswith(("Uplift", "Alt"))][["learner", "spearman_vs_true_uplift", "observed_uplift_top20_pts", "qini_coef", "qini_coef_ci95"]]
    am = json.load(open(o / "actions_metrics.json"))
    be_cost = am.get("breakeven_offer_cost_uplift_rule_vs_everyone")
    rand_line = list(np.linspace(0, float(qc["Oracle (true uplift)"].iloc[-1]), 101))     # random targeting: straight line to the all-contact total
    ser = {n: list(qc[n]) for n in names}
    ser["Random"] = rand_line
    top20 = ev.set_index("policy")
    p4 = ('<div class="card"><h2>Who buys because of the offer: uplift vs propensity targeting (June, randomized)</h2>'
          '<p class="sub">Cumulative incremental recharges (treated minus scaled control) as more subscribers are contacted, ranked by each score.</p>'
          + legend(names + ["Random"], ["var(--s1)", "var(--s2)", "var(--s3)", "var(--muted)"])
          + lines(list(qc.share_contacted), ser, ["var(--s1)", "var(--s2)", "var(--s3)", "var(--muted)"],
                  ["", "", 'stroke-dasharray="6 4"', 'stroke-dasharray="2 4"'], ylabel="incremental recharges", end_labels=False)
          + '<details><summary>Table view</summary>' + table(ev[["policy", "spearman_vs_true_uplift", "expected_incremental_recharges_top20", "observed_uplift_top20_pts", "qini_coef", "qini_coef_ci95"]]) + '</details></div>'
          '<div class="card"><h2>Predicted vs true uplift by quintile (uplift model, June)</h2>'
          + legend(["predicted", "true (simulator)", "observed (randomized)"], ["var(--s1)", "var(--s2)", "var(--s3)"])
          + hbars([f"Quintile {int(q) + 1}" + (" (highest)" if q == 4 else " (lowest)" if q == 0 else "") for q in quint.q],
                  {"pred": [max(v, 0) for v in quint.pred_uplift_pts], "true": list(quint.true_uplift_pts), "obs": [max(v, 0) for v in quint.observed_uplift_pts]},
                  ["var(--s1)", "var(--s2)", "var(--s3)"], fmt="{:.1f} pts")
          + '<p class="sub">Predictions are compared with the simulator\'s true effect and with the observed effect in the randomized data. Use the model to rank; check calibration again on real campaigns. Negative bars are shown as zero.</p></div>'
          '<div class="card"><h2>Net value of the Grow policy on the June holdout (true uplift, assumption costs)</h2>' + table(grow) + '</div>'
          f'<div class="card"><h2>When does targeting beat contacting everyone? Net value (SDG) by offer cost</h2>'
          f'<p class="sub">Break-even: the uplift rule beats contacting everyone from an offer cost of {be_cost} SDG upward (assumed cost: {cfg["actions"]["offer_cost_sdg"]} SDG).</p>'
          + table(sweep) + '</div>'
          f'<div class="card"><h2>Learner comparison (selected on April cross-validation only; June is the test)</h2>'
          + table(cv) + '<p class="sub">Selected: ' + esc(up["selected_learner"]) + '. Differences between the top learners are within cross-validation noise; all beat propensity targeting.</p>'
          + table(alt) + '</div>')
    # ---- page 5
    d = drift.copy()
    d["status"] = d.status
    p5 = ('<div class="card"><h2>How wrong can the assumed rates be? (break-even analysis)</h2>' + table(brk)
          + '<p class="sub">Break-even rate = the rate at which the action has zero expected value. "Robust share" = allocated subscribers still positive if the true rate is half the assumed one. Values still scale with the true rate.</p></div>'
          '<div class="card"><h2>CLV bake-off (holdout Oct-Dec)</h2>' + table(clv) + '</div>'
          '<div class="card"><h2>Propensity (test: Nov and Dec labels)</h2>'
          + table(pd.DataFrame(prop["test"]).T.reset_index().rename(columns={"index": "model"})) + '</div>'
          '<div class="card"><h2>Churn stand-in (Project 1 scores replace this when available)</h2>' + table(pd.DataFrame([churn]).drop(columns=["note"]))
          + '</div><div class="card"><h2>Drift check (PSI: &lt;0.1 OK, 0.1-0.25 WATCH, &gt;0.25 ALERT)</h2>' + table(d)
          + '<p class="sub">Score PSI compares against the baseline saved on the first run, so it is 0 until the next monthly run. tenure_days shifts every month by construction.</p></div>')

    pages = [("Segments", p1), ("Value map", p2), ("Action list", p3), ("Uplift vs propensity", p4), ("Model health", p5)]
    nav = "".join(f'<button role="tab" aria-selected="{"true" if i == 0 else "false"}" data-t="{i}">{esc(n)}</button>' for i, (n, _) in enumerate(pages))
    body = "".join(f'<section class="{"on" if i == 0 else ""}" id="t{i}">{h}</section>' for i, (_, h) in enumerate(pages))
    js = ("document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button').forEach(x=>x.setAttribute('aria-selected',x===b));"
          "document.querySelectorAll('section').forEach((s,i)=>s.classList.toggle('on',i==b.dataset.t))});")
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>Customer Value Map</title><style>{CSS}</style></head><body><header><h1>Customer Value Map and Next-Best-Action</h1>'
           f'<p class="sub">Prepaid subscribers, as of {cfg["segmentation"]["as_of"]}. All data is simulated; results demonstrate the method, not real-world performance.</p>'
           f'<p class="warn">Expected values use assumed offer costs and save/win-back rates (config: actions). The churn input is a stand-in until Project 1 scores are connected.</p></header>'
           f'<nav role="tablist">{nav}</nav><main>{body}</main><script>{js}</script></body></html>')
    (o / "dashboard.html").write_text(doc, encoding="utf-8")
    print("wrote", o / "dashboard.html", f"({len(doc) / 1e6:.2f} MB)")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent.parent / "config" / "customer_value.yaml")
