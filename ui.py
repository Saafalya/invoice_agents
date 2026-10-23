"""Look and feel for the Streamlit app: theme CSS and small HTML building blocks.

The signature element is the decision stamp, styled like the rubber stamps on paper invoices.
Everything else stays quiet: navy ink on ledger white, colour only where it carries meaning.
"""
from __future__ import annotations

import html

import streamlit as st

INK, CARBON, PAPER, RULE = "#1B2A4A", "#2B59C3", "#FBFCFA", "#DDE3EA"
STAMP = {  # decision -> (colour, label)
    "APPROVED": ("#1E7A4C", "Approved"),
    "HOLD": ("#A15C07", "Hold"),
    "ESCALATED": ("#B42318", "Escalated"),
    "REJECTED": ("#7A1F1F", "Rejected"),
    "PENDING REVIEW": ("#2B59C3", "Needs review"),
}
AGENTS = ["Planner", "Extraction", "Critic", "Matching", "Fraud", "Decision", "Human review"]
NODE_TO_AGENT = {"extractor": "Extraction", "critic_extract": "Critic", "matcher": "Matching",
                 "fraud": "Fraud", "critic_match": "Critic", "decider": "Decision",
                 "review": "Human review"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp td, .stApp th, .stApp summary {{
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}}
/* icons keep their own font: never override [data-testid="stIconMaterial"] */
code, pre, .stCode {{ font-family: 'IBM Plex Mono', ui-monospace, monospace !important; }}
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; color: {INK}; }}

.hdr {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem;
        border-bottom: 2px solid {INK}; padding-bottom: .9rem; margin-bottom: 1.1rem; flex-wrap: wrap; }}
.hdr h1 {{ font-size: 2rem; font-weight: 600; margin: 0; padding: 0; line-height: 1.15; }}
.hdr p {{ margin: .3rem 0 0; color: #4A5568; max-width: 62ch; }}
.chips {{ display: flex; gap: .5rem; flex-wrap: wrap; }}
.chip {{ border: 1px solid {RULE}; border-radius: 999px; padding: .25rem .7rem; font-size: .82rem;
         color: {INK}; background: white; white-space: nowrap; }}
.chip b {{ font-weight: 600; }}
.dot {{ display: inline-block; width: .5rem; height: .5rem; border-radius: 50%; margin-right: .4rem;
        vertical-align: middle; }}

.stamp {{ display: inline-block; font-weight: 600; text-transform: uppercase; letter-spacing: .08em;
          font-size: .78rem; padding: .18rem .55rem; border: 2px solid currentColor; border-radius: 4px;
          box-shadow: inset 0 0 0 1px white, inset 0 0 0 2.5px currentColor; transform: rotate(-3deg);
          background: white; line-height: 1.3; }}
.stamp.big {{ font-size: 1.05rem; padding: .3rem .8rem; }}

.pipe {{ display: flex; gap: 0; margin: .3rem 0 1rem; overflow-x: auto; }}
.pipe div {{ flex: 1 1 0; min-width: 92px; text-align: center; font-size: .8rem; padding: .45rem .3rem;
             border: 1px solid {RULE}; border-left: none; background: white; color: #64748B; }}
.pipe div:first-child {{ border-left: 1px solid {RULE}; border-radius: 6px 0 0 6px; }}
.pipe div:last-child {{ border-radius: 0 6px 6px 0; }}
.pipe div span {{ display: block; font-size: .7rem; color: #94A3B8; }}
.pipe div.on {{ background: {CARBON}; color: white; border-color: {CARBON}; }}
.pipe div.on span {{ color: #DCE6FA; }}
.pipe div.done {{ color: {INK}; }}

.case {{ display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; }}
.case .who {{ font-size: 1.15rem; font-weight: 600; color: {INK}; }}
.case .meta {{ color: #4A5568; font-size: .9rem; }}
.amt {{ font-size: 1.4rem; font-weight: 600; font-variant-numeric: tabular-nums; color: {INK}; }}
.meter {{ height: 8px; border-radius: 4px; background: #EEF1F5; overflow: hidden; margin: .25rem 0 .15rem; }}
.meter i {{ display: block; height: 100%; border-radius: 4px; }}
.small {{ font-size: .8rem; color: #64748B; }}
.brief {{ border-left: 3px solid {CARBON}; background: #F3F6FD; padding: .7rem .9rem; margin: .6rem 0;
          color: {INK}; border-radius: 0 6px 6px 0; }}
.brief b {{ font-weight: 600; }}
.trace {{ font-family: 'IBM Plex Mono', monospace; font-size: .8rem; line-height: 1.55;
          background: white; border: 1px solid {RULE}; border-radius: 6px; padding: .6rem .8rem;
          max-height: 360px; overflow-y: auto; }}
.trace .a {{ font-weight: 500; }}
.empty {{ text-align: center; padding: 2.2rem 1rem; border: 1px dashed {RULE}; border-radius: 8px;
          color: #4A5568; background: white; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
"""

AGENT_COLOURS = {"Planner Agent": "#475569", "Extraction Agent": CARBON, "Critic Agent": "#A15C07",
                 "Matching Agent": "#0F766E", "Fraud Agent": "#B42318", "Decision Agent": INK,
                 "Human Review": "#6D28D9"}


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def stamp(decision: str, big: bool = False) -> str:
    colour, label = STAMP.get(decision, ("#475569", decision.title()))
    return f'<span class="stamp{" big" if big else ""}" style="color:{colour}">{esc(label)}</span>'


def risk_colour(score: int) -> str:
    return "#1E7A4C" if score < 20 else "#A15C07" if score < 40 else "#B42318"


def risk_meter(score: int) -> str:
    return (f'<div class="meter"><i style="width:{max(score, 2)}%;background:{risk_colour(score)}"></i></div>'
            f'<div class="small">Fraud risk {score}/100</div>')


def header(engine: str, model: str, n_inv: int, pending: int):
    ollama = engine.startswith("Ollama")
    eng = f"Ollama model <b>{esc(model)}</b>" if ollama else "<b>Rule-based</b> mode (no AI model)"
    colour = "#1E7A4C" if ollama else "#94A3B8"
    st.markdown(f"""
<div class="hdr">
  <div><h1>Invoice check</h1>
  <p>AI agents check every vendor invoice against its purchase order and goods receipt before it is
  paid, flag fraud, and ask a person when it matters. All data is synthetic.</p></div>
  <div class="chips">
    <span class="chip"><span class="dot" style="background:{colour}"></span>{eng}</span>
    <span class="chip"><b>{n_inv}</b> invoices in batch</span>
    <span class="chip"><b>{pending}</b> waiting for review</span>
  </div>
</div>""", unsafe_allow_html=True)


def pipeline(active: str | None = None, done: set | None = None) -> str:
    done = done or set()
    cells = []
    for i, a in enumerate(AGENTS, 1):
        cls = "on" if a == active else "done" if a in done else ""
        cells.append(f'<div class="{cls}"><span>Step {i}</span>{a}</div>')
    return f'<div class="pipe">{"".join(cells)}</div>'


def trace(lines: list[tuple[str, str, str]]) -> str:
    """lines: (prefix, agent, message)."""
    rows = []
    for prefix, agent, msg in lines:
        c = AGENT_COLOURS.get(agent, INK)
        rows.append(f'<div>{esc(prefix)}<span class="a" style="color:{c}">{esc(agent)}</span>: {esc(msg)}</div>')
    return f'<div class="trace">{"".join(rows)}</div>'


def case_head(r: dict, decision: str) -> str:
    return f"""
<div class="case">
  <div><div class="who">{esc(r['vendor'])}</div>
  <div class="meta">{esc(r['invoice_no'])}, due in {esc(r['days_left'])} days</div></div>
  <div style="min-width:180px">{risk_meter(r['risk_score'])}</div>
  <div style="text-align:right"><div class="amt">INR {r['total']:,.2f}</div>{stamp(decision, big=True)}</div>
</div>"""


def brief(summary: str) -> str:
    text = esc(summary)
    for key in ("Problem:", "Stake:", "Vendor history:", "Check before deciding:", "Check:"):
        text = text.replace(key, f"<br><b>{key}</b>" if not text.startswith(key) else f"<b>{key}</b>", 1)
    return f'<div class="brief">{text}</div>'


def empty(msg: str) -> str:
    return f'<div class="empty">{esc(msg)}</div>'
