"""Look and feel: an Indian accounts office, not a SaaS dashboard.

The page is a sheet of ledger paper (pale green, faint ruling, the red double rule that marks
money columns). Decisions are violet rubber stamps, the ink colour Indian offices use, with red
ink kept for escalations and rejections. Escalated invoices are payment vouchers; the agent
pipeline is a routing slip that gets ticked off; totals are control totals.
"""
from __future__ import annotations

import html

import streamlit as st

PAPER = "#F3F6EE"     # ledger paper
RULE = "#C9D8BF"      # printed ruling
BINDING = "#2F5D46"   # ledger cloth binding
INK = "#24314D"       # blue-black fountain-pen ink
MARGIN = "#C23B32"    # red money-column rule
VIOLET = "#5B3F99"    # rubber-stamp ink

# decision -> (ink colour, short label for tables)
STAMP = {
    "APPROVED": (VIOLET, "Passed"),
    "HOLD": (VIOLET, "On hold"),
    "ESCALATED": (MARGIN, "Escalated"),
    "REJECTED": (MARGIN, "Rejected"),
    "PENDING REVIEW": (MARGIN, "Awaiting review"),
}
STAMP_TEXT = {"APPROVED": "Passed for payment", "HOLD": "On hold", "ESCALATED": "Escalated",
              "REJECTED": "Rejected", "PENDING REVIEW": "For review"}
AGENTS = ["Planner", "Extraction", "Critic", "Matching", "Fraud", "Decision", "Human review"]
NODE_TO_AGENT = {"extractor": "Extraction", "critic_extract": "Critic", "matcher": "Matching",
                 "fraud": "Fraud", "critic_match": "Critic", "decider": "Decision",
                 "review": "Human review"}
AGENT_COLOURS = {"Planner Agent": "#5B6B60", "Extraction Agent": INK, "Critic Agent": "#8A5A12",
                 "Matching Agent": BINDING, "Fraud Agent": MARGIN, "Decision Agent": INK,
                 "Human Review": VIOLET}

CSS = f"""
<style>
@font-face {{ font-family: 'Archivo'; src: url('app/static/fonts/archivo.woff2') format('woff2');
  font-weight: 100 900; font-stretch: 62% 125%; font-display: swap; }}
@font-face {{ font-family: 'Courier Prime'; src: url('app/static/fonts/courier-prime-400.woff2') format('woff2');
  font-weight: 400; font-display: swap; }}
@font-face {{ font-family: 'Courier Prime'; src: url('app/static/fonts/courier-prime-700.woff2') format('woff2');
  font-weight: 700; font-display: swap; }}

.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp td, .stApp th, .stApp summary {{ font-family: 'Archivo', system-ui, sans-serif; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
  font-family: 'Archivo', sans-serif; font-stretch: 72%; font-weight: 650; color: {INK};
  letter-spacing: 0; }}
.stApp code, .stApp pre {{ font-family: 'Courier Prime', ui-monospace, monospace !important; }}

/* the sheet: faint ruling, red double margin */
.stApp {{ background-color: {PAPER};
  background-image: repeating-linear-gradient(to bottom, transparent 0 31px, rgba(47,93,70,.09) 31px 32px); }}
[data-testid="stMain"] {{ background-image: linear-gradient(to right, transparent 0 28px, {MARGIN} 28px 29px,
  transparent 29px 32px, {MARGIN} 32px 33px, transparent 33px); }}
.block-container {{ max-width: 1240px; padding-top: 2rem; }}
section[data-testid="stSidebar"] {{ background: #E5ECDE; border-right: 6px solid {BINDING}; }}

/* tabs as ledger index tabs */
.stTabs [role="tablist"] {{ gap: 4px; border-bottom: 1px solid {BINDING}; box-shadow: none; }}
.stTabs [data-testid="stTab"] {{ background: #E3EBDC; border: 1px solid {RULE}; border-bottom: none !important;
  border-radius: 5px 5px 0 0; padding: .4rem .95rem; margin-bottom: -1px; box-shadow: none !important; }}
.stTabs [data-testid="stTab"][aria-selected="true"] {{ background: {PAPER}; border-color: {BINDING};
  box-shadow: 0 1px 0 {PAPER} !important; }}
.stTabs [data-testid="stTab"]::after, .stTabs [data-testid="stTab"]::before {{ display: none !important; }}
.stTabs [data-testid="stTab"] p {{ font-stretch: 82%; font-weight: 600; color: {INK}; }}
.stTabs [data-testid="stTab"][aria-selected="false"] p {{ color: #5B6B60; }}

.stButton button, .stDownloadButton button {{ border-radius: 3px; font-weight: 600; }}
:focus-visible {{ outline: 2px solid {VIOLET} !important; outline-offset: 2px; }}

/* form header */
.form {{ display: grid; grid-template-columns: 1fr auto; gap: 1.5rem; align-items: end;
  border-bottom: 4px double {BINDING}; padding-bottom: .8rem; margin-bottom: 1.2rem; }}
.form h1 {{ font-size: 2.6rem; line-height: 1; margin: 0; padding: 0; font-stretch: 68%; }}
.form p {{ margin: .45rem 0 0; color: #45534A; max-width: 60ch; }}
.fields {{ display: grid; grid-template-columns: repeat(4, auto); gap: .2rem 1.4rem; }}
.fields div {{ font-size: .74rem; color: #5B6B60; }}
.fields b {{ display: block; font-size: .98rem; font-weight: 600; color: {INK}; min-width: 6.5rem;
  border-bottom: 1px dotted #7D8C80; padding: .1rem 0 .15rem; font-variant-numeric: tabular-nums; }}
@media (max-width: 900px) {{ .form {{ grid-template-columns: 1fr; }} .fields {{ grid-template-columns: repeat(2, auto); }} }}

/* ledger tables: ruled rows, money column behind a red double rule */
table.ledger {{ width: 100% !important; display: table !important; border-collapse: collapse; background: rgba(255,255,255,.55);
  font-variant-numeric: tabular-nums; margin: .2rem 0 .8rem; }}
[data-testid="stMarkdownContainer"] div:has(> table.ledger) {{ width: 100%; }}
table.ledger caption {{ caption-side: top; text-align: left; font-stretch: 75%; font-weight: 650;
  font-size: 1.05rem; color: {INK}; padding-bottom: .25rem; }}
table.ledger th {{ text-align: left; font-weight: 500; font-size: .78rem; color: #5B6B60;
  border-bottom: 2px solid {BINDING}; padding: .3rem .6rem; }}
table.ledger td {{ border-bottom: 1px solid {RULE}; padding: .38rem .6rem; color: {INK}; }}
table.ledger td.amt, table.ledger th.amt {{ text-align: right; border-left: 4px double {MARGIN};
  font-weight: 600; white-space: nowrap; }}
table.ledger td small {{ color: #5B6B60; font-weight: 400; margin-left: .35rem; }}
table.ledger tr.foot td {{ border-top: 2px solid {INK}; border-bottom: 4px double {INK}; font-weight: 700; }}
table.ledger.wide td {{ font-size: 1.35rem; font-weight: 600; }}
table.ledger td.num, table.ledger th.num {{ text-align: right; }}

/* routing slip */
.slip {{ display: flex; border: 1px solid {BINDING}; background: rgba(255,255,255,.6); margin: .4rem 0 1rem;
  overflow-x: auto; }}
.slip div {{ flex: 1 1 0; min-width: 96px; padding: .4rem .55rem; border-right: 1px solid {RULE};
  font-size: .85rem; color: #5B6B60; }}
.slip div:last-child {{ border-right: none; }}
.slip i {{ font-style: normal; display: inline-block; width: 1rem; height: 1rem; border: 1.5px solid #7D8C80;
  margin-right: .4rem; vertical-align: -3px; text-align: center; line-height: .9rem; font-weight: 700; }}
.slip .done {{ color: {INK}; }}
.slip .done i {{ border-color: {VIOLET}; color: {VIOLET}; }}
.slip .on {{ color: {INK}; background: #ECE7F6; box-shadow: inset 0 -3px 0 {VIOLET}; }}
.slip .on i {{ border-color: {VIOLET}; }}

/* rubber stamp */
.stamp {{ display: inline-block; font-family: 'Archivo', sans-serif; font-stretch: 80%; font-weight: 700;
  text-transform: uppercase; letter-spacing: .06em; font-size: .8rem; line-height: 1.2;
  padding: .2rem .55rem; border: 2.5px solid currentColor; border-radius: 3px; transform: rotate(-4deg);
  box-shadow: inset 0 0 0 1.5px {PAPER}, inset 0 0 0 3px currentColor; opacity: .88; mix-blend-mode: multiply; }}
.stamp.big {{ font-size: 1.15rem; padding: .35rem .8rem; transform: rotate(-7deg); }}

/* payment voucher (wraps a keyed st.container) */
[class*="st-key-voucher"] {{ background: #FFFFFF; border: 1px solid {RULE}; border-top: 2px dashed #9AAA98;
  padding: 1rem 1.2rem 1.1rem; margin-bottom: 1.4rem; box-shadow: 0 1px 0 {RULE}; }}
.vhead {{ display: grid; grid-template-columns: 1.6fr 1fr 1fr 1.1fr auto; gap: 1rem; align-items: end;
  border-bottom: 2px solid {BINDING}; padding-bottom: .55rem; }}
.vhead div {{ font-size: .74rem; color: #5B6B60; }}
.vhead b {{ display: block; font-size: 1.02rem; font-weight: 600; color: {INK}; }}
.vhead .amt b {{ font-size: 1.3rem; text-align: right; font-variant-numeric: tabular-nums; }}
.vhead .amt {{ border-left: 4px double {MARGIN}; padding-left: .7rem; text-align: right; }}
@media (max-width: 900px) {{ .vhead {{ grid-template-columns: 1fr 1fr; }} }}
.risk {{ display: flex; align-items: center; gap: .6rem; margin: .6rem 0 .2rem; font-size: .82rem; color: #5B6B60; }}
.risk span.bar {{ flex: 0 0 180px; height: 9px; border: 1px solid #9AAA98; background:
  repeating-linear-gradient(90deg, transparent 0 8px, rgba(0,0,0,.06) 8px 9px); }}
.risk span.bar i {{ display: block; height: 100%; }}
.brief {{ margin: .7rem 0 .4rem; color: {INK}; line-height: 1.6; }}
.brief b {{ font-weight: 650; font-stretch: 80%; color: {BINDING}; }}

.trace {{ font-family: 'Courier Prime', monospace; font-size: .84rem; line-height: 1.5;
  background: #FFFFFF; border: 1px solid {RULE}; border-left: 4px double {MARGIN}; padding: .6rem .8rem;
  max-height: 360px; overflow-y: auto; color: {INK}; }}
.trace .a {{ font-weight: 700; }}
.empty {{ padding: 1.6rem 1.2rem; border: 1px dashed #9AAA98; background: rgba(255,255,255,.6);
  color: {INK}; max-width: 60ch; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def stamp(decision: str, big: bool = False) -> str:
    colour = STAMP.get(decision, (INK, ""))[0]
    text = STAMP_TEXT.get(decision, decision.title())
    return f'<span class="stamp{" big" if big else ""}" style="color:{colour}">{esc(text)}</span>'


def risk_colour(score: int) -> str:
    return BINDING if score < 20 else "#B7791F" if score < 40 else MARGIN


def risk_meter(score: int) -> str:
    return (f'<div class="risk"><span class="bar"><i style="width:{max(score, 2)}%;'
            f'background:{risk_colour(score)}"></i></span>Fraud risk {score} of 100</div>')


def header(engine: str, model: str, batch_no: int, date: str, pending: int):
    checker = f"Ollama {esc(model)}" if engine.startswith("Ollama") else "Rules only"
    st.markdown(f"""
<div class="form">
  <div><h1>Invoice register</h1>
  <p>Every vendor invoice is matched against its purchase order and goods receipt before it is
  passed for payment. Agents do the checking; a person decides the doubtful ones. Synthetic data.</p></div>
  <div class="fields">
    <div>Batch no.<b>{batch_no}</b></div>
    <div>Dated<b>{esc(date)}</b></div>
    <div>Checked by<b>{checker}</b></div>
    <div>Awaiting review<b>{pending}</b></div>
  </div>
</div>""", unsafe_allow_html=True)


def ledger(rows, caption: str = "", head=("Particulars", "Amount")) -> str:
    """rows: (particulars, amount, note). A row whose particulars start with '=' is a footing."""
    body = []
    for name, value, *note in rows:
        cls = ' class="foot"' if name.startswith("=") else ""
        n = f"<small>{esc(note[0])}</small>" if note and note[0] else ""
        body.append(f'<tr{cls}><td>{esc(name.lstrip("="))}</td><td class="amt">{esc(value)}{n}</td></tr>')
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return (f'<table class="ledger">{cap}<tr><th>{esc(head[0])}</th><th class="amt">{esc(head[1])}</th></tr>'
            f'{"".join(body)}</table>')


def footing(cells) -> str:
    """One wide row of control totals: [(label, value), ...]."""
    last = len(cells) - 1      # only the money column sits behind the red rule
    cls = lambda i: "amt" if i == last else "num" if i else ""
    th = "".join(f'<th class="{cls(i)}">{esc(l)}</th>' for i, (l, _) in enumerate(cells))
    td = "".join(f'<td class="{cls(i)}">{esc(v)}</td>' for i, (_, v) in enumerate(cells))
    return f'<table class="ledger wide"><tr>{th}</tr><tr>{td}</tr></table>'


def pipeline(active: str | None = None, done: set | None = None) -> str:
    done = done or set()
    cells = []
    for i, a in enumerate(AGENTS, 1):
        cls = "on" if a == active else "done" if a in done else ""
        tick = "&#10003;" if a in done and a != active else ""
        cells.append(f'<div class="{cls}"><i>{tick}</i>{i}. {a}</div>')
    return f'<div class="slip">{"".join(cells)}</div>'


def trace(lines) -> str:
    """lines: (prefix, agent, message)."""
    rows = []
    for prefix, agent, msg in lines:
        c = AGENT_COLOURS.get(agent, INK)
        rows.append(f'<div>{esc(prefix)}<span class="a" style="color:{c}">{esc(agent)}</span>: {esc(msg)}</div>')
    return f'<div class="trace">{"".join(rows)}</div>'


def voucher_head(r: dict) -> str:
    return f"""
<div class="vhead">
  <div>Vendor<b>{esc(r['vendor'])}</b></div>
  <div>Invoice no.<b>{esc(r['invoice_no'])}</b></div>
  <div>Due in<b>{esc(r['days_left'])} days</b></div>
  <div class="amt">Amount (INR)<b>{r['total']:,.2f}</b></div>
  <div>{stamp(r['decision'], big=True)}</div>
</div>{risk_meter(r['risk_score'])}"""


def brief(summary: str) -> str:
    text = esc(summary)
    for key in ("Problem:", "Stake:", "Vendor history:", "Check before deciding:", "Check:"):
        if key in text:
            lead = "" if text.startswith(key) else "<br>"
            text = text.replace(key, f"{lead}<b>{key}</b>", 1)
    return f'<div class="brief">{text}</div>'


def empty(msg: str) -> str:
    return f'<div class="empty">{esc(msg)}</div>'
