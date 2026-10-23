"""Streamlit PoC: agentic three-way invoice matching for accounts payable.

Run with:  streamlit run app.py
"""
import time
from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st

import ui
from config import APPROVED, ESCALATED, HOLD, REJECTED
from data_gen import generate_batch
from llm import get_llm
from memory import VendorMemory
from orchestrator import HAS_LANGGRAPH, Ctx, evaluate, finish_review, process_batch, process_one
from pdf_tools import invoice_to_pdf, pdf_to_text

st.set_page_config(page_title="Invoice check", page_icon="🧾", layout="wide")
ui.inject()

PENDING = "PENDING REVIEW"
MEMORY_FILE = "vendor_memory.json"

ARCHITECTURE_DOT = """
digraph G {
  rankdir=TB; nodesep=0.45; ranksep=0.55; newrank=true;
  node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#6366f1", fontname="Helvetica", fontsize=11];
  edge [color="#64748b", fontname="Helvetica", fontsize=9];

  batch   [label="Invoice batch\\n(text from email / OCR)", shape=folder, fillcolor="#f1f5f9", color="#94a3b8"];
  planner [label="Planner Agent\\nprioritise + sequence"];
  extract [label="Extraction Agent\\nfast parser / LLM -> schema"];
  critic1 [label="Critic Agent\\nvalidate extraction", fillcolor="#fff7ed", color="#f97316"];
  matcher [label="Matching Agent\\n3-way match tools"];
  fraud   [label="Fraud Agent\\nlookalike vendor - bank change\\namount spike - risk score", fillcolor="#fef2f2", color="#ef4444"];
  critic2 [label="Critic Agent\\ncross-check variance", fillcolor="#fff7ed", color="#f97316"];
  decider [label="Decision Agent\\napprove / hold / escalate"];
  review  [label="Human Review (HITL)\\nLangGraph interrupt:\\napprove / reject / hold", fillcolor="#e0e7ff", color="#4f46e5"];
  out     [label="Payment run  |  Vendor query  |  Rejected", shape=note, fillcolor="#ecfdf5", color="#10b981"];
  memory  [label="Vendor memory\\nhistory - bank on file\\nescalations - overrides", shape=cylinder, fillcolor="#fef9c3", color="#ca8a04"];
  data    [label="Tools and data\\nPO DB - GRN DB - Vendor master\\nDuplicate registry - Business rules", shape=cylinder, fillcolor="#f1f5f9", color="#94a3b8"];
  llm     [label="LLM (Ollama or mock)\\nextraction + explanations", shape=ellipse, fillcolor="#faf5ff", color="#a855f7"];

  batch -> planner -> extract -> critic1;
  critic1 -> extract [label="rejected: retry\\n(robust parser)", style=dashed, color="#f97316"];
  critic1 -> matcher [label="valid"];
  critic1 -> decider [label="invalid after retry", style=dashed, constraint=false];
  matcher -> fraud -> critic2 -> decider;
  decider -> review [label="escalated: pause"];
  decider -> out [label="approve / hold"];
  review -> out;
  fraud -> memory [style=dotted, dir=back];
  review -> memory [label="learns from reviewer", style=dotted, color="#ca8a04", constraint=false];
  data -> matcher [style=dotted, dir=both];
  llm -> extract [style=dotted];
  llm -> decider [style=dotted, constraint=false];
  {rank=same; llm; data}
  {rank=same; batch; planner; extract; critic1; matcher}
  {rank=same; fraud; critic2; decider; review; out}
}
"""



# --------------------------------------------------------------------------- helpers
def timed(gen):
    """Yield (event, seconds spent inside the generator) so UI sleeps are not counted."""
    it = iter(gen)
    while True:
        t0 = time.perf_counter()
        try:
            ev = next(it)
        except StopIteration:
            return
        yield ev, time.perf_counter() - t0


def final(r):
    return r["final_decision"] or PENDING


def label(decision):
    return ui.STAMP.get(decision, ("", decision.title()))[1]


def results_df(results):
    return pd.DataFrame([{
        "#": r["idx"], "Invoice": r["invoice_no"], "Vendor": r["vendor"],
        "Total (INR)": r["total"], "Priority": r["priority"].title(), "Agent said": label(r["decision"]),
        "Status": label(final(r)), "Risk": r["risk_score"],
        "Issues": ", ".join(sorted({i["type"].replace("_", " ").lower() for i in r["issues"]})) or "-",
        "At risk (INR)": r["at_risk"], "Attempts": r["attempts"],
    } for r in results])


COLUMNS = {
    "Total (INR)": st.column_config.NumberColumn(format="%.2f"),
    "At risk (INR)": st.column_config.NumberColumn(format="%.2f"),
    "Risk": st.column_config.ProgressColumn("Fraud risk", min_value=0, max_value=100, format="%d"),
}
LABEL_COLOUR = {lab: col for col, lab in ui.STAMP.values()}


def table(df, **kw):
    """Dataframe with the decision columns coloured like their stamps."""
    cols = [c for c in ("Status", "Agent said") if c in df.columns]
    styled = df.style.map(lambda v: f"color: {LABEL_COLOUR.get(v, '#1B2A4A')}; font-weight: 600",
                          subset=cols)
    st.dataframe(styled, width="stretch", hide_index=True, column_config=COLUMNS, **kw)


def get_memory(persist: bool) -> VendorMemory:
    mem = st.session_state.get("memory")
    if mem is None or bool(mem.path) != persist:
        mem = st.session_state.memory = VendorMemory(MEMORY_FILE if persist else None)
    return mem


def new_ctx(llm, batch, hitl) -> Ctx:
    mem = st.session_state.memory
    mem.seed(batch.history, batch.bank_db)
    return Ctx(llm, batch.vendors, batch.po_db, batch.grn_db, batch.today, memory=mem, hitl=hitl)


def show_invoice(r, key=""):
    """One invoice in detail: findings first, then the document, the trace and the data."""
    t1, t2, t3, t4 = st.tabs(["Findings", "Invoice", "Agent trace", "Extracted data"])
    with t1:
        st.markdown(f"{ui.stamp(r['decision'])}&nbsp;&nbsp;{ui.esc(r['reason'])}", unsafe_allow_html=True)
        if r["issues"]:
            st.dataframe(pd.DataFrame([{"Issue": i["type"].replace("_", " ").capitalize(),
                                        "What the agents found": i["detail"]} for i in r["issues"]]),
                         width="stretch", hide_index=True)
        else:
            st.caption("No issues: vendor, PO, goods receipt, prices, tax and duplicates all check out.")
        if r["risk_signals"]:
            st.markdown(ui.risk_meter(r["risk_score"]), unsafe_allow_html=True)
            if r.get("risk_note"):
                st.markdown(f"**Fraud Agent:** {r['risk_note']}")
            st.dataframe(pd.DataFrame([{"Signal": s["signal"].replace("_", " ").capitalize(),
                                        "Points": s["points"], "Why": s["detail"]}
                                       for s in r["risk_signals"]]),
                         width="stretch", hide_index=True)
        if r.get("llm_review"):
            st.caption("Critic Agent, AI second opinion: " + "; ".join(r["llm_review"]))
    with t2:
        st.code(r["raw"], language=None)
    with t3:
        st.markdown(ui.trace([("", e["agent"], e["msg"]) for e in r["trace"]]), unsafe_allow_html=True)
    with t4:
        st.json(r["extract"] or {}, expanded=False)


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Settings")
    with st.expander(":material/smart_toy: AI model", expanded=True):
        engine = st.radio("Engine", ["Mock (no setup needed)", "Ollama (local model)"],
                          label_visibility="collapsed")
        model = st.text_input("Ollama model", "llama3.2") if engine.startswith("Ollama") else ""
        use_graph = st.toggle("Orchestrate with LangGraph", value=HAS_LANGGRAPH, disabled=not HAS_LANGGRAPH)
        if not HAS_LANGGRAPH:
            st.caption("langgraph is not installed, using the built-in runner (same workflow).")
        delay = st.slider("Animation speed (s per step)", 0.0, 0.5, 0.08, 0.02)

    with st.expander(":material/receipt_long: Invoice batch"):
        n_inv = st.slider("Invoices in batch", 10, 60, 24)
        err_pct = st.slider("Share with seeded errors", 0.10, 0.60, 0.35, 0.05)
        seed = st.number_input("Random seed", 0, 9999, 42)
        regenerate = st.button("Generate new batch", width="stretch")

    with st.expander(":material/person_check: Human review", expanded=True):
        hitl = st.toggle("Pause escalated invoices for a reviewer", value=True)
        reviewer = st.text_input("Your name (for the audit trail)", "Finance reviewer")

    with st.expander(":material/storefront: Vendor memory"):
        persist = st.toggle(f"Remember vendors across sessions", value=False,
                            help=f"Saves to {MEMORY_FILE}")
        memory = get_memory(persist)
        if st.button("Reset vendor memory", width="stretch"):
            memory.vendors = {}
            memory.save()

    with st.expander(":material/schedule: Time-saving assumptions"):
        manual_min = st.number_input("Manual review, min per invoice", 1.0, 30.0, 6.0, 0.5)
        review_min = st.number_input("Review of a flagged invoice, min", 1.0, 30.0, 4.0, 0.5)

if regenerate or "batch" not in st.session_state:
    st.session_state.batch = generate_batch(n_inv, err_pct, int(seed))
    for k in ("results", "ctx", "uploads"):
        st.session_state.pop(k, None)

batch = st.session_state.batch
memory.seed(batch.history, batch.bank_db)

n_pending = sum(r["status"] == "PENDING_REVIEW"
                for r in st.session_state.get("results", []) + st.session_state.get("uploads", []))
ui.header(engine, model, len(batch.invoices), n_pending)

tab_run, tab_review, tab_results, tab_upload, tab_memory, tab_arch = st.tabs([
    ":material/play_circle: Run", f":material/gavel: Review queue ({n_pending})",
    ":material/insights: Results", ":material/upload_file: Upload PDF",
    ":material/storefront: Vendor memory", ":material/account_tree: How it works"])

# --------------------------------------------------------------------------- tab: run
with tab_run:
    k = st.columns(4)
    k[0].metric("Invoices waiting", len(batch.invoices), border=True)
    k[1].metric("Total value (INR)", f"{sum(i['truth']['total'] for i in batch.invoices):,.0f}", border=True)
    k[2].metric("Vendors", len({i['truth']['vendor'] for i in batch.invoices}), border=True)
    k[3].metric("Purchase orders on file", len(batch.po_db), border=True)

    left, right = st.columns([3, 1], vertical_alignment="center")
    left.markdown("Each invoice must match its **purchase order** (what we ordered) and **goods receipt** "
                  "(what arrived) before it is paid. Some invoices in this batch hide errors and fraud.")
    run = right.button("Check all invoices", type="primary", icon=":material/play_arrow:", width="stretch")
    pipe_box = st.empty()
    pipe_box.markdown(ui.pipeline(), unsafe_allow_html=True)

    if run:
        llm, warn = get_llm("ollama" if engine.startswith("Ollama") else "mock", model)
        if warn:
            st.warning(warn, icon=":material/warning:")
        ctx = st.session_state.ctx = new_ctx(llm, batch, hitl)
        st.session_state.uploads = []
        total = len(batch.invoices)
        bar = st.progress(0.0, text="Planner Agent is sorting the batch by due date...")
        c1, c2 = st.columns([3, 2])
        c1.markdown("**Live agent trace**")
        trace_box = c1.empty()
        c2.markdown("**Decisions so far**")
        table_box = c2.empty()

        lines, results, proc, done = [], [], 0.0, {"Planner"}
        for ev, dt in timed(process_batch(batch, ctx, use_graph)):
            proc += dt
            if ev["type"] == "plan":
                entries, prefix = ev["log"], ""
                pipe_box.markdown(ui.pipeline("Planner"), unsafe_allow_html=True)
            elif ev["type"] == "step":
                entries, prefix = ev["update"].get("log", []), f"#{ev['idx']:02d} "
                agent = ui.NODE_TO_AGENT.get(ev["node"])
                done.add(agent)
                pipe_box.markdown(ui.pipeline(agent, done), unsafe_allow_html=True)
            else:
                entries, prefix = [], ""
            lines += [(prefix, e["agent"], e["msg"]) for e in entries]
            trace_box.markdown(ui.trace(lines[-14:]), unsafe_allow_html=True)
            if ev["type"] == "result":
                results.append(ev["result"])
                bar.progress(len(results) / total, text=f"{len(results)} of {total} invoices checked")
                with table_box.container():
                    table(results_df(results)[["Invoice", "Status", "Risk"]].iloc[::-1], height=330)
            elif ev["type"] == "step":
                time.sleep(delay)

        st.session_state.results = results
        st.session_state.proc_seconds = proc
        st.session_state.engine_name = llm.name
        bar.progress(1.0, text="All invoices checked")
        pipe_box.markdown(ui.pipeline(None, set(ui.AGENTS)), unsafe_allow_html=True)
        waiting = sum(r["status"] == "PENDING_REVIEW" for r in results)
        if waiting:
            st.success(f"Done. {waiting} invoice(s) need your decision in the Review queue.",
                       icon=":material/gavel:")
            st.button("Update the review count")   # rerun so the tab label shows the count
        else:
            st.success("Done. See the Results tab.", icon=":material/check_circle:")

    with st.expander("See the incoming invoices and the data the agents use"):
        reveal = st.toggle("Show the answer key (seeded errors)")
        st.dataframe(pd.DataFrame([{
            "#": i["idx"], "Invoice": i["truth"]["invoice_no"], "Vendor": i["truth"]["vendor"],
            "Total (INR)": i["truth"]["total"],
            **({"Seeded error": i["truth"]["kind"]} if reveal else {}),
        } for i in batch.invoices]), width="stretch", hide_index=True, height=240,
            column_config=COLUMNS)
        st.markdown("**Purchase orders and goods receipts**")
        st.dataframe(pd.DataFrame([
            {"PO": po, "Vendor": d["vendor"], "SKU": sku, "Ordered": l["qty"], "PO price": l["price"],
             "Received (GRN)": batch.grn_db[po].get(sku)}
            for po, d in batch.po_db.items() for sku, l in d["lines"].items()
        ]), width="stretch", hide_index=True, height=240)
        st.caption("Approved vendors: " + ", ".join(batch.vendors))
        st.markdown("**A sample invoice**")
        st.code(batch.invoices[0]["text"], language=None)

# --------------------------------------------------------------------------- tab: review queue
with tab_review:
    ctx = st.session_state.get("ctx")
    lists = {"results": st.session_state.get("results", []), "uploads": st.session_state.get("uploads", [])}
    queue = [(name, i, r) for name, rs in lists.items() for i, r in enumerate(rs)
             if r["status"] == "PENDING_REVIEW"]
    if queue:
        st.caption("Highest risk first. Your decision resumes the paused workflow and teaches the vendor "
                   "memory, so an approved bank change becomes the new account on file.")
    elif lists["results"]:
        st.markdown(ui.empty("All caught up. Nothing is waiting for a decision."), unsafe_allow_html=True)
    else:
        st.markdown(ui.empty("Nothing to review yet. Go to Run and click 'Check all invoices'. "
                             "Escalated invoices land here."), unsafe_allow_html=True)
    for name, i, r in sorted(queue, key=lambda q: -q[2]["risk_score"]):
        with st.container(border=True):
            st.markdown(ui.case_head(r, r["decision"]), unsafe_allow_html=True)
            if r.get("case_summary"):
                st.markdown(ui.brief(r["case_summary"]), unsafe_allow_html=True)
            with st.expander("Details: findings, invoice, agent trace"):
                show_invoice(r)
            note = st.text_input("Note for the audit trail (optional)", key=f"note-{name}-{i}",
                                 placeholder="e.g. Called vendor, bank change confirmed")
            cols = st.columns(3)
            # Equal weight on purpose: highlighting "Approve" would nudge reviewers to pay risky invoices.
            buttons = (("APPROVE", "Approve payment", ":material/check_circle:", "secondary"),
                       ("HOLD", "Hold for vendor", ":material/pause_circle:", "secondary"),
                       ("REJECT", "Reject", ":material/block:", "secondary"))
            for col, (action, text, icon, kind) in zip(cols, buttons):
                if col.button(text, key=f"{action}-{name}-{i}", icon=icon, type=kind, width="stretch"):
                    answer = {"action": action, "note": note, "reviewer": reviewer}
                    st.session_state[name][i] = finish_review(r, answer, ctx)
                    st.toast(f"{r['invoice_no']}: {text.lower()} recorded", icon=icon)
                    st.rerun()

    done = [r for rs in lists.values() for r in rs if r["reviewed_by"] not in ("", "auto")]
    if done:
        st.markdown("#### Audit trail")
        table(pd.DataFrame([{
            "Invoice": r["invoice_no"], "Vendor": r["vendor"], "Agent said": label(r["decision"]),
            "Status": label(r["final_decision"]), "Reviewer": r["reviewed_by"], "Note": r["review_note"],
        } for r in done]))

# --------------------------------------------------------------------------- tab: results
with tab_results:
    res = st.session_state.get("results")
    if not res:
        st.markdown(ui.empty("No results yet. Go to Run and click 'Check all invoices'."),
                    unsafe_allow_html=True)
    else:
        m = evaluate(res)
        counts = Counter(final(r) for r in res)
        st.caption(f"Engine: {st.session_state.engine_name}")

        c = st.columns(5)
        for col, key in zip(c, (APPROVED, HOLD, ESCALATED, REJECTED, PENDING)):
            col.metric(label(key), counts.get(key, 0), border=True)

        st.markdown("#### How well did the agents do?")
        v = st.columns(4)
        v[0].metric("Seeded errors caught", f"{m['caught']}/{m['seeded']}", f"{m['recall']:.0%} recall",
                    delta_color="off", border=True)
        v[1].metric("Fraud cases escalated", f"{m['fraud_caught']}/{m['fraud_seeded']}", border=True,
                    help="Lookalike vendor, bank-account change and amount spike, seeded in the batch")
        v[2].metric("Clean invoices wrongly flagged", f"{m['false_positives']}/{m['clean']}", border=True)
        v[3].metric("Bad extractions fixed by the Critic", f"{m['recovered_by_retry']}/{m['retried']}",
                    border=True)

        st.markdown("#### Money and time")
        flagged = m["n"] - sum(r["decision"] == APPROVED for r in res)
        manual_total = m["n"] * manual_min
        assisted = st.session_state.proc_seconds / 60 + flagged * review_min
        saved = manual_total - assisted
        w = st.columns(4)
        w[0].metric("Payments stopped (INR)", f"{m['blocked_value']:,.0f}", border=True)
        w[1].metric("Overpayment caught (INR)", f"{m['detected_at_risk']:,.0f}",
                    f"seeded: {m['true_at_risk']:,.0f}", delta_color="off", border=True)
        w[2].metric("Manual effort (min)", f"{manual_total:,.0f}", border=True)
        w[3].metric("With agents (min)", f"{assisted:,.0f}",
                    f"{saved / manual_total:.0%} saved" if manual_total else None, delta_color="off",
                    border=True)
        st.caption("Effort figures are estimates from the sidebar assumptions. Agent compute time is "
                   "measured (animation excluded); people only review flagged invoices.")

        ch1, ch2 = st.columns(2)
        dec = pd.DataFrame([{"Status": label(k), "Invoices": n} for k, n in counts.items()])
        ch1.markdown("**Decisions**")
        ch1.altair_chart(alt.Chart(dec).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("Invoices:Q", title=None), y=alt.Y("Status:N", sort="-x", title=None),
            color=alt.Color("Status:N", legend=None, scale=alt.Scale(
                domain=list(LABEL_COLOUR), range=list(LABEL_COLOUR.values())))).properties(height=200),
            width="stretch")
        types = Counter(i["type"].replace("_", " ").capitalize() for r in res for i in r["issues"])
        ch2.markdown("**Issues found**")
        if types:
            iss = pd.DataFrame([{"Issue": t, "Count": n} for t, n in types.items()])
            ch2.altair_chart(alt.Chart(iss).mark_bar(cornerRadiusEnd=3, color=ui.INK).encode(
                x=alt.X("Count:Q", title=None), y=alt.Y("Issue:N", sort="-x", title=None)
            ).properties(height=200), width="stretch")

        st.markdown("#### All invoices")
        options = [label(k) for k in (APPROVED, HOLD, REJECTED, PENDING)]
        pick = st.pills("Show", options, default=options, selection_mode="multi")
        df = results_df(sorted(res, key=lambda r: r["idx"]))
        df = df[df["Status"].isin(pick or [])]
        table(df)
        st.download_button("Download as CSV", df.to_csv(index=False), "ap_results.csv", "text/csv",
                           icon=":material/download:")

        st.markdown("#### Look at one invoice")
        ordered = sorted(res, key=lambda r: r["idx"])
        r = st.selectbox("Invoice", ordered, label_visibility="collapsed",
                         format_func=lambda r: f"#{r['idx']:02d}  {r['invoice_no']}  {r['vendor']}  ({label(final(r))})")
        show_invoice(r)
        st.caption(f"Answer key: {r['truth']['kind']}")

# --------------------------------------------------------------------------- tab: upload PDF
with tab_upload:
    st.markdown("Upload text-based PDF invoices. They go through the same agents and are checked against "
                "this batch's purchase orders, goods receipts and vendor memory. In rule-based mode only "
                "this demo layout can be read; pick the Ollama model for other layouts.")
    with st.container(border=True):
        st.markdown("**No PDF handy?** Download one of this batch's invoices, then upload it. An invoice the "
                    "agents already checked will be caught as a duplicate.")
        sc1, sc2 = st.columns([3, 1], vertical_alignment="bottom")
        pick_inv = sc1.selectbox("Sample invoice", batch.invoices, format_func=lambda i: (
            f"#{i['idx']:02d} {i['truth']['invoice_no']}  {i['truth']['vendor']}"
            + (f"  [{i['truth']['kind']}]" if reveal else "")))
        sc2.download_button("Download PDF", invoice_to_pdf(pick_inv["text"], pick_inv["truth"]["invoice_no"]),
                            f"{pick_inv['truth']['invoice_no']}.pdf", "application/pdf",
                            icon=":material/download:", width="stretch")

    files = st.file_uploader("Invoice PDFs", type=["pdf"], accept_multiple_files=True)
    if files and st.button("Check uploaded invoices", type="primary", icon=":material/play_arrow:"):
        ctx = st.session_state.get("ctx")
        if ctx is None:
            llm, warn = get_llm("ollama" if engine.startswith("Ollama") else "mock", model)
            if warn:
                st.warning(warn, icon=":material/warning:")
            ctx = st.session_state.ctx = new_ctx(llm, batch, hitl)
        ctx.hitl = hitl
        uploads = st.session_state.setdefault("uploads", [])
        for f in files:
            text = pdf_to_text(f.getvalue())
            if not text.strip():
                st.error(f"{f.name} has no text layer, so it is probably a scanned image. "
                         "Upload a text-based PDF; scanned invoices are not supported yet.")
                continue
            uploads.append(process_one(text, ctx, use_graph, f.name))
        st.rerun()

    ups = st.session_state.get("uploads", [])
    if ups:
        table(results_df(ups).drop(columns=["#", "Priority"]).assign(
            File=[u["truth"].get("source", "") for u in ups]))
        u = st.selectbox("Look at an upload", ups,
                         format_func=lambda u: f"{u['truth'].get('source')}  ({label(final(u))})")
        show_invoice(u)

# --------------------------------------------------------------------------- tab: memory
with tab_memory:
    st.markdown("The Fraud Agent reads this memory and every review decision writes to it. Approved "
                "totals extend a vendor's history, escalations raise its risk, and a reviewer-approved "
                "bank change becomes the account on file.")
    st.dataframe(pd.DataFrame(memory.table()), width="stretch", hide_index=True)
    st.caption(f"Saved to {MEMORY_FILE}." if memory.path
               else "Kept for this browser session only. Turn on 'Remember vendors' in the sidebar to keep it.")

# --------------------------------------------------------------------------- tab: architecture
with tab_arch:
    st.graphviz_chart(ARCHITECTURE_DOT, width="stretch")
    st.markdown("""
| Agent | What it does | Tools it uses |
|---|---|---|
| **Planner** | Sorts the batch by due date and amount, flags urgent invoices | header peek |
| **Extraction** | Turns invoice text into a typed schema (AI model or parser); retries with a robust parser if rejected | `parse_invoice`, LLM |
| **Critic** | Rejects extractions that don't add up or contain values not in the document; asks the AI model for a second opinion; checks every money variance is explained | `validate_extraction`, `grounding_problems`, LLM |
| **Matching** | Invoice vs purchase order vs goods receipt: price, quantity, tax, vendor and duplicates | `three_way_match`, PO/GRN data, duplicate registry |
| **Fraud** | Lookalike vendors, bank changes, amount spikes, round amounts; 0-100 risk score, explained in plain words | `fraud_checks`, vendor memory, LLM |
| **Decision** | Applies the business rules, writes the reason and a case summary for reviewers | `decide`, LLM |
| **Human review** | Pauses escalations (LangGraph `interrupt`), resumes with your answer and teaches the vendor memory | checkpointer, vendor memory |

**Decision rules:** no issues means *approved*. Duplicate, unknown or lookalike vendor, bank change, missing PO,
vendor mismatch or unexplained variance means *escalated*. A fraud risk score of 40 or more means *escalated*.
Other mismatches are *held* for vendor correction, or *escalated* if 10,000 INR or more is at risk.
Escalated invoices wait for a person to approve, hold or reject.
""")
