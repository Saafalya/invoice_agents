"""Streamlit PoC: agentic three-way invoice matching for accounts payable.

Run with:  streamlit run app.py
"""
import time
from collections import Counter

import pandas as pd
import streamlit as st

from config import APPROVED, ESCALATED, HOLD, REJECTED
from data_gen import generate_batch
from llm import get_llm
from memory import VendorMemory
from orchestrator import HAS_LANGGRAPH, Ctx, evaluate, finish_review, process_batch, process_one
from pdf_tools import invoice_to_pdf, pdf_to_text

st.set_page_config(page_title="AP Three-Way Match Agents", page_icon="🧾", layout="wide")

PENDING = "PENDING REVIEW"
COLORS = {APPROVED: "#d1fae5", HOLD: "#fef3c7", ESCALATED: "#fee2e2", REJECTED: "#fecaca",
          PENDING: "#e0e7ff"}
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


def results_df(results):
    return pd.DataFrame([{
        "#": r["idx"], "Invoice": r["invoice_no"], "Vendor": r["vendor"],
        "Total (INR)": r["total"], "Priority": r["priority"], "Decision": r["decision"],
        "Final": final(r), "Risk": r["risk_score"],
        "Issues": ", ".join(sorted({i["type"] for i in r["issues"]})) or "-",
        "At risk (INR)": r["at_risk"], "Attempts": r["attempts"],
    } for r in results])


def style(df):
    fmt = {c: "{:,.2f}" for c in ("Total (INR)", "At risk (INR)") if c in df.columns}
    key = "Final" if "Final" in df.columns else "Decision"
    return (df.style
            .apply(lambda row: [f"background-color: {COLORS[row[key]]}; color: #111"] * len(row), axis=1)
            .format(fmt))


def get_memory(persist: bool) -> VendorMemory:
    mem = st.session_state.get("memory")
    if mem is None or bool(mem.path) != persist:
        mem = st.session_state.memory = VendorMemory(MEMORY_FILE if persist else None)
    return mem


def new_ctx(llm, batch, hitl) -> Ctx:
    mem = st.session_state.memory
    mem.seed(batch.history, batch.bank_db)
    return Ctx(llm, batch.vendors, batch.po_db, batch.grn_db, batch.today, memory=mem, hitl=hitl)


def show_invoice(r):
    a, b = st.columns(2)
    a.markdown("**Raw invoice**")
    a.code(r["raw"], language=None)
    b.markdown("**Agent decision**")
    b.info(f"{r['decision']}: {r['reason']}")
    if r["issues"]:
        b.dataframe(pd.DataFrame(r["issues"]), width="stretch", hide_index=True)
    if r["risk_signals"]:
        b.markdown(f"**Fraud risk score: {r['risk_score']}/100**")
        if r.get("risk_note"):
            b.warning(f"Fraud Agent: {r['risk_note']}")
        b.dataframe(pd.DataFrame(r["risk_signals"]), width="stretch", hide_index=True)
    if r.get("llm_review"):
        b.caption("Critic Agent (LLM second opinion): " + "; ".join(r["llm_review"]))
    b.markdown("**Agent trace**")
    b.code("\n".join(f"{e['agent']}: {e['msg']}" for e in r["trace"]), language=None)
    with st.expander("Extracted data (validated schema)", expanded=False):
        st.json(r["extract"] or {})


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Batch")
    n_inv = st.slider("Invoices in batch", 10, 60, 24)
    err_pct = st.slider("Seeded error rate", 0.10, 0.60, 0.35, 0.05)
    seed = st.number_input("Random seed", 0, 9999, 42)
    regenerate = st.button("Generate new batch")

    st.header("Agent engine")
    engine = st.radio("LLM backend", ["Mock (no setup needed)", "Ollama (local model)"])
    model = st.text_input("Ollama model", "llama3.2") if engine.startswith("Ollama") else ""
    use_graph = st.checkbox("Orchestrate with LangGraph", value=HAS_LANGGRAPH, disabled=not HAS_LANGGRAPH)
    if not HAS_LANGGRAPH:
        st.caption("langgraph is not installed, using the built-in runner (same workflow).")
    delay = st.slider("Demo delay per step (s)", 0.0, 0.5, 0.08, 0.02)

    st.header("Human in the loop")
    hitl = st.checkbox("Pause escalated invoices for a reviewer", value=True)
    reviewer = st.text_input("Reviewer name", "Finance reviewer")

    st.header("Agent memory")
    persist = st.checkbox(f"Remember vendors across sessions ({MEMORY_FILE})", value=False)
    memory = get_memory(persist)
    if st.button("Reset vendor memory"):
        memory.vendors = {}
        memory.save()

    st.header("Time-saving assumptions")
    manual_min = st.number_input("Manual review, min per invoice", 1.0, 30.0, 6.0, 0.5)
    review_min = st.number_input("Human review of a flagged invoice, min", 1.0, 30.0, 4.0, 0.5)

if regenerate or "batch" not in st.session_state:
    st.session_state.batch = generate_batch(n_inv, err_pct, int(seed))
    for k in ("results", "ctx", "uploads"):
        st.session_state.pop(k, None)

batch = st.session_state.batch
memory.seed(batch.history, batch.bank_db)

st.title("🧾 Invoice Three-Way Match Agents")
st.caption("Multi-agent accounts-payable workflow: plan, extract, validate, match, detect fraud, decide, "
           "and ask a human when it matters. All data is synthetic.")

n_pending = sum(r["status"] == "PENDING_REVIEW"
                for r in st.session_state.get("results", []) + st.session_state.get("uploads", []))
tab_run, tab_review, tab_results, tab_upload, tab_memory, tab_arch = st.tabs([
    "1. Run agents", f"2. Review queue ({n_pending})", "3. Results and validation", "4. Upload PDF",
    "5. Vendor memory", "6. Architecture"])

# --------------------------------------------------------------------------- tab 1: run
with tab_run:
    st.subheader("Incoming batch")
    st.write("Each invoice must agree with its purchase order (what we ordered) and goods receipt note "
             "(what actually arrived) before it is paid. Some invoices in this batch contain seeded errors.")
    reveal = st.checkbox("Reveal answer key (seeded errors)")
    preview = pd.DataFrame([{
        "#": i["idx"], "Invoice": i["truth"]["invoice_no"], "Vendor": i["truth"]["vendor"],
        "Total (INR)": i["truth"]["total"],
        **({"Seeded error": i["truth"]["kind"]} if reveal else {}),
    } for i in batch.invoices])
    st.dataframe(preview, width="stretch", hide_index=True, height=240)
    with st.expander("Master data the agents look up (POs, goods receipts, vendors)"):
        st.dataframe(pd.DataFrame([
            {"PO": po, "Vendor": d["vendor"], "SKU": sku, "Ordered": l["qty"], "PO price": l["price"],
             "Received (GRN)": batch.grn_db[po].get(sku)}
            for po, d in batch.po_db.items() for sku, l in d["lines"].items()
        ]), width="stretch", hide_index=True)
        st.write("Approved vendors:", ", ".join(batch.vendors))
    with st.expander("Sample invoice text"):
        st.code(batch.invoices[0]["text"], language=None)

    if st.button("▶ Run agents", type="primary"):
        llm, warn = get_llm("ollama" if engine.startswith("Ollama") else "mock", model)
        if warn:
            st.warning(warn)
        ctx = st.session_state.ctx = new_ctx(llm, batch, hitl)
        st.session_state.uploads = []
        total = len(batch.invoices)
        bar = st.progress(0.0, text="Planner Agent is sequencing the batch...")
        left, right = st.columns([3, 2])
        left.markdown("**Live agent trace**")
        trace_box = left.empty()
        right.markdown("**Decisions so far**")
        table_box = right.empty()

        lines, results, proc = [], [], 0.0
        for ev, dt in timed(process_batch(batch, ctx, use_graph)):
            proc += dt
            if ev["type"] == "plan":
                entries, prefix = ev["log"], ""
            elif ev["type"] == "step":
                entries, prefix = ev["update"].get("log", []), f"#{ev['idx']:02d} "
            else:
                entries, prefix = [], ""
            lines += [f"{prefix}{e['agent']}: {e['msg']}" for e in entries]
            trace_box.code("\n".join(lines[-16:]), language=None)
            if ev["type"] == "result":
                results.append(ev["result"])
                bar.progress(len(results) / total, text=f"{len(results)}/{total} invoices processed")
                table_box.dataframe(
                    style(results_df(results)[["Invoice", "Final", "Risk", "At risk (INR)"]].iloc[::-1]),
                    width="stretch", hide_index=True, height=300)
            elif ev["type"] == "step":
                time.sleep(delay)

        st.session_state.results = results
        st.session_state.proc_seconds = proc
        st.session_state.engine_name = llm.name
        bar.progress(1.0, text="Done")
        waiting = sum(r["status"] == "PENDING_REVIEW" for r in results)
        st.success("Batch complete. " + (f"{waiting} escalated invoice(s) are paused in the Review queue."
                                         if waiting else "Open the 'Results and validation' tab."))
        if waiting:
            st.button("Open the review queue")         # rerun so the tab badge shows the count

# --------------------------------------------------------------------------- tab 2: review queue
with tab_review:
    st.subheader("Human-in-the-loop review")
    st.write("Escalated invoices stop at a LangGraph `interrupt`. The graph state is checkpointed; your "
             "answer resumes it. Your decision is written to vendor memory, so the agents learn from it "
             "(for example, an approved bank change becomes the new account on file).")
    ctx = st.session_state.get("ctx")
    lists = {"results": st.session_state.get("results", []), "uploads": st.session_state.get("uploads", [])}
    queue = [(name, i, r) for name, rs in lists.items() for i, r in enumerate(rs)
             if r["status"] == "PENDING_REVIEW"]
    if not queue:
        st.info("Nothing waiting. Run the agents with 'Pause escalated invoices' switched on.")
    for name, i, r in sorted(queue, key=lambda q: -q[2]["risk_score"]):
        with st.container(border=True):
            st.markdown(f"### {r['invoice_no']} - {r['vendor']} - INR {r['total']:,.2f}  "
                        f"(risk {r['risk_score']}/100)")
            show_invoice(r)
            note = st.text_input("Note for the audit trail", key=f"note-{name}-{i}")
            cols = st.columns(3)
            for col, action, label in zip(cols, ("APPROVE", "HOLD", "REJECT"),
                                          ("✅ Approve payment", "⏸ Hold for vendor", "⛔ Reject")):
                if col.button(label, key=f"{action}-{name}-{i}"):
                    answer = {"action": action, "note": note, "reviewer": reviewer}
                    st.session_state[name][i] = finish_review(r, answer, ctx)
                    st.rerun()

    done = [r for rs in lists.values() for r in rs if r["reviewed_by"] not in ("", "auto")]
    if done:
        st.subheader("Audit trail")
        st.dataframe(pd.DataFrame([{
            "Invoice": r["invoice_no"], "Vendor": r["vendor"], "Agent said": r["decision"],
            "Final": r["final_decision"], "Reviewer": r["reviewed_by"], "Note": r["review_note"],
        } for r in done]), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- tab 2: results
with tab_results:
    res = st.session_state.get("results")
    if not res:
        st.info("Run the agents first (tab 1).")
    else:
        m = evaluate(res)
        counts = Counter(r["decision"] for r in res)
        st.caption(f"Engine: {st.session_state.engine_name}")

        c = st.columns(4)
        c[0].metric("Invoices processed", m["n"])
        c[1].metric("Approved", counts[APPROVED])
        c[2].metric("Held", counts[HOLD])
        c[3].metric("Escalated", counts[ESCALATED])

        h = st.columns(4)
        h[0].metric("Fraud cases escalated", f"{m['fraud_caught']}/{m['fraud_seeded']}",
                    help="Lookalike vendor, bank-account change and amount spike, seeded in the batch")
        h[1].metric("Awaiting human review", m["pending_review"])
        h[2].metric("Reviewed by a human", m["reviewed"])
        h[3].metric("Human overrides", m["overrides"],
                    help="Final decision differs from the agent's decision")

        st.subheader("Validation against the seeded answer key")
        v = st.columns(4)
        v[0].metric("Seeded errors caught", f"{m['caught']}/{m['seeded']}", f"{m['recall']:.0%} recall", delta_color="off")
        v[1].metric("Correct issue type found", f"{m['type_correct']}/{m['seeded']}")
        v[2].metric("Clean invoices wrongly flagged", f"{m['false_positives']}/{m['clean']}")
        v[3].metric("Bad extractions fixed by Critic loop", f"{m['recovered_by_retry']}/{m['retried']}")

        st.subheader("Money and time")
        flagged = m["n"] - counts[APPROVED]
        manual_total = m["n"] * manual_min
        assisted = st.session_state.proc_seconds / 60 + flagged * review_min
        saved = manual_total - assisted
        w = st.columns(4)
        w[0].metric("Payment value blocked (INR)", f"{m['blocked_value']:,.0f}")
        w[1].metric("Overpayment risk detected (INR)", f"{m['detected_at_risk']:,.0f}",
                    f"true seeded: {m['true_at_risk']:,.0f}", delta_color="off")
        w[2].metric("Manual effort (min)", f"{manual_total:,.0f}")
        w[3].metric("Agent-assisted effort (min)", f"{assisted:,.0f}",
                    f"{saved / manual_total:.0%} saved" if manual_total else None, delta_color="off")
        st.caption("Effort figures are estimates from the sidebar assumptions. Agent compute time is measured "
                   "(demo delay excluded); humans only review flagged invoices.")

        ch1, ch2 = st.columns(2)
        ch1.markdown("**Decisions**")
        ch1.bar_chart(pd.Series(counts, name="invoices"))
        types = Counter(i["type"] for r in res for i in r["issues"])
        ch2.markdown("**Issues found by type**")
        if types:
            ch2.bar_chart(pd.Series(types, name="issues"))

        st.subheader("All decisions")
        options = [APPROVED, HOLD, REJECTED, PENDING]
        pick = st.multiselect("Filter by final decision", options, default=options)
        df = results_df(sorted(res, key=lambda r: r["idx"]))
        df = df[df["Final"].isin(pick)]
        st.dataframe(style(df), width="stretch", hide_index=True)
        st.download_button("Download results (CSV)", df.to_csv(index=False), "ap_results.csv", "text/csv")

        st.subheader("Inspect one invoice")
        ordered = sorted(res, key=lambda r: r["idx"])
        r = st.selectbox("Invoice", ordered,
                         format_func=lambda r: f"#{r['idx']:02d}  {r['invoice_no']}  -  {r['vendor']}  [{r['decision']}]")
        show_invoice(r)
        st.caption(f"Answer key: {r['truth']['kind']}")

# --------------------------------------------------------------------------- tab 4: upload PDF
with tab_upload:
    st.subheader("Run the agents on PDF invoices")
    st.write("Upload text-based PDF invoices. They go through the same agents, checked against this "
             "batch's POs, goods receipts, vendor master and memory. The mock backend reads this demo "
             "layout; for other layouts pick the Ollama backend so the LLM does the extraction.")
    st.markdown("**No PDF handy?** Download one of this batch's invoices as a PDF, then upload it. "
                "Uploading an invoice the agents already processed shows the duplicate check in action.")
    sc1, sc2 = st.columns([3, 1])
    pick_inv = sc1.selectbox("Sample invoice", batch.invoices, format_func=lambda i: (
        f"#{i['idx']:02d} {i['truth']['invoice_no']} - {i['truth']['vendor']}"
        + (f" [{i['truth']['kind']}]" if reveal else "")))
    sc2.download_button("Download PDF", invoice_to_pdf(pick_inv["text"], pick_inv["truth"]["invoice_no"]),
                        f"{pick_inv['truth']['invoice_no']}.pdf", "application/pdf")

    files = st.file_uploader("Invoice PDFs", type=["pdf"], accept_multiple_files=True)
    if files and st.button("▶ Run agents on uploads", type="primary"):
        ctx = st.session_state.get("ctx")
        if ctx is None:
            llm, warn = get_llm("ollama" if engine.startswith("Ollama") else "mock", model)
            if warn:
                st.warning(warn)
            ctx = st.session_state.ctx = new_ctx(llm, batch, hitl)
        ctx.hitl = hitl
        uploads = st.session_state.setdefault("uploads", [])
        for f in files:
            text = pdf_to_text(f.getvalue())
            if not text.strip():
                st.error(f"{f.name}: no text layer found (scanned image?). OCR is not supported yet.")
                continue
            uploads.append(process_one(text, ctx, use_graph, f.name))
        st.rerun()

    ups = st.session_state.get("uploads", [])
    if ups:
        st.dataframe(style(results_df(ups).drop(columns=["#", "Priority"]).assign(
            File=[u["truth"].get("source", "") for u in ups])), width="stretch", hide_index=True)
        u = st.selectbox("Inspect upload", ups, format_func=lambda u: f"{u['truth'].get('source')} [{final(u)}]")
        show_invoice(u)

# --------------------------------------------------------------------------- tab 5: memory
with tab_memory:
    st.subheader("Vendor memory (long-term)")
    st.write("The Fraud Agent reads this memory; the Human Review step writes to it. It starts from the "
             "vendor master and past approved invoices, then learns from every decision: approved totals "
             "extend the vendor's history, escalations raise its risk, and a reviewer-approved bank change "
             "becomes the account on file.")
    st.dataframe(pd.DataFrame(memory.table()), width="stretch", hide_index=True)
    st.caption("Saved to vendor_memory.json when 'Remember vendors across sessions' is on." if memory.path
               else "Kept for this browser session only.")

# --------------------------------------------------------------------------- tab 3: architecture
with tab_arch:
    st.subheader("Agent workflow")
    st.graphviz_chart(ARCHITECTURE_DOT, width="stretch")
    st.markdown("""
| Agent | Responsibility | Tools it uses |
|---|---|---|
| **Planner** | Sequences the batch by due date and amount, flags urgent invoices | header peek |
| **Extraction** | Turns invoice text into a typed schema (LLM or parser); retries with a robust parser if rejected | `parse_invoice`, LLM |
| **Critic** | (1) Rejects extractions whose arithmetic does not add up. (2) Checks every money variance is explained by a named issue | `validate_extraction` |
| **Matching** | Three-way match of invoice, PO and GRN; price, quantity, tax, vendor and duplicate checks | `three_way_match`, PO/GRN DB, duplicate registry |
| **Fraud** | Lookalike vendor names, bank-account changes, amount spikes vs history, round amounts; gives a 0-100 risk score with reasons | `fraud_checks`, vendor memory |
| **Decision** | Applies business rules, then writes a plain-language reason | `decide`, LLM |
| **Human Review** | Pauses escalations with a LangGraph `interrupt`; resumes with the reviewer's answer and teaches the memory | checkpointer, vendor memory |

**Decision rules:** no issues -> *approved*. Duplicate, unknown or lookalike vendor, bank change, missing PO, vendor mismatch or
unexplained variance -> *escalated*. Fraud risk score 40 or more -> *escalated*. Other mismatches -> *held* for vendor correction,
or *escalated* if the amount at risk is 10,000 INR or more. Escalated invoices wait for a human: approve, hold or reject.
""")
