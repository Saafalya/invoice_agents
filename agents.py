"""The agents. Each node takes (state, ctx) and returns a partial state update."""
from __future__ import annotations

from config import (APPROVED, ESCALATED, HOLD, MAX_EXTRACT_ATTEMPTS, REJECTED, URGENT_DAYS,
                    VARIANCE_TOLERANCE)
from tools import (decide, fraud_checks, grounding_problems, parse_invoice, peek_header,
                   three_way_match, validate_extraction)

EXPLAINED_BY = {"PRICE_MISMATCH", "QTY_EXCESS", "TAX_ERROR", "UNORDERED_ITEM", "DUPLICATE",
                "NO_PO", "UNKNOWN_VENDOR", "VENDOR_MISMATCH", "LOOKALIKE_VENDOR"}
WHOLE_PAYMENT_AT_RISK = {"LOOKALIKE_VENDOR", "BANK_CHANGE"}
REVIEW_ACTIONS = {"APPROVE": APPROVED, "REJECT": REJECTED, "HOLD": HOLD}


def _log(agent: str, msg: str) -> dict:
    return {"agent": agent, "msg": msg}


# --------------------------------------------------------------------------- Planner (batch level)
def planner_agent(raws: list[str], today) -> list[dict]:
    """Sequence the batch: nearest due date first, then biggest amount, then arrival order."""
    plan = []
    for idx, raw in enumerate(raws):
        h = peek_header(raw, today)
        urgent = h["days_left"] <= URGENT_DAYS
        plan.append({
            "idx": idx, **h,
            "priority": "URGENT" if urgent else "NORMAL",
            "checks": ["extract", "validate", "3-way match", "duplicate", "tax"],
        })
    plan.sort(key=lambda p: (p["days_left"], -p["total_hint"], p["idx"]))
    return plan


# --------------------------------------------------------------------------- Extraction Agent
def extractor_agent(state, ctx):
    attempt = state.get("attempt", 0) + 1
    name = "Extraction Agent"
    ext, method = None, ""
    if attempt == 1 and ctx.llm.can_extract:
        ext, method = ctx.llm.extract(state["raw"]), f"LLM ({ctx.llm.name})"
        if ext is None:
            method = "LLM returned invalid JSON -> fast parser"
    if ext is None:
        robust = attempt > 1
        method = method or ("robust parser (handles thousand separators)" if robust else "fast parser")
        try:
            ext = parse_invoice(state["raw"], robust=robust)
        except Exception as e:  # noqa: BLE001
            return {"attempt": attempt, "extract": None,
                    "log": [_log(name, f"attempt {attempt} via {method}: parse failed ({e})")]}
    return {"attempt": attempt, "extract": ext,
            "log": [_log(name, f"attempt {attempt} via {method}: {ext.invoice_no} | {ext.vendor} | total {ext.total:,.2f}")]}


# --------------------------------------------------------------------------- Critic Agent (1)
def critic_extract_agent(state, ctx):
    name, ext, raw = "Critic Agent", state.get("extract"), state["raw"]
    problems = ["extraction produced no result"] if ext is None else validate_extraction(ext)
    if ext is not None:
        if f"{ext.total:,.2f}" not in raw and f"{ext.total:.2f}" not in raw:
            problems.append(f"extracted total {ext.total:,.2f} does not appear in the document")
        problems += grounding_problems(ext, raw)      # hard rule: no invented field values
    retry = state["attempt"] < MAX_EXTRACT_ATTEMPTS
    if problems:
        msg = (f"extraction REJECTED ({problems[0]}) -> asking Extraction Agent to retry" if retry
               else f"extraction REJECTED again ({problems[0]}) -> cannot trust data, escalating")
        return {"extract_ok": False, "critic_issues": problems, "llm_review": [],
                "log": [_log(name, msg)]}

    # Rules passed. Ask the LLM for a second opinion (the mock returns nothing).
    review = ctx.llm.review_extraction(raw, ext)
    if review and retry:
        return {"extract_ok": False, "critic_issues": review, "llm_review": review,
                "log": [_log(name, f"rules passed, but LLM doubts it ({review[0]}) -> retrying with parser")]}
    if review:      # advisory only: a model's doubt alone never escalates an invoice
        return {"extract_ok": True, "critic_issues": [], "llm_review": review,
                "log": [_log(name, f"extraction PASSED rules; LLM note for reviewer: {review[0]}")]}
    return {"extract_ok": True, "critic_issues": [], "llm_review": [],
            "log": [_log(name, "extraction PASSED arithmetic, format and grounding checks")]}


def route_after_extract_critic(state) -> str:
    if state["extract_ok"]:
        return "matcher"
    return "extractor" if state["attempt"] < MAX_EXTRACT_ATTEMPTS else "decider"


# --------------------------------------------------------------------------- Matching Agent
def matcher_agent(state, ctx):
    inv = state["extract"]
    issues, expected_total, at_risk = three_way_match(inv, ctx.vendors, ctx.po_db, ctx.grn_db, ctx.seen)
    msg = (f"{len(issues)} issue(s): " + ", ".join(sorted({i['type'] for i in issues}))
           if issues else "invoice, PO and GRN agree")
    return {"issues": issues, "expected_total": expected_total, "at_risk": at_risk,
            "log": [_log("Matching Agent", f"3-way match done -> {msg}")]}


# --------------------------------------------------------------------------- Fraud Agent
def fraud_agent(state, ctx):
    """Looks for fraud a three-way match cannot see, using the vendor memory."""
    inv, issues = state["extract"], list(state["issues"])
    new, signals, score = fraud_checks(inv, issues, ctx.vendors, ctx.memory)
    at_risk = state["at_risk"]
    if {i["type"] for i in new} & WHOLE_PAYMENT_AT_RISK:
        at_risk = inv.total
    found = ", ".join(i["type"] for i in new) or "no fraud signals"
    risk_signals = [{"signal": k, "points": p, "detail": d} for k, p, d in signals]
    # The rules set the score; the LLM only explains it in plain words for the reviewer.
    note = ctx.llm.explain_risk({"vendor": inv.vendor, "risk_score": score, "signals": risk_signals})
    return {"issues": issues + new, "risk_score": score, "risk_signals": risk_signals,
            "risk_note": note, "at_risk": at_risk,
            "log": [_log("Fraud Agent", f"{found}; risk score {score}/100")]}


# --------------------------------------------------------------------------- Critic Agent (2)
def critic_match_agent(state, ctx):
    """Cross-check: any money variance must be explained by at least one named issue."""
    inv, issues = state["extract"], list(state["issues"])
    variance = round(inv.total - state["expected_total"], 2)
    types = {i["type"] for i in issues}
    if abs(variance) > VARIANCE_TOLERANCE and not (types & EXPLAINED_BY):
        issues.append({"type": "UNEXPLAINED_VARIANCE",
                       "detail": f"total differs from PO/GRN expectation by {variance:,.2f} with no matching rule"})
        msg = f"variance {variance:,.2f} is unexplained -> added UNEXPLAINED_VARIANCE"
    else:
        msg = f"variance {variance:,.2f} is consistent with findings"
    return {"issues": issues, "log": [_log("Critic Agent", msg)]}


# --------------------------------------------------------------------------- Decision Agent
def decision_agent(state, ctx):
    ext, issues = state.get("extract"), state.get("issues", [])
    at_risk = state.get("at_risk", ext.total if ext else 0.0)
    decision, rule = decide(issues, at_risk, state.get("extract_ok", False), state.get("risk_score", 0))
    reason = ctx.llm.explain({
        "decision": decision, "issues": issues, "at_risk": at_risk,
        "invoice_no": ext.invoice_no if ext else "?", "total": ext.total if ext else 0.0,
    })
    return {"decision": decision, "reason": reason, "at_risk": at_risk,
            "log": [_log("Decision Agent", f"{decision} ({rule})")]}


# --------------------------------------------------------------------------- Human Review (HITL)
def needs_human(state, ctx) -> bool:
    return bool(ctx.hitl) and state.get("decision") == ESCALATED


def review_request(state) -> dict:
    """What the reviewer sees when the graph pauses."""
    ext = state.get("extract")
    return {"invoice_no": ext.invoice_no if ext else "(unreadable)",
            "vendor": ext.vendor if ext else "(unreadable)", "total": ext.total if ext else 0.0,
            "reason": state.get("reason", ""), "issues": state.get("issues", []),
            "risk_score": state.get("risk_score", 0), "risk_note": state.get("risk_note", ""),
            "llm_review": state.get("llm_review", [])}


def human_review_agent(state, ctx):
    """Pauses escalated invoices for a person, then writes the outcome to vendor memory.

    In LangGraph mode `ctx.ask_human` is `langgraph.types.interrupt`: the graph checkpoints and
    stops here, and resumes when the reviewer answers. LangGraph re-runs this node on resume,
    so nothing with side effects happens before the answer is in.
    """
    ext, decision = state.get("extract"), state["decision"]
    answer = None
    if needs_human(state, ctx):
        answer = ctx.ask_human(review_request(state), state)
        final, by = REVIEW_ACTIONS[answer["action"]], answer.get("reviewer") or "reviewer"
        note = f" - \"{answer['note']}\"" if answer.get("note") else ""
        msg = f"{by} reviewed: {answer['action']} -> {final}{note}"
    else:
        final, by = decision, "auto"
        msg = f"no review needed -> {final}"
    if ext is not None:
        ctx.memory.record(ext.vendor, ext.total, decision, final)
        types = {i["type"] for i in state.get("issues", [])}
        if answer and final == APPROVED and "BANK_CHANGE" in types:
            ctx.memory.accept_bank(ext.vendor, ext.bank_account)
            msg += "; new bank account saved to vendor memory"
    return {"final_decision": final, "reviewed_by": by,
            "review_note": (answer or {}).get("note", ""),
            "log": [_log("Human Review", msg)]}
