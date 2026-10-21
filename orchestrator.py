"""Workflow wiring: LangGraph if installed, otherwise an equivalent plain-Python runner."""
from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, TypedDict

from agents import (critic_extract_agent, critic_match_agent, decision_agent, extractor_agent,
                    fraud_agent, human_review_agent, matcher_agent, needs_human, planner_agent,
                    route_after_extract_critic)
from config import APPROVED, ESCALATED
from memory import VendorMemory

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.graph import END, StateGraph
    from langgraph.types import Command, interrupt
    HAS_LANGGRAPH = True
except ImportError:  # the PoC still runs, just without LangGraph
    HAS_LANGGRAPH = False


class State(TypedDict, total=False):
    raw: str
    plan: dict
    attempt: int
    extract: Any
    extract_ok: bool
    critic_issues: list
    issues: list
    expected_total: float
    at_risk: float
    decision: str
    reason: str
    risk_score: int
    risk_signals: list
    final_decision: str
    reviewed_by: str
    review_note: str
    human_answer: dict      # plain-Python runner only: the reviewer's answer on resume
    log: Annotated[list, operator.add]


@dataclass
class Ctx:
    llm: Any
    vendors: list
    po_db: dict
    grn_db: dict
    today: Any
    seen: set = field(default_factory=set)      # duplicate registry for this run
    memory: Any = None                          # VendorMemory (long-term, across runs)
    hitl: bool = False                          # pause escalations for a human reviewer
    graph: Any = None                           # compiled LangGraph, kept for resuming

    def __post_init__(self):
        if self.memory is None:
            self.memory = VendorMemory()

    def ask_human(self, payload: dict, state: dict) -> dict:
        if self.graph is not None:
            return interrupt(payload)               # checkpoint and pause the graph here
        return state["human_answer"]                # plain runner: answer was put in state


NODES = {
    "extractor": extractor_agent,
    "critic_extract": critic_extract_agent,
    "matcher": matcher_agent,
    "fraud": fraud_agent,
    "critic_match": critic_match_agent,
    "decider": decision_agent,
    "review": human_review_agent,
}


def build_graph(ctx: Ctx):
    g = StateGraph(State)
    def bind(fn):
        return lambda state: fn(state, ctx)

    for name, fn in NODES.items():
        g.add_node(name, bind(fn))
    g.set_entry_point("extractor")
    g.add_edge("extractor", "critic_extract")
    g.add_conditional_edges("critic_extract", route_after_extract_critic,
                            {"extractor": "extractor", "matcher": "matcher", "decider": "decider"})
    g.add_edge("matcher", "fraud")
    g.add_edge("fraud", "critic_match")
    g.add_edge("critic_match", "decider")
    g.add_edge("decider", "review")
    g.add_edge("review", END)
    return g.compile(checkpointer=_checkpointer())      # checkpoints make interrupt/resume work


def _checkpointer():
    """In-memory checkpoints; our Pydantic schema is allow-listed for safe deserialisation."""
    try:
        serde = JsonPlusSerializer(allowed_msgpack_modules=[("schemas", "InvoiceExtract"),
                                                            ("schemas", "LineItem")])
    except TypeError:                     # older langgraph without the allow-list option
        return InMemorySaver()
    return InMemorySaver(serde=serde)


def _merge(state: dict, upd: dict) -> dict:
    new = dict(state)
    for k, v in upd.items():
        new[k] = list(state.get("log", [])) + list(v) if k == "log" else v
    return new


def _next(node: str, state: dict) -> Optional[str]:
    """Plain-Python mirror of the LangGraph edges above."""
    if node == "critic_extract":
        return route_after_extract_critic(state)
    return {"extractor": "critic_extract", "matcher": "fraud", "fraud": "critic_match",
            "critic_match": "decider", "decider": "review", "review": None}[node]


PAUSED = "__interrupt__"     # node name yielded when the graph stops for a human


def _stream(graph, inp, ctx: Ctx, state: dict, thread_id: str):
    for chunk in graph.stream(inp, {"configurable": {"thread_id": thread_id}}, stream_mode="updates"):
        for node, upd in chunk.items():
            if node == PAUSED:
                yield PAUSED, {"log": []}, state
                return
            state = _merge(state, upd)
            yield node, upd, state


def _walk(node: Optional[str], ctx: Ctx, state: dict):
    while node:
        if node == "review" and needs_human(state, ctx) and "human_answer" not in state:
            yield PAUSED, {"log": []}, state
            return
        upd = NODES[node](state, ctx)
        state = _merge(state, upd)
        yield node, upd, state
        node = _next(node, state)


def run_invoice(raw: str, plan_item: dict, ctx: Ctx, thread_id: str = ""):
    """Yield (node, update, state_so_far) for each agent step on one invoice.

    If a human must review it, the last item is (PAUSED, ..., state): call resume_invoice later.
    """
    state: dict = {"raw": raw, "plan": plan_item, "attempt": 0, "log": []}
    if ctx.graph is not None:
        yield from _stream(ctx.graph, state, ctx, state, thread_id or str(uuid.uuid4()))
    else:
        yield from _walk("extractor", ctx, state)


def resume_invoice(pending: dict, answer: dict, ctx: Ctx):
    """Continue a paused invoice with the reviewer's answer. Yields steps like run_invoice."""
    state = pending["state"]
    if ctx.graph is not None:
        yield from _stream(ctx.graph, Command(resume=answer), ctx, state, pending["thread_id"])
    else:
        yield from _walk("review", ctx, {**state, "human_answer": answer})


def _to_result(inv: dict, item: dict, state: dict, thread_id: str = "") -> dict:
    ext = state.get("extract")
    paused = "final_decision" not in state
    return {
        "idx": item["idx"], "priority": item["priority"], "days_left": item["days_left"],
        "invoice_no": ext.invoice_no if ext else "(unreadable)",
        "vendor": ext.vendor if ext else "(unreadable)",
        "total": ext.total if ext else item["total_hint"],
        "decision": state["decision"], "reason": state["reason"],
        "issues": state.get("issues", []), "critic_issues": state.get("critic_issues", []),
        "at_risk": round(state.get("at_risk", 0.0), 2), "attempts": state.get("attempt", 1),
        "extract": ext.model_dump() if ext else None,
        "raw": inv["text"], "trace": state["log"], "truth": inv["truth"],
        "risk_score": state.get("risk_score", 0), "risk_signals": state.get("risk_signals", []),
        "final_decision": state.get("final_decision"), "reviewed_by": state.get("reviewed_by", ""),
        "review_note": state.get("review_note", ""),
        "status": "PENDING_REVIEW" if paused else "DONE",
        "pending": {"state": state, "thread_id": thread_id, "inv": inv, "item": item} if paused else None,
    }


def finish_review(result: dict, answer: dict, ctx: Ctx) -> dict:
    """Apply a reviewer's answer to a paused result and return the updated result."""
    p = result["pending"]
    state = p["state"]
    for _node, _upd, state in resume_invoice(p, answer, ctx):
        pass
    return _to_result(p["inv"], p["item"], state, p["thread_id"])


def process_batch(batch, ctx: Ctx, use_langgraph: bool = True):
    """Generator of UI events: plan -> (step* -> result)*."""
    ctx.memory.seed(batch.history, batch.bank_db)
    plan = planner_agent([i["text"] for i in batch.invoices], ctx.today)
    n_urgent = sum(p["priority"] == "URGENT" for p in plan)
    yield {"type": "plan", "plan": plan, "log": [{
        "agent": "Planner Agent",
        "msg": f"sequenced {len(plan)} invoices by due date and amount; {n_urgent} urgent (due within 7 days)"}]}
    ctx.graph = build_graph(ctx) if (use_langgraph and HAS_LANGGRAPH) else None
    run_id = uuid.uuid4().hex[:8]
    for pos, item in enumerate(plan):
        inv, state, tid = batch.invoices[item["idx"]], {}, f"{run_id}-{item['idx']}"
        for node, upd, state in run_invoice(inv["text"], item, ctx, tid):
            if node != PAUSED:
                yield {"type": "step", "pos": pos, "idx": item["idx"], "node": node, "update": upd}
        yield {"type": "result", "result": _to_result(inv, item, state, tid)}


def process_one(text: str, ctx: Ctx, use_langgraph: bool = True, name: str = "upload") -> dict:
    """Run a single invoice (e.g. an uploaded PDF) through the same agents."""
    if use_langgraph and HAS_LANGGRAPH and ctx.graph is None:
        ctx.graph = build_graph(ctx)
    item = {**planner_agent([text], ctx.today)[0], "idx": -1}
    inv = {"idx": -1, "text": text, "truth": {"kind": "UPLOADED", "expected_issues": [],
                                              "true_at_risk": 0.0, "source": name}}
    state, tid = {}, f"upload-{uuid.uuid4().hex[:8]}"
    for _node, _upd, state in run_invoice(text, item, ctx, tid):
        pass
    return _to_result(inv, item, state, tid)


FRAUD_KINDS = {"LOOKALIKE_VENDOR", "BANK_CHANGE", "AMOUNT_SPIKE"}


def evaluate(results: list[dict]) -> dict:
    """Compare agent decisions with the hidden seeded ground truth."""
    seeded = [r for r in results if r["truth"]["expected_issues"]]
    clean = [r for r in results if not r["truth"]["expected_issues"]]
    caught = [r for r in seeded if r["decision"] != APPROVED]
    exact = [r for r in seeded
             if set(r["truth"]["expected_issues"]) <= {i["type"] for i in r["issues"]}]
    false_pos = [r for r in clean if r["decision"] != APPROVED]
    return {
        "n": len(results), "seeded": len(seeded), "caught": len(caught),
        "recall": len(caught) / len(seeded) if seeded else 1.0,
        "type_correct": len(exact), "clean": len(clean), "false_positives": len(false_pos),
        "retried": sum(r["attempts"] > 1 for r in results),
        "recovered_by_retry": sum(r["attempts"] > 1 and r["extract"] is not None
                                  and not r["critic_issues"] for r in results),
        "true_at_risk": sum(r["truth"]["true_at_risk"] for r in seeded),
        "detected_at_risk": sum(r["at_risk"] for r in results if r["decision"] != APPROVED),
        "blocked_value": sum(r["total"] for r in results if r["decision"] != APPROVED),
        "fraud_seeded": sum(r["truth"]["kind"] in FRAUD_KINDS for r in results),
        "fraud_caught": sum(r["truth"]["kind"] in FRAUD_KINDS and r["decision"] == ESCALATED
                            for r in results),
        "pending_review": sum(r["status"] == "PENDING_REVIEW" for r in results),
        "reviewed": sum(r["reviewed_by"] not in ("", "auto") for r in results),
        "overrides": sum(r["final_decision"] not in (None, r["decision"]) for r in results),
    }
