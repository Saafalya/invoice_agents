"""Headless run: python run_cli.py [--ollama MODEL] [--no-langgraph] [--hitl] [--memory FILE]"""
import argparse
import time

from data_gen import generate_batch
from llm import get_llm
from memory import VendorMemory
from orchestrator import HAS_LANGGRAPH, Ctx, evaluate, finish_review, process_batch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--ollama", default=None, help="Ollama model name, e.g. llama3.2")
ap.add_argument("--no-langgraph", action="store_true")
ap.add_argument("--verbose", action="store_true")
ap.add_argument("--hitl", action="store_true", help="ask a human (you) about each escalated invoice")
ap.add_argument("--memory", default=None, help="vendor memory JSON file to load and keep learning in")
a = ap.parse_args()

batch = generate_batch(a.n, seed=a.seed)
llm, warn = get_llm("ollama" if a.ollama else "mock", a.ollama or "")
if warn:
    print("WARNING:", warn)
ctx = Ctx(llm, batch.vendors, batch.po_db, batch.grn_db, batch.today,
          memory=VendorMemory(a.memory), hitl=a.hitl)
print(f"LLM: {llm.name} | LangGraph: {HAS_LANGGRAPH and not a.no_langgraph}")

results, t0 = [], time.perf_counter()
for ev in process_batch(batch, ctx, use_langgraph=not a.no_langgraph):
    if ev["type"] == "plan":
        print(f"[Planner Agent] {ev['log'][0]['msg']}")
    elif ev["type"] == "step" and a.verbose:
        for e in ev["update"].get("log", []):
            print(f"  #{ev['idx']:02d} {e['agent']}: {e['msg']}")
    elif ev["type"] == "result":
        r = ev["result"]
        results.append(r)
        print(f"#{r['idx']:02d} {r['invoice_no']:9} {r['decision']:9} risk={r['risk_score']:>3} "
              f"at_risk={r['at_risk']:>10,.2f}  truth={r['truth']['kind']}")

pending = [r for r in results if r["status"] == "PENDING_REVIEW"]
if pending:
    print(f"\n{len(pending)} invoice(s) paused for human review. Answer a=approve, r=reject, h=hold.")
for r in pending:
    print(f"\n{r['invoice_no']} from {r['vendor']}, INR {r['total']:,.2f}, risk {r['risk_score']}/100")
    print("  " + r["reason"])
    if r["risk_signals"]:
        print("  Fraud Agent: " + r["risk_note"])
    choice = ""
    while choice not in ("a", "r", "h"):
        choice = input("  decision [a/r/h]: ").strip().lower()[:1]
    note = input("  note (optional): ").strip()
    answer = {"action": {"a": "APPROVE", "r": "REJECT", "h": "HOLD"}[choice], "note": note, "reviewer": "cli"}
    results[results.index(r)] = done = finish_review(r, answer, ctx)
    print(f"  -> final decision {done['final_decision']}")

m = evaluate(results)
print(f"\nProcessed {m['n']} in {time.perf_counter() - t0:.2f}s")
print(f"Seeded errors caught: {m['caught']}/{m['seeded']} ({m['recall']:.0%}) | issue type correct: {m['type_correct']}/{m['seeded']}")
print(f"False positives: {m['false_positives']}/{m['clean']} | retried extractions: {m['retried']} (recovered {m['recovered_by_retry']})")
print(f"Amount at risk: detected {m['detected_at_risk']:,.2f} vs true {m['true_at_risk']:,.2f}")
print(f"Fraud cases escalated: {m['fraud_caught']}/{m['fraud_seeded']} | human reviews: {m['reviewed']} "
      f"(overrides {m['overrides']})")
