"""End-to-end checks of the agent pipeline. Run with: pytest"""
import pytest

from data_gen import generate_batch
from llm import MockLLM
from memory import VendorMemory
from orchestrator import HAS_LANGGRAPH, Ctx, evaluate, finish_review, process_batch, process_one
from pdf_tools import invoice_to_pdf, pdf_to_text
from schemas import InvoiceExtract
from tools import fraud_checks

MODES = [False] + ([True] if HAS_LANGGRAPH else [])


def run(seed=42, n=24, err=0.35, hitl=False, langgraph=False, memory=None):
    b = generate_batch(n, err, seed)
    ctx = Ctx(MockLLM(), b.vendors, b.po_db, b.grn_db, b.today, memory=memory, hitl=hitl)
    res = [e["result"] for e in process_batch(b, ctx, use_langgraph=langgraph) if e["type"] == "result"]
    return b, ctx, res


@pytest.mark.parametrize("langgraph", MODES)
def test_catches_every_seeded_error_without_false_alarms(langgraph):
    _, _, res = run(langgraph=langgraph)
    m = evaluate(res)
    assert m["caught"] == m["seeded"] and m["type_correct"] == m["seeded"]
    assert m["false_positives"] == 0
    assert m["fraud_caught"] == m["fraud_seeded"] == 3
    assert m["detected_at_risk"] == pytest.approx(m["true_at_risk"])


def test_langgraph_and_plain_runner_agree():
    if not HAS_LANGGRAPH:
        pytest.skip("langgraph not installed")
    a = [(r["invoice_no"], r["decision"], r["risk_score"]) for r in run(langgraph=False)[2]]
    b = [(r["invoice_no"], r["decision"], r["risk_score"]) for r in run(langgraph=True)[2]]
    assert a == b


@pytest.mark.parametrize("seed", range(5))
def test_fraud_cases_are_escalated_with_the_right_signal(seed):
    _, _, res = run(seed=seed, n=40, err=0.5)
    for r in res:
        kind = r["truth"]["kind"]
        if kind in ("LOOKALIKE_VENDOR", "BANK_CHANGE", "AMOUNT_SPIKE"):
            assert r["decision"] == "ESCALATED"
            assert kind in {i["type"] for i in r["issues"]}
            assert r["risk_score"] >= 40


@pytest.mark.parametrize("langgraph", MODES)
def test_human_review_pauses_and_resumes(langgraph):
    _, ctx, res = run(hitl=True, langgraph=langgraph)
    pending = [r for r in res if r["status"] == "PENDING_REVIEW"]
    assert pending and all(r["decision"] == "ESCALATED" for r in pending)
    assert all(r["final_decision"] for r in res if r["status"] == "DONE")

    r = pending[0]
    done = finish_review(r, {"action": "REJECT", "note": "fake vendor", "reviewer": "test"}, ctx)
    assert done["status"] == "DONE" and done["final_decision"] == "REJECTED"
    assert done["reviewed_by"] == "test" and done["review_note"] == "fake vendor"
    assert done["trace"][-1]["agent"] == "Human Review"


@pytest.mark.parametrize("langgraph", MODES)
def test_approved_bank_change_is_learned(langgraph):
    _, ctx, res = run(hitl=True, langgraph=langgraph)
    r = next(r for r in res if r["truth"]["kind"] == "BANK_CHANGE")
    new_acct = r["extract"]["bank_account"]
    assert ctx.memory.profile(r["vendor"])["bank"] != new_acct
    finish_review(r, {"action": "APPROVE", "note": "", "reviewer": "test"}, ctx)
    assert ctx.memory.profile(r["vendor"])["bank"] == new_acct

    # the next invoice paid to that account is no longer flagged
    _, signals, _ = fraud_checks(InvoiceExtract(**r["extract"]), [], ctx.vendors, ctx.memory)
    assert "BANK_CHANGE" not in {k for k, _, _ in signals}


def test_memory_persists_to_disk(tmp_path):
    path = tmp_path / "mem.json"
    _, ctx, _ = run(memory=VendorMemory(str(path)))
    seen = {v: p["invoices"] for v, p in ctx.memory.vendors.items()}
    reloaded = VendorMemory(str(path))
    assert {v: p["invoices"] for v, p in reloaded.vendors.items()} == seen
    assert sum(seen.values()) > 0


def test_pdf_round_trip_and_duplicate_upload():
    b, ctx, _ = run(hitl=True)
    inv = next(i for i in b.invoices if i["truth"]["kind"] == "CLEAN")
    text = pdf_to_text(invoice_to_pdf(inv["text"]))
    assert text == inv["text"]
    r = process_one(text, ctx, False, "dup.pdf")
    assert "DUPLICATE" in {i["type"] for i in r["issues"]}


def test_unreadable_upload_goes_to_a_human():
    b = generate_batch(24, 0.35, 42)
    ctx = Ctx(MockLLM(), b.vendors, b.po_db, b.grn_db, b.today, hitl=True)
    r = process_one("not an invoice at all", ctx, False)
    assert r["decision"] == "ESCALATED" and r["status"] == "PENDING_REVIEW"


def test_fraud_agent_writes_a_risk_note():
    _, _, res = run()
    fraud = [r for r in res if r["truth"]["kind"] in {"LOOKALIKE_VENDOR", "BANK_CHANGE", "AMOUNT_SPIKE"}]
    assert fraud and all(r["risk_note"].startswith("Risk ") for r in fraud)
    clean = [r for r in res if not r["risk_signals"]]
    assert all(r["risk_note"] == "No fraud signals found." for r in clean)


def test_risk_note_guard_rejects_invented_numbers():
    from llm import adds_numbers
    facts = "Risk 70/100 for Acme: pay-to account 1234 differs from 9876 (+70)."
    assert not adds_numbers("Bank account changed from 9876 to 1234; call Acme first.", facts)
    assert adds_numbers("Acme has sent 5 risky invoices worth 2,50,000.", facts)
