"""LLM backends. MockLLM needs nothing; OllamaLLM talks to a local Ollama server."""
from __future__ import annotations

import json
import re

from schemas import InvoiceExtract


def template_explanation(p: dict) -> str:
    if p["decision"] == "APPROVED":
        return ("All checks passed (vendor, PO, GRN quantities, prices, tax, duplicate). "
                "Safe to schedule for payment.")
    reasons = "; ".join(i["detail"] for i in p["issues"]) or "extraction could not be validated"
    who = "Escalated to a finance reviewer" if p["decision"] == "ESCALATED" else "Held for vendor correction"
    return f"{who}: {reasons}. Amount at risk: INR {p['at_risk']:,.2f}."


def template_risk_note(p: dict) -> str:
    """Fraud note without an LLM: list the signals behind the score."""
    if not p["signals"]:
        return "No fraud signals found."
    parts = "; ".join(f"{s['detail']} (+{s['points']})" for s in p["signals"])
    return f"Risk {p['risk_score']}/100 for {p['vendor']}: {parts}."


_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set:
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}


def adds_numbers(out: str, facts: str) -> bool:
    """True if the LLM wrote a number that is not in the facts (a likely hallucination)."""
    return bool(_numbers(out) - _numbers(facts))


class MockLLM:
    name = "mock (rule-based, no LLM)"
    can_extract = False

    def extract(self, text: str):
        return None

    def explain(self, payload: dict) -> str:
        return template_explanation(payload)

    def explain_risk(self, payload: dict) -> str:
        return template_risk_note(payload)

    def review_extraction(self, text: str, ext) -> list:
        return []                              # no LLM: the Critic relies on its rule checks


class OllamaLLM:
    can_extract = True

    def __init__(self, model: str = "llama3.2"):
        from langchain_ollama import ChatOllama
        self.name = f"ollama:{model}"
        self.chat = ChatOllama(model=model, temperature=0)
        self.json_chat = ChatOllama(model=model, temperature=0, format="json")

    def ping(self):
        self.chat.invoke("Reply with the single word: ok")

    def extract(self, text: str):
        prompt = (
            "Extract this invoice into JSON with exactly these keys: invoice_no, vendor, "
            "po_number, invoice_date, due_date, subtotal, tax, tax_rate (percent, e.g. 18), "
            "total, bank_account (pay-to account number, or null), "
            "lines (list of {sku, description, qty, unit_price, amount}). "
            "Use plain numbers without thousands separators. Return JSON only.\n\nINVOICE:\n" + text
        )
        try:
            data = json.loads(self.json_chat.invoke(prompt).content)
            data["tax_rate"] = float(data.get("tax_rate", 0)) / 100
            return InvoiceExtract(**data)
        except Exception:
            return None            # the Extraction Agent falls back to its parser

    def explain(self, payload: dict) -> str:
        facts = template_explanation(payload)
        prompt = (
            "You are an accounts-payable assistant. Rewrite the finding below as 1-2 clear "
            "sentences for a finance reviewer. Do not add facts or numbers.\n\n" + facts
        )
        try:
            out = self.chat.invoke(prompt).content.strip()
            return re.sub(r"\s+", " ", out) or facts
        except Exception:
            return facts

    def explain_risk(self, payload: dict) -> str:
        facts = template_risk_note(payload)
        if not payload["signals"]:
            return facts                       # nothing to explain, skip the LLM call
        prompt = (
            "You are a fraud analyst in accounts payable. In 2 short sentences, tell a finance "
            "reviewer why this invoice looks risky and what to verify before paying (for example, "
            "call the vendor on a known number). Use only the facts below. Do not add new numbers.\n\n"
            + facts
        )
        try:
            out = re.sub(r"\s+", " ", self.chat.invoke(prompt).content.strip())
            if not out or adds_numbers(out, facts):
                return facts                   # guard: no invented figures reach the reviewer
            return out
        except Exception:
            return facts

    def review_extraction(self, text: str, ext) -> list:
        """Second opinion: does the extracted data match the invoice? Returns short complaints."""
        prompt = (
            "You check invoice data extraction. Compare the EXTRACTED JSON with the INVOICE text. "
            "Look for values in the wrong field (for example invoice number and PO number swapped), "
            "wrong vendor, wrong bank account, or missing line items. Ignore number formatting. "
            'Reply with JSON only: {"ok": true} or {"ok": false, "problems": '
            '[{"field": "<json key>", "issue": "<short reason>"}]}.\n\n'
            f"INVOICE:\n{text}\n\nEXTRACTED:\n{ext.model_dump_json()}"
        )
        try:
            return parse_review(self.json_chat.invoke(prompt).content)
        except Exception:
            return []                          # model down or slow: rule checks still apply


FIELDS = set(InvoiceExtract.model_fields)


def parse_review(content: str) -> list:
    """Keep only well-formed complaints about real fields; anything else is ignored."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, dict) or data.get("ok") is True:
        return []
    out = []
    for p in data.get("problems") or []:
        if isinstance(p, dict) and p.get("field") in FIELDS and str(p.get("issue", "")).strip():
            out.append(f"{p['field']}: {str(p['issue']).strip()[:160]}")
    return out[:3]


def get_llm(engine: str, model: str = "llama3.2"):
    """Return (llm, warning_or_None). Falls back to the mock if Ollama is unavailable."""
    if engine != "ollama":
        return MockLLM(), None
    try:
        llm = OllamaLLM(model)
        llm.ping()
        return llm, None
    except Exception as e:  # noqa: BLE001
        return MockLLM(), f"Could not reach Ollama model '{model}' ({type(e).__name__}). Using the mock LLM."
