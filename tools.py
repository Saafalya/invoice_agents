"""Deterministic tools the agents call: parsing, validation, three-way matching, decision rules."""
from __future__ import annotations

import re
from datetime import date
from difflib import SequenceMatcher

from config import (ALWAYS_ESCALATE, APPROVED, ESCALATED, ESCALATE_AT_RISK, FRAUD_ESCALATE_SCORE,
                    HOLD, LOOKALIKE_SIMILARITY, PRICE_TOLERANCE, RISK_WEIGHTS, SPIKE_FACTOR,
                    SPIKE_MIN_HISTORY, TAX_RATE, VARIANCE_TOLERANCE)
from schemas import InvoiceExtract, LineItem


# --------------------------------------------------------------------------- parsing
def parse_invoice(text: str, robust: bool = True) -> InvoiceExtract:
    """Regex invoice parser.

    robust=False is the 'fast' parser: it does not understand thousand separators, so
    '1,250.00' is read as '1'. That is deliberate: it gives the Critic something real to catch.
    """
    num = r"[\d,]+(?:\.\d+)?" if robust else r"\d+(?:\.\d+)?"

    def to_f(s: str) -> float:
        return float(s.replace(",", ""))

    def grab(pattern: str):
        m = re.search(pattern, text, re.M)
        if not m:
            raise ValueError(f"field not found: {pattern}")
        return m

    lines = [
        LineItem(sku=m.group(1), description=m.group(2).strip(), qty=int(m.group(3)),
                 unit_price=to_f(m.group(4)), amount=to_f(m.group(5)))
        for m in re.finditer(
            rf"^(\S+) \| (.+?) \| Qty (\d+) \| Unit ({num}).*?Amount ({num})", text, re.M)
    ]
    if not lines:
        raise ValueError("no line items found")
    tax_m = grab(rf"^Tax \((\d+)%\):\s*({num})")
    bank_m = re.search(r"^Bank A/C:\s*(\S+)", text, re.M)
    return InvoiceExtract(
        invoice_no=grab(r"^Invoice No:\s*(\S+)").group(1),
        vendor=grab(r"^Vendor:\s*(.+)$").group(1).strip(),
        po_number=grab(r"^PO Number:\s*(\S+)").group(1),
        invoice_date=grab(r"^Invoice Date:\s*(\S+)").group(1),
        due_date=grab(r"^Due Date:\s*(\S+)").group(1),
        lines=lines,
        subtotal=to_f(grab(rf"^Subtotal:\s*({num})").group(1)),
        tax=to_f(tax_m.group(2)),
        tax_rate=int(tax_m.group(1)) / 100,
        total=to_f(grab(rf"^Total:\s*({num})").group(1)),
        bank_account=bank_m.group(1) if bank_m else None,
    )


def peek_header(text: str, today: date) -> dict:
    """Cheap look at due date and total, used by the Planner to prioritise."""
    due = re.search(r"^Due Date:\s*(\S+)", text, re.M)
    tot = re.search(r"^Total:\s*([\d,]+(?:\.\d+)?)", text, re.M)
    days_left = (date.fromisoformat(due.group(1)) - today).days if due else 999
    return {"days_left": days_left, "total_hint": float(tot.group(1).replace(",", "")) if tot else 0.0}


# --------------------------------------------------------------------------- validation
def validate_extraction(ext: InvoiceExtract) -> list[str]:
    """Arithmetic and format checks the Critic uses to accept or reject an extraction."""
    problems = []
    for l in ext.lines:
        if abs(l.qty * l.unit_price - l.amount) > 0.01:
            problems.append(f"{l.sku}: qty x unit ({l.qty * l.unit_price:,.2f}) != amount ({l.amount:,.2f})")
    if abs(sum(l.amount for l in ext.lines) - ext.subtotal) > 0.01:
        problems.append(f"sum of lines ({sum(l.amount for l in ext.lines):,.2f}) != subtotal ({ext.subtotal:,.2f})")
    if abs(ext.subtotal * ext.tax_rate - ext.tax) > 0.05:
        problems.append(f"tax ({ext.tax:,.2f}) inconsistent with stated rate {ext.tax_rate:.0%}")
    if abs(ext.subtotal + ext.tax - ext.total) > 0.01:
        problems.append(f"subtotal + tax ({ext.subtotal + ext.tax:,.2f}) != total ({ext.total:,.2f})")
    for label, value in (("invoice_date", ext.invoice_date), ("due_date", ext.due_date)):
        try:
            date.fromisoformat(value)
        except ValueError:
            problems.append(f"{label} '{value}' is not a valid date")
    return problems


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def grounding_problems(ext: InvoiceExtract, raw: str) -> list[str]:
    """Every key text field must appear in the document, so an LLM cannot invent values.

    Dates are skipped: an LLM may rightly rewrite 31/10/2026 as 2026-10-31.
    """
    doc = _norm(raw)
    fields = {"invoice_no": ext.invoice_no, "vendor": ext.vendor, "po_number": ext.po_number,
              "bank_account": ext.bank_account}
    return [f"{name} '{value}' does not appear in the document"
            for name, value in fields.items() if value and _norm(str(value)) not in doc]


# --------------------------------------------------------------------------- matching
def three_way_match(inv: InvoiceExtract, vendors, po_db, grn_db, seen: set):
    """Compare invoice vs PO vs GRN. Returns (issues, expected_total, amount_at_risk)."""
    issues = []

    def add(kind, detail):
        issues.append({"type": kind, "detail": detail})

    vendor_ok = inv.vendor in vendors
    if not vendor_ok:
        add("UNKNOWN_VENDOR", f"'{inv.vendor}' is not in the approved vendor master")

    key = (inv.vendor.lower(), inv.invoice_no.upper())
    if key in seen:
        add("DUPLICATE", f"invoice {inv.invoice_no} from {inv.vendor} was already processed in this batch")
    else:
        seen.add(key)

    expected_sub, po = 0.0, po_db.get(inv.po_number)
    if po is None:
        add("NO_PO", f"purchase order {inv.po_number} does not exist")
    else:
        if vendor_ok and po["vendor"] != inv.vendor:
            add("VENDOR_MISMATCH", f"{inv.po_number} was issued to {po['vendor']}, not {inv.vendor}")
        grn = grn_db.get(inv.po_number, {})
        for l in inv.lines:
            pl = po["lines"].get(l.sku)
            if pl is None:
                add("UNORDERED_ITEM", f"{l.sku} is not on {inv.po_number}")
                continue
            received = grn.get(l.sku, 0)
            expected_sub += pl["price"] * received
            if abs(l.unit_price - pl["price"]) > PRICE_TOLERANCE:
                pct = (l.unit_price / pl["price"] - 1) * 100
                add("PRICE_MISMATCH",
                    f"{l.sku}: billed {l.unit_price:,.2f} vs PO price {pl['price']:,.2f} ({pct:+.1f}%)")
            if l.qty > received:
                add("QTY_EXCESS", f"{l.sku}: billed {l.qty} but GRN shows only {received} received")

    if abs(inv.tax_rate - TAX_RATE) > 1e-6:
        add("TAX_ERROR", f"tax charged at {inv.tax_rate:.0%}, expected {TAX_RATE:.0%}")

    expected_total = round(expected_sub * (1 + TAX_RATE), 2)
    if {i["type"] for i in issues} & {"DUPLICATE", "UNKNOWN_VENDOR", "NO_PO"}:
        at_risk = inv.total                                  # whole payment is questionable
    else:
        at_risk = max(round(inv.total - expected_total, 2), 0.0)
    return issues, expected_total, at_risk


# --------------------------------------------------------------------------- fraud signals
def closest_vendor(name: str, vendors) -> tuple[str, float]:
    """Most similar approved vendor name and its similarity (0-1)."""
    scored = [(v, SequenceMatcher(None, name.lower(), v.lower()).ratio()) for v in vendors]
    return max(scored, key=lambda x: x[1]) if scored else ("", 0.0)


def fraud_checks(inv: InvoiceExtract, issues: list, vendors, memory):
    """Signals a three-way match cannot see. Returns (new_issues, signals, risk_score).

    `signals` lists every point added to the score, so the reviewer sees why it is high.
    """
    new, signals = [], []

    def add(kind, detail):
        new.append({"type": kind, "detail": detail})

    if inv.vendor not in vendors:
        match, sim = closest_vendor(inv.vendor, vendors)
        if sim >= LOOKALIKE_SIMILARITY:
            add("LOOKALIKE_VENDOR", f"'{inv.vendor}' is {sim:.0%} similar to approved vendor "
                                    f"'{match}': possible impersonation")
    profile = memory.profile(inv.vendor)
    if profile and inv.bank_account and inv.bank_account != profile["bank"]:
        add("BANK_CHANGE", f"pay-to account {inv.bank_account} differs from account on file "
                           f"{profile['bank']}")
    st = memory.stats(inv.vendor)
    if st and st[0] >= SPIKE_MIN_HISTORY and inv.total > SPIKE_FACTOR * st[2]:
        n, _avg, biggest = st
        add("AMOUNT_SPIKE", f"total {inv.total:,.2f} is {inv.total / biggest:.1f}x the largest of this "
                            f"vendor's {n} past approved invoices ({biggest:,.2f})")
    if inv.total >= 1000 and inv.total % 1000 == 0:
        signals.append(("ROUND_AMOUNT", RISK_WEIGHTS["ROUND_AMOUNT"], "suspiciously round total"))

    for i in issues + new:
        if i["type"] in RISK_WEIGHTS:
            signals.append((i["type"], RISK_WEIGHTS[i["type"]], i["detail"]))
    hist = memory.history_points(inv.vendor)
    if hist:
        signals.append(("VENDOR_HISTORY", hist, "vendor's past invoices were often escalated"))
    score = min(100, sum(pts for _, pts, _ in signals))
    return new, signals, score


def decide(issues: list, at_risk: float, extract_ok: bool, risk_score: int = 0):
    """Business rules for the Decision Agent. Returns (decision, rule_that_fired)."""
    if not extract_ok:
        return ESCALATED, "extraction could not be validated"
    types = {i["type"] for i in issues}
    if not types:
        return APPROVED, "no issues"
    if types & ALWAYS_ESCALATE:
        return ESCALATED, "fraud / control risk: " + ", ".join(sorted(types & ALWAYS_ESCALATE))
    if risk_score >= FRAUD_ESCALATE_SCORE:
        return ESCALATED, f"fraud risk score {risk_score} >= {FRAUD_ESCALATE_SCORE}"
    if at_risk >= ESCALATE_AT_RISK:
        return ESCALATED, f"amount at risk >= {ESCALATE_AT_RISK:,.0f}"
    return HOLD, "low-value mismatch: return to vendor for correction"
