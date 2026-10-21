"""Synthetic data: vendors, POs, goods receipts (GRN) and invoice text with seeded errors.

Every invoice carries a hidden `truth` record, so the PoC can report real accuracy numbers.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from config import TAX_RATE

APPROVED_VENDORS = [
    "Acme Supplies Pvt Ltd", "Bharat Office Mart", "Sahyadri Industrial Tools",
    "Konkan Packaging Co", "Deccan Electronics", "Pune Stationers Ltd",
    "Western Ghats Logistics", "Nashik Hardware Hub",
]
UNKNOWN_VENDORS = ["Zenith Trading Co", "Quickfix Enterprises", "Orbit Supplies LLP"]

CATALOG = {  # sku: (description, list price in INR)
    "SKU-101": ("Laptop Stand", 1250.00),
    "SKU-102": ("Wireless Mouse", 799.00),
    "SKU-103": ("A4 Paper Ream (500)", 285.00),
    "SKU-104": ("Safety Helmet", 640.00),
    "SKU-105": ("Packing Tape Carton", 1980.00),
    "SKU-106": ("LED Panel Light", 1450.00),
    "SKU-107": ("Cordless Drill", 4200.00),
    "SKU-108": ("Network Switch 24p", 3650.00),
    "SKU-109": ("Pallet Wrap Roll", 520.00),
    "SKU-110": ("Ergonomic Chair", 6800.00),
}

ERROR_CYCLE = [
    "PRICE_MISMATCH", "QTY_EXCESS", "DUPLICATE", "UNKNOWN_VENDOR",
    "TAX_ERROR", "LOOKALIKE_VENDOR", "BANK_CHANGE", "AMOUNT_SPIKE",
    "PRICE_MISMATCH", "QTY_EXCESS", "DUPLICATE",
]
HISTORY_LEN = 12        # past approved invoices per vendor, used to seed the agents' memory


@dataclass
class Batch:
    invoices: list      # [{"idx", "text", "truth"}]
    vendors: list       # approved vendor master
    po_db: dict         # po_no -> {"vendor", "lines": {sku: {"qty", "price"}}}
    grn_db: dict        # po_no -> {sku: received_qty}
    today: date
    bank_db: dict = field(default_factory=dict)   # vendor -> bank account on file
    history: dict = field(default_factory=dict)   # vendor -> past approved invoice totals


def _fmt(x: float, comma: bool) -> str:
    return f"{x:,.2f}" if comma else f"{x:.2f}"


def lookalike(name: str, rng) -> str:
    """Misspell an approved vendor's name the way invoice fraudsters do (one letter off)."""
    words = name.split()
    w = max(range(len(words)), key=lambda i: len(words[i]))
    word = words[w]
    doubled = [i for i in range(1, len(word)) if word[i] == word[i - 1]]
    if doubled:
        i = rng.choice(doubled)
        word = word[:i] + word[i + 1:]                      # Supplies -> Suplies
    else:
        i = rng.randrange(1, len(word) - 2)
        word = word[:i] + word[i + 1] + word[i] + word[i + 2:]   # swap two letters
    return " ".join(words[:w] + [word] + words[w + 1:])


def _account(rng) -> str:
    return "50100" + "".join(str(rng.randint(0, 9)) for _ in range(9))


def _random_lines(rng):
    skus = rng.sample(list(CATALOG), rng.randint(1, 3))
    return {s: {"qty": rng.randint(5, 20), "price": round(CATALOG[s][1] * rng.uniform(0.9, 1.1), 2)}
            for s in skus}


def _total(po_lines) -> float:
    return round(sum(l["qty"] * l["price"] for l in po_lines.values()) * (1 + TAX_RATE), 2)


def render_invoice(vendor, inv_no, po_no, inv_date, due_date, lines, tax_rate, comma, bank=None):
    """Return (invoice_text, total). `comma=True` uses thousand separators (1,250.00)."""
    rows, subtotal = [], 0.0
    for sku, qty, price in lines:
        amount = round(qty * price, 2)
        subtotal += amount
        rows.append(
            f"{sku} | {CATALOG[sku][0]} | Qty {qty} | "
            f"Unit {_fmt(price, comma)} | Amount {_fmt(amount, comma)}"
        )
    subtotal = round(subtotal, 2)
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)
    text = "\n".join([
        "TAX INVOICE",
        f"Vendor: {vendor}",
        *([f"Bank A/C: {bank}"] if bank else []),
        f"Invoice No: {inv_no}",
        f"PO Number: {po_no}",
        f"Invoice Date: {inv_date}",
        f"Due Date: {due_date}",
        "Items:",
        *rows,
        f"Subtotal: {_fmt(subtotal, comma)}",
        f"Tax ({round(tax_rate * 100)}%): {_fmt(tax, comma)}",
        f"Total: {_fmt(total, comma)}",
    ])
    return text, total


def _build_invoice(i, kind, rng, today, po_db, grn_db, bank_db, history):
    vendor = rng.choice(UNKNOWN_VENDORS if kind == "UNKNOWN_VENDOR" else APPROVED_VENDORS)
    po_vendor, bank = vendor, bank_db.get(vendor)
    inv_no, po_no = f"INV-{7000 + i}", f"PO-{5000 + i}"
    po_lines = _random_lines(rng)
    skus = list(po_lines)
    if kind == "AMOUNT_SPIKE":                    # same vendor, ~7-9x its biggest bill so far
        factor = max(2, round(rng.uniform(7, 9) * max(history[vendor]) / _total(po_lines)))
        for l in po_lines.values():
            l["qty"] *= factor
    received = {s: l["qty"] for s, l in po_lines.items()}
    inv_lines = [[s, l["qty"], l["price"]] for s, l in po_lines.items()]
    tax_rate, expected = TAX_RATE, []

    if kind == "PRICE_MISMATCH":
        j = rng.randrange(len(inv_lines))
        inv_lines[j][2] = round(inv_lines[j][2] * (1 + rng.uniform(0.08, 0.25)), 2)
        expected = ["PRICE_MISMATCH"]
    elif kind == "QTY_EXCESS":
        s = rng.choice(skus)
        received[s] = po_lines[s]["qty"] - rng.randint(1, 3)   # billed more than received
        expected = ["QTY_EXCESS"]
    elif kind == "TAX_ERROR":
        tax_rate = rng.choice([0.22, 0.28])
        expected = ["TAX_ERROR"]
    elif kind == "UNKNOWN_VENDOR":
        po_no = f"PO-{9100 + rng.randint(0, 800)}"              # PO does not exist
        bank = _account(rng)
        expected = ["NO_PO", "UNKNOWN_VENDOR"]
    elif kind == "LOOKALIKE_VENDOR":                            # fraudster copies a real PO
        vendor, bank = lookalike(po_vendor, rng), _account(rng)
        expected = ["UNKNOWN_VENDOR", "LOOKALIKE_VENDOR"]
    elif kind == "BANK_CHANGE":                                 # "please pay our new account"
        bank = _account(rng)
        expected = ["BANK_CHANGE"]
    elif kind == "AMOUNT_SPIKE":
        expected = ["AMOUNT_SPIKE"]

    if kind != "UNKNOWN_VENDOR":
        po_db[po_no] = {"vendor": po_vendor, "lines": po_lines}
        grn_db[po_no] = received

    due = today + timedelta(days=rng.randint(-3, 45))
    text, total = render_invoice(
        vendor, inv_no, po_no, (due - timedelta(days=30)).isoformat(), due.isoformat(),
        [tuple(l) for l in inv_lines], tax_rate, comma=rng.random() < 0.4, bank=bank,
    )
    correct = sum(po_lines[s]["price"] * received[s] for s in skus) * (1 + TAX_RATE)
    if kind in ("CLEAN", "AMOUNT_SPIKE"):          # a spike is suspicious, not an overpayment
        at_risk = 0.0
    elif kind in ("UNKNOWN_VENDOR", "LOOKALIKE_VENDOR", "BANK_CHANGE"):
        at_risk = total                            # the whole payment could go to a fraudster
    else:
        at_risk = max(round(total - correct, 2), 0.0)
    return {"text": text, "truth": {
        "kind": kind, "expected_issues": expected, "true_at_risk": at_risk,
        "invoice_no": inv_no, "vendor": vendor, "total": total,
    }}


def generate_batch(n=24, error_pct=0.35, seed=42, today=None) -> Batch:
    rng = random.Random(seed)
    today = today or date.today()
    n = max(n, 5)
    error_pct = min(max(error_pct, 0.0), 0.7)
    n_err = round(n * error_pct)
    error_kinds = [ERROR_CYCLE[i % len(ERROR_CYCLE)] for i in range(n_err)]
    n_dup = error_kinds.count("DUPLICATE")
    base_kinds = [k for k in error_kinds if k != "DUPLICATE"]
    base_kinds += ["CLEAN"] * (n - n_dup - len(base_kinds))
    rng.shuffle(base_kinds)

    bank_db = {v: _account(rng) for v in APPROVED_VENDORS}
    history = {v: [_total(_random_lines(rng)) for _ in range(HISTORY_LEN)] for v in APPROVED_VENDORS}
    po_db, grn_db = {}, {}
    records = [_build_invoice(i, k, rng, today, po_db, grn_db, bank_db, history)
               for i, k in enumerate(base_kinds)]

    clean = [r for r in records if r["truth"]["kind"] == "CLEAN"]
    for _ in range(n_dup):                       # re-submitted copies of earlier clean invoices
        if not clean:
            break
        src = rng.choice(clean)
        dup = {"text": src["text"], "truth": {
            **src["truth"], "kind": "DUPLICATE",
            "expected_issues": ["DUPLICATE"], "true_at_risk": src["truth"]["total"],
        }}
        pos = next(i for i, r in enumerate(records) if r is src)
        records.insert(rng.randint(pos + 1, len(records)), dup)

    for idx, r in enumerate(records):
        r["idx"] = idx                            # arrival order
    return Batch(records, list(APPROVED_VENDORS), po_db, grn_db, today, bank_db, history)
