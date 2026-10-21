"""Long-term vendor memory the agents read and the human reviewer teaches.

Each vendor profile keeps past approved totals, the bank account on file and how often its
invoices were escalated or overridden. It is a plain JSON file, so it survives app restarts.
"""
from __future__ import annotations

import json
import os
from statistics import mean

from config import APPROVED, ESCALATED, VENDOR_HISTORY_MAX_POINTS

MAX_TOTALS = 50     # keep the most recent approved totals per vendor


class VendorMemory:
    def __init__(self, path: str | None = None):
        self.path = path
        self.vendors: dict = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.vendors = json.load(f).get("vendors", {})

    def seed(self, history: dict, bank_db: dict):
        """Load the starting picture from the vendor master; never overwrites what was learned."""
        for v, acct in bank_db.items():
            self.vendors.setdefault(v, {
                "totals": list(history.get(v, [])), "bank": acct, "invoices": 0,
                "escalations": 0, "overrides": 0, "bank_changes_approved": 0,
            })
        self.save()

    def profile(self, vendor: str):
        return self.vendors.get(vendor)

    def stats(self, vendor: str):
        """(count, mean, max) of past approved totals, or None if there is no history."""
        p = self.profile(vendor)
        if not p or not p["totals"]:
            return None
        t = p["totals"]
        return len(t), mean(t), max(t)

    def history_points(self, vendor: str) -> int:
        """Risk points from the vendor's track record: share of its invoices that were escalated."""
        p = self.profile(vendor)
        if not p or p["invoices"] < 3:
            return 0
        return round(VENDOR_HISTORY_MAX_POINTS * p["escalations"] / p["invoices"])

    def record(self, vendor: str, total: float, agent_decision: str, final_decision: str):
        p = self.profile(vendor)
        if p is None:
            return                                   # unknown vendors get no profile
        p["invoices"] += 1
        p["escalations"] += agent_decision == ESCALATED
        p["overrides"] += agent_decision != final_decision
        if final_decision == APPROVED:
            p["totals"] = (p["totals"] + [round(total, 2)])[-MAX_TOTALS:]
        self.save()

    def accept_bank(self, vendor: str, account: str):
        """A reviewer confirmed the new account, so future invoices to it are not flagged."""
        p = self.profile(vendor)
        if p is not None and account:
            p["bank"] = account
            p["bank_changes_approved"] += 1
            self.save()

    def table(self) -> list[dict]:
        rows = []
        for v, p in sorted(self.vendors.items()):
            s = self.stats(v)
            rows.append({
                "Vendor": v, "Bank on file": p["bank"], "Invoices seen": p["invoices"],
                "Avg approved (INR)": round(s[1], 2) if s else None, "History size": s[0] if s else 0,
                "Escalations": p["escalations"], "Human overrides": p["overrides"],
                "Bank changes approved": p["bank_changes_approved"],
                "History risk pts": self.history_points(v),
            })
        return rows

    def save(self):
        if self.path:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"vendors": self.vendors}, f, indent=1)
