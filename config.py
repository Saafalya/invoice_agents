"""Central settings for the AP three-way-match PoC."""

TAX_RATE = 0.18               # GST rate the company expects on every invoice
PRICE_TOLERANCE = 0.01        # INR; allowed unit-price difference vs PO
VARIANCE_TOLERANCE = 1.00     # INR; allowed total difference vs expected
ESCALATE_AT_RISK = 10_000.0   # INR; mismatches above this go to a human
MAX_EXTRACT_ATTEMPTS = 2      # Extraction Agent retries once after Critic rejection
URGENT_DAYS = 7               # invoices due within this many days are fast-tracked

FRAUD_ESCALATE_SCORE = 40     # Fraud Agent risk score (0-100) at which a human must look
LOOKALIKE_SIMILARITY = 0.85   # vendor-name similarity that counts as a lookalike of an approved vendor
SPIKE_MIN_HISTORY = 5         # past invoices needed before the Fraud Agent judges amount spikes
SPIKE_FACTOR = 3.0            # spike = total above this multiple of the vendor's largest past invoice

APPROVED, HOLD, ESCALATED, REJECTED = "APPROVED", "HOLD", "ESCALATED", "REJECTED"

# Issue types that always need a human, whatever the amount (fraud / control risks)
ALWAYS_ESCALATE = {
    "DUPLICATE", "UNKNOWN_VENDOR", "NO_PO", "VENDOR_MISMATCH", "UNEXPLAINED_VARIANCE",
    "LOOKALIKE_VENDOR", "BANK_CHANGE",
}

# Points each signal adds to the Fraud Agent's risk score (capped at 100)
RISK_WEIGHTS = {
    "BANK_CHANGE": 70, "LOOKALIKE_VENDOR": 60, "AMOUNT_SPIKE": 40, "UNKNOWN_VENDOR": 30,
    "DUPLICATE": 30, "ROUND_AMOUNT": 15,
}
VENDOR_HISTORY_MAX_POINTS = 20  # extra points for a vendor whose past invoices were often escalated
