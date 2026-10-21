"""PDF in and out: read text from an uploaded invoice PDF, and render sample invoice PDFs.

Text PDFs only (what accounting software exports). Scanned images would need OCR first.
"""
from __future__ import annotations

import io
import re


def pdf_to_text(data: bytes) -> str:
    """Extract the text layer of a PDF, one line per printed line."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    lines = (re.sub(r"[ \t]+", " ", l).strip() for l in text.splitlines())
    return "\n".join(l for l in lines if l)


def invoice_to_pdf(text: str, title: str = "") -> bytes:
    """Render invoice text as a simple one-page PDF (used to make demo uploads)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setTitle(title or "Invoice")
    width, height = A4
    y = height - 60
    for i, line in enumerate(text.splitlines()):
        c.setFont("Helvetica-Bold" if i == 0 else "Courier", 14 if i == 0 else 9)
        c.drawString(50, y, line)
        y -= 22 if i == 0 else 14
    c.showPage()
    c.save()
    return buf.getvalue()
