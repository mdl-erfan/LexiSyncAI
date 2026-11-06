"""Shared test helpers: build small PDFs and Page objects so tests need no real corpus."""
from pathlib import Path

import pymupdf
import pytest

from ingestion.parser import Page

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

HEADER = "ACME Master Services Agreement"


def make_pdf(path: Path, pages: list[list[str] | None], header: str = "", footer: bool = True) -> Path:
    """Write a PDF. Each entry is a list of body lines, or None for an image-only (scanned) page.

    Text pages get an optional header at the top and "Page i of n" at the bottom.
    """
    doc = pymupdf.open()
    total = len(pages)
    for i, lines in enumerate(pages, start=1):
        page = doc.new_page()
        if lines is None:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
            pix.set_rect(pix.irect, (200, 200, 200))
            page.insert_image(pymupdf.Rect(100, 100, 300, 300), pixmap=pix)
            continue
        if header:
            page.insert_text((72, 40), header, fontsize=9)
        y = 100
        for line in lines:
            page.insert_text((72, y), line, fontsize=10)
            y += 16
        if footer:
            page.insert_text((72, 800), f"Page {i} of {total}", fontsize=9)
    doc.save(path)
    doc.close()
    return path

def make_pages(*page_lines: list[str], doc_name: str = "doc.pdf") -> list[Page]:
    """Build Page objects directly (no PDF), one per list of lines."""
    return [
        Page(doc_name=doc_name, page_no=i, page_label=str(i), lines=list(lines), is_scanned=False)
        for i, lines in enumerate(page_lines, start=1)
    ]

# A small contract used by the parser and pipeline tests.
CONTRACT_PAGES: list[list[str] | None] = [
    [
        "MASTER SERVICES AGREEMENT",
        "Section 7 Limitation of Liability",
        "7.1 General. Neither party is liable for indirect damages.",
        "7.2 Liability Cap. The total liability of the Supplier is limited to EUR 2,000,000.",
    ],
    [
        "7.3 Exceptions. The cap does not apply to wilful misconduct.",
        "Section 9 Indemnification",
        "9.1 Indemnity. The Supplier shall indemnify the Customer.",
    ],
    ["9.2 Indemnity Amount. Total indemnities are EUR 3,500,000 [***]."],
    ["9.3 Governing Law. This agreement is governed by German law."],
    None,  # scanned page
]


@pytest.fixture
def contract_pdf(tmp_path) -> Path:
    return make_pdf(tmp_path / "contract.pdf", CONTRACT_PAGES, header=HEADER)