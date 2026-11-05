"""Step1: read a PDF page by page into clean lines of text."""
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# A page with fewer characters than this is treated as scanned (image-only).
MIN_CHARS = 20

@dataclass
class Page:
    doc_name: str      # file name, e.g. "gdpr_2016_679.pdf"
    page_no: int       # PDF page index, starting at 1
    page_label: str    # printed page label if the PDF defines one
    lines: list[str]   # non-empty text lines, in reading order
    is_scanned: bool   # True if the page has (almost) no extractable text

def parse_pdf(path: str | Path) -> list[Page]:
    """Return one Page object per page of the PDF."""
    path = Path(path)
    pages = []
    with pymupdf.open(path) as doc:
        for i, pdf_page in enumerate(doc):
            text = pdf_page.get_text("text", sort=True)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            pages.append(
                Page(
                    doc_name=path.name,
                    page_no=i + 1,
                    page_label=pdf_page.get_label() or str(i + 1),
                    lines=lines,
                    is_scanned=len(text.strip()) < MIN_CHARS,
                )
            )
    return pages

if __name__ == "__main__":
    import sys

    pages = parse_pdf(sys.argv[1])
    scanned = [p.page_no for p in pages if p.is_scanned]
    print(f"{pages[0].doc_name}: {len(pages)} pages, scanned pages: {scanned or 'none'}")
    page_to_show = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    sample = pages[min(page_to_show, len(pages)) - 1]
    print(f"\n--- page {sample.page_no} (label {sample.page_label}), first 8 lines ---")
    for line in sample.lines[:8]:
        print(line)

