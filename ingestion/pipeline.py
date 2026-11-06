"""Step 4: one entry point for ingestion: PDF -> cleaned pages -> clause-aware chunks."""
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from ingestion.chunker import Chunk, chunk_pages
from ingestion.cleaning import remove_headers_footers
from ingestion.parser import parse_pdf


@dataclass
class IngestedDocument:
    doc_name: str
    n_pages: int
    scanned_pages: list[int]     # page numbers with no extractable text (need OCR)
    removed_lines: int           # header/footer/page-number lines dropped
    chunks: list[Chunk]

    @property
    def sections(self) -> list[str]:
        """Distinct section labels in document order (the detected structure)."""
        seen: dict[str, None] = {}
        for c in self.chunks:
            if c.section:
                seen.setdefault(c.section)
        return list(seen)

    def summary(self) -> dict:
        """Small, JSON-friendly description of the document, for the UI and logs."""
        return {
            "doc_name": self.doc_name,
            "pages": self.n_pages,
            "scanned_pages": self.scanned_pages,
            "removed_lines": self.removed_lines,
            "chunks": len(self.chunks),
            "labelled_chunks": sum(1 for c in self.chunks if c.section),
            "sections": len(self.sections),
        }


def ingest_pdf(path: str | Path) -> IngestedDocument:
    """Parse, clean and chunk one PDF."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    try:
        pages = parse_pdf(path)
    except pymupdf.FileDataError as exc:
        raise ValueError(f"{path.name} is not a readable PDF") from exc
    if not pages:
        raise ValueError(f"{path.name} has no pages")

    cleaned = remove_headers_footers(pages)
    removed = sum(len(before.lines) - len(after.lines) for before, after in zip(pages, cleaned))
    return IngestedDocument(
        doc_name=path.name,
        n_pages=len(pages),
        scanned_pages=[p.page_no for p in pages if p.is_scanned],
        removed_lines=removed,
        chunks=chunk_pages(cleaned),
    )


def ingest_many(paths: list[str | Path]) -> list[IngestedDocument]:
    """Ingest several PDFs (one case). File names must be unique: chunk ids use them."""
    names = [Path(p).name for p in paths]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"Duplicate file names would give clashing chunk ids: {duplicates}")
    return [ingest_pdf(p) for p in paths]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("usage: python -m ingestion.pipeline FILE.pdf [FILE2.pdf ...]")
    for doc in ingest_many(sys.argv[1:]):
        s = doc.summary()
        print(f"{s['doc_name']}: {s['pages']} pages, scanned pages: {s['scanned_pages'] or 'none'}")
        print(f"  removed {s['removed_lines']} header/footer lines")
        print(f"  {s['chunks']} chunks ({s['labelled_chunks']} labelled, {s['sections']} distinct sections)")