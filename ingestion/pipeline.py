"""Step 4: one entry point for ingestion: PDF -> cleaned pages -> clause-aware chunks."""
from pathlib import Path

import pymupdf
from dataclasses import dataclass

from ingestion.chunker import Chunk, chunk_pages
from ingestion.cleaning import remove_headers_footers
from ingestion.parser import parse_pdf




def ingest_pdf(path: str | Path) -> list[Chunk]:
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
    chunks = chunk_pages(cleaned)
    return chunks


if __name__ == "__main__":
    import sys

    chunks = ingest_pdf(sys.argv[1])
    print(f"{len(chunks)} chunks")