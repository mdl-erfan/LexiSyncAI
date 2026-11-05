"""Step 2: remove repeated headers, footers and page numbers from parsed pages."""
import re
from collections import Counter
from dataclasses import replace

from ingestion.parser import Page

# How many lines at the top and bottom of each page count as the "edge".
EDGE_LINES = 3
# A line is a header/footer if it repeats on at least this share of pages...
MIN_SHARE = 0.3
# ...and on at least this many pages (so short documents aren't over-cleaned).
MIN_PAGES = 3

PAGE_NUMBER = re.compile(r"(page\s*)?[-–]?\s*\d+\s*[-–]?(\s*(of|/)\s*\d+)?", re.IGNORECASE)


def _key(line: str) -> str:
    """Normalize a line so the same header on different pages gives the same key."""
    line = re.sub(r"\d+", "#", line.lower())   # "L 119/82" and "L 119/83" -> "l #/#"
    return " ".join(sorted(line.split()))       # same words in a different order -> same key


def _is_page_number(line: str) -> bool:
    """True for lines like "12", "Page 3", "- 4 -" or "5 of 20"."""
    return PAGE_NUMBER.fullmatch(line.strip()) is not None

def find_repeated(pages: list[Page]) -> set[str]:
    """Return the keys of lines that repeat at the page edges across many pages."""
    text_pages = [p for p in pages if not p.is_scanned]
    counts = Counter()
    for p in text_pages:
        edge = p.lines[:EDGE_LINES] + p.lines[-EDGE_LINES:]
        counts.update({_key(line) for line in edge})   # a set: count each page once
    needed = max(MIN_PAGES, MIN_SHARE * len(text_pages))
    return {key for key, n in counts.items() if n >= needed}


def remove_headers_footers(pages: list[Page]) -> list[Page]:
    """Return new pages with repeated edge lines and page numbers removed."""
    repeated = find_repeated(pages)
    cleaned = []
    for p in pages:
        n = len(p.lines)
        keep = []
        for idx, line in enumerate(p.lines):
            at_edge = idx < EDGE_LINES or idx >= n - EDGE_LINES
            if at_edge and (_key(line) in repeated or _is_page_number(line)):
                continue
            keep.append(line)
        cleaned.append(replace(p, lines=keep))
    return cleaned


if __name__ == "__main__":
    import sys

    from ingestion.parser import parse_pdf

    pages = parse_pdf(sys.argv[1])
    cleaned = remove_headers_footers(pages)
    removed = sum(len(a.lines) - len(b.lines) for a, b in zip(pages, cleaned))
    print(f"Removed {removed} header/footer lines from {len(pages)} pages")

    page_to_show = max(1, int(sys.argv[2])) if len(sys.argv) > 2 else 1
    before = pages[min(page_to_show, len(pages)) - 1]
    after = cleaned[min(page_to_show, len(cleaned)) - 1]
    print(f"\n--- page {after.page_no}: first 3 lines BEFORE ---")
    print("\n".join(before.lines[:3]))
    print(f"\n--- page {after.page_no}: first 3 lines AFTER ---")
    print("\n".join(after.lines[:3]))