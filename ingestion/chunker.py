"""Step 3: split cleaned pages into clause-aware chunks with page and section metadata."""
import re
from dataclasses import dataclass, field

from ingestion.parser import Page

# Chunks longer than this many characters are split at line boundaries.
MAX_CHARS = 1500

# Headings: "Article 83", "Section 4", "§ 9 Liability and Indemnification", "CHAPTER VIII"
HEADING = re.compile(
    r"(?P<kind>article|section|§|chapter)\s*(?P<num>\d+[a-z]?|[ivxlc]+)"
    r"(?:\s*[-–:.]?\s+(?P<title>[A-Za-z].{0,80}))?",
    re.IGNORECASE,
)
# Numbered clauses: "9.1   Exclusion of Liability for Sunrise. ..."
CLAUSE = re.compile(r"(?P<num>\d+(?:\.\d+)+)\.?\s+(?P<rest>\S.*)")
# Numbered paragraphs inside a heading: "4.   Infringements of the following ..."
PARAGRAPH = re.compile(r"(?P<num>\d{1,3})\.\s+\S")
# Numbered items before any heading (e.g. recitals): "(12) Whereas ..."
ITEM = re.compile(r"\((?P<num>\d{1,3})\)\s+\S")
# Table-of-contents lines: a big gap or dot leaders, then a page number at the end
TOC_LINE = re.compile(r".*(\.{3,}|\s{3,})\d+\s*")


@dataclass
class Chunk:
    doc_name: str
    section: str                 # "Article 83(4)", "9.1", "(12)", or "" if unknown
    heading: str                 # "Article 83 General conditions for imposing administrative fines"
    lines: list[str] = field(default_factory=list)
    line_pages: list[int] = field(default_factory=list)   # page number of each line
    heading_only: bool = False   # True while the chunk holds just a heading
    chunk_id: str = ""

    @property
    def text(self) -> str:
        return " ".join(self.lines)

    @property
    def page_start(self) -> int:
        return self.line_pages[0]

    @property
    def page_end(self) -> int:
        return self.line_pages[-1]


def _heading_label(line: str) -> str | None:
    """Return a label like "Article 83" if the line is a heading, else None."""
    if len(line) > 100 or TOC_LINE.fullmatch(line):
        return None
    m = HEADING.fullmatch(line)
    if not m or line[-1] in ".,;:":    # "Article 263 TFEU." ends a wrapped sentence
        return None
    kind = m.group("kind")
    kind = "§" if kind == "§" else kind.capitalize()
    return f"{kind} {m.group('num')}"


def _clause_title(number: str, rest: str) -> str:
    """'9.1', 'Exclusion of Liability for Sunrise. Unless ...' -> '9.1 Exclusion of Liability for Sunrise'"""
    title = rest.split(". ")[0]
    return f"{number} {title[:80]}"


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    """Turn cleaned pages into clause-aware chunks."""
    doc_name = pages[0].doc_name if pages else ""
    chunks: list[Chunk] = []
    current: Chunk | None = None
    parent = ""          # label that numbered paragraphs belong to, e.g. "Article 83"
    top_heading = ""     # last big heading, e.g. "§ 9 Liability and Indemnification"
    heading = ""         # heading text for the chunk being built
    expect_title = False

    def new_chunk(section: str) -> None:
        nonlocal current
        if current and current.heading_only:
            # Merge a heading-only chunk ("Article 83" + title) into the next one.
            current.section, current.heading, current.heading_only = section, heading, False
            return
        if current and current.lines:
            chunks.append(current)
        current = Chunk(doc_name, section, heading)

    for page in pages:
        for line in page.lines:
            label = _heading_label(line)
            clause = None if label or TOC_LINE.fullmatch(line) else CLAUSE.fullmatch(line)
            para = PARAGRAPH.match(line)
            item = ITEM.match(line)
            is_title = expect_title and not (label or clause or para or item)
            expect_title = False

            if label:
                parent, heading, top_heading = label, line, line
                new_chunk(label)
                current.heading_only = True
                expect_title = True
            elif is_title:
                heading = top_heading = f"{heading} {line}"   # "Article 83" + its title line
                current.heading = heading
            elif clause:
                parent = clause.group("num")
                title = _clause_title(parent, clause.group("rest"))
                heading = f"{top_heading} > {title}" if top_heading else title
                new_chunk(parent)
            elif para and parent:
                new_chunk(f"{parent}({para.group('num')})")
            elif item and not parent:
                new_chunk(f"({item.group('num')})")
            elif current is None:
                new_chunk("")

            current.lines.append(line)
            current.line_pages.append(page.page_no)

    if current and current.lines:
        chunks.append(current)

    chunks = _split_long(chunks)
    for i, c in enumerate(chunks):
        c.chunk_id = f"{doc_name}#{i}"
    return chunks


def _split_long(chunks: list[Chunk]) -> list[Chunk]:
    """Split chunks above MAX_CHARS into pieces at line boundaries, keeping the label."""
    result = []
    for c in chunks:
        if len(c.text) <= MAX_CHARS:
            result.append(c)
            continue
        piece = None
        for line, page_no in zip(c.lines, c.line_pages):
            if piece is None or len(piece.text) + len(line) > MAX_CHARS:
                piece = Chunk(c.doc_name, c.section, c.heading)
                result.append(piece)
            piece.lines.append(line)
            piece.line_pages.append(page_no)
    return result


if __name__ == "__main__":
    import sys

    from ingestion.cleaning import remove_headers_footers
    from ingestion.parser import parse_pdf

    chunks = chunk_pages(remove_headers_footers(parse_pdf(sys.argv[1])))
    labelled = sum(1 for c in chunks if c.section)
    print(f"{len(chunks)} chunks, {labelled} with a section label")

    wanted = sys.argv[2] if len(sys.argv) > 2 else None
    for c in chunks:
        if wanted is None or c.section == wanted:
            pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"p.{c.page_start}-{c.page_end}"
            print(f"\n[{c.chunk_id}] {c.section or '(no section)'} | {pages}")
            print(f"  heading: {c.heading}")
            print(f"  text:    {c.text[:200]}...")
            if wanted is None:
                break