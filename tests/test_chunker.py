from ingestion.chunker import MAX_CHARS, chunk_pages
from tests.conftest import make_pages


def sections(chunks):
    return [c.section for c in chunks]


def test_empty_input_gives_no_chunks():
    assert chunk_pages([]) == []


def test_article_heading_title_is_merged_and_paragraphs_get_labels():
    pages = make_pages(
        [
            "Article 83",
            "General conditions for imposing administrative fines",
            "1. Each supervisory authority shall ensure that fines are effective.",
            "4. Infringements of the following provisions shall be subject to fines up to 10 000 000 EUR.",
        ],
        ["5. Infringements of the following provisions shall be subject to fines up to 20 000 000 EUR."],
    )
    chunks = chunk_pages(pages)
    assert sections(chunks) == ["Article 83(1)", "Article 83(4)", "Article 83(5)"]
    assert chunks[0].heading.startswith("Article 83 General conditions")
    assert (chunks[1].page_start, chunks[2].page_start) == (1, 2)


def test_numbered_clauses_inside_a_section_heading():
    pages = make_pages(
        [
            "§ 9 Liability and Indemnification",
            "9.1 Exclusion of Liability for Sunrise. Unless otherwise stated, Sunrise is not liable.",
            "9.3 Cap on liability. The cap is [***] per year.",
        ]
    )
    chunks = chunk_pages(pages)
    assert sections(chunks) == ["9.1", "9.3"]
    assert "Exclusion of Liability" in chunks[0].heading
    assert chunks[0].heading.startswith("§ 9 Liability and Indemnification")


def test_redaction_markers_are_kept():
    chunks = chunk_pages(make_pages(["9.3 Cap on liability. The cap is [***] per year."]))
    assert "[***]" in chunks[0].text


def test_chunk_spanning_two_pages_has_start_and_end_page():
    pages = make_pages(
        ["7.2 Liability Cap. The total liability of the Supplier is limited"],
        ["to the fees paid in the preceding twelve months.", "7.3 Exceptions. Wilful misconduct."],
    )
    cap, exceptions = chunk_pages(pages)
    assert (cap.section, cap.page_start, cap.page_end) == ("7.2", 1, 2)
    assert (exceptions.section, exceptions.page_start, exceptions.page_end) == ("7.3", 2, 2)


def test_table_of_contents_lines_do_not_become_headings():
    pages = make_pages(
        ["Table of contents", "Article 1 ........ 5", "Article 2 ........ 6"],
        ["Article 1", "Subject-matter", "1. This Regulation lays down rules."],
    )
    chunks = chunk_pages(pages)
    assert sections(chunks) == ["", "Article 1(1)"]
    assert "Article 2 ........ 6" in chunks[0].lines


def test_numbered_items_before_any_heading_are_labelled():
    chunks = chunk_pages(make_pages(["(12) Whereas data protection is a fundamental right."]))
    assert sections(chunks) == ["(12)"]


def test_paragraph_numbers_without_a_heading_get_no_label():
    assert sections(chunk_pages(make_pages(["1. A loose numbered line."]))) == [""]


def test_long_chunks_are_split_and_keep_their_section():
    long_clause = ["9.1 Long clause. " + "x" * 80] + [f"line {i} " + "y" * 90 for i in range(40)]
    chunks = chunk_pages(make_pages(long_clause))
    assert len(chunks) >= 3
    assert set(sections(chunks)) == {"9.1"}
    assert all(len(c.text) <= MAX_CHARS for c in chunks)


def test_chunk_ids_are_unique_and_sequential():
    chunks = chunk_pages(make_pages(["9.1 One. text", "9.2 Two. text", "9.3 Three. text"], doc_name="a.pdf"))
    assert [c.chunk_id for c in chunks] == ["a.pdf#0", "a.pdf#1", "a.pdf#2"]


def test_no_text_is_lost_or_reordered():
    pages = make_pages(
        ["Article 83", "General conditions", "1. First paragraph.", "4. Fourth paragraph."],
        ["Continuation of the fourth paragraph.", "§ 9 Liability", "9.1 Clause one. Some text."],
        ["9.2 Clause two. More text."],
    )
    chunks = chunk_pages(pages)
    assert [line for c in chunks for line in c.lines] == [line for p in pages for line in p.lines]
    assert all(len(c.lines) == len(c.line_pages) for c in chunks)

def test_article_without_numbered_paragraphs_keeps_its_own_label():
    pages = make_pages([
        "Article 4", "Definitions", "For the purposes of this Regulation:",
        "(1) 'personal data' means any information relating to a natural person.",
        "Article 5", "Principles relating to processing of personal data",
        "1. Personal data shall be processed lawfully.",
    ])
    chunks = chunk_pages(pages)
    assert [c.section for c in chunks] == ["Article 4", "Article 5(1)"]
    assert chunks[0].heading == "Article 4 Definitions"