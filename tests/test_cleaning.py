import pytest

from ingestion.cleaning import _is_page_number, remove_headers_footers
from ingestion.parser import Page
from tests.conftest import make_pages

WORDS = ["apple", "banana", "cherry", "damson", "elder", "fig"]


def body(i: int) -> list[str]:
    # Differ in words, not digits: the cleaner maps every digit to "#" when comparing lines.
    w = WORDS[i % len(WORDS)]
    return [f"The {w} provision applies to all suppliers", f"Another {w} obligation binds the customer"]


def test_repeated_header_and_footer_are_removed():
    pages = make_pages(*[["ACME Corp", *body(i), "Confidential"] for i in range(1, 6)])
    cleaned = remove_headers_footers(pages)
    for i, page in enumerate(cleaned, start=1):
        assert page.lines == body(i)


def test_header_with_changing_numbers_is_removed():
    # Like the Official Journal header: "L 119/82", "L 119/83", ...
    pages = make_pages(*[[f"L 119/{80 + i}", *body(i)] for i in range(1, 7)])
    for page in remove_headers_footers(pages):
        assert not page.lines[0].startswith("L 119")


@pytest.mark.parametrize("line", ["12", "Page 3", "- 4 -", "5 of 20", "Page 7 of 30"])
def test_page_number_formats_are_recognised(line):
    assert _is_page_number(line)


@pytest.mark.parametrize("line", ["Article 83", "9.3 Governing Law", "EUR 2,000,000", "Section 7"])
def test_real_content_is_not_a_page_number(line):
    assert not _is_page_number(line)


def test_page_number_at_the_edge_is_removed_even_if_unique():
    pages = make_pages(["Opening text here", "12"], ["More body text here", "13"])
    assert [p.lines for p in remove_headers_footers(pages)] == [["Opening text here"], ["More body text here"]]


def test_number_in_the_middle_of_a_page_is_kept():
    lines = ["a first line", "second line", "third line", "fourth line", "12",
             "fifth line", "sixth line", "seventh line", "eighth line"]
    cleaned = remove_headers_footers(make_pages(lines))
    assert "12" in cleaned[0].lines


def test_short_documents_are_not_over_cleaned():
    # Two pages sharing a first line is not enough evidence of a header (MIN_PAGES = 3).
    pages = make_pages(["Shared title line", *body(1)], ["Shared title line", *body(2)])
    assert all(p.lines[0] == "Shared title line" for p in remove_headers_footers(pages))


def test_scanned_pages_are_ignored_and_untouched():
    text_pages = make_pages(*[["ACME Corp", *body(i)] for i in range(1, 5)])
    scanned = Page(doc_name="doc.pdf", page_no=5, page_label="5", lines=[], is_scanned=True)
    cleaned = remove_headers_footers([*text_pages, scanned])
    assert cleaned[4].lines == [] and cleaned[4].is_scanned
    assert all("ACME Corp" not in p.lines for p in cleaned[:4])


def test_input_pages_are_not_mutated():
    pages = make_pages(*[["ACME Corp", *body(i)] for i in range(1, 5)])
    before = [list(p.lines) for p in pages]
    remove_headers_footers(pages)
    assert [p.lines for p in pages] == before