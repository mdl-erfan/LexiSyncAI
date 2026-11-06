from ingestion.parser import parse_pdf
from tests.conftest import CONTRACT_PAGES, HEADER


def test_one_page_object_per_pdf_page(contract_pdf):
    pages = parse_pdf(contract_pdf)
    assert len(pages) == len(CONTRACT_PAGES)
    assert [p.page_no for p in pages] == [1, 2, 3, 4, 5]
    assert all(p.doc_name == "contract.pdf" for p in pages)


def test_page_label_falls_back_to_page_number(contract_pdf):
    assert [p.page_label for p in parse_pdf(contract_pdf)] == ["1", "2", "3", "4", "5"]


def test_lines_are_in_reading_order(contract_pdf):
    first = parse_pdf(contract_pdf)[0]
    assert first.lines == [HEADER, *CONTRACT_PAGES[0], "Page 1 of 5"]


def test_lines_are_stripped_and_non_empty(contract_pdf):
    for page in parse_pdf(contract_pdf):
        assert all(line and line == line.strip() for line in page.lines)


def test_scanned_page_is_flagged(contract_pdf):
    pages = parse_pdf(contract_pdf)
    assert [p.page_no for p in pages if p.is_scanned] == [5]
    assert pages[4].lines == []


def test_text_pages_are_not_flagged(contract_pdf):
    assert not any(p.is_scanned for p in parse_pdf(contract_pdf)[:4])