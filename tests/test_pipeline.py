import pytest

from ingestion.chunker import MAX_CHARS
from ingestion.pipeline import ingest_many, ingest_pdf
from tests.conftest import CONTRACT_PAGES, HEADER, make_pdf


def chunk_by_section(doc, section):
    matches = [c for c in doc.chunks if c.section == section]
    assert matches, f"no chunk with section {section!r}; got {doc.sections}"
    return matches[0]


def test_document_summary(contract_pdf):
    doc = ingest_pdf(contract_pdf)
    assert doc.doc_name == "contract.pdf"
    assert doc.n_pages == 5
    assert doc.scanned_pages == [5]
    assert doc.summary()["chunks"] == len(doc.chunks) > 0


def test_headers_footers_and_page_numbers_are_removed(contract_pdf):
    doc = ingest_pdf(contract_pdf)
    assert doc.removed_lines == 8   # 4 headers + 4 "Page i of 5" footers
    text = " ".join(c.text for c in doc.chunks)
    assert HEADER not in text
    assert "Page 1 of 5" not in text


def test_clauses_land_on_the_right_pages(contract_pdf):
    doc = ingest_pdf(contract_pdf)
    assert chunk_by_section(doc, "7.2").page_start == 1
    assert chunk_by_section(doc, "7.3").page_start == 2
    assert chunk_by_section(doc, "9.2").page_start == 3
    assert chunk_by_section(doc, "9.3").page_start == 4
    assert "2,000,000" in chunk_by_section(doc, "7.2").text


def test_redactions_survive_the_whole_pipeline(contract_pdf):
    assert "[***]" in chunk_by_section(ingest_pdf(contract_pdf), "9.2").text


def test_detected_sections_in_document_order(contract_pdf):
    assert ingest_pdf(contract_pdf).sections == ["7.1", "7.2", "7.3", "9.1", "9.2", "9.3"]


def test_every_chunk_has_valid_pages_and_size(contract_pdf):
    doc = ingest_pdf(contract_pdf)
    for c in doc.chunks:
        assert 1 <= c.page_start <= c.page_end <= doc.n_pages
        assert c.page_end not in doc.scanned_pages
        assert len(c.text) <= MAX_CHARS or len(c.lines) == 1


def test_ingestion_is_deterministic(contract_pdf):
    first, second = ingest_pdf(contract_pdf), ingest_pdf(contract_pdf)
    assert [(c.chunk_id, c.section, c.text) for c in first.chunks] == [
        (c.chunk_id, c.section, c.text) for c in second.chunks
    ]


def test_fully_scanned_pdf_gives_no_chunks(tmp_path):
    path = make_pdf(tmp_path / "scan.pdf", [None, None])
    doc = ingest_pdf(path)
    assert doc.chunks == [] and doc.scanned_pages == [1, 2]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_pdf(tmp_path / "nope.pdf")


def test_non_pdf_file_raises_clear_error(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_text("this is not a pdf")
    with pytest.raises(ValueError, match="not a readable PDF"):
        ingest_pdf(bad)


def test_ingest_many_keeps_documents_separate(tmp_path):
    a = make_pdf(tmp_path / "a.pdf", CONTRACT_PAGES[:4], header=HEADER)
    b = make_pdf(tmp_path / "b.pdf", CONTRACT_PAGES[:4], header=HEADER)
    docs = ingest_many([a, b])
    ids = [c.chunk_id for d in docs for c in d.chunks]
    assert len(ids) == len(set(ids))
    assert {c.doc_name for c in docs[1].chunks} == {"b.pdf"}


def test_ingest_many_rejects_duplicate_file_names(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    a = make_pdf(tmp_path / "x" / "same.pdf", CONTRACT_PAGES[:4])
    b = make_pdf(tmp_path / "y" / "same.pdf", CONTRACT_PAGES[:4])
    with pytest.raises(ValueError, match="Duplicate file names"):
        ingest_many([a, b])