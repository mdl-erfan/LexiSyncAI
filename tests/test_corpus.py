"""Checks on the real test documents. Skipped automatically if data/raw/ lacks the PDFs."""
import pytest

from ingestion.chunker import MAX_CHARS
from ingestion.pipeline import ingest_pdf
from tests.conftest import RAW_DIR

GDPR = "gdpr_2016_679.pdf"
SUNRISE = "vendor_contract.pdf"


def load(name):
    path = RAW_DIR / name
    if not path.exists():
        pytest.skip(f"{path} not found (corpus PDFs are not committed)")
    return ingest_pdf(path)


@pytest.fixture(scope="module")
def gdpr():
    return load(GDPR)


@pytest.fixture(scope="module")
def sunrise():
    return load(SUNRISE)


def pages_of(doc, section):
    return {c.page_start for c in doc.chunks if c.section == section}


def test_gdpr_has_88_pages_and_no_scanned_pages(gdpr):
    assert gdpr.n_pages == 88
    assert gdpr.scanned_pages == []


def test_gdpr_fine_tiers_are_on_the_expected_pages(gdpr):
    assert 82 in pages_of(gdpr, "Article 83(4)")
    assert 83 in pages_of(gdpr, "Article 83(5)")


def test_sunrise_clause_9_3_and_redactions(sunrise):
    assert 22 in pages_of(sunrise, "9.3")
    assert any("[***]" in c.text for c in sunrise.chunks)


@pytest.mark.parametrize("doc_fixture", ["gdpr", "sunrise"])
def test_chunk_invariants_on_real_documents(doc_fixture, request):
    doc = request.getfixturevalue(doc_fixture)
    ids = [c.chunk_id for c in doc.chunks]
    assert len(ids) == len(set(ids))
    for c in doc.chunks:
        assert 1 <= c.page_start <= c.page_end <= doc.n_pages
        assert len(c.text) <= MAX_CHARS or len(c.lines) == 1