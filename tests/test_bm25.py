from ingestion.chunker import chunk_pages
from retrieval.bm25 import BM25Index
from tests.conftest import make_pages


def gdpr_like_chunks():
    return chunk_pages(make_pages(
        [
            "Article 83",
            "General conditions for imposing administrative fines",
            "4. Infringements of the following provisions shall be subject to fines up to 10 000 000 EUR.",
            "5. Infringements of the following provisions shall be subject to fines up to 20 000 000 EUR.",
        ],
        ["Article 17", "Right to erasure", "1. The data subject shall have the right to obtain the erasure of personal data."],
        ["§ 9 Liability", "9.3 Cap on liability. The cap is [***] per year."],
    ))


def sections(hits):
    return [c.section for c, _ in hits]


def test_exact_amount_finds_the_right_paragraph():
    index = BM25Index(gdpr_like_chunks())
    assert sections(index.search("fines up to 20,000,000 EUR"))[0] == "Article 83(5)"


def test_reference_from_the_section_label_is_searchable():
    # "83(4)" never appears in the chunk text, only in its section label.
    index = BM25Index(gdpr_like_chunks())
    assert sections(index.search("Article 83(4)"))[0] == "Article 83(4)"


def test_clause_number_and_redaction_are_searchable():
    index = BM25Index(gdpr_like_chunks())
    assert sections(index.search("clause 9.3"))[0] == "9.3"
    assert sections(index.search("[***] cap"))[0] == "9.3"


def test_topic_words_find_the_article():
    index = BM25Index(gdpr_like_chunks())
    assert sections(index.search("right to erasure"))[0] == "Article 17(1)"


def test_unrelated_query_returns_nothing():
    assert BM25Index(gdpr_like_chunks()).search("quantum physics") == []


def test_results_are_sorted_and_limited_to_k():
    hits = BM25Index(gdpr_like_chunks()).search("fines infringements EUR", k=2)
    assert len(hits) <= 2
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_empty_index_returns_nothing():
    assert BM25Index([]).search("anything") == []