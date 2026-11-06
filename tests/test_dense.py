"""Dense index tests with a tiny fake embedding model: fast, no download, fully predictable."""
import numpy as np

from ingestion.chunker import chunk_pages
from retrieval.dense import QUERY_PREFIX, DenseIndex
from tests.conftest import make_pages

VOCAB = ["fine", "erasure", "liability", "payment"]


class FakeModel:
    """Embeds text as normalised counts of a few topic words. Records every text it sees."""

    def __init__(self):
        self.seen: list[str] = []

    def embed(self, texts, batch_size=32):
        for text in texts:
            self.seen.append(text)
            v = np.array([text.lower().count(w) for w in VOCAB], dtype=float) + 1e-3
            yield v / np.linalg.norm(v)


def sample_chunks():
    return chunk_pages(make_pages(
        ["9.1 Fines. A fine applies to any fine-worthy breach."],
        ["9.2 Erasure. Data erasure on request."],
        ["9.3 Liability. Liability is capped."],
    ))


def test_one_vector_per_chunk():
    index = DenseIndex(sample_chunks(), FakeModel())
    assert index.embeddings.shape == (3, len(VOCAB))
    assert index.embeddings.dtype == np.float32


def test_chunks_are_embedded_with_section_and_heading():
    model = FakeModel()
    DenseIndex(sample_chunks(), model)
    assert model.seen[0].startswith("9.1 9.1 Fines")


def test_query_gets_the_bge_prefix():
    model = FakeModel()
    DenseIndex(sample_chunks(), model).search("liability", k=1)
    assert model.seen[-1] == QUERY_PREFIX + "liability"


def test_most_similar_chunk_comes_first():
    index = DenseIndex(sample_chunks(), FakeModel())
    assert index.search("how big is the liability", k=1)[0][0].section == "9.3"
    assert index.search("erasure request", k=1)[0][0].section == "9.2"


def test_always_returns_k_results_sorted_by_score():
    hits = DenseIndex(sample_chunks(), FakeModel()).search("something unrelated", k=3)
    assert len(hits) == 3   # unlike BM25, dense never returns an empty list
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_empty_index_returns_nothing():
    assert DenseIndex([], FakeModel()).search("anything") == []