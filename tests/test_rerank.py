"""Reranker tests with a fake cross-encoder: fast, no download, fully predictable."""
from ingestion.chunker import Chunk
from retrieval.dense import chunk_text
from retrieval.rerank import RerankedIndex, Reranker


def chunk(name: str, text: str) -> Chunk:
    return Chunk(doc_name="doc.pdf", section=name, heading=name, lines=[text], line_pages=[1],
                 chunk_id=f"doc.pdf#{name}")


FINES = chunk("83(4)", "administrative fines up to 10 000 000 EUR")
ORDER = chunk("83(6)", "fines for non-compliance with an order of the regulator")
ERASURE = chunk("17(1)", "right to erasure of personal data")


class FakeCrossEncoder:
    """Scores a chunk by how many question words it contains. Records what it was given."""

    def __init__(self):
        self.calls = []

    def rerank(self, query, documents, batch_size=64):
        documents = list(documents)
        self.calls.append((query, documents))
        words = query.lower().split()
        for doc in documents:
            yield float(sum(w in doc.lower() for w in words))


def sections(hits):
    return [c.section for c, _ in hits]


def test_best_matching_chunk_comes_first():
    hits = Reranker(FakeCrossEncoder()).rerank("fines regulator order", [FINES, ORDER, ERASURE])
    assert sections(hits) == ["83(6)", "83(4)", "17(1)"]


def test_scores_are_sorted_and_plain_floats():
    hits = Reranker(FakeCrossEncoder()).rerank("fines regulator order", [FINES, ORDER, ERASURE])
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(type(s) is float for s in scores)


def test_top_n_limits_the_result():
    hits = Reranker(FakeCrossEncoder()).rerank("fines", [FINES, ORDER, ERASURE], top_n=2)
    assert len(hits) == 2


def test_model_reads_the_same_text_as_the_dense_index():
    model = FakeCrossEncoder()
    Reranker(model).rerank("fines", [FINES])
    assert model.calls == [("fines", [chunk_text(FINES)])]


def test_no_chunks_means_no_model_call():
    model = FakeCrossEncoder()
    assert Reranker(model).rerank("fines", []) == []
    assert model.calls == []


def test_model_name_is_recorded():
    reranker = Reranker(FakeCrossEncoder(), model_name="BAAI/bge-reranker-base")
    assert reranker.model_name == "BAAI/bge-reranker-base"


class FakeHybrid:
    def __init__(self, ranking):
        self.ranking = ranking
        self.asked_k = None

    def search(self, query, k=20):
        self.asked_k = k
        return [(c, 0.0) for c in self.ranking[:k]]


def test_reranked_index_reorders_hybrid_candidates():
    # Hybrid puts the right chunk last; the reranker must lift it to the top.
    hybrid = FakeHybrid([ERASURE, FINES, ORDER])
    hits = RerankedIndex(hybrid, Reranker(FakeCrossEncoder()), candidates=3).search("order regulator", k=1)
    assert sections(hits) == ["83(6)"]
    assert hybrid.asked_k == 3