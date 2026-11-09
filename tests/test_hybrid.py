import pytest

from ingestion.chunker import Chunk
from retrieval.hybrid import RRF_K, HybridIndex, rrf_merge


def chunk(name: str) -> Chunk:
    return Chunk(doc_name="doc.pdf", section=name, heading=name, chunk_id=f"doc.pdf#{name}")


A, B, C, D = chunk("A"), chunk("B"), chunk("C"), chunk("D")


def ids(hits):
    return [c.section for c, _ in hits]


def test_chunk_found_by_both_lists_wins():
    # B is 2nd and 1st; A is 1st and 3rd; D and C appear in one list only.
    assert ids(rrf_merge([[A, B, C], [B, D, A]])) == ["B", "A", "D", "C"]


def test_rrf_score_formula():
    (_, score), = rrf_merge([[A], [A]])
    assert score == pytest.approx(2 / (RRF_K + 1))


def test_chunk_from_only_one_list_is_kept():
    assert set(ids(rrf_merge([[A], [B]]))) == {"A", "B"}


def test_top_n_limits_the_result():
    assert len(rrf_merge([[A, B, C, D]], top_n=2)) == 2


def test_empty_rankings_give_nothing():
    assert rrf_merge([[], []]) == []


class FakeIndex:
    def __init__(self, ranking):
        self.ranking = ranking
        self.asked_k = None

    def search(self, query, k=5):
        self.asked_k = k
        return [(c, 1.0) for c in self.ranking[:k]]


def test_hybrid_index_merges_both_indexes():
    bm25, dense = FakeIndex([A, B, C]), FakeIndex([B, D, A])
    hits = HybridIndex(bm25, dense, candidates=3).search("question", k=4)
    assert ids(hits) == ["B", "A", "D", "C"]
    assert bm25.asked_k == dense.asked_k == 3


def test_hybrid_index_with_empty_indexes():
    assert HybridIndex(FakeIndex([]), FakeIndex([])).search("question") == []