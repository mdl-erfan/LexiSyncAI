"""Step 6A: merge BM25 and dense rankings with Reciprocal Rank Fusion (RRF)."""
from ingestion.chunker import Chunk
from retrieval.bm25 import BM25Index
from retrieval.dense import DenseIndex

# Constant from the original RRF paper (Cormack et al., 2009).
RRF_K = 60


def rrf_merge(rankings: list[list[Chunk]], k: int = RRF_K, top_n: int = 20) -> list[tuple[Chunk, float]]:
    """Combine several ranked chunk lists into one, using only rank positions."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1 / (k + rank)
            chunks[chunk.chunk_id] = chunk
    best = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_n]
    return [(chunks[cid], scores[cid]) for cid in best]

class HybridIndex:
    def __init__(self, bm25: BM25Index, dense: DenseIndex, candidates: int = 20):
        self.bm25 = bm25
        self.dense = dense
        self.candidates = candidates

    def search(self, query: str, k: int = 20) -> list[tuple[Chunk, float]]:
        """Top k chunks after merging BM25 and dense results with RRF."""
        bm25_hits = [c for c, _ in self.bm25.search(query, k=self.candidates)]
        dense_hits = [c for c, _ in self.dense.search(query, k=self.candidates)]
        return rrf_merge([bm25_hits, dense_hits], top_n=k)