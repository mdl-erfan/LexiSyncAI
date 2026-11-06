"""Step 5B: keyword search over chunks with BM25."""
from rank_bm25 import BM25Okapi

from ingestion.chunker import Chunk
from retrieval.tokenizer import tokenize


def chunk_tokens(chunk: Chunk) -> list[str]:
    """Tokens for one chunk: its section label, heading and text."""
    return tokenize(f"{chunk.section} {chunk.heading} {chunk.text}")


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = list(chunks)
         # BM25Okapi cannot be built from zero documents (e.g. a fully scanned PDF).
        self.bm25 = BM25Okapi([chunk_tokens(c) for c in self.chunks]) if self.chunks else None

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Return up to k (chunk, score) pairs, best first. Chunks with score 0 are left out."""
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in best if scores[i] > 0]
