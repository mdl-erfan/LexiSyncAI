"""Step 6B: re-score hybrid candidates with a cross-encoder and keep the best."""
from fastembed.rerank.cross_encoder import TextCrossEncoder

from ingestion.chunker import Chunk
from retrieval.dense import chunk_text
from retrieval.hybrid import HybridIndex

MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"

_models: dict[str, TextCrossEncoder] = {}


def load_reranker(model_name: str = MODEL_NAME) -> TextCrossEncoder:
    """Load a cross-encoder once per model name and reuse it."""
    if model_name not in _models:
        _models[model_name] = TextCrossEncoder(model_name)
    return _models[model_name]


class Reranker:
    def __init__(self, model: TextCrossEncoder | None = None, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = model or load_reranker(model_name)

    def rerank(self, query: str, chunks: list[Chunk], top_n: int = 5) -> list[tuple[Chunk, float]]:
        """Score each (query, chunk) pair and return the top_n chunks, best first."""
        if not chunks:
            return []
        scores = list(self.model.rerank(query, [chunk_text(c) for c in chunks]))
        ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [(c, float(s)) for c, s in ranked[:top_n]]
class RerankedIndex:
    """Full retrieval: hybrid (BM25 + dense + RRF) candidates, then cross-encoder reranking."""

    def __init__(self, hybrid: HybridIndex, reranker: Reranker, candidates: int = 20):
        self.hybrid = hybrid
        self.reranker = reranker
        self.candidates = candidates

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        candidates = [c for c, _ in self.hybrid.search(query, k=self.candidates)]
        return self.reranker.rerank(query, candidates, top_n=k)