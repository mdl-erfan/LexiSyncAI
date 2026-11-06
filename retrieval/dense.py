"""Step 5C: semantic search over chunks with sentence embeddings (bge-small via fastembed/ONNX)."""
import numpy as np
from fastembed import TextEmbedding

from ingestion.chunker import Chunk

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge models expect this instruction in front of a search query (not in front of passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model: TextEmbedding | None = None


def load_model() -> TextEmbedding:
    """Load the embedding model once and reuse it."""
    global _model
    if _model is None:
        _model = TextEmbedding(MODEL_NAME)
    return _model


def chunk_text(chunk: Chunk) -> str:
    """Text that gets embedded for one chunk: label and heading for context, then the body."""
    return f"{chunk.section} {chunk.heading}\n{chunk.text}"


class DenseIndex:
    def __init__(self, chunks: list[Chunk], model: TextEmbedding | None = None):
        self.chunks = list(chunks)
        self.model = model or load_model()
        vectors = self.model.embed([chunk_text(c) for c in self.chunks], batch_size=32)
        self.embeddings = np.array(list(vectors), dtype=np.float32)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the k most similar (chunk, cosine score) pairs, best first."""
        q = next(iter(self.model.embed([QUERY_PREFIX + query])))
        scores = self.embeddings @ q
        best = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in best]