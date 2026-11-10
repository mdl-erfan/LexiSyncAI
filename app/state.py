"""Cached, shared resources for the Streamlit app (loaded once per server process)."""
from datetime import datetime
from pathlib import Path

import ollama
import streamlit as st

from generation.llm import MODEL
from ingestion.pipeline import IngestedDocument
from retrieval.bm25 import BM25Index
from retrieval.dense import DenseIndex, load_model
from retrieval.hybrid import HybridIndex
from retrieval.rerank import RerankedIndex, Reranker

# <repo root>/data/raw/uploads  (this file is <repo root>/app/state.py)
UPLOADS_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "uploads"


@st.cache_resource(show_spinner="Loading embedding and reranking models...")
def get_models():
    """Embedding model (bge-small) and reranker (MiniLM), loaded once."""
    return load_model(), Reranker()


@st.cache_resource(show_spinner=f"Loading {MODEL} into memory...")
def warm_up_llm() -> bool:
    """An empty prompt makes Ollama load the model without generating text."""
    ollama.generate(model=MODEL, prompt="", keep_alive="30m")
    return True


def save_uploads(uploaded_files) -> list[Path]:
    """Write Streamlit uploads to data/raw/uploads/case_<timestamp>/, keeping the file names."""
    folder = UPLOADS_DIR / f"case_{datetime.now():%Y%m%d_%H%M%S}"
    folder.mkdir(parents=True, exist_ok=True)   # creates the folders on the first run
    paths = []
    for f in uploaded_files:
        path = folder / Path(f.name).name   # .name drops any directory part
        path.write_bytes(f.getvalue())
        paths.append(path)
    return paths


def build_index(docs: list[IngestedDocument], embedder, reranker) -> RerankedIndex:
    """BM25 + dense over all chunks of the case, merged with RRF, then reranked."""
    chunks = [c for doc in docs for c in doc.chunks]
    bm25 = BM25Index(chunks)
    dense = DenseIndex(chunks, model=embedder)   # the slow step: embeds every chunk
    return RerankedIndex(HybridIndex(bm25, dense), reranker)