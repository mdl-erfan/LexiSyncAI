import time

import streamlit as st

from app.state import build_index, get_models, save_uploads, warm_up_llm
from app.views import show_answer, show_structure
from generation.answer import answer_question
from ingestion.pipeline import ingest_many

st.set_page_config(page_title="LexiSyncAI", layout="wide")

st.title("LexiSyncAI")
st.caption("Evidence-grounded legal document analyst: cited answers or an explicit refusal.")

embedder, reranker = get_models()
try:
    warm_up_llm()
    st.sidebar.success("Models ready (Ollama: llama3.2 loaded)")
except Exception as exc:  # Ollama not running, model not pulled, etc.
    st.sidebar.warning(f"Ollama is not ready: {exc}")

# --- 1. Create a case: upload 2-5 PDFs -------------------------------------
st.header("1. Case documents")
uploads = st.file_uploader(
    "Upload 1-5 public legal PDFs (one case)", type="pdf", accept_multiple_files=True
)
ready = 1 <= len(uploads) <= 5
if uploads and not ready:
    st.warning(f"A case needs 1-5 PDFs, you uploaded {len(uploads)}.")

if st.button("Index case", disabled=not ready):
    # Drop the previous case first, so a failure never leaves a half-old, half-new state.
    for key in ("paths", "docs", "index", "documents_searched", "answer"):
        st.session_state.pop(key, None)
    try:
        with st.status("Indexing case...", expanded=True) as status:
            t0 = time.perf_counter()
            st.write("Reading PDFs and cutting them into clauses...")
            paths = save_uploads(uploads)
            docs = ingest_many(paths)
            n_chunks = sum(len(d.chunks) for d in docs)
            st.write(f"Ingestion done in {time.perf_counter() - t0:.1f} s ({n_chunks} chunks)")
            if n_chunks == 0:
                raise ValueError("No extractable text found (scanned PDFs?). Nothing to index.")

            t1 = time.perf_counter()
            st.write("Building keyword + semantic index (the semantic part takes a while)...")
            index = build_index(docs, embedder, reranker)
            st.write(f"Index built in {time.perf_counter() - t1:.1f} s")
            status.update(label="Case indexed", state="complete", expanded=False)
        st.session_state.update(
            paths=paths,
            docs=docs,
            index=index,
            documents_searched=[d.doc_name for d in docs],
        )
    except ValueError as exc:   # unreadable PDF, duplicate names, no text
        st.error(str(exc))

# Shown on every rerun, because the results live in session_state.
if "docs" in st.session_state:
    show_structure(st.session_state["docs"])

# --- 2. Ask a question -----------------------------------------------------
st.header("2. Ask a question")
if "index" not in st.session_state:
    st.info("Index a case first.")
else:
    with st.form("ask_form"):
        question = st.text_input(
            "Question about the documents",
            placeholder="Under the GDPR, what is the maximum fine for the higher tier?",
        )
        asked = st.form_submit_button("Ask")
    if asked and question.strip():
        with st.spinner("Searching, reranking and writing a cited answer..."):
            st.session_state["answer"] = answer_question(
                question.strip(),
                st.session_state["index"],
                st.session_state["documents_searched"],
            )
    if "answer" in st.session_state:
        pdf_paths = {p.name: p for p in st.session_state["paths"]}
        show_answer(st.session_state["answer"], pdf_paths)

st.sidebar.write("Index: ready ✅" if "index" in st.session_state else "Index: not built")