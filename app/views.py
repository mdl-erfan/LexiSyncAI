"""Display helpers for the Streamlit app (no state, no model calls)."""
import re
from pathlib import Path

import pymupdf
import streamlit as st

from generation.answer import Answer
from ingestion.pipeline import IngestedDocument

BADGE_COLORS = {"HIGH": "green", "MEDIUM": "orange", "LOW": "yellow", "INSUFFICIENT": "red"}

# Plain-language version of Answer.reason, for the refusal box.
REASON_TEXT = {
    "reranker score below threshold":
        "Layer 1 (retrieval): no passage was relevant enough, so the LLM was not asked.",
    "LLM judged the evidence insufficient":
        "Layer 2 (LLM check): the model read the best passages and judged that they "
        "do not answer the question.",
    "answer had no valid citation":
        "Grounding check: the draft answer could not be tied to any retrieved passage, "
        "even after one retry.",
    "LLM output was not valid JSON":
        "The model's reply could not be read, so the system refused instead of guessing.",
}


# --- Detected structure (8.5) ----------------------------------------------

def _page_list(pages: list[int], limit: int = 10) -> str:
    """'3, 7, 8' or '3, 7, 8, ... (+12 more)' for long lists."""
    shown = ", ".join(str(p) for p in pages[:limit])
    return shown if len(pages) <= limit else f"{shown}, ... (+{len(pages) - limit} more)"


def show_structure(docs: list[IngestedDocument]) -> None:
    """One card per document: size, detected sections, scanned pages flagged."""
    st.subheader("Detected structure")
    for doc in docs:
        s = doc.summary()
        with st.container(border=True):
            st.markdown(f"**{s['doc_name']}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pages", s["pages"])
            c2.metric("Chunks", s["chunks"])
            c3.metric("Sections", s["sections"])
            c4.metric("Scanned pages", len(s["scanned_pages"]))

            if s["scanned_pages"]:
                st.warning(
                    f"Pages {_page_list(s['scanned_pages'])} have no extractable text "
                    "(scanned images). They are flagged and cannot be searched."
                )
            st.caption(
                f"{s['labelled_chunks']} of {s['chunks']} chunks carry a section label; "
                f"{s['removed_lines']} header/footer lines removed."
            )

            with st.expander("Section list"):
                rows: dict[str, dict] = {}          # first chunk of each section label
                for chunk in doc.chunks:
                    if chunk.section and chunk.section not in rows:
                        rows[chunk.section] = {
                            "Section": chunk.section,
                            "Heading": chunk.heading,
                            "Page": chunk.page_start,
                        }
                st.dataframe(list(rows.values()), hide_index=True)


# --- Answer (8.6, 8.7) and refusal (8.8) -----------------------------------

def clean_answer_text(answer: Answer) -> str:
    """Answer text without invented [n] markers, safe to render as markdown."""
    text = answer.text
    for n in answer.invalid_citations:
        text = re.sub(rf"\s*\[{n}\]", "", text)   # also removes the space before it
    return text.replace("$", r"\$")               # '$' would start a math formula


@st.cache_data(show_spinner=False)
def render_page(pdf_path: str, page_no: int, zoom: float = 1.5) -> bytes:
    """One PDF page as PNG bytes. page_no starts at 1, like Chunk.page_start."""
    with pymupdf.open(pdf_path) as doc:
        pix = doc[page_no - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        return pix.tobytes("png")


def show_abstention(answer: Answer, footer: str) -> None:
    """Refusal: badge, which documents were searched, and why it refused."""
    st.badge(answer.badge.value, color=BADGE_COLORS[answer.badge.value])
    st.warning("**Not answered: the documents do not contain enough evidence "
               "to answer this question.**")
    st.markdown("**Documents searched:**")
    for name in answer.documents_searched:
        st.markdown(f"- `{name}`")   # backticks: '_' in file names would turn into italics
    st.caption("Why: " + REASON_TEXT.get(answer.reason, answer.reason))
    if answer.raw_answer:   # only set when the grounding check rejected a draft
        with st.expander("Rejected draft answer"):
            st.text(answer.raw_answer)
    st.caption(footer)


def show_answer(answer: Answer, pdf_paths: dict[str, Path]) -> None:
    """Badge + answer on the left, cited passages and PDF pages on the right."""
    st.subheader("Answer")
    score = "n/a" if answer.top_score is None else f"{answer.top_score:.2f}"
    footer = f"{answer.latency_s:.1f} s · best reranker score {score}"

    if answer.abstained:
        show_abstention(answer, footer)
        return

    label = answer.badge.value
    left, right = st.columns(2, gap="large")
    with left:
        st.badge(label, color=BADGE_COLORS[label])
        st.markdown(clean_answer_text(answer))
        if answer.invalid_citations:
            n = len(answer.invalid_citations)
            st.caption(f"{n} invalid citation marker(s) removed "
                       "(they pointed to passages that were never retrieved).")
        st.caption(footer)
    with right:
        st.markdown("**Cited evidence**")
        for c in answer.citations:
            with st.expander(f"[{c.number}] {c.label}", expanded=True):
                passage_tab, page_tab = st.tabs(["Passage", "PDF page"])
                with passage_tab:
                    st.text(c.text)   # raw text: legal text can contain *, $, [***]
                with page_tab:
                    path = pdf_paths.get(c.doc_name)
                    if path is None:
                        st.caption("Original PDF not available.")
                    else:
                        # a passage can span two pages (e.g. Article 17(1), pp. 43-44)
                        for page_no in range(c.page_start, c.page_end + 1):
                            st.image(render_page(str(path), page_no),
                                     caption=f"{c.doc_name}, p. {page_no}")