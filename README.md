# LexiSyncAI

Evidence-grounded legal document analyst: upload 2–5 legal PDFs, ask questions,
and get plain-language answers with page and clause citations, or an explicit
"insufficient evidence" when the documents don't support an answer.

Final project for Deep Learning & GenAI (DA3), SRH Hochschule Hamburg.

## Status

- [x] Step 1: PDF parser (page text, page labels, scanned-page detection)
- [x] Step 2: Header/footer removal
- [x] Step 3: Clause-aware chunking
- [x] Step 4: Single `ingest_pdf` pipeline function + automated tests
- [ ] Step 5: BM25 (clause-number-aware tokenizer) + dense indexes
- [ ] Step 6: Hybrid retrieval (RRF) + cross-encoder reranker, recall@k per configuration
- [ ] Step 7: Grounded answers with page/clause citations, abstention, confidence badge
- [ ] Step 8: Streamlit UI (upload, detected structure, cited page side by side)
- [ ] Step 9: Evaluation (15+ held-out questions, no-retrieval vs retrieval comparison)
- [ ] Step 10: Packaging (Docker/Makefile, pinned dependencies, fixed seeds)

## Architecture

Ingestion pipeline (implemented):

```
PDF -> parser.py -> cleaning.py -> chunker.py -> list of chunks
       \______________ pipeline.py: ingest_pdf(path) ______________/
```

| Module | What it does |
|---|---|
| `ingestion/parser.py` | Reads every page into clean text lines. Keeps the page number and page label, and flags scanned (image-only) pages. |
| `ingestion/cleaning.py` | Removes headers, footers and page numbers by detecting lines that repeat at the top or bottom of many pages. Works on any document without knowing its wording.