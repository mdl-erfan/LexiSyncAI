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
- [x] Step 5: BM25 + dense indexes
  - [x] 5A: Legal-aware tokenizer
  - [x] 5B: BM25 index
  - [x] 5C: Dense index (bge-small)
  - [x] 5E: Retrieval tests
- [ ] Step 6: Hybrid retrieval (RRF) + cross-encoder reranker, recall@k per configuration
- [ ] Step 7: Grounded answers with page/clause citations, abstention, confidence badge (Llama 3.2 via Ollama)
- [ ] Step 8: Streamlit UI (upload, detected structure, cited page side by side)
- [ ] Step 9: Evaluation (15+ held-out questions, no-retrieval vs retrieval comparison)
- [ ] Step 10: Packaging (Docker/Makefile, pinned dependencies, fixed seeds)

## Architecture

```
PDF -> parser.py -> cleaning.py -> chunker.py -> chunks -> BM25 index  (keywords)
       \___________ pipeline.py: ingest_pdf(path) ___________/       -> dense index (meaning)
```

### Ingestion (`ingestion/`)

| Module | What it does |
|---|---|
| `parser.py` | Reads every page into clean text lines. Keeps the page number and page label, and flags scanned (image-only) pages. |
| `cleaning.py` | Removes headers, footers and page numbers by detecting lines that repeat at the top or bottom of many pages. Works on any document without knowing its wording. |
| `chunker.py` | Cuts the document into clause-sized chunks. Each chunk carries a section label (`Article 83(4)`, `9.3`), a readable heading, and its start and end page. Merges heading-only pieces, ignores the table of contents, splits chunks over 1,500 characters, and never drops text. |
| `pipeline.py` | Single entry point. `ingest_pdf(path)` runs parse, clean and chunk and returns an `IngestedDocument` (chunks, page count, scanned pages, removed lines, detected sections). `ingest_many(paths)` handles a whole case. Raises clear errors for missing or unreadable files and duplicate file names. |

### Retrieval (`retrieval/`)

| Module | What it does |
|---|---|
| `tokenizer.py` | Turns text into search tokens while keeping legal references intact: amounts (`10 000 000` and `10,000,000` → `10000000`), references (`83(4)`, `6(1)(a)`), clause numbers (`9.3`) and redactions (`[***]` → `[redacted]`). |
| `bm25.py` | Keyword search with BM25 over each chunk's section label, heading and text. Strong on exact references and amounts. |
| `dense.py` | Semantic search with the pretrained `BAAI/bge-small-en-v1.5` embedding model (384-dimensional vectors, cosine similarity). Strong on paraphrased questions. |

Both indexes return an empty result for a document without text (e.g. a fully
scanned PDF) instead of crashing.

**Embedding runtime:** `bge-small` runs as an optimised ONNX export via
[fastembed](https://github.com/qdrant/fastembed) rather than PyTorch. PyTorch no
longer ships releases for Intel Macs (last version: 2.2.2), while ONNX Runtime
does, and it is fast on CPU. The model downloads from the Hugging Face Hub
(~67 MB) on first use and is cached afterwards.

## Setup

```bash
conda create -n lexiSyncAI python=3.12 -y
conda activate lexiSyncAI
pip install -r requirements.txt
```

All dependencies are pinned in `requirements.txt`.

## Test corpus

Place the PDFs in `data/raw/` (they are not committed):

| File | Source |
|---|---|
| `gdpr_2016_679.pdf` | [EUR-Lex, Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj) |
| `vendor_contract.pdf` | SEC EDGAR, Sunrise Communications AG, F-4 (2024-08-26), Exhibit 10.4 |

## Usage

Run all commands from the repository root.

**Ingest** one or more PDFs and print a summary for each:

```bash
python -m ingestion.pipeline data/raw/gdpr_2016_679.pdf data/raw/vendor_contract.pdf
```

**Search** from Python:

```python
from ingestion.pipeline import ingest_pdf
from retrieval.bm25 import BM25Index
from retrieval.dense import DenseIndex

doc = ingest_pdf("data/raw/gdpr_2016_679.pdf")
bm25 = BM25Index(doc.chunks)
dense = DenseIndex(doc.chunks)          # embeds all chunks once

for chunk, score in bm25.search("Article 83(4)", k=3):
    print(chunk.section, chunk.page_start, score)

for chunk, score in dense.search("How much can a company be fined?", k=3):
    print(chunk.section, chunk.page_start, score)
```

Each ingestion stage also has its own command-line check:

```bash
python -m ingestion.parser data/raw/gdpr_2016_679.pdf 82                 # pages, scanned-page detection
python -m ingestion.cleaning data/raw/gdpr_2016_679.pdf 82               # lines removed, page before/after
python -m ingestion.chunker data/raw/gdpr_2016_679.pdf "Article 83(4)"   # chunks with a given label
python -m ingestion.chunker data/raw/vendor_contract.pdf "9.3"
```

## Tests

```bash
python -m pytest
```

78 tests, all passing.

| File | Tests | What it covers |
|---|---|---|
| `test_parser.py` | 6 | Page numbering, page-label fallback, reading order, scanned-page flagging |
| `test_cleaning.py` | 16 | Repeated headers/footers, numbered headers, page-number formats, short documents, no mutation of input |
| `test_chunker.py` | 11 | Section labels, heading merging, page ranges, table of contents, splitting long chunks, no text lost |
| `test_pipeline.py` | 12 | End-to-end on a generated PDF, error handling, determinism, multi-document ingestion |
| `test_corpus.py` | 5 | Real documents: Art. 83(4) on p. 82, Art. 83(5) on p. 83, clause 9.3 on p. 22, `[***]` kept |
| `test_tokenizer.py` | 15 | Amount normalisation, legal references, clause numbers, redactions, stopwords |
| `test_bm25.py` | 7 | Exact amounts, references from section labels, unrelated queries, empty index |
| `test_dense.py` | 6 | Uses a fake model (no download): vector shape, query prefix, ranking, empty index |

The unit tests generate their own small PDFs and use a fake embedding model, so
they run on any machine without the corpus or a model download.
`test_corpus.py` skips itself if the PDFs are not in `data/raw/`.

## Results so far

### Ingestion

| | GDPR | Sunrise contract |
|---|---|---|
| Pages | 88 | 45 |
| Scanned pages | 0 | 0 |
| Header/footer lines removed | 88 | 44 (page numbers) |
| Chunks | 620 | 183 |
| Labelled chunks | 619 | 177 |
| Distinct sections | 545 | 155 |
| Key checks | Art. 83(4) on p. 82, Art. 83(5) on p. 83 | Clause 9.3 on p. 22, `[***]` redactions kept |

The few unlabelled chunks are title and preamble text before the first recital
or clause. Distinct sections are fewer than chunks because long clauses are
split into pieces that keep the same label.

### BM25 vs dense retrieval (GDPR, top 3)

| Query | BM25 | Dense | Better |
|---|---|---|---|
| fines up to 20 000 000 EUR | 83(6), 83(5), 83(4) | (151), 83(6), 83(5) | BM25: exact amount |
| Article 83(4) | **83(4)**, 83(1), 83(8) | 83(9), 83(8), 58(2) | BM25: dense misses the reference |
| How much can a company be fined for ignoring the regulator? | (157), (28), 57(2) | **83(6)**, 83(4), (148) | Dense: paraphrase, no shared keywords |
| quantum physics | no results | 3 chunks, scores 0.52–0.55 | BM25 knows there is no match |

Each method fails where the other succeeds, which motivates hybrid retrieval
(Step 6). Dense scores are compressed (right and wrong hits differ by ~0.01),
so a cross-encoder reranker decides the final order.

### Latency (Intel MacBook Pro, CPU only)

| Operation | Time |
|---|---|
| BM25 index, GDPR (620 chunks) | < 1 s |
| Dense index, GDPR (620 chunks) | 48 s |
| Embedding one question | < 1 s |

## Known issues

- Header detection treats digits as interchangeable (`L 119/82` = `L 119/83`),
  so on short documents, body lines that differ only by a number could be
  mistaken for headers. Not observed on the test corpus.
- Dense indexes are rebuilt on every program start (48 s for GDPR); saving them
  to `data/indexes/` is planned if evaluation runs need it.
- Scanned pages are flagged but not OCR'd (optional bonus feature).

## Project layout

```
ingestion/    PDF parsing, header/footer cleaning, clause-aware chunking, pipeline
retrieval/    Tokenizer, BM25 index, dense index (hybrid + reranker planned)
generation/   Cited answers, abstention, confidence badge (planned)
app/          Streamlit UI (planned)
evaluation/   Question set, recall@k, citation precision, abstention accuracy (planned)
tests/        Automated tests (pytest, 78 tests)
data/raw/     Source PDFs (not committed)
```