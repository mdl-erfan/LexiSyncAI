# LexiSyncAI

Evidence-grounded legal document analyst: upload 2–5 legal PDFs, ask questions,
and get plain-language answers with page and clause citations, or an explicit
"insufficient evidence" when the documents don't support an answer.

Final project for Deep Learning & GenAI (DA3), SRH Hochschule Hamburg. Due 12 Nov 2026.

## Status

- [x] Step 1: PDF parser (page text, page labels, scanned-page detection)
- [x] Step 2: Header/footer removal
- [x] Step 3: Clause-aware chunking
- [ ] Step 4: Single `ingest_pdf` pipeline function + automated tests
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
```

| Module | What it does |
|---|---|
| `ingestion/parser.py` | Reads every page into clean text lines. Keeps the page number and page label, and flags scanned (image-only) pages. |
| `ingestion/cleaning.py` | Removes headers, footers and page numbers by detecting lines that repeat at the top or bottom of many pages. Works on any document without knowing its wording. |
| `ingestion/chunker.py` | Cuts the document into clause-sized chunks. Each chunk carries a section label (`Article 83(4)`, `9.3`), a readable heading, and its start and end page. Merges heading-only pieces, ignores the table of contents, splits chunks over 1,500 characters, and never drops text. |

Retrieval, generation, UI and evaluation live in `retrieval/`, `generation/`,
`app/` and `evaluation/` and are still to be built.

## Setup

```bash
conda create -n lexiSyncAI python=3.12 -y
conda activate lexiSyncAI
pip install -r requirements.txt
```

Dependencies are pinned in `requirements.txt` (currently `pymupdf==1.28.2`).

## Test corpus

Place the PDFs in `data/raw/` (they are not committed):

| File | Source |
|---|---|
| `gdpr_2016_679.pdf` | [EUR-Lex, Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj) |
| `vendor_contract.pdf` | SEC EDGAR, Sunrise Communications AG, F-4 (2024-08-26), Exhibit 10.4 |

## Usage

Run all commands from the repository root. Each stage has a small command-line
check so you can inspect its output on its own.

**1. Parser:** show parsed pages and scanned-page detection. The optional
argument is a page number to print.

```bash
python -m ingestion.parser data/raw/gdpr_2016_679.pdf 82
```

**2. Header/footer removal:** report how many lines were removed and show a
page before and after. The optional argument is a page number.

```bash
python -m ingestion.cleaning data/raw/gdpr_2016_679.pdf 82
```

**3. Chunker:** print the chunk count and the first chunk. The optional
argument is a section label, which prints every chunk with that label.

```bash
python -m ingestion.chunker data/raw/gdpr_2016_679.pdf
python -m ingestion.chunker data/raw/gdpr_2016_679.pdf "Article 83(4)"
python -m ingestion.chunker data/raw/vendor_contract.pdf "9.3"
```

## Results so far

Ingestion on the two test documents:

| | GDPR (88 pages) | Sunrise contract (45 pages) |
|---|---|---|
| Header/footer lines removed | 88 | 44 (page numbers) |
| Chunks | 620 | 183 |
| Key checks | Art. 83(4) on p. 82, Art. 83(5) on p. 83 | Clause 9.3 on p. 22, `[***]` redactions kept |

## Known issues

- Numbers with space-separated thousands (`10 000 000`) can be split into
  separate tokens. To be handled in the BM25 tokenizer (Step 5).
- No automated tests yet; the checks above were done by hand. Added in Step 4.
- Scanned pages are flagged but not OCR'd (optional bonus feature).

## Project layout

```
ingestion/    PDF parsing, header/footer cleaning, clause-aware chunking
retrieval/    BM25, dense embeddings, hybrid merge, reranker (planned)
generation/   Cited answers, abstention, confidence badge (planned)
app/          Streamlit UI (planned)
evaluation/   Question set, recall@k, citation precision, abstention accuracy (planned)
tests/        Automated tests (planned)
data/raw/     Source PDFs (not committed)
<<<<<<< HEAD
```
=======
```
>>>>>>> 36a2e1f (Update README with full ingestion usage, results and corrected roadmap)
