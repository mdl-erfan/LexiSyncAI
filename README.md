# LexiSyncAI

Evidence-grounded legal document analyst: upload 2–5 legal PDFs, ask questions,
and get plain-language answers with page and clause citations, or an explicit
"insufficient evidence" when the documents don't support an answer.

## Status

- [x] Step 1: PDF parser (page text, page labels, scanned-page detection)
- [ ] Step 2: Header/footer removal
- [ ] Step 3: Clause-aware chunking
- [ ] Step 4: BM25 + dense indexes
- [ ] Step 5: Hybrid retrieval + reranker
- [ ] Step 6: Grounded answers, citations, abstention
- [ ] Step 7: UI
- [ ] Step 8: Evaluation

## Setup

```bash
conda create -n lexiSyncAI python=3.12 -y
conda activate lexiSyncAI
pip install -r requirements.txt
```

## Test corpus

Place the PDFs in `data/raw/` (they are not committed):

| File | Source |
|---|---|
| `gdpr_2016_679.pdf` | [EUR-Lex, Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj) |
| `vendor_contract.pdf` | SEC EDGAR, Sunrise Communications AG, F-4 (2024-08-26), Exhibit 10.4 |

## Usage

Inspect how a PDF is parsed (optionally pass a page number):

```bash
python -m ingestion.parser data/raw/gdpr_2016_679.pdf 82
```