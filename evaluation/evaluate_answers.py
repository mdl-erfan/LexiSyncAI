"""Step 7g: run answer_question() on every evaluation question and show abstention behaviour."""
import json
from pathlib import Path

from generation.answer import answer_question
from ingestion.pipeline import ingest_many
from retrieval.bm25 import BM25Index
from retrieval.dense import DenseIndex
from retrieval.hybrid import HybridIndex
from retrieval.rerank import RerankedIndex, Reranker

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "evaluation" / "questions.json"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "evaluation" / "results" / "answers.json"


def main() -> None:
    questions = json.loads(QUESTIONS.read_text())
    paths = sorted({RAW / q["doc"] for q in questions})
    chunks = [c for d in ingest_many(paths) for c in d.chunks]
    index = RerankedIndex(HybridIndex(BM25Index(chunks), DenseIndex(chunks)), Reranker())
    doc_names = [p.name for p in paths]

    rows = []
    for q in questions:
        a = answer_question(q["question"], index, doc_names)
        score = "none" if a.top_score is None else f"{a.top_score:.2f}"
        print(f"{q['id']:<9}{q['type']:<13}{a.badge.value:<13}{str(a.abstained):<7}"
              f"{score:>7} {a.latency_s:>6.1f}s  {a.reason}")
        rows.append({
            "id": q["id"], "type": q["type"], "question": q["question"],
            "badge": a.badge.value, "abstained": a.abstained, "reason": a.reason,
            "top_score": a.top_score, "latency_s": round(a.latency_s, 2),
            "answer": a.text, "citations": [c.label for c in a.citations],
            "invalid_citations": a.invalid_citations, "expected": q["answer"],
        })

    unans = [r for r in rows if r["type"] == "unanswerable"]
    ans = [r for r in rows if r["type"] != "unanswerable"]
    print(f"\nAbstained on unanswerable: {sum(r['abstained'] for r in unans)}/{len(unans)}")
    print(f"Wrongly abstained on answerable: {sum(r['abstained'] for r in ans)}/{len(ans)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"Saved to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()