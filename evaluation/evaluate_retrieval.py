"""Step 6D: retrieval evaluation - recall@k per configuration, latency, and reranker scores for the abstention threshold."""
import json
import time
from pathlib import Path

from ingestion.chunker import Chunk
from ingestion.pipeline import ingest_many
from retrieval.bm25 import BM25Index
from retrieval.dense import DenseIndex
from retrieval.hybrid import HybridIndex
from retrieval.rerank import RerankedIndex, Reranker

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "evaluation" / "questions.json"
RAW_DIR = ROOT / "data" / "raw"
RESULTS = ROOT / "evaluation" / "results" / "retrieval.json"

KS = [1, 3, 5, 10]
RERANKERS = {
    "Hybrid + MiniLM": "Xenova/ms-marco-MiniLM-L-6-v2",
    "Hybrid + bge-reranker": "BAAI/bge-reranker-base",
}


def covers(chunk: Chunk, doc: str, gold: dict, strict: bool) -> bool:
    """Does this chunk contain the gold evidence? strict=True also requires the right section label."""
    on_page = chunk.doc_name == doc and chunk.page_start <= gold["page"] <= chunk.page_end
    return on_page and (not strict or chunk.section == gold["section"])


def recall_at_k(chunks: list[Chunk], question: dict, k: int, strict: bool = False) -> float:
    """Share of the question's gold locations found in the top k chunks (0.5 = one of two for multi-hop)."""
    top = chunks[:k]
    found = sum(any(covers(c, question["doc"], g, strict) for c in top) for g in question["gold"])
    return found / len(question["gold"])

def build_configs(chunks: list[Chunk]) -> dict:
    bm25 = BM25Index(chunks)
    t = time.time()
    dense = DenseIndex(chunks)
    print(f"Dense index: {len(chunks)} chunks in {time.time() - t:.1f}s")
    hybrid = HybridIndex(bm25, dense)
    configs = {"BM25": bm25, "Dense": dense, "Hybrid (RRF)": hybrid}
    for name, model_name in RERANKERS.items():
        configs[name] = RerankedIndex(hybrid, Reranker(model_name=model_name))
    return configs


def evaluate(configs: dict, questions: list[dict]) -> dict:
    results = {}
    for name, index in configs.items():
        rows = []
        for q in questions:
            t = time.time()
            hits = index.search(q["question"], k=max(KS))
            seconds = time.time() - t
            chunks = [c for c, _ in hits]
            row = {"id": q["id"], "type": q["type"], "seconds": seconds,
                   "best_score": hits[0][1] if hits else None,
                   "top": [f"{c.doc_name}|{c.section}|p.{c.page_start}" for c in chunks[:5]]}
            if q["gold"]:
                row["recall"] = {k: recall_at_k(chunks, q, k) for k in KS}
                row["recall_strict"] = {k: recall_at_k(chunks, q, k, strict=True) for k in KS}
            rows.append(row)
        results[name] = rows
    return results

def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def print_tables(results: dict) -> None:
    for key, title in [("recall", "Recall@k (page level)"), ("recall_strict", "Recall@k (page + section)")]:
        print(f"\n{title}, answerable questions only")
        print(f"{'Configuration':24}" + "".join(f"{'@' + str(k):>8}" for k in KS) + f"{'s/question':>12}")
        for name, rows in results.items():
            answerable = [r for r in rows if key in r]
            cells = "".join(f"{mean([r[key][k] for r in answerable]):>8.2f}" for k in KS)
            print(f"{name:24}{cells}{mean([r['seconds'] for r in rows]):>12.2f}")


def threshold_report(results: dict) -> None:
    print("\nBest reranker score per question (abstention threshold)")
    for name in RERANKERS:
        rows = results[name]
        ans = [r["best_score"] for r in rows if r["type"] != "unanswerable"]
        unans = [r["best_score"] for r in rows if r["type"] == "unanswerable"]
        print(f"  {name}: answerable min {min(ans):.2f} / unanswerable max {max(unans):.2f}")
        if max(unans) < min(ans):
            print(f"    separable: a threshold of {(max(unans) + min(ans)) / 2:.2f} splits them")
        else:
            overlap = [r["id"] for r in rows
                       if (r["type"] == "unanswerable" and r["best_score"] >= min(ans))
                       or (r["type"] != "unanswerable" and r["best_score"] <= max(unans))]
            print(f"    overlap, check these questions: {overlap}")


def main() -> None:
    questions = json.loads(QUESTIONS.read_text())
    docs = ingest_many(sorted({RAW_DIR / q["doc"] for q in questions}))
    chunks = [c for d in docs for c in d.chunks]
    print(f"{len(questions)} questions, {len(docs)} documents, {len(chunks)} chunks")

    results = evaluate(build_configs(chunks), questions)
    print_tables(results)
    threshold_report(results)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\nPer-question details saved to {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()